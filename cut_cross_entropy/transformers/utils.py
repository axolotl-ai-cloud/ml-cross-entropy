# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
import transformers

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard
except ImportError:
    try:
        from torch.distributed._tensor import DTensor, Shard
    except ImportError:
        DTensor = None
        Shard = None

from contextlib import contextmanager, nullcontext

import torch.nn as nn

from cut_cross_entropy import VocabParallelOptions, linear_cross_entropy
from cut_cross_entropy.cce_utils import CCEPreset

TransformersModelT = TypeVar("TransformersModelT", bound=transformers.PreTrainedModel)

# Error message for unimplemented remote model loading
REMOTE_MODEL_NOT_IMPLEMENTED_ERROR = (
    "Remote model loading patching not yet implemented for {model_type}. "
    "Please create an issue at https://github.com/axolotl-ai-cloud/axolotl/issues "
    "to request support for this model."
)


@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device,
    therefore creating an empty model. Useful when just initializing the model would blow the available RAM.

    This is a minimal implementation adapted from accelerate.init_empty_weights to avoid the accelerate dependency.
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__.copy()
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to("meta"), **kwargs)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


class CCEKwargs(CCEPreset):
    impl: str
    reduction: str


@dataclass
class PatchOptions:
    impl: str
    reduction: str
    filter_eps: float | str | None
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool
    train_only: bool
    c_grad_chunk_size: int = 0

    def to_kwargs(self) -> CCEKwargs:
        return CCEKwargs(
            impl=self.impl,
            reduction=self.reduction,
            filter_eps=self.filter_eps,
            accum_e_fp32=self.accum_e_fp32,
            accum_c_fp32=self.accum_c_fp32,
            filter_e_grad=self.filter_e_grad,
            filter_c_grad=self.filter_c_grad,
        )

    def use_lce(self, labels: torch.Tensor | None, training: bool) -> bool:
        if labels is None:
            return False

        if not training and self.train_only:
            return False

        return True


def apply_lce(
    e: torch.Tensor,
    c: torch.Tensor,
    labels: torch.Tensor,
    opts: PatchOptions,
    bias: torch.Tensor | None = None,
    softcap: float | None = None,
    shift_labels: torch.Tensor | None = None,
    **loss_kwargs,
) -> torch.Tensor:
    num_items_in_batch = loss_kwargs.get("num_items_in_batch", None)
    cce_kwargs = opts.to_kwargs()
    if num_items_in_batch is not None and cce_kwargs["reduction"] == "mean":
        cce_kwargs["reduction"] = "sum"
    else:
        num_items_in_batch = None

    if isinstance(c, DTensor):
        # Get the device mesh and process group from the DTensor
        device_mesh = c.device_mesh

        vocab_dim = 0  # or whichever dim is vocab-sharded
        process_group = device_mesh.get_group("tp")

        # Get the local shard info
        placement = c.placements[vocab_dim]  # Assuming vocab is sharded on this dim
        if isinstance(placement, Shard):
            # Calculate this rank's vocabulary range
            vocab_size = c.size(vocab_dim)  # this is actually the size of the unsharded tensor

            vocab_parallel_options = VocabParallelOptions.from_vocab(
                vocab_size,
                process_group,
                reduce_e_grad=True,
            )
            cce_kwargs["vocab_parallel_options"] = vocab_parallel_options

        c_local = c.to_local()
    else:
        c_local = c

    # Under DeepSpeed ZeRO-3, lm_head.weight is sharded across ranks.
    # Pass the original nn.Parameter references so the backward pass can
    # re-gather the full tensors via GatheredParameters.
    zero3_params: list[torch.nn.Parameter] = []
    if hasattr(c, "ds_id"):
        zero3_params.append(c)
    if bias is not None and hasattr(bias, "ds_id"):
        zero3_params.append(bias)
    if zero3_params:
        cce_kwargs["zero3_params"] = zero3_params

    if c.dtype == torch.bfloat16 and e.dtype == torch.float32:
        # specifically only handling the case we've seen with DoRA where it outputs float32 when the weights are bfloat16
        e = e.to(c.dtype)

    # Sequence/context parallelism supplies pre-shifted `shift_labels`, which upstream's
    # ForCausalLMLoss uses as-is instead of shifting `labels`. Mirror that: shifting again
    # would train position i to predict token i+2.
    if shift_labels is not None:
        targets, shift = shift_labels, 0
    else:
        targets, shift = labels, True

    loss = linear_cross_entropy(
        e,
        c_local,
        targets.to(e.device),
        bias=bias,
        shift=shift,
        softcap=softcap,
        c_grad_chunk_size=opts.c_grad_chunk_size,
        **cce_kwargs,
    )

    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch

    return loss


def _is_lora_linear(module: nn.Module) -> bool:
    return all(
        hasattr(module, a) for a in ("lora_A", "lora_B", "scaling", "base_layer", "active_adapters")
    )


def _base_weight(base_layer: nn.Module) -> torch.Tensor:
    weight = base_layer.weight
    if type(weight).__name__ in ("Params4bit", "Int8Params") or hasattr(base_layer, "W_q"):
        from peft.utils.integrations import dequantize_module_weight

        weight = dequantize_module_weight(base_layer)
    return weight


def _zero3_gather(params: list[torch.Tensor]):
    if not any(hasattr(p, "ds_id") for p in params):
        return nullcontext()
    from deepspeed.runtime.zero.partition_parameters import GatheredParameters, ZeroParamStatus

    # A parameter DeepSpeed already holds (tied to the embedding, persistent, or prefetched)
    # is still claimed by its submodule; re-gathering it would fail on release.
    ds_params = [
        p for p in params if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]
    if not ds_params:
        return nullcontext()
    return GatheredParameters(ds_params, modifier_rank=None)


def _lora_lm_head_inputs(
    e: torch.Tensor, lm_head: nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Fold the active LoRA adapters of ``lm_head`` into an augmented (e, c, bias) triple.

    ``logits = e @ W^T + sum_i scaling_i * dropout_i(e) @ A_i^T @ B_i^T`` is exactly
    ``[e, z_1, ..., z_k] @ [W, B_1, ..., B_k]^T`` with ``z_i = scaling_i * A_i(dropout_i(e))``,
    so the unmaterialised-logit kernel sees one wider classifier. Gradients reach ``A_i``
    through ``z_i`` and ``B_i`` through the concatenated classifier. Returns ``None`` when
    the module is not a LoRA layer or its adapters are merged/disabled, in which case
    ``lm_head.weight`` already is the effective classifier.
    """
    if not _is_lora_linear(lm_head):
        return None
    if lm_head.disable_adapters:
        if lm_head.merged:
            lm_head.unmerge()
        return None
    if lm_head.merged:
        return None
    adapters = [a for a in lm_head.active_adapters if a in lm_head.lora_A]
    if not adapters:
        return None

    base_layer = lm_head.get_base_layer()
    weight = _base_weight(base_layer)
    if DTensor is not None and isinstance(weight, DTensor):
        raise NotImplementedError(
            "CCE does not support a vocab-parallel (DTensor) lm_head with LoRA adapters."
        )

    variants = getattr(lm_head, "lora_variant", {})
    gather_params: list[torch.Tensor] = [weight]
    if base_layer.bias is not None:
        gather_params.append(base_layer.bias)

    with _zero3_gather(gather_params):
        kdtype = weight.dtype
        if e.dtype == torch.float32 and kdtype in (torch.bfloat16, torch.float16):
            e = e.to(kdtype)
        bias = base_layer.bias
        e_blocks: list[torch.Tensor] = [e]
        c_blocks: list[torch.Tensor] = [weight]

        for name in adapters:
            lora_A = lm_head.lora_A[name]
            lora_B = lm_head.lora_B[name]
            dropout = lm_head.lora_dropout[name]
            scaling = lm_head.scaling[name]
            if getattr(lm_head, "cast_input_dtype_enabled", True):
                x = e.to(lora_A.weight.dtype)
            else:
                x = e
            variant = variants.get(name)

            # Read B through its forward rather than `.weight`: under FSDP2 (axolotl shards
            # lora_A/lora_B/magnitude as their own units) and ZeRO-3 the module call is what
            # unshards the parameter and registers its gradient hooks.
            eye = torch.eye(lora_A.out_features, device=x.device, dtype=x.dtype)
            b_t = lora_B(eye)
            lora_bias = None
            if lora_B.bias is not None:
                lora_bias = lora_B(eye.new_zeros(1, eye.size(0)))[0]
                b_t = b_t - lora_bias
            b_weight = b_t.t()

            if variant is None and getattr(lm_head, "use_dora", {}).get(name):
                raise NotImplementedError("CCE needs peft>=0.16 for DoRA on lm_head.")
            if variant is None:
                e_blocks.append((lora_A(dropout(x)) * scaling).to(kdtype))
                c_blocks.append(b_weight.to(kdtype))
                if lora_bias is not None:
                    lora_bias = lora_bias * scaling
                    bias = lora_bias if bias is None else bias + lora_bias
                continue

            if "Dora" not in type(variant).__name__:
                raise NotImplementedError(
                    f"CCE does not support LoRA variant {type(variant).__name__} on lm_head."
                )
            if lora_bias is not None:
                raise NotImplementedError("CCE does not support DoRA with lora_bias on lm_head.")
            # Mirror peft's DoraLinearVariant.forward: the weight norm is a constant, and the
            # running output (base plus every adapter applied so far) is row-scaled by
            # magnitude / norm before this adapter's LoRA term is added. The scale comes out
            # of the magnitude module's own forward (zero input, unit base_result gives
            # `magnitude / norm - 1`) for the same sharding reason as B above.
            dora = lm_head.lora_magnitude_vector[name]
            unit = torch.ones(1, weight.size(0), device=x.device, dtype=x.dtype)
            if base_layer.bias is not None:
                unit = unit + base_layer.bias.to(x.dtype)
            mag_norm_scale = (
                dora(
                    x.new_zeros(1, x.size(-1)),
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=base_layer,
                    base_result=unit,
                    adapter_name=name,
                )[0]
                + 1
            )[:, None]
            if isinstance(dropout, nn.Identity) or not lm_head.training:
                c_blocks = [(mag_norm_scale * block).to(kdtype) for block in c_blocks]
                xd = x
            else:
                # peft recomputes only the base projection, on the dropped-out input.
                xd = dropout(x)
                e_blocks.append(xd.to(kdtype))
                c_blocks.append(((mag_norm_scale - 1) * weight).to(kdtype))
            e_blocks.append((lora_A(xd) * scaling).to(kdtype))
            c_blocks.append((mag_norm_scale * b_weight).to(kdtype))

        e_aug = torch.cat(e_blocks, dim=-1)
        c_aug = torch.cat(c_blocks, dim=-1)

    return e_aug, c_aug, bias


def apply_lce_lm_head(
    e: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    opts: PatchOptions,
    softcap: float | None = None,
    shift_labels: torch.Tensor | None = None,
    **loss_kwargs,
) -> torch.Tensor:
    """``apply_lce`` over an ``lm_head`` module rather than its weight.

    PEFT's ``LoraLayer.weight`` resolves to the frozen base weight, so passing
    ``lm_head.weight`` silently drops any LoRA adapter on the head. This folds the
    adapters into the classifier (see ``_lora_lm_head_inputs``) and otherwise behaves
    like ``apply_lce(e, lm_head.weight, ..., bias=lm_head.bias)``.
    """
    lora_inputs = _lora_lm_head_inputs(e, lm_head)
    if lora_inputs is None:
        c = lm_head.weight
        bias = getattr(lm_head, "bias", None)
    else:
        e, c, bias = lora_inputs

    return apply_lce(
        e,
        c,
        labels,
        opts,
        bias=bias,
        softcap=softcap,
        shift_labels=shift_labels,
        **loss_kwargs,
    )


def patch_remote_model_class(
    remote_model_id: str,
    class_name: str,
    patch_fn: Callable,
) -> None:
    """
    Load a model class and patch a specific class method.

    Args:
        remote_model_id: The HuggingFace model ID to load remote code from
        class_name: Name of the class to patch (e.g., "KimiLinearForCausalLM")
        patch_fn: Function to patch the class method (e.g., forward function)
    """
    import importlib

    # Load the model configuration. For trust_remote_code models this also
    # triggers download of the remote configuration code.
    model_config = transformers.AutoConfig.from_pretrained(remote_model_id, trust_remote_code=True)

    # Derive the modeling module name from the config.
    parts = model_config.__class__.__module__.split(".")
    parts[-1] = parts[-1].replace("configuration_", "modeling_", 1)
    module_name = ".".join(parts)

    try:
        # Try import from installed transformer package first
        model_class = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        from transformers.dynamic_module_utils import get_class_in_module

        # trust_remote_code model: trigger the remote-code download into the HF
        # modules cache, then resolve the class from there.
        with init_empty_weights():
            transformers.AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)

        # get_class_in_module can be patched downstream (for ex: in Axolotl).
        model_class = get_class_in_module(class_name, module_name)

    # Patch the forward method.
    setattr(model_class, "forward", patch_fn)
