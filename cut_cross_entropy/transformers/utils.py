# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import importlib
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

from contextlib import contextmanager

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

    if c.dtype == torch.bfloat16 and e.dtype == torch.float32:
        # specifically only handling the case we've seen with DoRA where it outputs float32 when the weights are bfloat16
        e = e.to(c.dtype)

    loss = linear_cross_entropy(
        e,
        c_local,
        labels.to(e.device),
        bias=bias,
        shift=True,
        softcap=softcap,
        **cce_kwargs,
    )

    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch

    return loss


def patch_remote_model_class(
    remote_model_id: str,
    class_name: str,
    patch_fn: Callable,
) -> None:
    """
    Load remote model code and patch a specific class method.

    Args:
        remote_model_id: The HuggingFace model ID to load remote code from
        class_name: Name of the class to patch (e.g., "KimiLinearForCausalLM")
        patch_fn: Function to patch the class method (e.g., forward function)
    """
    from transformers.dynamic_module_utils import get_class_in_module

    # Load the remote model configuration to trigger remote code download
    model_config = transformers.AutoConfig.from_pretrained(remote_model_id, trust_remote_code=True)

    # Get the auto class to download modeling code without loading weights
    with init_empty_weights():
        transformers.AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)

    # Derive the module name from the config
    parts = model_config.__class__.__module__.split(".")
    parts[-1] = parts[-1].replace("configuration_", "modeling_", 1)
    module_name = ".".join(parts)

    # Use get_class_in_module. This can be patched downstream (for ex: in Axolotl).
    model_class = get_class_in_module(class_name, module_name)

    # Patch the forward method
    setattr(model_class, "forward", patch_fn)
