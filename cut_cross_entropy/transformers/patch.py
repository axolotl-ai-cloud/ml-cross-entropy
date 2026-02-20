# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import importlib
import inspect
from typing import overload

from transformers import PretrainedConfig, PreTrainedModel

from cut_cross_entropy.cce_utils import LinearCrossEntropyImpl
from cut_cross_entropy.linear_cross_entropy import LCE_IMPL_DEFAULT

from .utils import PatchOptions, TransformersModelT

AXOLOTL_CCE_FORK = 1

PATCH_FNS = {
    "afmoe": ("cut_cross_entropy.transformers.afmoe", "patch_afmoe"),
    "apertus": ("cut_cross_entropy.transformers.apertus", "patch_apertus"),
    "arcee": ("cut_cross_entropy.transformers.arcee", "patch_arcee"),
    "cohere": ("cut_cross_entropy.transformers.cohere", "patch_cohere"),
    "cohere2": ("cut_cross_entropy.transformers.cohere", "patch_cohere2"),
    "deepseek_v3": ("cut_cross_entropy.transformers.deepseek_v3", "patch_deepseek_v3"),
    "exaone4": ("cut_cross_entropy.transformers.exaone4", "patch_exaone4"),
    "gemma": ("cut_cross_entropy.transformers.gemma", "patch_gemma"),
    "gemma2": ("cut_cross_entropy.transformers.gemma3", "patch_gemma2"),
    "gemma3": ("cut_cross_entropy.transformers.gemma3", "patch_gemma3"),
    "gemma3_text": ("cut_cross_entropy.transformers.gemma3", "patch_gemma3_text"),
    "gemma3n": ("cut_cross_entropy.transformers.gemma3n", "patch_gemma3n"),
    "gemma3n_text": ("cut_cross_entropy.transformers.gemma3n", "patch_gemma3n_text"),
    "glm": ("cut_cross_entropy.transformers.glm4", "patch_glm"),
    "glm4": ("cut_cross_entropy.transformers.glm4", "patch_glm4"),
    "glm4_moe": ("cut_cross_entropy.transformers.glm4", "patch_glm4_moe"),
    "glm4_moe_lite": ("cut_cross_entropy.transformers.glm4_moe_lite", "patch_glm4_moe_lite"),
    "glm_moe_dsa": ("cut_cross_entropy.transformers.glm_moe_dsa", "patch_glm_moe_dsa"),
    "glm46v": ("cut_cross_entropy.transformers.glm46v", "patch_glm46v"),
    "glm4v": ("cut_cross_entropy.transformers.glm4v", "patch_glm4v"),
    "glm_image": ("cut_cross_entropy.transformers.glm_image", "patch_glm_image"),
    "glm4v_moe": ("cut_cross_entropy.transformers.glm4v", "patch_glm4v_moe"),
    "gpt_oss": ("cut_cross_entropy.transformers.gpt_oss", "patch_gpt_oss"),
    "granite": ("cut_cross_entropy.transformers.granite", "patch_granite"),
    "granitemoe": ("cut_cross_entropy.transformers.granitemoe", "patch_granitemoe"),
    "granitemoeshared": ("cut_cross_entropy.transformers.granitemoe", "patch_granitemoeshared"),
    "granitemoehybrid": ("cut_cross_entropy.transformers.granitemoe", "patch_granitemoehybrid"),
    "hunyuan_v1_dense": ("cut_cross_entropy.transformers.hunyuan_v1", "patch_hunyuan_v1_dense"),
    "hunyuan_v1_moe": ("cut_cross_entropy.transformers.hunyuan_v1", "patch_hunyuan_v1_moe"),
    "internvl": ("cut_cross_entropy.transformers.internvl", "patch_internvl"),
    "kimi_linear": ("cut_cross_entropy.transformers.kimi_linear", "patch_kimi_linear"),
    "lfm2": ("cut_cross_entropy.transformers.lfm2", "patch_lfm2"),
    "lfm2_moe": ("cut_cross_entropy.transformers.lfm2_moe", "patch_lfm2_moe"),
    "lfm2_vl": ("cut_cross_entropy.transformers.lfm2_vl", "patch_lfm2_vl"),
    "llama": ("cut_cross_entropy.transformers.llama", "patch_llama"),
    "llama4": ("cut_cross_entropy.transformers.llama4", "patch_llama4"),
    "llava": ("cut_cross_entropy.transformers.llava", "patch_llava"),
    "llama4_text": ("cut_cross_entropy.transformers.llama4", "patch_llama4_text"),
    "ministral": ("cut_cross_entropy.transformers.ministral3", "patch_ministral"),
    "ministral3": ("cut_cross_entropy.transformers.ministral3", "patch_ministral3"),
    "mistral": ("cut_cross_entropy.transformers.mistral", "patch_mistral"),
    "mistral3": ("cut_cross_entropy.transformers.mistral3", "patch_mistral3"),
    "mixtral": ("cut_cross_entropy.transformers.mixtral", "patch_mixtral"),
    "mllama": ("cut_cross_entropy.transformers.mllama", "patch_mllama"),
    "olmo": ("cut_cross_entropy.transformers.olmo3", "patch_olmo"),
    "olmo2": ("cut_cross_entropy.transformers.olmo3", "patch_olmo2"),
    "olmo3": ("cut_cross_entropy.transformers.olmo3", "patch_olmo3"),
    "olmoe": ("cut_cross_entropy.transformers.olmoe", "patch_olmoe"),
    "phi": ("cut_cross_entropy.transformers.phi", "patch_phi"),
    "phi3": ("cut_cross_entropy.transformers.phi3", "patch_phi3"),
    "phi4_multimodal": ("cut_cross_entropy.transformers.phi4_multimodal", "patch_phi4_multimodal"),
    "qwen2": ("cut_cross_entropy.transformers.qwen2", "patch_qwen2"),
    "qwen2_moe": ("cut_cross_entropy.transformers.qwen2_moe", "patch_qwen2_moe"),
    "qwen2_vl": ("cut_cross_entropy.transformers.qwen2_vl", "patch_qwen2_vl"),
    "qwen2_5_vl": ("cut_cross_entropy.transformers.qwen2_5_vl", "patch_qwen2_5_vl"),
    "qwen3": ("cut_cross_entropy.transformers.qwen3", "patch_qwen3"),
    "qwen3_5": ("cut_cross_entropy.transformers.qwen3_5", "patch_qwen3_5"),
    "qwen3_5_vl": ("cut_cross_entropy.transformers.qwen3_5", "patch_qwen3_5_vl"),
    "qwen3_5_moe": ("cut_cross_entropy.transformers.qwen3_5_moe", "patch_qwen3_5_moe"),
    "qwen3_5_moe_vl": ("cut_cross_entropy.transformers.qwen3_5_moe", "patch_qwen3_5_moe_vl"),
    "qwen3_moe": ("cut_cross_entropy.transformers.qwen3_moe", "patch_qwen3_moe"),
    "qwen3_vl": ("cut_cross_entropy.transformers.qwen3_vl", "patch_qwen3_vl"),
    "qwen3_vl_moe": ("cut_cross_entropy.transformers.qwen3_vl", "patch_qwen3_vl_moe"),
    "qwen3_next": ("cut_cross_entropy.transformers.qwen3_next", "patch_qwen3_next"),
    "smollm3": ("cut_cross_entropy.transformers.smollm3", "patch_smollm3"),
    "seed_oss": ("cut_cross_entropy.transformers.seed_oss", "patch_seed_oss"),
    "step3p5": ("cut_cross_entropy.transformers.step3p5", "patch_step3p5"),
    "voxtral": ("cut_cross_entropy.transformers.voxtral", "patch_voxtral"),
}


def _get_patch_fn(model_type: str):
    """Lazy import of patch function.

    Backward compatible: if PATCH_FNS[model_type] is already a callable
    (e.g. set by a downstream library), return it directly. If it's a
    tuple, lazy import it via importlib.
    """
    if model_type not in PATCH_FNS:
        raise RuntimeError(f"Unknown model type {model_type}")

    patch_spec = PATCH_FNS[model_type]

    # Backward compatibility: if already a callable, return directly
    if callable(patch_spec):
        return patch_spec

    # New API: tuple of (module_path, function_name)
    module_path, fn_name = patch_spec

    try:
        module = importlib.import_module(module_path)
        patch_fn = getattr(module, fn_name)
        # Cache the imported function for subsequent calls
        PATCH_FNS[model_type] = patch_fn
        return patch_fn
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"CCE cannot import {model_type}. "
            f"Please ensure your transformers version supports {model_type}. Error: {e}"
        )


@overload
def cce_patch(
    model_type_or_model: str | PretrainedConfig,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    train_only: bool = False,
) -> None: ...


@overload
def cce_patch(
    model_type_or_model: TransformersModelT,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    train_only: bool = False,
) -> TransformersModelT: ...


def cce_patch(
    model_type_or_model: str | TransformersModelT | PretrainedConfig,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    train_only: bool = False,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    if impl not in (v.name.lower() for v in LinearCrossEntropyImpl):
        raise ValueError(f"Unknown {impl=}")

    if isinstance(model_type_or_model, PreTrainedModel):
        if hasattr(model_type_or_model, "config"):
            model_type = model_type_or_model.config.model_type
        else:
            raise ValueError(
                "model_type_or_model is a PreTrainedModel but does not have a config attribute"
            )
    elif isinstance(model_type_or_model, PretrainedConfig):
        model_type = model_type_or_model.model_type
    else:
        model_type = model_type_or_model

    patch_options = PatchOptions(
        impl=impl,
        reduction=reduction,
        filter_eps=filter_eps,
        accum_e_fp32=accum_e_fp32,
        accum_c_fp32=accum_c_fp32,
        filter_e_grad=filter_e_grad,
        filter_c_grad=filter_c_grad,
        train_only=train_only,
    )

    patch_fn = _get_patch_fn(model_type)

    # Check if patch_fn supports remote_model_id parameter
    sig = inspect.signature(patch_fn)
    if "remote_model_id" in sig.parameters:
        return patch_fn(model_type_or_model, patch_options, remote_model_id)
    else:
        return patch_fn(model_type_or_model, patch_options)
