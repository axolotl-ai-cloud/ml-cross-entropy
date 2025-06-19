# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import overload

from transformers import PreTrainedModel, PretrainedConfig

from cut_cross_entropy.cce_utils import LinearCrossEntropyImpl
from cut_cross_entropy.linear_cross_entropy import LCE_IMPL_DEFAULT

from .cohere import patch_cohere, patch_cohere2
from .gemma import patch_gemma
from .gemma3 import patch_gemma2, patch_gemma3, patch_gemma3_text
from .glm4 import patch_glm, patch_glm4
from .llama import patch_llama
from .llama4 import patch_llama4, patch_llama4_text
from .mistral import patch_mistral
from .mistral3 import patch_mistral3
from .mllama import patch_mllama
from .phi import patch_phi
from .phi3 import patch_phi3
from .phi4_multimodal import patch_phi4_multimodal
from .qwen2 import patch_qwen2
from .qwen2_5_vl import patch_qwen2_5_vl
from .qwen2_moe import patch_qwen2_moe
from .qwen2_vl import patch_qwen2_vl
from .qwen3 import patch_qwen3
from .qwen3_moe import patch_qwen3_moe

from .utils import PatchOptions, TransformersModelT

PATCH_FNS = {
    "llama": patch_llama,
    "llama4": patch_llama4,
    "llama4_text": patch_llama4_text,
    "mllama": patch_mllama,
    "phi": patch_phi,
    "phi3": patch_phi3,
    "phi4_multimodal": patch_phi4_multimodal,
    "gemma": patch_gemma,
    "gemma2": patch_gemma2,
    "gemma3": patch_gemma3,
    "gemma3_text": patch_gemma3_text,
    "mistral": patch_mistral,
    "mistral3": patch_mistral3,
    "qwen2": patch_qwen2,
    "qwen2_moe": patch_qwen2_moe,
    "qwen2_vl": patch_qwen2_vl,
    "qwen2_5_vl": patch_qwen2_5_vl,
    "qwen3": patch_qwen3,
    "qwen3_moe": patch_qwen3_moe,
    "cohere": patch_cohere,
    "cohere2": patch_cohere2,
    "glm": patch_glm,
    "glm4": patch_glm4,
}


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

    if model_type in PATCH_FNS:
        return PATCH_FNS[model_type](model_type_or_model, patch_options)
    else:
        raise RuntimeError(f"Unknown model type {model_type}")
