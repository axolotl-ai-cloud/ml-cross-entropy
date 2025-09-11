# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import overload

from transformers import PretrainedConfig, PreTrainedModel

from cut_cross_entropy.cce_utils import LinearCrossEntropyImpl
from cut_cross_entropy.linear_cross_entropy import LCE_IMPL_DEFAULT

from .arcee import patch_arcee
from .cohere import patch_cohere, patch_cohere2
from .deepseek_v3 import patch_deepseek_v3
from .gemma import patch_gemma
from .gemma3 import patch_gemma2, patch_gemma3, patch_gemma3_text
from .gemma3n import patch_gemma3n, patch_gemma3n_text
from .glm4 import patch_glm, patch_glm4, patch_glm4_moe
from .gpt_oss import patch_gpt_oss
from .granite import patch_granite
from .granitemoe import patch_granitemoe
from .llama import patch_llama
from .llama4 import patch_llama4, patch_llama4_text
from .mistral import patch_mistral
from .mistral3 import patch_mistral3
from .mixtral import patch_mixtral
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
from .smollm3 import patch_smollm3
from .utils import PatchOptions, TransformersModelT
from .voxtral import patch_voxtral

try:
    from .seed_oss import patch_seed_oss
except ImportError:
    patch_seed_oss = None

try:
    from .apertus import patch_apertus
except ImportError:
    patch_apertus = None

try:
    from .hunyuan_v1 import patch_hunyuan_v1_dense, patch_hunyuan_v1_moe
except ImportError:
    patch_hunyuan_v1_dense = None
    patch_hunyuan_v1_moe = None

try:
    from .glm4v import patch_glm4v, patch_glm4v_moe
except ImportError:
    patch_glm4v = None
    patch_glm4v_moe = None

try:
    from .qwen3_next import patch_qwen3_next
except ImportError:
    patch_qwen3_next = None

AXOLOTL_CCE_FORK = 1

PATCH_FNS = {
    "apertus": patch_apertus,
    "arcee": patch_arcee,
    "cohere": patch_cohere,
    "cohere2": patch_cohere2,
    "deepseek_v3": patch_deepseek_v3,
    "gemma": patch_gemma,
    "gemma2": patch_gemma2,
    "gemma3": patch_gemma3,
    "gemma3_text": patch_gemma3_text,
    "gemma3n": patch_gemma3n,
    "gemma3n_text": patch_gemma3n_text,
    "glm": patch_glm,
    "glm4": patch_glm4,
    "glm4_moe": patch_glm4_moe,
    "glm4v": patch_glm4v,
    "glm4v_moe": patch_glm4v_moe,
    "gpt_oss": patch_gpt_oss,
    "granite": patch_granite,
    "granitemoe": patch_granitemoe,
    "hunyuan_v1_dense": patch_hunyuan_v1_dense,
    "hunyuan_v1_moe": patch_hunyuan_v1_moe,
    "llama": patch_llama,
    "llama4": patch_llama4,
    "llama4_text": patch_llama4_text,
    "mistral": patch_mistral,
    "mistral3": patch_mistral3,
    "mixtral": patch_mixtral,
    "mllama": patch_mllama,
    "phi": patch_phi,
    "phi3": patch_phi3,
    "phi4_multimodal": patch_phi4_multimodal,
    "qwen2": patch_qwen2,
    "qwen2_moe": patch_qwen2_moe,
    "qwen2_vl": patch_qwen2_vl,
    "qwen2_5_vl": patch_qwen2_5_vl,
    "qwen3": patch_qwen3,
    "qwen3_moe": patch_qwen3_moe,
    "qwen3_next": patch_qwen3_next,
    "smollm3": patch_smollm3,
    "seed_oss": patch_seed_oss,
    "voxtral": patch_voxtral,
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
        if PATCH_FNS[model_type] is None:
            raise ValueError(
                "CCE cannot import the related modeling class."
                f"Please ensure your transformers version support {model_type}"
            )

        return PATCH_FNS[model_type](model_type_or_model, patch_options)
    else:
        raise RuntimeError(f"Unknown model type {model_type}")
