"""Mistral CCE patch. Mistral inherits Llama. Adapted from transformers 4.52.4."""

from types import MethodType

import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
)


def patch_mistral(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the llama patch file
    from . import llama as llama_patch

    llama_patch._PATCH_OPTS = patch_options

    cce_forward = llama_patch.cce_forward

    from transformers.models.mistral import modeling_mistral

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_mistral.MistralForCausalLM
        ), f"Expected a MistralForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_mistral.MistralForCausalLM.forward = cce_forward
    return None
