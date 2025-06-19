"""Phi CCE patch. Phi inherits Llama. Adapted from transformers 4.52.4."""

from types import MethodType

import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
)


def patch_phi(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the llama patch file
    from . import llama as llama_patch

    llama_patch._PATCH_OPTS = patch_options

    cce_forward = llama_patch.cce_forward

    from transformers.models.phi import modeling_phi

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_phi.PhiForCausalLM
        ), f"Expected a PhiForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_phi.PhiForCausalLM.forward = cce_forward
    return None
