"""Afmoe CCE patch."""

from types import MethodType
from typing import Optional, Union

import torch
import transformers
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_afmoe(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    token_type_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, MoeCausalLMOutputWithPast]:
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state

    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None

        loss = apply_lce(
            hidden_states,
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    else:
        # Only compute necessary logits
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    return MoeCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def patch_afmoe(
    maybe_model: TransformersModelT,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    """Patch Afmoe for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    # Handle remote model patching
    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="AfmoeForCausalLM",
            patch_fn=cce_forward_afmoe,
        )
        return None

    # Handle already instantiated model
    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class_name = maybe_model.__class__.__name__
        if model_class_name == "AfmoeForCausalLM":
            maybe_model.forward = MethodType(cce_forward_afmoe, maybe_model)
            return maybe_model
        else:
            raise ValueError(f"Expected AfmoeForCausalLM, got {model_class_name}")

    # Try to import and patch the class directly from transformers
    try:
        from transformers.models.afmoe import modeling_afmoe
        modeling_afmoe.AfmoeForCausalLM.forward = cce_forward_afmoe
    except ImportError:
        raise ImportError(
            "Could not find module to patch. Either ensure remote code is enabled "
            "or check if transformers has the modeling code integrated for this model type"
        )

    return None
