"""Phi-4 Multimodal CCE patch. Adapted from transformers 4.52.4."""

from types import MethodType
from typing import Optional, Union

import torch
import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

_PATCH_OPTS: PatchOptions | None = None

def cce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    image_pixel_values: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[torch.LongTensor] = None,
    image_attention_mask=None,
    audio_input_features: Optional[torch.FloatTensor] = None,
    audio_embed_sizes=None,
    audio_attention_mask=None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> CausalLMOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        image_pixel_values=image_pixel_values,
        image_sizes=image_sizes,
        image_attention_mask=image_attention_mask,
        audio_input_features=audio_input_features,
        audio_embed_sizes=audio_embed_sizes,
        audio_attention_mask=audio_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    
    loss = None
    logits = None

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states[:, slice_indices, :],
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def patch_phi4_multimodal(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.phi4_multimodal import modeling_phi4_multimodal

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_phi4_multimodal.Phi4MultimodalForCausalLM
        ), f"Expected a Phi4MultimodalForCausalLM model. Got {type(maybe_model)}."

        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model


    modeling_phi4_multimodal.Phi4MultimodalForCausalLM.forward = cce_forward
    return None