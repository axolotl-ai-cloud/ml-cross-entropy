"""Mistral and Mistral3 CCE patch. Adapted from transformers 4.52.4."""

from types import MethodType
from typing import Optional, Tuple, Union

import torch
import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
)
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3CausalLMOutputWithPast,
    KwargsForCausalLM,
)
from transformers.processing_utils import Unpack

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_multimodal(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, Mistral3CausalLMOutputWithPast]:

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        image_sizes=image_sizes,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    loss = None
    logits = None

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )

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
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Mistral3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )


def patch_mistral3(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.mistral3 import modeling_mistral3

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_mistral3.Mistral3ForConditionalGeneration
        ), f"Expected a Mistral3ForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_mistral3.Mistral3ForConditionalGeneration.forward = cce_forward_multimodal
    return None
