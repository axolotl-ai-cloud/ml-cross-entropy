"""Step3p5 CCE patch. Adapted from stepfun-ai/Step-3.5-Flash revision 8fb8cbc."""

from types import MethodType
from typing import Optional, Union

import torch
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_step3p5(
    self,
    input_ids: torch.LongTensor = None,
    num_patches=None,
    patch_pixel_values=None,
    patch_newline_mask=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    outputs = self.model(
        input_ids=input_ids,
        num_patches=num_patches,
        patch_pixel_values=patch_pixel_values,
        patch_newline_mask=patch_newline_mask,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
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
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_step3p5(
    maybe_model: TransformersModelT,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    """Patch Step3p5 for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    # Handle remote model patching
    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Step3p5ForCausalLM",
            patch_fn=cce_forward_step3p5,
        )
        return None

    # Handle already instantiated model
    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class_name = maybe_model.__class__.__name__
        if model_class_name == "Step3p5ForCausalLM":
            maybe_model.forward = MethodType(cce_forward_step3p5, maybe_model)
            return maybe_model
        else:
            raise ValueError(f"Expected Step3p5ForCausalLM, got {model_class_name}")

    # Try to import and patch the class directly from transformers
    try:
        from transformers.models.step3p5 import modeling_step3p5
        modeling_step3p5.Step3p5ForCausalLM.forward = cce_forward_step3p5
    except ImportError:
        raise ImportError(
            "Could not find module to patch. Either ensure remote code is enabled "
            "or check if transformers has the modeling code integrated for this model type"
        )

    return None
