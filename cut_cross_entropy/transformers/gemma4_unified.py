"""Gemma4Unified (text and multimodal) CCE patch. Adapted from transformers 5.10.1."""

# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import Optional, Union

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedCausalLMOutputWithPast,
)

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Gemma4UnifiedCausalLMOutputWithPast:
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    loss = None
    logits = None

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states[:, slice_indices, :],
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            softcap=getattr(self.config, "final_logit_softcapping", None),
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    return Gemma4UnifiedCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        shared_kv_states=outputs.shared_kv_states,
    )


def cce_forward_multimodal(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_position_ids: Optional[torch.LongTensor] = None,
    video_position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Gemma4UnifiedCausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it doesn't collide with the explicit return_dict=True below.
    lm_kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        return_dict=True,
        **lm_kwargs,
    )

    hidden_states = outputs.last_hidden_state
    loss = None
    logits = None

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states[:, slice_indices, :],
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            softcap=getattr(self.config.get_text_config(), "final_logit_softcapping", None),
            **lm_kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if (
            final_logit_softcapping := self.config.get_text_config().final_logit_softcapping
        ) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        if labels is not None:
            loss = self.loss_function(
                logits, labels, self.config.get_text_config().vocab_size, **lm_kwargs
            )

    return Gemma4UnifiedCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
        shared_kv_states=outputs.shared_kv_states,
    )


def patch_gemma4_unified_text(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Gemma4UnifiedForCausalLM",
            patch_fn=cce_forward,
        )
        return None

    from transformers.models.gemma4_unified import modeling_gemma4_unified

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_gemma4_unified.Gemma4UnifiedForCausalLM), (
            f"Expected a Gemma4UnifiedForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_gemma4_unified.Gemma4UnifiedForCausalLM.forward = cce_forward
    return None


def patch_gemma4_unified(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Gemma4UnifiedForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.gemma4_unified import modeling_gemma4_unified

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration
        ), f"Expected a Gemma4UnifiedForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration.forward = cce_forward_multimodal
    return None
