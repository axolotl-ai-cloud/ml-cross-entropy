"""Lfm2VL CCE patch. Adapted from transformers 4.57.0."""

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
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlCausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_multimodal(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    spatial_shapes: Optional[torch.Tensor] = None,
    pixel_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
):
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        spatial_shapes=spatial_shapes,
        pixel_attention_mask=pixel_attention_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

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

    return Lfm2VlCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )


def patch_lfm2_vl(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Lfm2VlForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.lfm2_vl import modeling_lfm2_vl

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_lfm2_vl.Lfm2VlForConditionalGeneration), (
            f"Expected a Lfm2VlForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_lfm2_vl.Lfm2VlForConditionalGeneration.forward = cce_forward_multimodal
    return None
