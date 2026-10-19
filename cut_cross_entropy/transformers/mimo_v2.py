"""MiMoV2 CCE patch. Adapted from XiaomiMiMo/MiMo-V2.5 revision 6365158."""

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

# The text-only MiMo-V2.5-Pro remote code shares this class name and model_type; its forward is
# this one minus the multimodal embedding merge, which only runs when multimodal inputs are passed.

from types import MethodType
from typing import Optional, Union

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce_lm_head,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_mimo_v2(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    image_embeds: Optional[torch.Tensor] = None,
    video_pixel_values: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    video_embeds: Optional[torch.Tensor] = None,
    audio_codes: Optional[torch.Tensor] = None,
    audio_embeds: Optional[torch.Tensor] = None,
    **kwargs,
) -> CausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    if inputs_embeds is None and input_ids is not None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if any(
            x is not None
            for x in [
                pixel_values,
                image_embeds,
                video_pixel_values,
                video_embeds,
                audio_codes,
                audio_embeds,
            ]
        ):
            inputs_embeds = self._get_multimodal_embeds(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_embeds=image_embeds,
                video_pixel_values=video_pixel_values,
                video_grid_thw=video_grid_thw,
                video_embeds=video_embeds,
                audio_codes=audio_codes,
                audio_embeds=audio_embeds,
            )
        input_ids = None

    outputs: BaseModelOutputWithPast = self.model(
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

    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce_lm_head(
            hidden_states[:, slice_indices, :],
            self.lm_head,
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
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_mimo_v2(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="MiMoV2ForCausalLM",
            patch_fn=cce_forward_mimo_v2,
        )
        return None

    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class_name = maybe_model.__class__.__name__
        if model_class_name == "MiMoV2ForCausalLM":
            maybe_model.forward = MethodType(cce_forward_mimo_v2, maybe_model)
            return maybe_model
        else:
            raise ValueError(f"Expected MiMoV2ForCausalLM, got {model_class_name}")

    try:
        from transformers.models.mimo_v2 import modeling_mimo_v2

        modeling_mimo_v2.MiMoV2ForCausalLM.forward = cce_forward_mimo_v2
    except ImportError:
        raise ImportError(
            "Could not find module to patch. Either ensure remote code is enabled "
            "or check if transformers has the modeling code integrated for this model type"
        )

    return None
