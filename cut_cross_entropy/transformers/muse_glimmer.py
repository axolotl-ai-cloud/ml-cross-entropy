"""MuseGlimmer CCE patch. Adapted from transformers 5.16.0.dev0."""

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

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerCausalLMOutputWithPast,
)

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_multimodal(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs,
) -> tuple | MuseGlimmerCausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
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

    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        # scale hidden_states by output_multiplier in-place of logits; the tanh softcap is
        # applied inside apply_lce, giving the same `T * tanh(logits * mult / T)`
        loss = apply_lce(
            hidden_states[:, slice_indices, :] * self.config.text_config.output_multiplier,
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            softcap=self.config.text_config.final_logit_softcapping,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # MuseGlimmer pre-scales logits by `output_multiplier` before the Gemma-style tanh softcap.
        # Together with `final_logit_softcapping = T`, this gives `T * tanh(logits * mult / T)`.
        logits = logits * self.config.text_config.output_multiplier
        logits = logits / self.config.text_config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.text_config.final_logit_softcapping

        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

    return MuseGlimmerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )


def patch_muse_glimmer(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="MuseGlimmerForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.muse_glimmer import modeling_muse_glimmer

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_muse_glimmer.MuseGlimmerForConditionalGeneration), (
            f"Expected a MuseGlimmerForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_muse_glimmer.MuseGlimmerForConditionalGeneration.forward = cce_forward_multimodal
    return None
