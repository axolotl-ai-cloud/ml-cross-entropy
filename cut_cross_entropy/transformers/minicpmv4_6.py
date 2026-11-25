"""MiniCPM-V 4.6, MiniCPM-V 4.7 and MiniCPM-OCR CCE patch. Adapted from transformers 5.24.1."""

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
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MultimodalCausalLMOutputWithPast,
)
from transformers.utils.deprecation import deprecate_kwarg

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce_lm_head,
    patch_remote_model_class,
    slice_logits_to_keep,
)

_PATCH_OPTS: PatchOptions | None = None


@deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
def cce_forward_multimodal(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    target_sizes: torch.IntTensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    target_sizes_videos: torch.IntTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    cache: list[torch.Tensor] | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    downsample_mode: str | None = None,
    mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
    **kwargs,
) -> tuple | MultimodalCausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        target_sizes=target_sizes,
        pixel_values_videos=pixel_values_videos,
        target_sizes_videos=target_sizes_videos,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        mm_encoder_outputs=mm_encoder_outputs,
        downsample_mode=downsample_mode,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = None
    loss = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce_lm_head(
            hidden_states,
            self.lm_head,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    return MultimodalCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        cache=outputs.cache,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_outputs=outputs.image_outputs,
        video_outputs=outputs.video_outputs,
    )


@deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
def cce_forward_multimodal_v4_7(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    target_sizes: torch.IntTensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    target_sizes_videos: torch.IntTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    cache: list[torch.Tensor] | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    downsample_mode: str | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
    **kwargs,
) -> tuple | MultimodalCausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        target_sizes=target_sizes,
        pixel_values_videos=pixel_values_videos,
        target_sizes_videos=target_sizes_videos,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        downsample_mode=downsample_mode,
        mm_token_type_ids=mm_token_type_ids,
        mm_encoder_outputs=mm_encoder_outputs,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = None
    loss = None

    hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce_lm_head(
            hidden_states,
            self.lm_head,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    return MultimodalCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        cache=outputs.cache,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_outputs=outputs.image_outputs,
        video_outputs=outputs.video_outputs,
    )


def patch_minicpmv4_6(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="MiniCPMV4_6ForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.minicpmv4_6 import modeling_minicpmv4_6

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_minicpmv4_6.MiniCPMV4_6ForConditionalGeneration), (
            f"Expected a MiniCPMV4_6ForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_minicpmv4_6.MiniCPMV4_6ForConditionalGeneration.forward = cce_forward_multimodal
    return None


def patch_minicpm_ocr(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="MiniCPMOcrForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.minicpm_ocr import modeling_minicpm_ocr

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_minicpm_ocr.MiniCPMOcrForConditionalGeneration), (
            f"Expected a MiniCPMOcrForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_minicpm_ocr.MiniCPMOcrForConditionalGeneration.forward = cce_forward_multimodal
    return None


def patch_minicpmv4_7(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="MiniCPMV4_7ForConditionalGeneration",
            patch_fn=cce_forward_multimodal_v4_7,
        )
        return None

    from transformers.models.minicpmv4_7 import modeling_minicpmv4_7

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_minicpmv4_7.MiniCPMV4_7ForConditionalGeneration), (
            f"Expected a MiniCPMV4_7ForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal_v4_7, maybe_model)
        return maybe_model

    modeling_minicpmv4_7.MiniCPMV4_7ForConditionalGeneration.forward = cce_forward_multimodal_v4_7
    return None
