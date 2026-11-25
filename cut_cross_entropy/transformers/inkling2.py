"""Inkling2 CCE patch. Adapted from transformers 5.24.1."""

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
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.inkling2.modeling_inkling2 import Inkling2CausalLMOutputWithPast
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
def cce_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    cache: Cache | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs,
) -> Inkling2CausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce_lm_head(
            hidden_states,
            self.lm_head,
            labels,
            _PATCH_OPTS,
            softcap=self.logit_softcap.softcap,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states).float()
        logits = self.logit_softcap(logits)

        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs
            )

    return Inkling2CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        cache=outputs.cache,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


@deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
def cce_forward_multimodal(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    cache: Cache | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
    **kwargs,
) -> tuple | Inkling2CausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        mm_encoder_outputs=mm_encoder_outputs,
        **kwargs,
    )

    hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce_lm_head(
            hidden_states,
            self.lm_head,
            labels,
            _PATCH_OPTS,
            softcap=self.logit_softcap.softcap,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states).float()
        logits = self.logit_softcap(logits)

        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs
            )

    return Inkling2CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        cache=outputs.cache,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_outputs=outputs.image_outputs,
        video_outputs=outputs.video_outputs,
        router_logits=outputs.router_logits,
    )


def patch_inkling2_text(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Inkling2ForCausalLM",
            patch_fn=cce_forward,
        )
        return None

    from transformers.models.inkling2 import modeling_inkling2

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_inkling2.Inkling2ForCausalLM), (
            f"Expected an Inkling2ForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_inkling2.Inkling2ForCausalLM.forward = cce_forward
    return None


def patch_inkling2(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Inkling2ForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.inkling2 import modeling_inkling2

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_inkling2.Inkling2ForConditionalGeneration), (
            f"Expected an Inkling2ForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_inkling2.Inkling2ForConditionalGeneration.forward = cce_forward_multimodal
    return None
