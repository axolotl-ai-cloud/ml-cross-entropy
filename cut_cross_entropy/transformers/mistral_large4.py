"""MistralLarge4 CCE patch. Adapted from transformers 5.21.0.dev0."""

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
from typing import Optional, Tuple, Union

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import slice_logits_to_keep
from transformers.models.mistral_large4.modeling_mistral_large4 import (
    MistralLarge4CausalLMOutputWithPast,
)

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce_lm_head,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_multimodal(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    image_sizes: Optional[torch.Tensor] = None,
    mm_encoder_outputs: Optional[dict[str, BaseModelOutputWithPooling]] = None,
    **kwargs,
) -> Union[Tuple, MistralLarge4CausalLMOutputWithPast]:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        image_sizes=image_sizes,
        mm_encoder_outputs=mm_encoder_outputs,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce_lm_head(
            hidden_states,
            self.lm_head,
            labels,
            _PATCH_OPTS,
            softcap=self.final_logit_softcapping,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

        # Logits softcapping
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    return MistralLarge4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        image_hidden_states=outputs.image_hidden_states,
    )


def patch_mistral_large4(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="MistralLarge4ForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.mistral_large4 import modeling_mistral_large4

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_mistral_large4.MistralLarge4ForConditionalGeneration
        ), f"Expected a MistralLarge4ForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_mistral_large4.MistralLarge4ForConditionalGeneration.forward = cce_forward_multimodal
    return None
