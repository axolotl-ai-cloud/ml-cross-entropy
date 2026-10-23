"""Step3p7 CCE patch. Adapted from transformers 5.20.1."""

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

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce_lm_head,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_step3p7(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_local: Optional[torch.FloatTensor] = None,
    num_local_patches: Optional[Union[list[int], torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    mm_encoder_outputs: Optional[dict] = None,
    **kwargs,
):
    from transformers.models.step3p7.modeling_step3p7 import Step3p7CausalLMOutputWithPast

    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_local=pixel_values_local,
        num_local_patches=num_local_patches,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        mm_encoder_outputs=mm_encoder_outputs,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None

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

    return Step3p7CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        router_logits=outputs.router_logits,
    )


def patch_step3p7(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    """Patch Step3p7 for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    from transformers.models.step3p7 import modeling_step3p7

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_step3p7.Step3p7ForConditionalGeneration), (
            f"Expected a Step3p7ForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_step3p7, maybe_model)
        return maybe_model

    modeling_step3p7.Step3p7ForConditionalGeneration.forward = cce_forward_step3p7
    return None
