"""NemotronH CCE patch. Adapted from nvidia/Nemotron-H-47B-Base-FP8."""

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
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_nemotron_h(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    labels: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
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
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_cache=use_cache,
        cache_position=cache_position,
        attention_mask=attention_mask,
    )

    hidden_states = outputs[0]

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
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_nemotron_h(
    maybe_model: TransformersModelT,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    """Patch NemotronH for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="NemotronHForCausalLM",
            patch_fn=cce_forward_nemotron_h,
        )
        return None

    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class_name = maybe_model.__class__.__name__
        if model_class_name == "NemotronHForCausalLM":
            maybe_model.forward = MethodType(cce_forward_nemotron_h, maybe_model)
            return maybe_model
        else:
            raise ValueError(f"Expected NemotronHForCausalLM, got {model_class_name}")

    # Try to import and patch the class directly from transformers
    try:
        from transformers.models.nemotron_h import modeling_nemotron_h

        modeling_nemotron_h.NemotronHForCausalLM.forward = cce_forward_nemotron_h
    except ImportError:
        raise ImportError(
            "Could not find module to patch. Either ensure remote code is enabled "
            "or check if transformers has the modeling code integrated for this model type"
        )

    return None
