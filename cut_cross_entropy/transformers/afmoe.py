"""Afmoe CCE patch. Adapted from transformers 5.20.1 and arcee-ai/Trinity-Nano-Preview revision b576b32 (remote code)."""

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
from typing import Callable, Optional, Union

import torch
import transformers
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce_lm_head,
    check_remote_forward_fingerprint,
    load_remote_model_class,
    slice_logits_to_keep,
)

_PATCH_OPTS: PatchOptions | None = None

# sha256 of the remote-code AfmoeForCausalLM.forward source at revision b576b32.
_REMOTE_FORWARD_SHA256 = "06cf0ad12b54b040d4f03d38ec49916e8174c6e6bca5de65f3f5c31ae769b9ef"


def cce_forward_afmoe_remote(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    token_type_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, MoeCausalLMOutputWithPast]:
    outputs = self.model(
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
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    return MoeCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def cce_forward_afmoe(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[tuple, MoeCausalLMOutputWithPast]:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state

    loss = None
    logits = None

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
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    return MoeCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def _select_forward(model_class: type) -> Callable:
    """Pick the CCE forward for a native or remote-code AfmoeForCausalLM class."""
    # A checkpoint without auto_map resolves to the native class even with a remote_model_id.
    if model_class.__module__.startswith("transformers.models.afmoe"):
        return cce_forward_afmoe
    if model_class.forward is not cce_forward_afmoe_remote:
        check_remote_forward_fingerprint(model_class, _REMOTE_FORWARD_SHA256)
    return cce_forward_afmoe_remote


def patch_afmoe(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    """Patch Afmoe for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    # Handle remote model patching
    if remote_model_id is not None:
        model_class = load_remote_model_class(remote_model_id, "AfmoeForCausalLM")
        model_class.forward = _select_forward(model_class)
        return None

    # Handle already instantiated model
    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class = type(maybe_model)
        if model_class.__name__ != "AfmoeForCausalLM":
            raise ValueError(f"Expected AfmoeForCausalLM, got {model_class.__name__}")
        maybe_model.forward = MethodType(_select_forward(model_class), maybe_model)
        return maybe_model

    # Try to import and patch the class directly from transformers
    try:
        from transformers.models.afmoe import modeling_afmoe

        modeling_afmoe.AfmoeForCausalLM.forward = cce_forward_afmoe
    except ImportError:
        raise ImportError(
            "Could not find module to patch. Either ensure remote code is enabled "
            "or check if transformers has the modeling code integrated for this model type"
        )

    return None
