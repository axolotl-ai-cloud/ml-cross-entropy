"""NemotronH CCE patch. Adapted from transformers 5.20.1 and nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 revision bf77c31 / nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 revision 2dc98e2 (remote code)."""

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

import inspect
import sys
from types import MethodType
from typing import Callable, Optional, Union

import torch
import transformers
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.utils import ModelOutput

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce_lm_head,
    check_remote_forward_fingerprint,
    load_remote_model_class,
    slice_logits_to_keep,
)

_PATCH_OPTS: PatchOptions | None = None

# sha256 of the remote-code NemotronHForCausalLM.forward source, one per remote-code variant.
# self.backbone + cache_params: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 revision bf77c3174f68ad409e1c2aa60daeb46e32d1c606.
_REMOTE_NANO_FORWARD_SHA256 = "dc7a2c7bc204b37f9b68f38af0db3c9b9ed3e8ac3abe953715419fe3346736f9"
# self.model + past_key_values: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 revision 2dc98e2afe4face0e4ce40972a915c45368bd34a.
_REMOTE_SUPER_FORWARD_SHA256 = "d961f2b7868516fc7ed981ef2ba3162b93f6e33b3e212149d6dc2db53d19d79a"


def cce_forward_nemotron_h(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[tuple, MoeCausalLMOutputWithPast]:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs[0]

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
        logits = self.lm_head(hidden_states).float()

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


def cce_forward_nemotron_h_remote_nano(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cache_params=None,
    labels: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, ModelOutput]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )

    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    nemotron_h_outputs = self.backbone(
        input_ids,
        cache_params=cache_params,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_cache=use_cache,
        cache_position=cache_position,
        attention_mask=attention_mask,
    )
    hidden_states = nemotron_h_outputs[0]

    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None

        loss = apply_lce_lm_head(
            hidden_states.to(self.lm_head.weight.dtype),
            self.lm_head,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + nemotron_h_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return sys.modules[type(self).__module__].NemotronHCausalLMOutput(
        loss=loss,
        logits=logits,
        cache_params=nemotron_h_outputs.cache_params,
        hidden_states=nemotron_h_outputs.hidden_states,
        attentions=nemotron_h_outputs.attentions,
    )


def cce_forward_nemotron_h_remote_super(
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
) -> Union[tuple, ModelOutput]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )

    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    nemotron_h_outputs = self.model(
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
    hidden_states = nemotron_h_outputs[0]

    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None

        loss = apply_lce_lm_head(
            hidden_states.to(self.lm_head.weight.dtype),
            self.lm_head,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + nemotron_h_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return sys.modules[type(self).__module__].NemotronHCausalLMOutput(
        loss=loss,
        logits=logits,
        past_key_values=nemotron_h_outputs.past_key_values,
        hidden_states=nemotron_h_outputs.hidden_states,
        attentions=nemotron_h_outputs.attentions,
    )


def _select_forward(model_class: type) -> Callable:
    """Pick the CCE forward for a native or remote-code NemotronHForCausalLM class."""
    if model_class.__module__.startswith("transformers.models.nemotron_h"):
        return cce_forward_nemotron_h
    if model_class.forward in (
        cce_forward_nemotron_h_remote_nano,
        cce_forward_nemotron_h_remote_super,
    ):
        return model_class.forward
    # The remote-code variants differ in backbone attribute and cache kwarg; pick by signature.
    if "cache_params" in inspect.signature(model_class.forward).parameters:
        forward, expected_sha256 = cce_forward_nemotron_h_remote_nano, _REMOTE_NANO_FORWARD_SHA256
    else:
        forward, expected_sha256 = cce_forward_nemotron_h_remote_super, _REMOTE_SUPER_FORWARD_SHA256
    check_remote_forward_fingerprint(model_class, expected_sha256)
    return forward


def patch_nemotron_h(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    """Patch NemotronH for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        model_class = load_remote_model_class(remote_model_id, "NemotronHForCausalLM")
        model_class.forward = _select_forward(model_class)
        return None

    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class = type(maybe_model)
        if model_class.__name__ != "NemotronHForCausalLM":
            raise ValueError(f"Expected NemotronHForCausalLM, got {model_class.__name__}")
        maybe_model.forward = MethodType(_select_forward(model_class), maybe_model)
        return maybe_model

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
