"""Step3p7 CCE patch. Adapted from transformers 5.20.1 and stepfun-ai/Step-3.7-Flash revision 5f62440 (remote code)."""

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

import sys
from types import MethodType
from typing import Callable, Optional, Union

import torch
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce_lm_head,
    check_remote_forward_fingerprint,
    load_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None

# sha256 of the remote-code Step3p7ForConditionalGeneration.forward source at revision 5f62440.
_REMOTE_FORWARD_SHA256 = "642fc6847484a5ba6a5833bcfabc472b11e17258e3f2554bf2e58276469e4e28"


def cce_forward_step3p7_remote(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: Optional[torch.Tensor] = None,
    num_patches=None,
    patch_pixel_values=None,
    patch_newline_mask=None,
    image_embeds: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
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

    # Mirrors upstream: `pixel_values` / `image_embeds` are named params here and are not
    # forwarded, so Step3p7Model._parse_and_validate_image_input bails early and images are
    # dropped (patch_pixel_values alone does not gate it). Kept as-is rather than diverging.
    outputs = self.model(
        input_ids=input_ids,
        num_patches=num_patches,
        patch_pixel_values=patch_pixel_values,
        patch_newline_mask=patch_newline_mask,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state

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
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

    output_cls = getattr(
        sys.modules.get(type(self).__module__),
        "Step3p7CausalLMOutputWithPast",
        CausalLMOutputWithPast,
    )
    return output_cls(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


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


def _select_forward(model_class: type) -> Callable:
    """Pick the CCE forward for a native or remote-code Step3p7ForConditionalGeneration class."""
    # A checkpoint without auto_map resolves to the native class even with a remote_model_id.
    if model_class.__module__.startswith("transformers.models.step3p7"):
        return cce_forward_step3p7
    if model_class.forward is not cce_forward_step3p7_remote:
        check_remote_forward_fingerprint(model_class, _REMOTE_FORWARD_SHA256)
    return cce_forward_step3p7_remote


def patch_step3p7(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    """Patch Step3p7 for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    # Handle remote model patching
    if remote_model_id is not None:
        model_class = load_remote_model_class(remote_model_id, "Step3p7ForConditionalGeneration")
        model_class.forward = _select_forward(model_class)
        return None

    # Handle already instantiated model
    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class = type(maybe_model)
        if model_class.__name__ != "Step3p7ForConditionalGeneration":
            raise ValueError(
                f"Expected Step3p7ForConditionalGeneration, got {model_class.__name__}"
            )
        maybe_model.forward = MethodType(_select_forward(model_class), maybe_model)
        return maybe_model

    # Try to import and patch the class directly from transformers
    try:
        from transformers.models.step3p7 import modeling_step3p7

        modeling_step3p7.Step3p7ForConditionalGeneration.forward = cce_forward_step3p7
    except ImportError:
        raise ImportError(
            "Could not find module to patch. Either ensure remote code is enabled "
            "or check if transformers has the modeling code integrated for this model type"
        )

    return None
