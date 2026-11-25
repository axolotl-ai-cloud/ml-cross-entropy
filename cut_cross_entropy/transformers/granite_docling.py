"""GraniteDocling CCE patch. Adapted from transformers 5.24.1."""

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
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.granite_docling.modeling_granite_docling import (
    GraniteDoclingCausalLMOutputWithPast,
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
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cache: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    tile_fine_mask: Optional[torch.BoolTensor] = None,
    pixel_attention_mask: Optional[torch.BoolTensor] = None,
    mm_encoder_outputs: Optional[dict] = None,
    labels: Optional[torch.LongTensor] = None,
    router_labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> GraniteDoclingCausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it doesn't collide with the explicit return_dict=True below.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        tile_fine_mask=tile_fine_mask,
        pixel_attention_mask=pixel_attention_mask,
        mm_encoder_outputs=mm_encoder_outputs,
        use_cache=use_cache,
        return_dict=True,
        **kwargs,
    )
    router_logits = None
    if outputs.image_outputs is not None:
        router_logits = outputs.image_outputs.router_logits

    hidden_states = outputs[0]
    hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce_lm_head(
            hidden_states / self.config.text_config.logits_scaling,
            self.lm_head,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)
        logits = logits / self.config.text_config.logits_scaling

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )
    if router_labels is not None and router_logits is not None:
        router_labels = router_labels.to(router_logits.device, router_logits.dtype).reshape(-1)
        router_loss = (
            self.config.density_router_loss_weight
            * nn.functional.binary_cross_entropy_with_logits(
                router_logits.reshape(-1), router_labels
            )
        )
        loss = router_loss if loss is None else loss + router_loss

    return GraniteDoclingCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        cache=outputs.cache,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_outputs=outputs.image_outputs,
        router_logits=router_logits,
    )


def patch_granite_docling(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="GraniteDoclingForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.granite_docling import modeling_granite_docling

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_granite_docling.GraniteDoclingForConditionalGeneration
        ), f"Expected a GraniteDoclingForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_granite_docling.GraniteDoclingForConditionalGeneration.forward = cce_forward_multimodal
    return None
