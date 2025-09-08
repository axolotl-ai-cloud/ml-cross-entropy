"""InternVL_Chat CCE patch. Adapted from InternVL3_5 remote modeling 9bb6a56."""

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
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)

_PATCH_OPTS: PatchOptions | None = None


def cce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    defer_logits_calculation: bool = False,
    **kwargs,
) -> CausalLMOutputWithPast:
    outputs: BaseModelOutputWithPast = self.model(
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

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )
    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states[:, slice_indices, :],
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            **kwargs,
        )
    elif _PATCH_OPTS is not None and defer_logits_calculation:
        # defer logits calculation to the Chat forward
        logits = hidden_states[:, slice_indices, :]
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def cce_multimodal_forward(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_flags: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    image_flags = image_flags.squeeze(-1)
    input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

    vit_embeds = self.extract_feature(pixel_values)
    vit_embeds = vit_embeds[image_flags == 1]
    # vit_batch_size = pixel_values.shape[0]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == self.img_context_token_id
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
    except:
        vit_embeds = vit_embeds.reshape(-1, C)
        # print(
        #     f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
        #     f"vit_embeds.shape={vit_embeds.shape}"
        # )
        n_token = min(selected.sum(), vit_embeds.size(0))
        input_embeds[selected][:n_token] = (
            input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]
        )

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        defer_logits_calculation=True,
    )

    hidden_states = outputs.last_hidden_state

    loss = None
    logits = None

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states,
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
        )
    else:
        logits = self.lm_head(hidden_states)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_internvl_chat(
    maybe_model: TransformersModelT,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    """Patch InternVLChat for CCE."""
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    # Handle remote model patching
    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="InternVLChatModel",
            patch_fn=cce_multimodal_forward,
        )

        # Load config to determine language model architecture
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(remote_model_id, trust_remote_code=True)

        # Determine text config
        if hasattr(model_config, "llm_config"):
            text_config = getattr(model_config, "llm_config")
        elif hasattr(model_config, "get_text_config"):
            text_config = model_config.get_text_config()
        else:
            raise ValueError("No text config found in model config.")

        # Patch the language model forward based on architecture
        architecture: str = text_config.architectures[0]
        if architecture == "LlamaForCausalLM":
            from transformers.models.llama import modeling_llama

            modeling_llama.LlamaForCausalLM.forward = cce_forward
        elif architecture == "Qwen2ForCausalLM":
            from transformers.models.qwen2 import modeling_qwen2

            modeling_qwen2.Qwen2ForCausalLM.forward = cce_forward
        elif architecture == "Qwen3MoeForCausalLM":
            from transformers.models.qwen3_moe import modeling_qwen3_moe

            modeling_qwen3_moe.Qwen3MoeForCausalLM.forward = cce_forward
        elif architecture == "Qwen3ForCausalLM":
            from transformers.models.qwen3 import modeling_qwen3

            modeling_qwen3.Qwen3ForCausalLM.forward = cce_forward
        else:
            raise NotImplementedError(f"Unsupported architecture in InternVL_Chat: {architecture}")

        return None

    # Handle already instantiated model
    if isinstance(maybe_model, transformers.PreTrainedModel):
        model_class_name = maybe_model.__class__.__name__
        if model_class_name == "InternVLChatModel":
            maybe_model.forward = MethodType(cce_multimodal_forward, maybe_model)
            # Also patch the language model forward
            maybe_model.language_model.forward = MethodType(cce_forward, maybe_model.language_model)
            return maybe_model
        else:
            raise ValueError(f"Expected InternVLChatModel, got {model_class_name}")

    return None
