"""Kimi Linear CCE patch. Adapted from moonshotai/Kimi-Linear-48B-A3B-Instruct revision fd1de63."""

import importlib
from types import MethodType
from typing import List, Optional

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
)

try:
    from accelerate import init_empty_weights
except ImportError:
    raise ImportError("We require `accelerate` package to patch kimi_linear.")

_PATCH_OPTS: PatchOptions | None = None


def cce_forward_kimi(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    generation_mode: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
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
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    if generation_mode:
        hidden_states = hidden_states[:, -1:]

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
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_kimi_linear(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, str):
        # Load the remote model configuration to trigger remote code download
        model_config = AutoConfig.from_pretrained(maybe_model, trust_remote_code=True)

        # Load model with empty weights to import the modeling module
        with init_empty_weights():
            AutoModelForCausalLM.from_pretrained(maybe_model, trust_remote_code=True)

        # Get the modeling module
        parts = model_config.__class__.__module__.split(".")
        parts[-1] = parts[-1].replace("configuration_", "modeling_", 1)
        module_name = ".".join(parts)
        modeling_kimi = importlib.import_module(module_name)

        # Patch the forward method of the class
        if hasattr(modeling_kimi, "KimiLinearForCausalLM"):
            modeling_kimi.KimiLinearForCausalLM.forward = cce_forward_kimi

        return None

    elif isinstance(maybe_model, transformers.PreTrainedModel):
        # Patch an already instantiated model
        model_class_name = maybe_model.__class__.__name__
        if model_class_name == "KimiLinearForCausalLM":
            maybe_model.forward = MethodType(cce_forward_kimi, maybe_model)
            return maybe_model
        else:
            raise ValueError(f"Expected KimiLinearForCausalLM, got {model_class_name}")

    elif isinstance(maybe_model, transformers.PretrainedConfig):
        # Config is already loaded
        if maybe_model.model_type == "kimi_linear":
            # Import the modeling module using the config
            parts = maybe_model.__class__.__module__.split(".")
            parts[-1] = parts[-1].replace("configuration_", "modeling_", 1)
            module_name = ".".join(parts)
            modeling_kimi = importlib.import_module(module_name)

            if hasattr(modeling_kimi, "KimiLinearForCausalLM"):
                modeling_kimi.KimiLinearForCausalLM.forward = cce_forward_kimi

            return None

    return None
