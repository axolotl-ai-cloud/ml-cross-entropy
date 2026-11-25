"""Inkling2Loop CCE patch. Inkling2Loop inherits Inkling2. Adapted from transformers 5.24.1."""

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

import functools
from types import MethodType

import transformers
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    Inkling2LoopCausalLMOutputWithPast,
)

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    patch_remote_model_class,
)

from . import inkling2 as inkling2_patch


# Same forward bodies as Inkling2's but a distinct output dataclass (same fields), so re-box.
@functools.wraps(inkling2_patch.cce_forward)
def cce_forward(self, *args, **kwargs) -> Inkling2LoopCausalLMOutputWithPast:
    return Inkling2LoopCausalLMOutputWithPast(**inkling2_patch.cce_forward(self, *args, **kwargs))


@functools.wraps(inkling2_patch.cce_forward_multimodal)
def cce_forward_multimodal(self, *args, **kwargs) -> Inkling2LoopCausalLMOutputWithPast:
    return Inkling2LoopCausalLMOutputWithPast(
        **inkling2_patch.cce_forward_multimodal(self, *args, **kwargs)
    )


def patch_inkling2_loop_text(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    inkling2_patch._PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Inkling2LoopForCausalLM",
            patch_fn=cce_forward,
        )
        return None

    from transformers.models.inkling2_loop import modeling_inkling2_loop

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_inkling2_loop.Inkling2LoopForCausalLM), (
            f"Expected an Inkling2LoopForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_inkling2_loop.Inkling2LoopForCausalLM.forward = cce_forward
    return None


def patch_inkling2_loop(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    inkling2_patch._PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Inkling2LoopForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.inkling2_loop import modeling_inkling2_loop

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_inkling2_loop.Inkling2LoopForConditionalGeneration
        ), f"Expected an Inkling2LoopForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_inkling2_loop.Inkling2LoopForConditionalGeneration.forward = cce_forward_multimodal
    return None
