"""GLM46V CCE patch. Adapted from transformers 5.15."""

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
from transformers.models.glm46v.modeling_glm46v import Glm46VCausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    patch_remote_model_class,
)

from . import glm4v as glm4v_patch


# GLM46V's forward body matches GLM4V's but returns its own output dataclass (same fields,
# unrelated type), so re-box rather than duplicating the forward.
@functools.wraps(glm4v_patch.cce_forward_multimodal)
def cce_forward_multimodal(self, *args, **kwargs) -> Glm46VCausalLMOutputWithPast:
    outputs = glm4v_patch.cce_forward_multimodal(self, *args, **kwargs)
    return Glm46VCausalLMOutputWithPast(**outputs)


def patch_glm46v(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the glm4v patch file
    glm4v_patch._PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Glm46VForConditionalGeneration",
            patch_fn=cce_forward_multimodal,
        )
        return None

    from transformers.models.glm46v import modeling_glm46v

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_glm46v.Glm46VForConditionalGeneration), (
            f"Expected a Glm46VForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_glm46v.Glm46VForConditionalGeneration.forward = cce_forward_multimodal
    return None
