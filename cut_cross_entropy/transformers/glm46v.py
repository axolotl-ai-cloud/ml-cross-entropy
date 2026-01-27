"""GLM46V CCE patch. GLM46V inherits GLM4V. Adapted from transformers 5.0.0."""

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

import transformers

from cut_cross_entropy.transformers.utils import (
    REMOTE_MODEL_NOT_IMPLEMENTED_ERROR,
    PatchOptions,
    TransformersModelT,
)


def patch_glm46v(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    if remote_model_id is not None:
        raise NotImplementedError(REMOTE_MODEL_NOT_IMPLEMENTED_ERROR.format(model_type="glm46v"))

    # Set the _PATCH_OPTS in the glm4v patch file
    from . import glm4v as glm4v_patch

    glm4v_patch._PATCH_OPTS = patch_options

    cce_forward_multimodal = glm4v_patch.cce_forward_multimodal

    from transformers.models.glm46v import modeling_glm46v

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_glm46v.Glm46VForConditionalGeneration), (
            f"Expected a Glm46VForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_glm46v.Glm46VForConditionalGeneration.forward = cce_forward_multimodal
    return None
