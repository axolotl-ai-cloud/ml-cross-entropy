"""GLM4V_Moe CCE patch. It inherits GLM4V. Adapted from transformers 4.56.1."""

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
    PatchOptions,
    TransformersModelT,
)


def patch_glm4v_moe(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the glm4v patch file
    from . import glm4v as glm4v_patch

    glm4v_patch._PATCH_OPTS = patch_options

    cce_forward_multimodal = glm4v_patch.cce_forward_multimodal

    from transformers.models.glm4v_moe import modeling_glm4v_moe

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_glm4v_moe.Glm4vMoeForConditionalGeneration
        ), f"Expected a Glm4vMoeForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)
        return maybe_model

    modeling_glm4v_moe.Glm4vMoeForConditionalGeneration.forward = cce_forward_multimodal
    return None
