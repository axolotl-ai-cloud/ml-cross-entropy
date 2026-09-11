"""InternVL CCE patch. Adapted from transformers 5.17."""

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
from transformers.models.internvl.modeling_internvl import InternVLCausalLMOutputWithPast

from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    patch_remote_model_class,
)

from . import llava as llava_patch


# InternVL's forward body is identical to Llava's, but it returns its own output dataclass
# (same fields, unrelated type), so re-box rather than duplicating the forward.
@functools.wraps(llava_patch.cce_forward)
def cce_forward(self, *args, **kwargs) -> InternVLCausalLMOutputWithPast:
    outputs = llava_patch.cce_forward(self, *args, **kwargs)
    return InternVLCausalLMOutputWithPast(**outputs)


def patch_internvl(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the llava patch file
    llava_patch._PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="InternVLForConditionalGeneration",
            patch_fn=cce_forward,
        )
        return None

    from transformers.models.internvl import modeling_internvl

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_internvl.InternVLForConditionalGeneration), (
            f"Expected a InternVLForConditionalGeneration model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_internvl.InternVLForConditionalGeneration.forward = cce_forward
    return None
