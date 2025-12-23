"""OLMO patch. It inherits from Llama. Adapted from transformers 4.57.1."""

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


def patch_olmo(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the llama patch file
    from . import llama as llama_patch

    llama_patch._PATCH_OPTS = patch_options

    cce_forward = llama_patch.cce_forward

    from transformers.models.olmo import modeling_olmo

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_olmo.OlmoForCausalLM), (
            f"Expected a OlmoForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_olmo.OlmoForCausalLM.forward = cce_forward
    return None


def patch_olmo2(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the llama patch file
    from . import llama as llama_patch

    llama_patch._PATCH_OPTS = patch_options

    cce_forward = llama_patch.cce_forward

    from transformers.models.olmo2 import modeling_olmo2

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_olmo2.Olmo2ForCausalLM), (
            f"Expected a Olmo2ForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_olmo2.Olmo2ForCausalLM.forward = cce_forward
    return None


def patch_olmo3(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the llama patch file
    from . import llama as llama_patch

    llama_patch._PATCH_OPTS = patch_options

    cce_forward = llama_patch.cce_forward

    from transformers.models.olmo3 import modeling_olmo3

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_olmo3.Olmo3ForCausalLM), (
            f"Expected a Olmo3ForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_olmo3.Olmo3ForCausalLM.forward = cce_forward
    return None
