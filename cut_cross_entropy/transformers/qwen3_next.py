"""Qwen3_next CCE patch. It inherits Mixtral. Adapted from transformers Qwen3_next PR."""

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


def patch_qwen3_next(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    if remote_model_id is not None:
        raise NotImplementedError(REMOTE_MODEL_NOT_IMPLEMENTED_ERROR.format(model_type="qwen3_next"))
    
    # Set the _PATCH_OPTS in the mixtral patch file
    from . import mixtral as mixtral_patch

    mixtral_patch._PATCH_OPTS = patch_options

    cce_forward = mixtral_patch.cce_forward

    from transformers.models.qwen3_next import modeling_qwen3_next

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_qwen3_next.Qwen3NextForCausalLM), (
            f"Expected a Qwen3NextForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_qwen3_next.Qwen3NextForCausalLM.forward = cce_forward
    return None
