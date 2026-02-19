"""Lfm2Moe CCE patch. Lfm2Moe inherits Mixtral. Adapted from transformers 4.57.0."""

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
    patch_remote_model_class,
)


def patch_lfm2_moe(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
    remote_model_id: str | None = None,
) -> TransformersModelT | None:
    # Set the _PATCH_OPTS in the mixtral patch file
    from . import mixtral as mixtral_patch

    mixtral_patch._PATCH_OPTS = patch_options

    cce_forward = mixtral_patch.cce_forward

    if remote_model_id is not None:
        patch_remote_model_class(
            remote_model_id=remote_model_id,
            class_name="Lfm2MoeForCausalLM",
            patch_fn=cce_forward,
        )
        return None

    from transformers.models.lfm2_moe import modeling_lfm2_moe

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(maybe_model, modeling_lfm2_moe.Lfm2MoeForCausalLM), (
            f"Expected a Lfm2MoeForCausalLM model. Got {type(maybe_model)}."
        )
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_lfm2_moe.Lfm2MoeForCausalLM.forward = cce_forward
    return None
