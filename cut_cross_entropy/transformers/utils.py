# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import TypeVar

import torch
import transformers
from torch.distributed.tensor import DTensor, Shard

from cut_cross_entropy import VocabParallelOptions, linear_cross_entropy
from cut_cross_entropy.cce_utils import CCEPreset

TransformersModelT = TypeVar("TransformersModelT", bound=transformers.PreTrainedModel)


class CCEKwargs(CCEPreset):
    impl: str
    reduction: str


@dataclass
class PatchOptions:
    impl: str
    reduction: str
    filter_eps: float | str | None
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool
    train_only: bool

    def to_kwargs(self) -> CCEKwargs:
        return CCEKwargs(
            impl=self.impl,
            reduction=self.reduction,
            filter_eps=self.filter_eps,
            accum_e_fp32=self.accum_e_fp32,
            accum_c_fp32=self.accum_c_fp32,
            filter_e_grad=self.filter_e_grad,
            filter_c_grad=self.filter_c_grad,
        )

    def use_lce(self, labels: torch.Tensor | None, training: bool) -> bool:
        if labels is None:
            return False

        if not training and self.train_only:
            return False

        return True


def apply_lce(
    e: torch.Tensor,
    c: torch.Tensor,
    labels: torch.Tensor,
    opts: PatchOptions,
    bias: torch.Tensor | None = None,
    softcap: float | None = None,
    **loss_kwargs,
) -> torch.Tensor:
    num_items_in_batch = loss_kwargs.get("num_items_in_batch", None)
    cce_kwargs = opts.to_kwargs()
    if num_items_in_batch is not None and cce_kwargs["reduction"] == "mean":
        cce_kwargs["reduction"] = "sum"
    else:
        num_items_in_batch = None

    if isinstance(c, DTensor):
        # Get the device mesh and process group from the DTensor
        device_mesh = c.device_mesh

        vocab_dim = 0  # or whichever dim is vocab-sharded
        process_group = device_mesh.get_group("tp")

        # Get the local shard info
        placement = c.placements[vocab_dim]  # Assuming vocab is sharded on this dim
        if isinstance(placement, Shard):
            # Calculate this rank's vocabulary range
            vocab_size = c.size(vocab_dim)  # this is actually the size of the unsharded tensor

            vocab_parallel_options = VocabParallelOptions.from_vocab(
                vocab_size,
                process_group,
                reduce_e_grad=True,
            )
            cce_kwargs["vocab_parallel_options"] = vocab_parallel_options

        c_local = c.to_local()
    else:
        c_local = c

    if c.dtype == torch.bfloat16 and e.dtype == torch.float32:
        # specifically only handling the case we've seen with DoRA where it outputs float32 when the weights are bfloat16
        e = e.to(c.dtype)

    loss = linear_cross_entropy(
        e,
        c_local,
        labels.to(e.device),
        bias=bias,
        shift=True,
        softcap=softcap,
        **cce_kwargs,
    )

    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch

    return loss
