# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import torch.nn.functional as F

from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.utils import (
    _build_flat_valids,
    handle_reduction_none,
    softcapping,
)
from cut_cross_entropy.vocab_parallel import (
    VocabParallelOptions,
    vocab_parallel_torch_compile_lce_apply,
)


@torch.compile(fullgraph=True)
def torch_compile_linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    softcap: float | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    e2: torch.Tensor | None = None,
    c2: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = e @ c.T
    if e2 is not None:
        assert c2 is not None
        logits = logits + e2 @ c2.T

    if bias is not None:
        logits = logits + bias

    if softcap is not None:
        logits = softcapping(logits, softcap)

    loss = F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction=reduction)

    return loss


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def torch_compile_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    vocab_parallel_options: VocabParallelOptions | None = None,
    e2: torch.Tensor | None = None,
    c2: torch.Tensor | None = None,
) -> torch.Tensor:
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)

    orig_b_size = targets.size()
    e = e.contiguous()
    targets = targets.contiguous()

    shift = int(shift)
    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()
    if e2 is not None:
        e2 = e2.contiguous().flatten(0, -2)

    if valids is not None:
        e = e[valids]
        if e2 is not None:
            e2 = e2[valids]
        targets = targets[(valids + shift) if shift != 0 else valids]

    if vocab_parallel_options is None:
        loss = torch_compile_linear_cross_entropy_apply(
            e,
            c,
            targets,
            bias,
            softcap,
            ignore_index=ignore_index,
            reduction=reduction,
            e2=e2,
            c2=c2,
        )
    else:
        loss = vocab_parallel_torch_compile_lce_apply(
            vocab_parallel_options, e, c, targets, bias, softcap, reduction, e2=e2, c2=c2
        )

    if reduction == "none":
        loss = handle_reduction_none(orig_b_size, valids, shift, loss)

        if shift != 0:
            loss = loss[..., shift:]

    return loss
