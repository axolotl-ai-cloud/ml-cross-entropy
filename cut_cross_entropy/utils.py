# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import importlib.metadata

import packaging.version
import torch


@torch.compile(fullgraph=True)
def softcapping(logits: torch.Tensor, softcap: float) -> torch.Tensor:
    return torch.tanh(logits / softcap) * softcap


def _handle_eps(filter_eps: float | str | None, dtype: torch.dtype) -> float | None:
    if filter_eps is None:
        return None
    elif isinstance(filter_eps, float):
        return filter_eps
    elif filter_eps == "auto":
        return torch.finfo(dtype).eps / 32
    else:
        raise RuntimeError(f"Unknown eps {filter_eps=}")


def _build_flat_valids(
    targets: torch.Tensor,
    ignore_index: int,
    shift: int,
) -> torch.Tensor | None:
    if shift != 0:
        targets = targets[..., shift:]
    else:
        targets = targets.flatten()

    valids = (targets != ignore_index).nonzero().to(torch.int32)

    if shift == 0:
        assert valids.size(1) == 1
        return valids.squeeze(1) if valids.numel() != targets.numel() else None

    for i in range(targets.ndim - 1):
        valids[:, i] *= targets.stride(i)

    assert targets.stride(-1) == 1

    return valids.sum(1)


def handle_reduction_none(
    batch_shape: torch.Size, valids: torch.Tensor | None, shift: int, loss: torch.Tensor
) -> torch.Tensor:
    if valids is None:
        return loss.view(batch_shape)

    full_loss = loss.new_zeros((batch_shape.numel(),))
    full_loss[(valids + shift) if shift != 0 else valids] = loss

    return full_loss.view(batch_shape)


@functools.cache
def is_package_greater_or_equal(package: str, version: str) -> bool:
    return packaging.version.parse(importlib.metadata.version(package)) >= packaging.version.parse(
        version
    )


@functools.cache
def is_torch_greater_or_equal_2_5() -> bool:
    return is_package_greater_or_equal("torch", "2.5")


def recommend_c_grad_chunk_size(
    num_tokens: int,
    vocab_size: int,
    hidden_size: int,
    *,
    device: torch.device | str | int | None = None,
    target_programs: int | None = None,
    max_scratch_bytes: int = 1 << 30,
) -> int:
    """Suggest an explicit classifier-gradient chunk size for the fixed CCE backward kernel.

    This heuristic balances parallel work against fp32 accumulator memory; it does
    not measure occupancy or guarantee optimal throughput. It does not modify the loss.

    :param num_tokens: Valid prediction tokens in one local microbatch, after shifting
        and masking. Use the total batch token count as an estimate if this is unknown.
    :param vocab_size: Rows in the classifier used by this rank's loss computation.
        For vocabulary parallelism, pass the local shard size; for DDP/FSDP, pass the
        classifier size seen during the forward pass. Do not multiply by world size.
    :param hidden_size: Classifier hidden dimension.
    :param device: CUDA device to inspect, defaulting to the current device.
    :param target_programs: Target number of backward GPU programs. Defaults to eight
        per SM on device. An explicit value avoids querying CUDA device properties.
    :param max_scratch_bytes: Maximum temporary fp32 classifier accumulator size,
        defaulting to 1 GiB. Must accommodate at least one vocabulary tile.
    :return: Power-of-two vocabulary tile count expressed in rows, or zero to use
        the full accumulator when the local classifier fits or num_tokens is zero.
    """
    from cut_cross_entropy.tl_autotune import _cce_backward_best_config

    for name, value, minimum in (
        ("num_tokens", num_tokens, 0),
        ("vocab_size", vocab_size, 1),
        ("hidden_size", hidden_size, 1),
        ("max_scratch_bytes", max_scratch_bytes, 1),
    ):
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if target_programs is not None and (
        not isinstance(target_programs, int) or target_programs < 1
    ):
        raise ValueError("target_programs must be a positive integer")

    if num_tokens == 0:
        return 0

    config = _cce_backward_best_config()
    block_b, block_v = config.kwargs["BLOCK_B"], config.kwargs["BLOCK_V"]
    memory_tiles = max_scratch_bytes // (4 * hidden_size * block_v)
    if memory_tiles < 1:
        raise ValueError("max_scratch_bytes must accommodate at least one vocabulary tile")
    if target_programs is None:
        target_programs = 8 * torch.cuda.get_device_properties(device).multi_processor_count
    token_tiles = (num_tokens + block_b - 1) // block_b
    vocab_tiles = (target_programs + token_tiles - 1) // token_tiles
    occupancy_tiles = 1 << (vocab_tiles - 1).bit_length()
    memory_tiles = 1 << (memory_tiles.bit_length() - 1)
    chunk_size = block_v * min(occupancy_tiles, memory_tiles)
    return chunk_size if chunk_size < vocab_size else 0
