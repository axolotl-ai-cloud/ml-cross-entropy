"""Distributed gradient checks for chunked classifier accumulation."""

import copy
import socket
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn as mp_spawn
from torch.nn.parallel import DistributedDataParallel as DDP

from cut_cross_entropy import VocabParallelOptions, linear_cross_entropy


def relative_error(actual, expected):
    return (
        (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    ).item()


def check(actual, expected, tolerance=0.01):
    error = relative_error(actual, expected)
    assert error < tolerance, f"relative error {error} >= {tolerance}"
    assert torch.isfinite(actual).all()
    return error


def vocabulary_parallel():
    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for filtering in (None, "auto"):
            for reduce_e in (False, True):
                torch.manual_seed(42)
                e = (torch.randn(2, 257, 128, device="cuda", dtype=dtype) / 4).requires_grad_()
                c = torch.randn(4099, 128, device="cuda", dtype=dtype).requires_grad_()
                labels = torch.randint(0, 4099, (2, 257), device="cuda")
                labels[:, 7::13] = -100
                vp = VocabParallelOptions.from_vocab(4099, reduce_e_grad=reduce_e)
                shard = c[vp.start : vp.stop].detach().clone().requires_grad_()
                expected = None
                for c_chunk in (0, 256):
                    loss = linear_cross_entropy(
                        e,
                        shard,
                        labels,
                        shift=1,
                        filter_eps=filtering,
                        accum_e_fp32=True,
                        accum_c_fp32=True,
                        c_grad_chunk_size=c_chunk,
                        vocab_parallel_options=vp,
                    )
                    ge, gc = torch.autograd.grad(loss, (e, shard))
                    if expected is None:
                        expected = (loss.detach(), ge.clone(), gc.clone())
                    result = {
                        "mode": "vocab_parallel",
                        "dtype": str(dtype),
                        "filter": filtering,
                        "reduce_e": reduce_e,
                        "c_chunk": c_chunk,
                        "loss_error": check(loss, expected[0]),
                        "e_error_vs_full_accum": check(ge, expected[1]),
                        "c_error_vs_full_accum": check(gc, expected[2]),
                    }
                    full_loss = linear_cross_entropy(
                        e,
                        c,
                        labels,
                        shift=1,
                        filter_eps=filtering,
                        accum_e_fp32=True,
                        accum_c_fp32=True,
                    )
                    full_ge, full_gc = torch.autograd.grad(full_loss, (e, c))
                    summed_ge = ge.float()
                    if not reduce_e:
                        dist.all_reduce(summed_ge)
                    result.update(
                        loss_error_vs_unsharded=check(loss, full_loss),
                        e_error_vs_unsharded=check(summed_ge, full_ge),
                        c_error_vs_unsharded=check(gc, full_gc[vp.start : vp.stop]),
                    )
                    results.append(result)
    return results


class TinyModel(torch.nn.Module):
    def __init__(self, c_chunk=0):
        super().__init__()
        self.projection = torch.nn.Linear(64, 128, bias=False)
        self.classifier = torch.nn.Parameter(torch.randn(4099, 128) / 8)
        self.c_chunk = c_chunk

    def forward(self, x, labels):
        e = self.projection(x).tanh()
        return linear_cross_entropy(
            e,
            self.classifier,
            labels,
            shift=1,
            accum_e_fp32=True,
            accum_c_fp32=True,
            c_grad_chunk_size=self.c_chunk,
        )


def data_parallel(rank, mode, c_chunk):
    torch.manual_seed(42)
    base = TinyModel().cuda()
    candidate = copy.deepcopy(base)
    candidate.c_chunk = c_chunk
    if mode == "ddp":
        base, candidate = [DDP(m.bfloat16(), device_ids=[rank]) for m in (base, candidate)]
    else:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        for model in (base, candidate):
            fully_shard(
                model,
                reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
    models = (base, candidate)
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
    result = {"mode": mode, "c_chunk": c_chunk, "steps": []}
    for step in range(2):
        torch.manual_seed(1000 + 10 * step + rank)
        batches = [
            (
                torch.randn(2, 257, 64, device="cuda", dtype=torch.bfloat16),
                torch.randint(0, 4099, (2, 257), device="cuda"),
            )
            for _ in range(2)
        ]
        for _, labels in batches:
            labels[:, 7::13] = -100
        losses = []
        for model, optimizer in zip(models, optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for microbatch, (x, labels) in enumerate(batches):
                if mode == "fsdp2":
                    model.set_requires_gradient_sync(microbatch == 1)
                context = model.no_sync() if mode == "ddp" and microbatch == 0 else nullcontext()
                with context:
                    loss = model(x, labels) / 2
                    loss.backward()
                total_loss += loss.detach()
            losses.append(total_loss)
        grad_errors = []
        for a, b in zip(base.parameters(), candidate.parameters(), strict=True):
            ga, gb = a.grad, b.grad
            if mode == "fsdp2":
                ga, gb = ga.full_tensor(), gb.full_tensor()
            grad_errors.append(check(gb, ga, 0.01))
        for optimizer in optimizers:
            optimizer.step()
        weight_errors = []
        for a, b in zip(base.parameters(), candidate.parameters(), strict=True):
            if mode == "fsdp2":
                a, b = a.full_tensor(), b.full_tensor()
            weight_errors.append(check(b, a, 0.001))
        result["steps"].append(
            {
                "loss_error": check(losses[1], losses[0]),
                "gradient_errors": grad_errors,
                "weight_errors": weight_errors,
            }
        )
    return result


def _run_distributed(rank: int, world_size: int, port: int, mode: str):
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
    )
    try:
        if mode == "vocab_parallel":
            vocabulary_parallel()
        else:
            data_parallel(rank, mode, 256)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires two CUDA devices")
@pytest.mark.skipif(not dist.is_nccl_available(), reason="Test requires NCCL")
@pytest.mark.parametrize("mode", ["vocab_parallel", "ddp", "fsdp2"])
def test_chunked_distributed(mode: str):
    if mode == "fsdp2":
        import torch.distributed.fsdp as fsdp

        if not hasattr(fsdp, "fully_shard"):
            pytest.skip("Test requires the public FSDP2 API")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp_spawn(_run_distributed, args=(2, port, mode), nprocs=2, join=True)
