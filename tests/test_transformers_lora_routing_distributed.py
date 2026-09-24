"""Routed lm_head LoRA under DDP and FSDP2 with lora_A / lora_B sharded as their own units."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn as mp_spawn

pytest.importorskip("peft")

ROW_MAP = {0: "a", 1: None, 2: "b"}


def _rank_ids(rank):
    # rank 1 never selects "b": its lora_A / lora_B forwards must still run in step
    return torch.tensor([2, 1, 0] if rank == 0 else [0, 1, 0], device="cuda")


def _worker(rank, world_size, port, mode):
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from test_transformers_lora_routing import (
        _opts,
        make_head,
        make_inputs,
        reference_loss,
        rel,
    )

    from cut_cross_entropy.transformers.utils import apply_lce_lm_head

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )
    try:
        head = make_head()
        # reference grads = mean over ranks of each rank's dense routed grads
        ref_grads = {}
        ref_losses = []
        for r in range(world_size):
            e, labels = make_inputs(seed=100 + r)
            loss = reference_loss(head, e, labels, _rank_ids(r), ROW_MAP)
            loss.backward()
            ref_losses.append(loss.detach())
            for name, p in head.named_parameters():
                if p.requires_grad and p.grad is not None:
                    ref_grads[name] = ref_grads.get(name, 0) + p.grad / world_size
            head.zero_grad()

        # the loss is called from inside the root module's forward, as the patched
        # CausalLM forwards do, so the root unit holds the base weight unsharded
        class _Model(torch.nn.Module):
            def __init__(self, lm_head):
                super().__init__()
                self.lm_head = lm_head

            def forward(self, e, labels, ids):
                return apply_lce_lm_head(
                    e, self.lm_head, labels, _opts(), adapter_ids=ids, adapter_map=ROW_MAP
                )

        model = _Model(head)
        if mode == "fsdp2":
            from torch.distributed.fsdp import fully_shard

            for name in ("a", "b"):
                fully_shard(head.lora_A[name])
                fully_shard(head.lora_B[name])
            fully_shard(model)

        e, labels = make_inputs(seed=100 + rank)
        loss = model(e, labels, _rank_ids(rank))
        assert rel(loss, ref_losses[rank]) < 2e-2, (rank, loss.item(), ref_losses[rank].item())
        loss.backward()
        if mode == "ddp":
            # plain data parallel: every rank holds full params, average the gradients
            for p in head.parameters():
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        for name, p in head.named_parameters():
            if not p.requires_grad:
                continue
            g = p.grad
            assert g is not None, (rank, name)
            if hasattr(g, "full_tensor"):
                g = g.full_tensor()
            expect = ref_grads[name]
            if isinstance(expect, int):
                assert g.abs().sum() == 0, (rank, name)
            else:
                assert rel(g, expect) < 5e-2, (rank, name, rel(g, expect))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires two CUDA devices")
@pytest.mark.skipif(not dist.is_nccl_available(), reason="Test requires NCCL")
@pytest.mark.parametrize("mode", ["ddp", "fsdp2"])
def test_routed_lora_distributed(mode):
    if mode == "fsdp2":
        import torch.distributed.fsdp as fsdp

        if not hasattr(fsdp, "fully_shard"):
            pytest.skip("Test requires the public FSDP2 API")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp_spawn(_worker, args=(2, port, mode), nprocs=2, join=True)
