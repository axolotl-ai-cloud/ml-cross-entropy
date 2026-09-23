"""Boundary cases for numerical checks and CUDA Compute Sanitizer memcheck."""

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("filter_eps", [None, "auto"])
@pytest.mark.parametrize("storage_offset", [0, 1])
@pytest.mark.parametrize(
    "tokens,vocab,hidden,chunk_size",
    [
        (1, 129, 63, 128),
        (127, 255, 65, 128),
        (128, 256, 128, 128),
        (129, 257, 123, 256),
        (257, 511, 127, 256),
    ],
)
def test_chunked_memory_boundaries(
    dtype, filter_eps, storage_offset, tokens, vocab, hidden, chunk_size
):
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    e_storage = torch.randn(tokens * hidden + storage_offset, device="cuda", dtype=dtype)
    c_storage = torch.randn(vocab * hidden + storage_offset, device="cuda", dtype=dtype)
    e_storage.div_(hidden**0.5)
    e = e_storage[storage_offset:].view(tokens, hidden).requires_grad_()
    c = c_storage[storage_offset:].view(vocab, hidden).requires_grad_()
    targets = torch.randint(0, vocab, (tokens + storage_offset,), device="cuda")[storage_offset:]
    targets[-1] = vocab - 1
    original_e, original_c, original_targets = (
        e.detach().clone(),
        c.detach().clone(),
        targets.clone(),
    )

    loss = linear_cross_entropy(
        e,
        c,
        targets,
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_eps=filter_eps,
        c_grad_chunk_size=chunk_size,
    )
    gradients = torch.autograd.grad(loss, (e, c))
    torch.cuda.synchronize()

    ef, cf = original_e.float().requires_grad_(), original_c.float().requires_grad_()
    expected_loss = F.cross_entropy(F.linear(ef, cf), targets)
    expected_gradients = torch.autograd.grad(expected_loss, (ef, cf))
    torch.testing.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual).all()
        assert (actual.float() - expected).norm() / expected.norm() < 0.01
    torch.testing.assert_close(e, original_e, rtol=0, atol=0)
    torch.testing.assert_close(c, original_c, rtol=0, atol=0)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0)
