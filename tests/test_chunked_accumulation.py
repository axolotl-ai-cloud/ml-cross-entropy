"""Numerical checks for vocabulary-chunked FP32 classifier accumulation."""

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy import LinearCrossEntropy, linear_cross_entropy

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("filter_eps", [None, "auto"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("with_extras", [False, True])
def test_chunked_gradients(dtype, filter_eps, reduction, with_extras):
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    e = (torch.randn(2, 129, 123, device="cuda", dtype=dtype) / 4).requires_grad_()
    c = torch.randn(1025, 123, device="cuda", dtype=dtype).requires_grad_()
    bias = torch.randn(1025, device="cuda", dtype=dtype).requires_grad_() if with_extras else None
    targets = torch.randint(0, 1025, (2, 129), device="cuda")
    if with_extras:
        targets[:, 7::11] = -100
    softcap = 20.0 if with_extras else None
    shift = int(with_extras)
    inputs = (e, c, bias) if bias is not None else (e, c)

    def run(chunk):
        loss = linear_cross_entropy(
            e,
            c,
            targets,
            bias=bias,
            softcap=softcap,
            shift=shift,
            reduction=reduction,
            filter_eps=filter_eps,
            accum_e_fp32=True,
            accum_c_fp32=True,
            c_grad_chunk_size=chunk,
        )
        return loss.detach(), torch.autograd.grad(loss.mean(), inputs)

    full_loss, full_grads = run(0)
    chunk_loss, chunk_grads = run(256)
    torch.testing.assert_close(chunk_loss, full_loss)
    for actual, expected in zip(chunk_grads, full_grads, strict=True):
        relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert relative_error < 0.003

    ef = e.detach().float().requires_grad_()
    cf = c.detach().float().requires_grad_()
    bf = bias.detach().float().requires_grad_() if bias is not None else None
    logits = F.linear(ef[:, :-shift] if shift else ef, cf, bf)
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    expected_loss = F.cross_entropy(
        logits.flatten(0, 1),
        targets[:, shift:].flatten(),
        reduction=reduction,
    )
    expected_grads = torch.autograd.grad(
        expected_loss.mean(), (ef, cf, bf) if bf is not None else (ef, cf)
    )
    for actual, expected in zip(chunk_grads, expected_grads, strict=True):
        relative_error = (actual.float() - expected).norm() / expected.norm()
        assert relative_error < 0.01


@pytest.mark.parametrize("chunk", [-1, 1, 129, 256.0, "256", None])
def test_invalid_chunk_size(chunk):
    e = torch.randn(2, 128, dtype=torch.bfloat16)
    c = torch.randn(256, 128, dtype=torch.bfloat16)
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="multiple of 128"):
        linear_cross_entropy(e, c, y, c_grad_chunk_size=chunk)


@skip_no_cuda
@pytest.mark.parametrize(
    "train_embedding,train_classifier", [(True, False), (False, True), (True, True)]
)
def test_chunked_partial_and_accumulated_gradients(train_embedding, train_classifier):
    torch.manual_seed(42)
    e = torch.randn(257, 128, device="cuda", dtype=torch.bfloat16).requires_grad_(train_embedding)
    c = torch.randn(1025, 128, device="cuda", dtype=torch.bfloat16).requires_grad_(train_classifier)
    targets = torch.randint(0, 1025, (257,), device="cuda")
    inputs = [t for t in (e, c) if t.requires_grad]
    loss = linear_cross_entropy(e, c, targets, accum_e_fp32=True, accum_c_fp32=True)
    expected = torch.autograd.grad(loss, inputs)
    for _ in range(2):
        linear_cross_entropy(
            e,
            c,
            targets,
            accum_e_fp32=True,
            accum_c_fp32=True,
            c_grad_chunk_size=256,
        ).backward()
    for tensor, grad in zip(inputs, expected, strict=True):
        relative_error = (tensor.grad.float() - 2 * grad.float()).norm() / (2 * grad.float().norm())
        assert relative_error < 0.003


@pytest.mark.parametrize("requirement", ["accumulation", "triton", "autotune", "contiguous"])
def test_chunking_requirements(requirement, monkeypatch):
    e = torch.randn(2, 128, dtype=torch.bfloat16)
    c = torch.randn(512, 128, dtype=torch.bfloat16).requires_grad_()
    y = torch.zeros(2, dtype=torch.long)
    accum_c_fp32 = requirement != "accumulation"
    if requirement == "triton":
        monkeypatch.setattr("cut_cross_entropy.cce.is_triton_greater_or_equal_3_2_0", lambda: False)
    if requirement == "autotune":
        monkeypatch.setattr("cut_cross_entropy.cce._AUTOTUNE", True)
    if requirement == "contiguous":
        c = c.T.contiguous().T.detach().requires_grad_()
    message = {
        "accumulation": "FP32 accumulation",
        "triton": "Triton >= 3.2",
        "autotune": "CCE_AUTOTUNE=0",
        "contiguous": "contiguous classifier",
    }[requirement]
    with pytest.raises(ValueError, match=message):
        linear_cross_entropy(e, c, y, accum_c_fp32=accum_c_fp32, c_grad_chunk_size=256)


def test_torch_compile_rejects_chunking():
    e = torch.randn(2, 128, dtype=torch.bfloat16)
    c = torch.randn(512, 128, dtype=torch.bfloat16)
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="only supported by CCE"):
        linear_cross_entropy(e, c, y, impl="torch_compile", c_grad_chunk_size=256)


@skip_no_cuda
@pytest.mark.parametrize("chunk_size", [128, 512, 1024])
def test_module_chunking(chunk_size):
    torch.manual_seed(42)
    e = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    c = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    targets = torch.randint(0, 512, (128,), device="cuda")
    module = LinearCrossEntropy(accum_e_fp32=True, accum_c_fp32=True, c_grad_chunk_size=chunk_size)
    loss = module(e, c, targets)
    grads = torch.autograd.grad(loss, (e, c))
    expected = linear_cross_entropy(e, c, targets, accum_e_fp32=True, accum_c_fp32=True)
    expected_grads = torch.autograd.grad(expected, (e, c))
    torch.testing.assert_close(loss, expected)
    for actual, reference in zip(grads, expected_grads, strict=True):
        assert (actual.float() - reference.float()).norm() / reference.float().norm() < 0.003
