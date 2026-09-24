"""linear_cross_entropy with a second operand pair: logits = e c^T + e2 c2^T + bias."""

import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _inputs(shape, d2, dtype, has_bias, invalids, c_requires_grad):
    torch.manual_seed(0)
    N, V, D = shape
    e = (torch.randn(4, N // 4, D, device="cuda", dtype=dtype) / 4).requires_grad_()
    c = torch.randn(V, D, device="cuda", dtype=dtype).requires_grad_(c_requires_grad)
    e2 = (torch.randn(4, N // 4, d2, device="cuda", dtype=dtype) / 4).requires_grad_()
    c2 = (torch.randn(V, d2, device="cuda", dtype=dtype) / 4).requires_grad_()
    bias = (torch.randn(V, device="cuda", dtype=dtype) / 4).requires_grad_() if has_bias else None
    targets = torch.randint(0, V, (4, N // 4), device="cuda")
    if invalids:
        targets[:, 3::7] = IGNORE_INDEX
    return e, c, e2, c2, bias, targets


def _reference(e, c, e2, c2, bias, targets, softcap, shift, reduction):
    if shift:
        e, e2, targets = e[:, :-1], e2[:, :-1], targets[:, 1:]
    logits = e.float() @ c.float().T + e2.float() @ c2.float().T
    if bias is not None:
        logits = logits + bias.float()
    if softcap is not None:
        logits = softcapping(logits, softcap)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, -2), targets.flatten(), ignore_index=IGNORE_INDEX, reduction=reduction
    ).view(targets.shape if reduction == "none" else ())


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce", "torch_compile"])
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 2e-2), (torch.float16, 5e-3)])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("d2", [8, 96])
@pytest.mark.parametrize("c_requires_grad", [False])
def test_second_operand_matches_reference(
    impl, dtype, tol, softcap, has_bias, shift, invalids, reduction, d2, c_requires_grad
):
    if impl == "torch_compile" and d2 != 8:
        pytest.skip("torch_compile is the reference path; one operand shape is enough")
    torch._dynamo.config.cache_size_limit = 256
    e, c, e2, c2, bias, targets = _inputs(
        (256, 507, 128), d2, dtype, has_bias, invalids, c_requires_grad
    )
    ref = _reference(e, c, e2, c2, bias, targets, softcap, shift, reduction)
    ref.sum().backward()
    ref_grads = [
        t.grad.clone() if t is not None and t.requires_grad else None for t in (e, c, e2, c2, bias)
    ]
    for t in (e, c, e2, c2, bias):
        if t is not None:
            t.grad = None

    loss = linear_cross_entropy(
        e,
        c,
        targets,
        bias=bias,
        softcap=softcap,
        shift=shift,
        reduction=reduction,
        impl=impl,
        filter_eps=None,
        accum_e_fp32=True,
        accum_c_fp32=True,
        e2=e2,
        c2=c2,
    )
    assert loss.shape == ref.shape
    loss.sum().backward()

    assert _rel(loss, ref) < tol, (loss, ref)
    for name, t, g in zip(("e", "c", "e2", "c2", "bias"), (e, c, e2, c2, bias), ref_grads):
        if g is None:
            assert t is None or t.grad is None, name
            continue
        assert t.grad is not None, name
        assert _rel(t.grad, g) < 5 * tol, (name, _rel(t.grad, g))


@skip_no_cuda
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 2e-2), (torch.float16, 5e-3)])
@pytest.mark.parametrize("d2", [8, 12])
def test_second_operand_with_trainable_c(dtype, tol, d2):
    e, c, e2, c2, bias, targets = _inputs((256, 507, 128), d2, dtype, True, True, True)
    ref = _reference(e, c, e2, c2, bias, targets, None, True, "mean")
    ref.backward()
    ref_grads = [t.grad.clone() for t in (e, c, e2, c2, bias)]
    for t in (e, c, e2, c2, bias):
        t.grad = None
    loss = linear_cross_entropy(
        e,
        c,
        targets,
        bias=bias,
        shift=True,
        filter_eps=None,
        accum_e_fp32=True,
        accum_c_fp32=True,
        e2=e2,
        c2=c2,
    )
    loss.backward()
    assert _rel(loss, ref) < tol
    for name, t, g in zip(("e", "c", "e2", "c2", "bias"), (e, c, e2, c2, bias), ref_grads):
        assert _rel(t.grad, g) < 5 * tol, (name, _rel(t.grad, g))


@skip_no_cuda
@pytest.mark.parametrize("filter_eps", ["auto", None])
@pytest.mark.parametrize("c_grad_chunk_size", [0, 128])
def test_second_operand_with_filtering_and_chunked_c(filter_eps, c_grad_chunk_size):
    e, c, e2, c2, bias, targets = _inputs((256, 507, 128), 8, torch.bfloat16, True, True, True)
    ref = _reference(e, c, e2, c2, bias, targets, None, True, "mean")
    ref.backward()
    ref_grads = [t.grad.clone() for t in (e, c, e2, c2, bias)]
    for t in (e, c, e2, c2, bias):
        t.grad = None
    loss = linear_cross_entropy(
        e,
        c,
        targets,
        bias=bias,
        shift=True,
        filter_eps=filter_eps,
        accum_e_fp32=True,
        accum_c_fp32=True,
        c_grad_chunk_size=c_grad_chunk_size,
        e2=e2,
        c2=c2,
    )
    loss.backward()
    assert _rel(loss, ref) < 2e-2
    for name, t, g in zip(("e", "c", "e2", "c2", "bias"), (e, c, e2, c2, bias), ref_grads):
        assert _rel(t.grad, g) < 1e-1, (name, _rel(t.grad, g))


@skip_no_cuda
def test_frozen_c_gets_no_classifier_gradient_buffer():
    from cut_cross_entropy.cce_backward import cce_backward_kernel

    e, c, e2, c2, bias, targets = _inputs((256, 507, 128), 8, torch.bfloat16, False, False, False)
    lse = torch.zeros(e.numel() // e.size(-1), device="cuda")
    de, dc, dbias, de2, dc2 = cce_backward_kernel(
        do=torch.ones((), device="cuda"),
        e=e.flatten(0, -2),
        c=c,
        bias=None,
        lse=lse,
        valids=None,
        softcap=None,
        filter_eps=None,
        filter_e_grad=False,
        filter_c_grad=False,
        targets=targets.flatten(),
        e2=e2.flatten(0, -2),
        c2=c2,
    )
    assert dc is None and dbias is None
    assert de.shape == e.flatten(0, -2).shape and de2.shape == e2.flatten(0, -2).shape
    assert dc2.shape == c2.shape


@skip_no_cuda
def test_rejects_partial_or_mismatched_second_operand():
    e, c, e2, c2, bias, targets = _inputs((256, 507, 128), 8, torch.bfloat16, False, False, True)
    with pytest.raises(ValueError, match="together"):
        linear_cross_entropy(e, c, targets, e2=e2)
    with pytest.raises(ValueError, match="do not match"):
        linear_cross_entropy(e, c, targets, e2=e2, c2=c2[:-1])
