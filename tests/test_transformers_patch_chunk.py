"""Checks that cce_patch and apply_lce plumb c_grad_chunk_size through to the kernel."""

import inspect

import pytest
import torch

from cut_cross_entropy.transformers.patch import cce_patch
from cut_cross_entropy.transformers.utils import PatchOptions, apply_lce

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _opts(**overrides) -> PatchOptions:
    base = dict(
        impl="cce",
        reduction="mean",
        filter_eps="auto",
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_e_grad=True,
        filter_c_grad=True,
        train_only=False,
    )
    base.update(overrides)
    return PatchOptions(**base)


def test_patch_options_default_is_unchunked():
    assert _opts().c_grad_chunk_size == 0


def test_patch_options_positional_construction_still_works():
    opts = PatchOptions("cce", "mean", "auto", False, False, True, True, False)
    assert opts.c_grad_chunk_size == 0
    assert "c_grad_chunk_size" not in opts.to_kwargs()


def test_cce_patch_positional_args_unchanged():
    sig = inspect.signature(cce_patch)
    params = list(sig.parameters)
    assert params[:10] == [
        "model_type_or_model",
        "impl",
        "reduction",
        "filter_eps",
        "accum_e_fp32",
        "accum_c_fp32",
        "filter_e_grad",
        "filter_c_grad",
        "train_only",
        "remote_model_id",
    ]
    assert params[-1] == "c_grad_chunk_size"
    assert sig.parameters["c_grad_chunk_size"].default == 0


@pytest.mark.parametrize("chunk", [-128, 100, "auto"])
def test_cce_patch_rejects_bad_chunk_size(chunk):
    with pytest.raises(ValueError, match="c_grad_chunk_size"):
        cce_patch("llama", accum_c_fp32=True, c_grad_chunk_size=chunk)


def test_cce_patch_requires_fp32_classifier_accumulation():
    with pytest.raises(ValueError, match="accum_c_fp32"):
        cce_patch("llama", c_grad_chunk_size=256)


def test_cce_patch_rejects_torch_compile_chunking():
    with pytest.raises(ValueError, match="CCE implementations"):
        cce_patch("llama", impl="torch_compile", accum_c_fp32=True, c_grad_chunk_size=256)


@skip_no_cuda
def test_apply_lce_chunk_matches_full():
    torch.manual_seed(0)
    e = (torch.randn(2, 129, 128, device="cuda", dtype=torch.bfloat16) / 4).requires_grad_()
    c = torch.randn(1025, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    labels = torch.randint(0, 1025, (2, 129), device="cuda")
    labels[:, 7::11] = -100

    def run(opts):
        loss = apply_lce(e, c, labels, opts)
        return loss, torch.autograd.grad(loss, (e, c))

    full_loss, full_grads = run(_opts())
    chunk_loss, chunk_grads = run(_opts(c_grad_chunk_size=256))
    assert chunk_loss.grad_fn.params.c_grad_chunk_size == 256

    torch.testing.assert_close(chunk_loss.detach(), full_loss.detach())
    for actual, expected in zip(chunk_grads, full_grads, strict=True):
        relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert relative_error < 0.003
