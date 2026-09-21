"""Selection and numerical checks for classifier-gradient chunk recommendations."""

import pytest
import torch

from cut_cross_entropy import LinearCrossEntropy, linear_cross_entropy, recommend_c_grad_chunk_size


@pytest.mark.parametrize(
    "tokens,vocab,hidden,sms,expected",
    [
        (512, 248320, 4096, 188, 65536),
        (2048, 248320, 4096, 188, 16384),
        (8192, 248320, 4096, 188, 4096),
        (32768, 248320, 4096, 188, 1024),
        (1, 248320, 4096, 188, 65536),
        (512, 248320, 8192, 188, 32768),
        (512, 248320, 5120, 188, 32768),
        (512, 248320, 4096, 84, 32768),
        (0, 248320, 4096, 188, 0),
        (512, 1025, 4096, 188, 0),
        (512, 65536, 4096, 188, 0),
        (512, 65537, 4096, 188, 65536),
        (1 << 20, 248320, 4096, 188, 128),
    ],
)
def test_recommended_chunk_size(tokens, vocab, hidden, sms, expected):
    assert recommend_c_grad_chunk_size(tokens, vocab, hidden, target_programs=8 * sms) == expected


@pytest.mark.parametrize("tokens", [1, 127, 128, 129, 512, 8192, 65536])
@pytest.mark.parametrize("hidden", [128, 4096, 8192, 16384])
def test_recommended_chunk_memory_bound(tokens, hidden):
    chunk = recommend_c_grad_chunk_size(tokens, 1 << 24, hidden, target_programs=8 * 188)
    assert chunk >= 128 and chunk % 128 == 0
    assert chunk * hidden * 4 <= 1 << 30


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("filter_eps", [None, "auto"])
def test_recommended_chunk_gradients(dtype, filter_eps):
    torch.manual_seed(42)
    e = (torch.randn(1, 513, 128, device="cuda", dtype=dtype) / 8).requires_grad_()
    labels = torch.zeros((1, 513), device="cuda", dtype=torch.long)
    labels[:, 7::13] = -100
    valid_tokens = (labels[:, 1:] != -100).count_nonzero().item()
    selected = recommend_c_grad_chunk_size(valid_tokens, 1 << 24, 128, device=e.device)
    c = torch.randn(selected + 1, 128, device="cuda", dtype=dtype).requires_grad_()
    labels[:, -1] = selected
    module = LinearCrossEntropy(
        shift=1,
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_eps=filter_eps,
        c_grad_chunk_size=selected,
    )
    loss = module(e, c, labels)
    assert loss.grad_fn.params.c_grad_chunk_size == selected
    gradients = torch.autograd.grad(loss, (e, c))
    expected = linear_cross_entropy(
        e,
        c,
        labels,
        shift=1,
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_eps=filter_eps,
        c_grad_chunk_size=0,
    )
    expected_gradients = torch.autograd.grad(expected, (e, c))
    torch.testing.assert_close(loss, expected)
    for actual, reference in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual).all()
        assert (actual.float() - reference.float()).norm() / reference.float().norm() < 0.003


def test_device_target_programs(monkeypatch):
    from types import SimpleNamespace

    devices = []

    def properties(device):
        devices.append(device)
        return SimpleNamespace(multi_processor_count=188)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert recommend_c_grad_chunk_size(2048, 248320, 4096, device="cuda:1") == 16384
    assert devices == ["cuda:1"]


def test_explicit_programs_do_not_query_device(monkeypatch):
    def unexpected(*args):
        raise AssertionError("Explicit targets must not query CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    assert recommend_c_grad_chunk_size(2048, 248320, 4096, target_programs=1504) == 16384
    assert recommend_c_grad_chunk_size(0, 248320, 4096) == 0


def test_local_vocab_shards():
    sizes = [124161, 65536, 65535]
    assert [recommend_c_grad_chunk_size(512, v, 4096, target_programs=1504) for v in sizes] == [
        65536,
        0,
        0,
    ]


def test_custom_scratch_budget():
    assert (
        recommend_c_grad_chunk_size(
            512,
            248320,
            4096,
            target_programs=1504,
            max_scratch_bytes=256 << 20,
        )
        == 16384
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_tokens": -1},
        {"num_tokens": 1.5},
        {"vocab_size": 0},
        {"hidden_size": 0},
        {"target_programs": 0},
        {"target_programs": 1.5},
        {"max_scratch_bytes": 0},
        {"max_scratch_bytes": 1},
    ],
)
def test_invalid_recommendation_inputs(overrides):
    kwargs = dict(num_tokens=512, vocab_size=248320, hidden_size=4096, target_programs=1504)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        recommend_c_grad_chunk_size(**kwargs)
