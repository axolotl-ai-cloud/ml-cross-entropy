"""Explicit per-row / per-token adapter routing in apply_lce_lm_head against dense routed logits."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cut_cross_entropy.transformers.utils import PatchOptions, apply_lce_lm_head

pytest.importorskip("peft")

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")

D, V = 96, 700
RANKS = {"a": 8, "b": 4}


def _opts(**overrides) -> PatchOptions:
    base = dict(
        impl="cce",
        reduction="mean",
        filter_eps=None,
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_e_grad=False,
        filter_c_grad=False,
        train_only=False,
    )
    base.update(overrides)
    return PatchOptions(**base)


class _Head(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.lm_head = nn.Linear(D, V, bias=bias, device="cuda", dtype=torch.bfloat16)


def make_head(bias=False, lora_bias=False, dropout=0.0, use_dora=False, seed=1):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    model = None
    for name, r in RANKS.items():
        cfg = LoraConfig(
            r=r,
            lora_alpha=2 * r,
            lora_dropout=dropout,
            lora_bias=lora_bias,
            use_dora=use_dora,
            target_modules=["lm_head"],
        )
        if model is None:
            model = get_peft_model(_Head(bias), cfg, adapter_name=name)
        else:
            model.add_adapter(name, cfg)
    head = model.base_model.model.lm_head
    for name in RANKS:
        nn.init.normal_(head.lora_B[name].weight, std=0.05)
        if lora_bias:
            nn.init.normal_(head.lora_B[name].bias, std=0.05)
    # only "a" is active; "b" is selected by routing alone, so keep it trainable
    head.set_adapter(["a"])
    for name, p in head.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
    return head


def make_inputs(B=3, S=33, seed=0, ignore=True):
    torch.manual_seed(seed)
    e = (torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16) / 4).requires_grad_()
    labels = torch.randint(0, V, (B, S), device="cuda")
    if ignore:
        labels[:, 4::9] = -100
        labels[0, -3:] = -100
    return e, labels


def routed_logits(head, e, ids, adapter_map):
    """Dense reference: base head plus the adapter each SOURCE position is routed to."""
    base = head.get_base_layer()
    x = e.float()
    logits = F.linear(x, base.weight.float(), None if base.bias is None else base.bias.float())
    if ids.ndim == 1:
        ids = ids[:, None].expand(e.shape[:-1])
    for key, name in adapter_map.items():
        if name is None:
            continue
        A, Bm, s = head.lora_A[name], head.lora_B[name], head.scaling[name]
        delta = (
            F.linear(
                F.linear(x, A.weight.float()),
                Bm.weight.float(),
                None if Bm.bias is None else Bm.bias.float(),
            )
            * s
        )
        logits = logits + delta * (ids == key)[..., None]
    return logits


def reference_loss(head, e, labels, ids, adapter_map, reduction="mean", num_items_in_batch=None):
    logits = routed_logits(head, e, ids, adapter_map)[:, :-1]
    targets = labels[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100, reduction="none"
    ).view(targets.shape)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if num_items_in_batch is not None:
        return loss.sum() / num_items_in_batch
    return loss.sum() / (targets != -100).sum()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def grads_of(head, e):
    out = {"e": e.grad}
    for name, p in head.named_parameters():
        if p.requires_grad:
            out[name] = p.grad
    return out


def assert_parity(
    head, e, labels, ids, adapter_map, reduction="mean", num_items_in_batch=None, **cce
):
    kwargs = {}
    if num_items_in_batch is not None:
        kwargs["num_items_in_batch"] = num_items_in_batch
    e.grad = None
    head.zero_grad()
    ref = reference_loss(head, e, labels, ids, adapter_map, reduction, num_items_in_batch)
    ref.sum().backward()
    ref_grads = grads_of(head, e)
    e.grad = None
    head.zero_grad()

    loss = apply_lce_lm_head(
        e,
        head,
        labels,
        _opts(reduction=reduction, **cce),
        adapter_ids=ids,
        adapter_map=adapter_map,
        **kwargs,
    )
    loss.sum().backward()
    grads = grads_of(head, e)

    assert loss.shape == ref.shape
    assert rel(loss, ref) < 2e-2, (loss, ref)
    for name, ref_grad in ref_grads.items():
        got = grads[name]
        if ref_grad is None:
            # adapter absent from this batch: it still took part, with a zero gradient
            assert got is not None and got.abs().sum() == 0, name
        else:
            assert got is not None, name
            assert rel(got, ref_grad) < 5e-2, (name, rel(got, ref_grad))
    return loss, ref


ROW_MAP = {0: "a", 1: None, 2: "b"}


@skip_no_cuda
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_row_routing_mixed_ranks_and_base_only(reduction):
    head = make_head()
    e, labels = make_inputs()
    ids = torch.tensor([2, 1, 0], device="cuda")
    assert_parity(head, e, labels, ids, ROW_MAP, reduction=reduction)


@skip_no_cuda
def test_token_routing_follows_source_position():
    head = make_head()
    e, labels = make_inputs()
    torch.manual_seed(3)
    ids = torch.randint(0, 3, e.shape[:-1], device="cuda")
    assert_parity(head, e, labels, ids, ROW_MAP)


@skip_no_cuda
def test_token_routing_causal_boundary():
    """Adapter switches at position t: the loss for target t+1 uses the adapter of source t."""
    head = make_head()
    e, labels = make_inputs(B=1, S=16, ignore=False)
    ids = torch.zeros(1, 16, dtype=torch.long, device="cuda")
    ids[0, 8:] = 2
    loss, ref = assert_parity(head, e, labels, ids, ROW_MAP, reduction="none")
    wrong_ref = reference_loss(head, e, labels, torch.roll(ids, 1, dims=1), ROW_MAP, "none")
    assert rel(loss, wrong_ref) > 1e-3, "a shifted routing would have to change the loss"


@skip_no_cuda
def test_permuted_batch_permutes_per_row_loss():
    head = make_head()
    e, labels = make_inputs()
    ids = torch.tensor([2, 1, 0], device="cuda")
    perm = torch.tensor([1, 2, 0], device="cuda")
    loss = apply_lce_lm_head(
        e, head, labels, _opts(reduction="none"), adapter_ids=ids, adapter_map=ROW_MAP
    )
    e2 = e.detach()[perm].requires_grad_()
    loss_p = apply_lce_lm_head(
        e2, head, labels[perm], _opts(reduction="none"), adapter_ids=ids[perm], adapter_map=ROW_MAP
    )
    assert rel(loss_p, loss[perm]) < 1e-4
    mean = apply_lce_lm_head(e, head, labels, _opts(), adapter_ids=ids, adapter_map=ROW_MAP)
    mean_p = apply_lce_lm_head(
        e2, head, labels[perm], _opts(), adapter_ids=ids[perm], adapter_map=ROW_MAP
    )
    assert rel(mean_p, mean) < 1e-4


@skip_no_cuda
def test_ignored_labels_keep_ids_valid_and_normalisation():
    head = make_head()
    e, labels = make_inputs()
    labels[1] = -100  # a whole base-only row ignored
    ids = torch.tensor([2, 1, 0], device="cuda")
    assert_parity(head, e, labels, ids, ROW_MAP, reduction="mean")
    n = int((labels[:, 1:] != -100).sum())
    assert_parity(head, e, labels, ids, ROW_MAP, reduction="mean", num_items_in_batch=n * 2)


@skip_no_cuda
def test_locally_absent_adapter_still_participates():
    head = make_head()
    e, labels = make_inputs()
    ids = torch.tensor([0, 1, 0], device="cuda")  # "b" is mapped but never selected
    assert_parity(head, e, labels, ids, ROW_MAP)
    assert head.lora_B["b"].weight.grad is not None


@skip_no_cuda
def test_all_base_only_rows():
    head = make_head()
    e, labels = make_inputs()
    ids = torch.ones(3, dtype=torch.long, device="cuda")
    assert_parity(head, e, labels, ids, ROW_MAP)


@skip_no_cuda
def test_shift_labels_no_double_shift():
    head = make_head()
    e, labels = make_inputs()
    ids = torch.tensor([2, 1, 0], device="cuda")
    ref = apply_lce_lm_head(
        e, head, labels, _opts(reduction="none"), adapter_ids=ids, adapter_map=ROW_MAP
    )
    pre = apply_lce_lm_head(
        e[:, :-1],
        head,
        labels,
        _opts(reduction="none"),
        shift_labels=labels[:, 1:],
        adapter_ids=ids,
        adapter_map=ROW_MAP,
    )
    assert rel(pre, ref) < 1e-4


@skip_no_cuda
def test_routed_lora_bias_and_base_bias():
    head = make_head(bias=True, lora_bias=True)
    e, labels = make_inputs()
    torch.manual_seed(5)
    ids = torch.randint(0, 3, e.shape[:-1], device="cuda")
    assert_parity(head, e, labels, ids, ROW_MAP)


@skip_no_cuda
def test_routed_dropout_eval_matches_and_train_runs():
    head = make_head(dropout=0.5)
    e, labels = make_inputs()
    ids = torch.tensor([2, 1, 0], device="cuda")
    head.eval()
    assert_parity(head, e, labels, ids, ROW_MAP)
    head.train()
    e.grad = None
    head.zero_grad()
    loss = apply_lce_lm_head(e, head, labels, _opts(), adapter_ids=ids, adapter_map=ROW_MAP)
    loss.backward()
    assert torch.isfinite(loss)
    assert head.lora_B["a"].weight.grad.abs().sum() > 0


@skip_no_cuda
def test_routing_does_not_touch_active_adapters_or_default_path():
    head = make_head()
    e, labels = make_inputs()
    ids = torch.tensor([2, 2, 2], device="cuda")
    before = list(head.active_adapters)
    apply_lce_lm_head(e, head, labels, _opts(), adapter_ids=ids, adapter_map=ROW_MAP)
    assert list(head.active_adapters) == before == ["a"]
    # without routing, the active adapter ("a") applies everywhere as before
    plain = apply_lce_lm_head(e, head, labels, _opts())
    ref = reference_loss(head, e, labels, torch.zeros(3, dtype=torch.long, device="cuda"), {0: "a"})
    assert rel(plain, ref) < 2e-2


@skip_no_cuda
@pytest.mark.parametrize(
    "ids, adapter_map, match",
    [
        (torch.tensor([0, 1, 7]), ROW_MAP, "not in adapter_map"),
        (torch.tensor([0, 1]), ROW_MAP, "shape"),
        (torch.zeros(3, 5, dtype=torch.long), ROW_MAP, "shape"),
        (torch.zeros(3, dtype=torch.float32), ROW_MAP, "integer"),
        (torch.zeros(3, dtype=torch.long), {0: "zzz"}, "unknown adapter"),
        (torch.zeros(3, dtype=torch.long), {"0": "a"}, "keys must be ints"),
    ],
)
def test_rejects_bad_routing(ids, adapter_map, match):
    head = make_head()
    e, labels = make_inputs()
    with pytest.raises(ValueError, match=match):
        apply_lce_lm_head(e, head, labels, _opts(), adapter_ids=ids.cuda(), adapter_map=adapter_map)


@skip_no_cuda
def test_rejects_partial_args_merged_dora_and_plain_head():
    head = make_head()
    e, labels = make_inputs()
    ids = torch.zeros(3, dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match="together"):
        apply_lce_lm_head(e, head, labels, _opts(), adapter_ids=ids)
    with pytest.raises(ValueError, match="together"):
        apply_lce_lm_head(e, head, labels, _opts(), adapter_map=ROW_MAP)
    head.merge()
    with pytest.raises(ValueError, match="unmerged"):
        apply_lce_lm_head(e, head, labels, _opts(), adapter_ids=ids, adapter_map=ROW_MAP)
    head.unmerge()
    plain = nn.Linear(D, V, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="PEFT LoRA"):
        apply_lce_lm_head(e, plain, labels, _opts(), adapter_ids=ids, adapter_map=ROW_MAP)
    dora = make_head(use_dora=True)
    with pytest.raises(NotImplementedError, match="DoRA"):
        apply_lce_lm_head(e, dora, labels, _opts(), adapter_ids=ids, adapter_map=ROW_MAP)
