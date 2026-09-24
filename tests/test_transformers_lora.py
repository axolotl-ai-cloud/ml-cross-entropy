"""apply_lce_lm_head must see PEFT LoRA adapters on lm_head; lm_head.weight alone silently drops them."""

import pathlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cut_cross_entropy.transformers.utils import PatchOptions, apply_lce, apply_lce_lm_head

peft = pytest.importorskip("peft")

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")

D, V, R = 128, 1000, 8


def _opts(**overrides) -> PatchOptions:
    base = dict(
        impl="cce",
        reduction="mean",
        filter_eps="auto",
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_e_grad=False,
        filter_c_grad=False,
        train_only=False,
    )
    base.update(overrides)
    return PatchOptions(**base)


def _inputs(seed=0):
    torch.manual_seed(seed)
    e = (torch.randn(2, 65, D, device="cuda", dtype=torch.bfloat16) / 4).requires_grad_()
    labels = torch.randint(0, V, (2, 65), device="cuda")
    labels[:, 5::11] = -100
    return e, labels


class _Head(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.lm_head = nn.Linear(D, V, bias=bias, device="cuda", dtype=torch.bfloat16)


def _lora_head(bias=False, dropout=0.0, use_dora=False, lora_bias=False, adapters=("default",)):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    cfg = LoraConfig(
        r=R,
        lora_alpha=2 * R,
        lora_dropout=dropout,
        use_dora=use_dora,
        lora_bias=lora_bias,
        target_modules=["lm_head"],
    )
    model = get_peft_model(_Head(bias), cfg, adapter_name=adapters[0])
    for name in adapters[1:]:
        model.add_adapter(name, cfg)
    model.base_model.set_adapter(list(adapters))
    head = model.base_model.model.lm_head
    for name in adapters:
        # lora_B is zero-initialised; give the adapter a real contribution and a real gradient.
        nn.init.normal_(head.lora_B[name].weight, std=0.05)
        if lora_bias:
            nn.init.normal_(head.lora_B[name].bias, std=0.05)
    return head


def _reference(head, e, labels):
    logits = head(e).float()
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100
    )


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _grads(head, e):
    out = {"e": e.grad}
    for name, p in head.named_parameters():
        if p.requires_grad:
            out[name] = p.grad
    return out


@skip_no_cuda
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"bias": True},
        {"lora_bias": True},
        {"use_dora": True},
        {"use_dora": True, "bias": True},
        {"adapters": ("a", "b")},
        {"adapters": ("a", "b"), "use_dora": True},
    ],
    ids=lambda k: "-".join(f"{a}={b}" for a, b in k.items()) or "plain",
)
def test_lora_lm_head_matches_dense_reference(kwargs):
    head = _lora_head(**kwargs)
    e, labels = _inputs()
    ref = _reference(head, e, labels)
    ref.backward()
    ref_grads = _grads(head, e)
    e.grad = None
    head.zero_grad()

    loss = apply_lce_lm_head(e, head, labels, _opts())
    loss.backward()
    grads = _grads(head, e)

    assert _rel(loss, ref) < 2e-2, (loss.item(), ref.item())
    assert set(grads) == set(ref_grads)
    for name, ref_grad in ref_grads.items():
        assert grads[name] is not None, name
        assert _rel(grads[name], ref_grad) < 5e-2, (name, _rel(grads[name], ref_grad))


@skip_no_cuda
def test_lora_dropout_matches_reference_in_eval_and_trains():
    head = _lora_head(dropout=0.5)
    e, labels = _inputs()
    head.eval()
    ref = _reference(head, e, labels)
    loss = apply_lce_lm_head(e, head, labels, _opts())
    assert _rel(loss, ref) < 2e-2

    head.train()
    loss = apply_lce_lm_head(e, head, labels, _opts())
    loss.backward()
    assert torch.isfinite(loss)
    assert head.lora_B["default"].weight.grad.abs().sum() > 0


@skip_no_cuda
def test_lora_dora_dropout_matches_reference_under_fixed_rng():
    head = _lora_head(dropout=0.5, use_dora=True)
    e, labels = _inputs()
    head.train()
    torch.manual_seed(7)
    ref = _reference(head, e, labels)
    torch.manual_seed(7)
    loss = apply_lce_lm_head(e, head, labels, _opts())
    assert _rel(loss, ref) < 2e-2


@skip_no_cuda
def test_merged_and_disabled_adapters_use_effective_weight():
    head = _lora_head()
    e, labels = _inputs()
    head.merge()
    merged = apply_lce_lm_head(e, head, labels, _opts())
    ref = _reference(head, e, labels)
    assert _rel(merged, ref) < 2e-2
    head.unmerge()

    head.enable_adapters(False)
    disabled = apply_lce_lm_head(e, head, labels, _opts())
    plain = apply_lce(e, head.base_layer.weight, labels, _opts())
    assert _rel(disabled, plain) < 1e-5


@skip_no_cuda
def test_plain_linear_threads_bias():
    torch.manual_seed(2)
    head = nn.Linear(D, V, bias=True, device="cuda", dtype=torch.bfloat16)
    e, labels = _inputs()
    loss = apply_lce_lm_head(e, head, labels, _opts())
    expected = apply_lce(e, head.weight, labels, _opts(), bias=head.bias)
    assert _rel(loss, expected) < 1e-5
    logits = head(e).float()
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1))
    assert _rel(loss, ref) < 2e-2


def test_modules_to_save_wrapper_exposes_trainable_copy():
    from peft.utils.other import ModulesToSaveWrapper

    head = nn.Linear(D, V, bias=False)
    wrapper = ModulesToSaveWrapper(head, "default")
    assert wrapper.weight is wrapper.modules_to_save["default"].weight
    assert wrapper.weight is not wrapper.original_module.weight


def test_no_patch_passes_lm_head_weight():
    """Every model patch must hand apply_lce_lm_head the module, never lm_head.weight."""
    root = pathlib.Path(__file__).resolve().parents[1] / "cut_cross_entropy" / "transformers"
    for path in root.glob("*.py"):
        if path.name in ("utils.py", "patch.py"):
            continue
        src = path.read_text()
        assert "apply_lce(" not in src, path.name
        for call in src.split("apply_lce_lm_head(")[1:]:
            args = call.split("\n        )")[0]
            assert "lm_head.weight," not in args, path.name


@skip_no_cuda
def test_patched_llama_trains_lm_head_lora():
    import transformers
    from peft import LoraConfig, get_peft_model

    from cut_cross_entropy.transformers.patch import cce_patch

    cfg = transformers.LlamaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).to("cuda", torch.bfloat16)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "lm_head"]
    )
    ids = torch.randint(0, 512, (2, 16), device="cuda")

    ref_model = get_peft_model(model, lora_cfg)
    lm_head = ref_model.base_model.model.lm_head
    nn.init.normal_(lm_head.lora_B["default"].weight, std=0.05)
    ref_loss = ref_model(input_ids=ids, labels=ids).loss
    ref_loss.backward()
    ref_grad = lm_head.lora_B["default"].weight.grad.clone()
    ref_model.zero_grad()

    cce_patch(ref_model.base_model.model)
    loss = ref_model(input_ids=ids, labels=ids).loss
    loss.backward()
    grad = lm_head.lora_B["default"].weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert _rel(loss, ref_loss) < 2e-2
    assert _rel(grad, ref_grad) < 5e-2
