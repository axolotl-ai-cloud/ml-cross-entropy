"""Checks that remote_model_id patching targets the class from_pretrained actually uses."""

import importlib
import json
import os
import subprocess
import sys
import warnings

import pytest
import transformers

from cut_cross_entropy.transformers.patch import cce_patch

_FAKE_REMOTE_FILES = {
    "configuration_fakellama.py": (
        "from transformers import LlamaConfig\n\n\n"
        "class FakeLlamaConfig(LlamaConfig):\n"
        '    model_type = "fakellama"\n'
    ),
    "modeling_fakellama.py": (
        "from transformers import LlamaForCausalLM\n\n"
        "from .configuration_fakellama import FakeLlamaConfig\n\n\n"
        "class FakeLlamaForCausalLM(LlamaForCausalLM):\n"
        "    config_class = FakeLlamaConfig\n"
    ),
    "config.json": json.dumps(
        {
            "model_type": "fakellama",
            "architectures": ["FakeLlamaForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_fakellama.FakeLlamaConfig",
                "AutoModelForCausalLM": "modeling_fakellama.FakeLlamaForCausalLM",
            },
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 64,
        }
    ),
}

_PATCH_THEN_LOAD = """
import sys

import transformers

from cut_cross_entropy.transformers.utils import patch_remote_model_class


def sentinel(self, *args, **kwargs):
    pass


patch_remote_model_class(sys.argv[1], "FakeLlamaForCausalLM", sentinel)
config = transformers.AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
assert type(model).forward is sentinel, f"{type(model).__module__} is not patched"
"""


def test_remote_code_class_patch_survives_warm_modules_cache(tmp_path):
    repo = tmp_path / "fake-llama"
    repo.mkdir()
    for name, content in _FAKE_REMOTE_FILES.items():
        (repo / name).write_text(content)

    # Fresh processes share one modules cache: the first run fills it, the second hits it warm.
    env = {**os.environ, "HF_MODULES_CACHE": str(tmp_path / "modules"), "HF_HUB_OFFLINE": "1"}
    for run in ("cold", "warm"):
        result = subprocess.run(
            [sys.executable, "-c", _PATCH_THEN_LOAD, str(repo)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{run} modules cache:\n{result.stderr}"


@pytest.mark.parametrize(
    "arch,class_name,native_forward",
    [
        ("afmoe", "AfmoeForCausalLM", "cce_forward_afmoe"),
        ("kimi_linear", "KimiLinearForCausalLM", "cce_forward_kimi"),
        ("step3p7", "Step3p7ForConditionalGeneration", "cce_forward_step3p7"),
    ],
)
def test_native_checkpoint_as_remote_model_id_gets_native_forward(
    tmp_path, monkeypatch, arch, class_name, native_forward
):
    modeling = importlib.import_module(f"transformers.models.{arch}.modeling_{arch}")
    model_class = getattr(modeling, class_name)
    monkeypatch.setattr(model_class, "forward", model_class.forward)  # restored after the test

    # A native-saved checkpoint has no auto_map, so it resolves to the native class.
    transformers.AutoConfig.for_model(arch).save_pretrained(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cce_patch(arch, remote_model_id=str(tmp_path))

    patch_module = importlib.import_module(f"cut_cross_entropy.transformers.{arch}")
    assert model_class.forward is getattr(patch_module, native_forward)
    assert not [w for w in caught if "remote" in str(w.message)]
