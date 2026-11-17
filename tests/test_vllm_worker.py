"""Custom worker lifecycle contracts using a stub of the pinned vLLM API."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from qorerl.recipes.spec import MXFP8_HADAMARD


@pytest.fixture
def worker_module(monkeypatch):
    events = []
    model = object()

    class Worker:
        # Matches vLLM 0.32.0's keyword-only API.
        def load_model(self, *, load_dummy_weights=False):
            events.append(("super_load", load_dummy_weights))
            self.model_runner = SimpleNamespace(model=model, model_memory_usage=512)

        def determine_available_memory(self):
            assert hasattr(self, "qorerl_recipe_installation")
            events.append(("profile", self.model_runner.model))
            return 1024

    upstream = ModuleType("vllm.v1.worker.gpu_worker")
    upstream.Worker = Worker
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    path = Path(__file__).resolve().parents[1] / "qorerl" / "engines" / "vllm_worker.py"
    spec = importlib.util.spec_from_file_location("_qorerl_worker_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, events, model


def test_worker_installs_recipe_after_load_before_memory_profile(worker_module, monkeypatch):
    module, events, model = worker_module
    installation = object()

    def install(loaded_model, recipe):
        assert events == [("super_load", False)]
        assert loaded_model is model
        assert recipe.name == MXFP8_HADAMARD
        assert recipe.layers == (1, 3)
        assert recipe.hadamard_block_size == 64
        assert recipe.seed == 19
        events.append(("install", loaded_model))
        return installation

    monkeypatch.setattr(module, "install", install)
    worker = module.QoreRLRecipeWorker()
    worker.vllm_config = SimpleNamespace(
        additional_config={
            "qorerl_recipe": {
                "name": MXFP8_HADAMARD,
                "options": {"layers": [1, 3], "block_size": 64, "seed": 19},
            }
        }
    )
    worker.load_model()
    assert worker.qorerl_recipe_installation is installation
    # vLLM's actual free-memory delta includes the caches; this diagnostic
    # weight-load snapshot must not be used to subtract them a second time.
    assert worker.model_runner.model_memory_usage == 512
    assert worker.determine_available_memory() == 1024
    assert events == [("super_load", False), ("install", model), ("profile", model)]


def test_worker_rejects_dummy_weights_before_loading(worker_module):
    module, events, _ = worker_module
    worker = module.QoreRLRecipeWorker()
    with pytest.raises(ValueError, match="canonical HF weights"):
        worker.load_model(load_dummy_weights=True)
    assert events == []
