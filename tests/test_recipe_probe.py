"""Observation plumbing tests with explicit stubs, not native MXFP8 measurements."""

import json
import math
import random
import weakref
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qorerl.engines.torch_test import TestLearner as CPULearner
from qorerl.observability import probe as probes
from qorerl.observability.config import RecipeObservationConfig
from qorerl.observability.probe import RecipeProbe
from qorerl.recipes import megatron_mxfp8
from qorerl.recipes.spec import MXFP8_HADAMARD, MXFP8_MLP, resolve_recipe
from qorerl.recipes.state import RecipeState, down_proj_name
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import TrainingBatch, Trajectory


def stub_result():
    # These sentinels test serialization, never quantization accuracy or speed.
    return {
        "bf16": {"latency_ms": 3.0, "output_relative_l2": 0.0},
        "mxfp8": {"latency_ms": 2.0, "output_relative_l2": 0.1},
        "mxfp8_hadamard": {"latency_ms": 4.0, "output_relative_l2": 0.05},
    }, {"backend": "explicit_unit_test_stub"}


def make_probe(*, layers=5, replay=None, **options):
    recipe = resolve_recipe(MXFP8_MLP)
    targets = {down_proj_name(layer): (32, 64) for layer in range(layers)}
    return RecipeProbe(
        RecipeObservationConfig(enabled=True, **options),
        recipe,
        targets,
        layers,
        replay=replay or (lambda *args, **kwargs: stub_result()),
    )


def capture_all(probe):
    for name in probe.targets:
        probe.capture(name, torch.ones(7, 64), torch.ones(32, 64))


def test_defaults_disabled_and_native_only_configuration():
    cfg = TrainConfig()
    assert not cfg.observation.enabled
    assert cfg.observation.every_steps == 20 and cfg.observation.layers_per_window == 2
    assert cfg.to_dict()["recipe_observation"] == {}
    assert TrainConfig(recipe_id=MXFP8_MLP, recipe_observation={"enabled": True}).observation.enabled
    with pytest.raises(ValueError, match="native Megatron MXFP8"):
        TrainConfig(recipe_observation={"enabled": True})
    with pytest.raises(ValueError, match="no CPU fallback"):
        TrainConfig(
            learner_backend="torch_test",
            rollout_backend="torch_test",
            recipe_id="fp32-test-v1",
            recipe_observation={"enabled": True},
        )


@pytest.mark.parametrize(
    "options",
    [
        None,
        [],
        {"extra": 1},
        {"enabled": 1},
        {"every_steps": 0},
        {"layers_per_window": -1},
        {"max_rows": True},
        {"repeats": 1.5},
        {"max_snapshot_bytes": 0},
    ],
)
def test_reject_unknown_or_invalid_options_even_when_disabled(options):
    with pytest.raises(ValueError):
        TrainConfig(recipe_observation=options)


def test_round_robin_windows_and_resume_use_preupdate_counters():
    probe = make_probe()
    for step, expected_layers in [(1, []), (19, []), (20, [0, 1]), (40, [2, 3]), (60, [4, 0])]:
        probe.begin(step=step, weight_version=step - 1, optimizer_steps=2 * (step - 1))
        capture_all(probe)
        events, seconds = probe.finish()
        assert sorted(event["layer"] for event in events) == sorted(expected_layers)
        assert seconds >= 0
        for event in events:
            assert event["status"] == "ok"
            assert event["step"] == step and event["weight_version"] == step - 1
            assert event["optimizer_steps"] == 2 * (step - 1)
            assert event["source"] == "measured" and event["phase"] == "learner.fprop"
            json.dumps(event, allow_nan=False)
    restored = make_probe()
    restored.begin(step=40, weight_version=39, optimizer_steps=78)
    capture_all(restored)
    assert [row["layer"] for row in restored.finish()[0]] == [2, 3]


def test_disabled_observer_does_not_capture_or_call_backend(monkeypatch):
    probe = make_probe(layers=1)
    probe.config = replace(probe.config, enabled=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled observation must do no snapshot or native work")

    monkeypatch.setattr(probes, "_bounded_rows", forbidden)
    probe.replay = forbidden
    probe.begin(step=20, weight_version=19, optimizer_steps=19)
    capture_all(probe)
    assert probe.finish() == ([], 0.0)


def test_snapshot_is_bounded_owned_detached_and_captured_once():
    calls = []

    def replay(x, weight, **options):
        assert not torch.is_grad_enabled()
        assert not x.requires_grad and not weight.requires_grad
        assert x.grad_fn is None and weight.grad_fn is None
        calls.append((x.clone(), weight.clone(), options))
        return stub_result()

    probe = make_probe(layers=1, max_rows=3, replay=replay)
    # Reshaping this noncontiguous activation would copy all 10 rows.
    x = torch.arange(5 * 2 * 64, dtype=torch.float32).reshape(5, 2, 64).transpose(0, 1)
    x.requires_grad_()
    weight = torch.ones(32, 64, requires_grad=True)
    expected_x = x.detach().reshape(-1, 64)[:3].clone()
    probe.begin(step=20, weight_version=19, optimizer_steps=8)
    probe.capture(down_proj_name(0), x, weight)
    with torch.no_grad():
        x.add_(1000)
        weight.add_(1000)
    probe.capture(down_proj_name(0), x, weight)
    events, _ = probe.finish()
    probe.capture(down_proj_name(0), x, weight)  # Later microbatch must be ignored.
    assert probe.finish() == ([], 0.0)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], expected_x, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], torch.ones_like(weight), rtol=0, atol=0)
    assert events[0]["original_rows"] == 10 and events[0]["shape_mnk"] == [3, 32, 64]
    assert events[0]["snapshot_bytes"] == (3 * 64 + 32 * 64) * 4
    assert probe._snapshots == {} and probe._snapshot_bytes == 0


def test_byte_cap_checked_before_copy_and_cumulative_across_layers(monkeypatch):
    copied = []
    original = probes._bounded_rows

    def counted(x, rows):
        copied.append(rows)
        return original(x, rows)

    monkeypatch.setattr(probes, "_bounded_rows", counted)
    one_snapshot = (3 * 64 + 32 * 64) * 4
    probe = make_probe(layers=2, max_rows=3, max_snapshot_bytes=one_snapshot)
    probe.begin(step=20, weight_version=19, optimizer_steps=19)
    capture_all(probe)
    rows, _ = probe.finish()
    assert copied == [3]
    assert sorted(event["status"] for event in rows) == ["ok", "skipped"]
    skipped = next(event for event in rows if event["status"] == "skipped")
    assert skipped["variants"] == {} and "cap exceeded" in skipped["error"]
    # A too-small budget cannot even perform the first bounded activation copy.
    copied.clear()
    probe = make_probe(layers=1, max_snapshot_bytes=1)
    probe.begin(step=20, weight_version=19, optimizer_steps=19)
    capture_all(probe)
    assert probe.finish()[0][0]["status"] == "skipped"
    assert not copied


def test_missing_execution_is_explicitly_skipped_and_disarmed():
    probe = make_probe(layers=1)
    probe.begin(step=20, weight_version=19, optimizer_steps=19)
    rows, _ = probe.finish()
    assert rows[0]["status"] == "skipped" and rows[0]["variants"] == {}
    assert "not executed" in rows[0]["error"]
    capture_all(probe)
    assert probe.finish() == ([], 0.0)


@pytest.mark.parametrize("failure", ["exception", "nonfinite", "missing_variant"])
def test_probe_failures_emit_error_without_fake_numbers_and_restore_rng(failure):
    def replay(*args, **kwargs):
        torch.rand(8)
        random.random()
        np.random.rand(8)
        if failure == "exception":
            raise ValueError("explicit stub failure")
        variants, environment = stub_result()
        if failure == "nonfinite":
            variants["mxfp8"]["latency_ms"] = float("nan")
        else:
            del variants["mxfp8"]
        return variants, environment

    probe = make_probe(layers=1, replay=replay)
    probe.begin(step=20, weight_version=19, optimizer_steps=19)
    capture_all(probe)
    torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
    events, seconds = probe.finish()
    assert seconds > 0 and events[0]["probe_seconds"] > 0
    assert events[0]["status"] == "error" and events[0]["variants"] == {}
    assert events[0]["error"]
    json.dumps(events[0], allow_nan=False)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])


def test_cuda_context_failure_is_not_swallowed():
    def fatal(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    probe = make_probe(layers=1, replay=fatal)
    probe.begin(step=20, weight_version=19, optimizer_steps=19)
    capture_all(probe)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        probe.finish()
    assert probe._snapshots == {}


def test_current_hadamard_transform_identity_is_retained():
    recipe = resolve_recipe(MXFP8_HADAMARD, {"layers": [3], "block_size": 32, "seed": 71})
    probe = RecipeProbe(
        RecipeObservationConfig(enabled=True),
        recipe,
        {down_proj_name(3): (32, 64)},
        5,
        replay=lambda *args, **kwargs: stub_result(),
    )
    probe.begin(step=20, weight_version=19, optimizer_steps=19)
    capture_all(probe)
    row = probe.finish()[0][0]
    assert row["hadamard_block_size"] == 32 and row["hadamard_seed"] == 74
    assert row["recipe_id"] == recipe.identity and row["layer"] == 3 and row["num_layers"] == 5


def test_megatron_adapter_captures_canonical_operands_before_active_rotation(monkeypatch):
    recipe = resolve_recipe(MXFP8_HADAMARD, {"block_size": 32, "seed": 9})
    module = torch.nn.Linear(64, 32, bias=False)
    x = torch.randn(3, 64, requires_grad=True)
    state = RecipeState(recipe)
    name = state.add_target(0, module.weight)
    installation = megatron_mxfp8.MegatronMXFP8Installation(state, {"backend": "test_stub"})
    captured, executed = [], []

    def capture(target, inputs, weight):
        assert target == name and inputs is x and weight is module.weight
        captured.append(1)

    def native_stub(inputs, weight):
        assert captured == [1]
        torch.testing.assert_close(inputs, state.rotate(x, name))
        torch.testing.assert_close(weight, state.rotate(module.weight, name))
        executed.append(1)
        return torch.nn.functional.linear(inputs, weight)

    installation.observer = SimpleNamespace(capture=capture)
    monkeypatch.setattr(megatron_mxfp8, "native_mxfp8_linear", native_stub)
    output, bias = installation.forward(module, x, name)
    output.sum().backward()
    assert bias is None and executed == [1] and installation.calls[name] == 1
    assert x.grad is not None and module.weight.grad is not None


def _cpu_training_case():
    cfg = TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        recipe_id="fp32-test-v1",
        micro_batch_size=1,
        update_epochs=2,
    )
    reference, learner = CPULearner(cfg), CPULearner(cfg)
    rows = [
        Trajectory(
            0, index, (1, 3), (token, 2), (-math.log(14),) * 2, str(index), "stop", learner.policy_version
        )
        for index, token in enumerate((5, 6))
    ]
    batch = TrainingBatch(rows, torch.tensor([1.0, 0.0]), torch.tensor([1.0, -1.0]), learner.policy_version)
    return reference, learner, batch


@pytest.mark.parametrize("active", [True, False])
def test_training_replay_after_backward_before_update_preserves_optimizer_and_graph(active):
    reference, learner, batch = _cpu_training_case()
    if not active:
        batch = replace(batch, advantages=torch.zeros_like(batch.advantages))
    initial_weight = learner.model.head.weight.detach().clone()
    calls, saved_roots = [], []

    class SavedTensor:
        def __init__(self, tensor):
            # Match autograd's packed value ownership without adding a Python
            # tensor -> grad_fn -> wrapper reference cycle of our own.
            self.tensor = tensor.detach()

    def pack(tensor):
        wrapper = SavedTensor(tensor)
        saved_roots.append(weakref.ref(wrapper))
        return wrapper

    def replay(x, weight, **kwargs):
        assert learner.optimizer_steps == 0 and learner.policy_version.weight_version == 0
        assert any(p.grad is not None for p in learner.parameters) == active
        assert any(p.grad is not None for p in learner.master_parameters) == active
        # Even an inactive rollout must release its forward graph before replay.
        assert saved_roots and all(reference() is None for reference in saved_roots)
        torch.testing.assert_close(weight, initial_weight, rtol=0, atol=0)
        assert not torch.is_grad_enabled() and not x.requires_grad and not weight.requires_grad
        torch.rand(8)
        random.random()
        # Mutating the owned snapshots must not mutate the graph or parameters.
        x.add_(9)
        weight.mul_(0)
        calls.append(1)
        return stub_result()

    recipe = resolve_recipe(MXFP8_MLP)
    # Deliberate test-only injection. Production TrainConfig rejects CPU probes.
    learner.recipe_probe = RecipeProbe(
        RecipeObservationConfig(enabled=True, every_steps=1, max_rows=2),
        recipe,
        {down_proj_name(0): tuple(learner.model.head.weight.shape)},
        1,
        replay=replay,
    )
    hook = learner.model.head.register_forward_pre_hook(
        lambda module, args: learner.recipe_probe.capture(down_proj_name(0), args[0], module.weight)
    )
    rng = torch.get_rng_state()
    expected = reference.update(batch)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda wrapper: wrapper.tensor):
        actual = learner.update(batch)
    hook.remove()
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(calls) == 1
    assert "recipe_observations" not in expected and "recipe_probe_seconds" not in expected
    event = actual.pop("recipe_observations")[0]
    assert actual.pop("recipe_probe_seconds") > 0
    assert actual == expected
    assert event["status"] == "ok" and event["step"] == 1 and event["weight_version"] == 0
    assert event["optimizer_steps"] == 0 and event["shape_mnk"][0] == 2
    torch.testing.assert_close(learner.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        learner.optimizer.state_dict(), reference.optimizer.state_dict(), rtol=0, atol=0
    )
    assert all(param.grad is None for param in learner.parameters)


def test_native_replay_does_not_emulate_on_cpu():
    with pytest.raises(RuntimeError, match="CUDA|no CPU fallback"):
        probes.native_replay(torch.ones(2, 64), torch.ones(32, 64), repeats=1, block_size=32, seed=0)


def test_timing_control_flow_warms_and_replays_all_full_variants_equally(monkeypatch):
    """Mock GPU boundaries solely to verify counts; these timings are not measurements."""
    from qorerl.recipes import te_mxfp8

    calls, rotations, events, syncs = [], [], [], []
    original_linear, original_apply = probes.F.linear, probes.PreparedHadamard.apply

    def linear(x, weight):
        assert not torch.is_grad_enabled()
        calls.append((x.detach().clone(), weight.detach().clone()))
        return original_linear(x, weight)

    def rotate(transform, x):
        rotations.append(x.shape)
        return original_apply(transform, x)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            events.append("record")

        def synchronize(self):
            syncs.append("event")

        def elapsed_time(self, end):
            return 1.25  # Explicit sentinel, not a CPU timing presented as GPU data.

    monkeypatch.setattr(te_mxfp8, "validate_native_mxfp8", lambda device: {"backend": "test_stub"})
    monkeypatch.setattr(te_mxfp8, "native_mxfp8_linear", linear)
    monkeypatch.setattr(probes.F, "linear", linear)
    monkeypatch.setattr(probes.PreparedHadamard, "apply", rotate)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: syncs.append("warmup")),
    )
    x, weight = torch.ones(3, 64).bfloat16(), torch.ones(32, 64).bfloat16()
    variants, environment = probes.native_replay(x, weight, repeats=4, block_size=32, seed=7)
    assert len(calls) == 3 * (1 + 4)
    assert len(rotations) == 2 * (1 + 4)  # Both X and W, every warmup/repeat.
    assert events == ["record"] * (2 * 3 * 4)
    assert syncs == ["warmup", *(["event"] * (3 * 4))]
    assert environment["backend"] == "test_stub"
    assert all(result["latency_ms"] == 1.25 for result in variants.values())
    assert variants["bf16"]["output_relative_l2"] == 0


@pytest.mark.parametrize("reference,output,expected", [(2.0, 3.0, 0.5), (0.0, 0.0, 0.0)])
def test_local_relative_error_definition(reference, output, expected):
    assert probes._relative_l2(torch.full((2, 2), output), torch.full((2, 2), reference)) == expected


@pytest.mark.parametrize("reference,output", [(0.0, 1.0), (float("nan"), 1.0), (1.0, float("inf"))])
def test_undefined_local_relative_error_is_not_zeroed(reference, output):
    with pytest.raises(FloatingPointError):
        probes._relative_l2(torch.full((2, 2), output), torch.full((2, 2), reference))
