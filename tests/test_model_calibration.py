"""Joint model selection, reference execution, and provisional-state isolation on CPU."""

import io
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from qorerl.engines.calibration_replay import CalibrationReplay
from qorerl.engines.factory import make_learner_engine, make_rollout_engine
from qorerl.recipes.calibration import CalibrationConfig
from qorerl.recipes.smoothquant import apply_candidate, prepare_smoothquant_candidate
from qorerl.recipes.state import down_proj_name
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import TrainingBatch
from qorerl.training.objective import compute_group_advantages
from qorerl.training.tasks import CopyTask, exact_match_rewards


def setup(*, recipe_options=None, **calibration):
    cfg = TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        model="cpu-fixture",
        recipe_id="mxfp8-smoothquant-mlp-down-v1",
        group_size=8,
        max_new_tokens=1,
        recipe_options=recipe_options or {},
        recipe_calibration={"evaluation_scope": "model_logits", **calibration},
    )
    learner, rollout = make_learner_engine(cfg), make_rollout_engine(cfg)
    policy = rollout.install_weights(learner.export_weights())
    prompts = CopyTask().batch(1, cfg.prompts_per_step, cfg.seed)
    trajectories = rollout.generate(prompts, policy, cfg.seed + 1000003)
    rewards = exact_match_rewards(prompts, trajectories)
    batch = TrainingBatch(trajectories, rewards, compute_group_advantages(rewards, cfg.group_size), policy)
    learner.update(batch)
    return cfg, learner, batch


def snapshot(learner):
    return torch.load(io.BytesIO(learner.checkpoint()), weights_only=True)


def assert_unchanged(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_unchanged(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_unchanged(a, b)
    else:
        assert actual == expected


def test_joint_comparison_uses_bf16_capture_and_actual_model_logits():
    cfg, learner, batch = setup()
    before = snapshot(learner)
    installed = learner.recipe_installation
    replay = CalibrationReplay(learner, batch.trajectories, cfg.calibration.max_rows)
    reference_inputs, reference = replay.run(reference=True, collect_inputs=True)
    current_inputs, current = replay.run(collect_inputs=True)
    # The downstream input changes when the upstream low-precision projection is bypassed.
    assert not torch.equal(reference_inputs[down_proj_name(1)], current_inputs[down_proj_name(1)])
    result = learner.prepare_candidate(batch)
    assert result.candidate is not None and result.comparison["status"] == "proposed"
    assert result.comparison["reference_execution"] == "cpu_fixture"
    assert all(row["current_relative_l2"] is None for row in result.evidence)
    assert len(result.candidate["scales"]) == 2  # Complete fitted combination, no local-error filter.
    proposal_state = deepcopy(learner.recipe_state)
    apply_candidate(cfg, proposal_state, result.candidate)
    candidate_inputs, candidate = replay.run(state=proposal_state, collect_inputs=True)
    assert not torch.equal(candidate_inputs[down_proj_name(1)], current_inputs[down_proj_name(1)])
    norm = reference.double().norm()
    for name, logits in (("current_relative_l2", current), ("candidate_relative_l2", candidate)):
        expected = ((logits.double() - reference.double()).norm() / norm).item()
        assert result.comparison[name] == pytest.approx(expected)
    assert current.shape[0] == result.comparison["heldout_rows"] == replay.heldout_rows
    assert result.comparison["candidate_relative_l2"] < result.comparison["current_relative_l2"]
    assert_unchanged(snapshot(learner), before)
    assert installed.state is learner.recipe_state and installed.calibrator is None
    assert not installed.reference_mode


@pytest.mark.parametrize(
    "overrides,status",
    [
        ({"min_relative_improvement": 1.0}, "no_gain"),
        ({"max_logits_relative_l2": 0.0}, "error_cap_exceeded"),
    ],
)
def test_joint_threshold_rejection_does_not_change_live_state(overrides, status):
    _, learner, batch = setup(**overrides)
    before = snapshot(learner)
    result = learner.prepare_candidate(batch)
    assert result.candidate is None and result.comparison["status"] == status
    assert any(row["status"] == "model_rejected" for row in result.evidence)
    assert_unchanged(snapshot(learner), before)


@pytest.mark.parametrize("failure", ["nan", "backend"])
def test_failed_candidate_replay_restores_state_rng_modes_and_observers(monkeypatch, failure):
    _, learner, batch = setup()
    installation = learner.recipe_installation
    active = learner.recipe_state.active_identity
    before = snapshot(learner)
    original_logits = learner.logits
    sentinel = object()
    installation.observer = sentinel
    learner.model.train()  # Replay must restore each module's incoming mode.
    learner.model.head.eval()
    modes = [m.training for m in learner.model.modules()]

    def logits(ids):
        torch.rand(1)  # Even a failing provisional run must preserve the caller's RNG.
        if installation.state.active_identity != active:
            if failure == "backend":
                raise RuntimeError("backend failure during candidate replay")
            return torch.full((1, ids.shape[1], 14), float("nan"))
        return original_logits(ids)

    monkeypatch.setattr(learner, "logits", logits)
    if failure == "backend":
        with pytest.raises(RuntimeError, match="backend failure"):
            learner.prepare_candidate(batch)
    else:
        result = learner.prepare_candidate(batch)
        assert result.candidate is None and result.comparison["status"] == "invalid_candidate"
        assert result.comparison["candidate_relative_l2"] is None
    assert installation.state is learner.recipe_state
    assert installation.observer is sentinel and installation.calibrator is None
    assert not installation.reference_mode
    assert [m.training for m in learner.model.modules()] == modes
    assert_unchanged(snapshot(learner), before)


def test_reference_bypasses_every_low_precision_target_including_noneditable_layers(monkeypatch):
    cfg, _, batch = setup()
    cfg = replace(cfg, recipe_options={"layers": [0, 1], "smoothquant_layers": [1]})
    learner = make_learner_engine(cfg)
    replay = CalibrationReplay(learner, batch.trajectories, 5)

    def fail(*args):
        raise RuntimeError("low precision was executed")

    monkeypatch.setattr(learner.recipe_installation, "linear", fail)
    samples, logits = replay.run(reference=True, collect_inputs=True)
    assert set(samples) == {down_proj_name(1)}
    assert samples[down_proj_name(1)].shape[0] == 5
    assert logits.shape[0] == 3
    with pytest.raises(RuntimeError, match="low precision was executed"):
        replay.run()
    assert not learner.recipe_installation.reference_mode


def test_joint_fit_can_activate_smoothing_only_inside_low_precision_scope():
    cfg, learner, batch = setup(
        recipe_options={"layers": [0, 1], "smoothquant_layers": [1], "allow_layer_expansion": True}
    )
    result = learner.prepare_candidate(batch)
    assert result.candidate is not None
    assert set(result.comparison["targets"]) == {down_proj_name(0), down_proj_name(1)}
    assert {row["target"]: row["action"] for row in result.evidence} == {
        down_proj_name(0): "activate",
        down_proj_name(1): "refresh",
    }
    assert set(learner.recipe_state.smoothing_factors) == {down_proj_name(1)}
    learner.apply_candidate(result.candidate)
    assert set(learner.recipe_state.smoothing_factors) == {down_proj_name(0), down_proj_name(1)}
    assert down_proj_name(2) not in learner.recipe_state.targets


def test_individually_better_projections_can_fail_the_joint_gate(monkeypatch):
    """Controlled cancellation: +1/-1 local errors cancel; +0.5/-0.25 do not."""
    from qorerl.recipes import smoothquant_model

    cfg, learner, batch = setup(discover_outliers=False)
    names = [down_proj_name(i) for i in (0, 1)]
    x = torch.full((4, 32), 4, dtype=torch.bfloat16)
    w = torch.eye(32, dtype=torch.bfloat16)
    # Each synthetic projection improves locally under its fitted factor S=2.
    for offset in (1.0, -1.0):

        def kernel(a, b):
            scaled = float(b[0, 0]) > 1
            error = (0.5 if offset > 0 else -0.25) if scaled else offset
            return torch.nn.functional.linear(a, b) + error

        factors, [row] = prepare_smoothquant_candidate(
            {names[0]: w},
            {names[0]: x},
            {names[0]: torch.ones(32)},
            CalibrationConfig(discover_outliers=False),
            fprop=kernel,
        )
        assert factors and row["candidate_relative_l2"] < row["current_relative_l2"]

    def fit(*args):
        return {n: torch.full((32,), 2.0) for n in names}, [
            {"target": n, "status": "fitted", "current_relative_l2": None, "candidate_relative_l2": None}
            for n in names
        ]

    class Replay:
        fit_rows, heldout_rows, reference_execution = 2, 2, "cpu_fixture"

        def run(self, *, reference=False, state=None, collect_inputs=False):
            base = torch.full((2, 32), 4.0)
            # Two identity layers: current +1 then -1; candidate +0.5 then -0.25.
            output = base if reference else (base + 0.5 - 0.25 if state else base + 1 - 1)
            return {n: x for n in names}, output

    monkeypatch.setattr(smoothquant_model, "_fit_joint_factors", fit)
    result = smoothquant_model.prepare_candidate(
        cfg, learner.recipe_state, learner.recipe_installation, Replay()
    )
    assert result.candidate is None and result.comparison["status"] == "no_gain"
    assert result.comparison["current_relative_l2"] == 0
    assert result.comparison["candidate_relative_l2"] > 0
    assert all(row["status"] == "model_rejected" for row in result.evidence)


@pytest.mark.parametrize(
    "options",
    [
        {"evaluation_scope": "unknown"},
        {"evaluation_scope": "model_logits", "discovery_only": True},
        {"evaluation_scope": "model_logits", "max_activation_relative_l2": 0.1},
        {"max_logits_relative_l2": 0.1},
        {"evaluation_scope": "model_logits", "max_logits_relative_l2": float("nan")},
    ],
)
def test_invalid_scope_and_threshold_combinations(options):
    with pytest.raises(ValueError):
        CalibrationConfig(**options)
