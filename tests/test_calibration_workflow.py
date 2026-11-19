"""Exercise review, resume, and external-scale publication across real Ray processes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import ray
import torch

from qorerl.recipes.calibration import make_scale_artifact
from qorerl.recipes.state import down_proj_name
from qorerl.runtime.calibration import artifact_context, crosses_interval, read_payload, read_review
from qorerl.runtime.ray_runtime import run_sync_grpo
from qorerl.training.config import TrainConfig

CONFIG = Path(__file__).resolve().parents[1] / "configs/grpo_cpu_smoothquant.json"


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def cluster():
    ray.init(num_cpus=3, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


@pytest.mark.parametrize("expand", [False, True], ids=["fixed-scope", "first-activation"])
def test_reviewed_change_and_auto_apply_yield_the_same_continuation(tmp_path, cluster, expand):
    options = {"layers": [1], "allow_layer_expansion": True} if expand else {"layers": [0, 1, 2]}
    cfg = replace(TrainConfig.load(CONFIG), steps=2, recipe_options=options)
    initial_targets = {down_proj_name(index) for index in options["layers"]}
    candidate_target = down_proj_name(0)
    paused = run_sync_grpo(replace(cfg, steps=1), tmp_path / "paused")
    assert paused["status"] == "awaiting_review" and paused["last_step"] == 1
    checkpoint = Path(paused["checkpoint"])
    review = json.loads((checkpoint / "candidate.json").read_text())
    evidence = {row["target"]: row for row in review["evidence"]}
    assert evidence[candidate_target]["action"] == ("activate" if expand else "refresh")
    assert evidence[candidate_target]["status"] == "proposed"
    existing = evidence[down_proj_name(1)]
    assert existing["action"] == "refresh" and existing["status"] == "no_gain"
    assert evidence[down_proj_name(2)]["status"] == "not_selected"
    assert set(review["scale_artifact"]["scales"]) == {candidate_target}
    assert len(paused["execution"]) == cfg.rollout_workers + 1
    for worker in paused["execution"]:
        assert set(worker["targets"]) == initial_targets
        assert worker["identity"] == paused["policy_version"]["recipe_id"]
        if expand:
            assert set(worker["eligible_targets"]) == set(evidence)
    with pytest.raises(ValueError, match="--accept-candidate"):
        run_sync_grpo(cfg, tmp_path / "missing", resume=checkpoint)
    with pytest.raises(ValueError, match="different calibration candidate"):
        run_sync_grpo(cfg, tmp_path / "wrong", resume=checkpoint, accept_candidate="wrong-id")
    # A decision at the final boundary must not require taking another optimizer step.
    rejected = run_sync_grpo(
        replace(cfg, steps=1),
        tmp_path / "rejected",
        resume=checkpoint,
        reject_candidate=paused["candidate_id"],
    )
    assert rejected["status"] == "completed" and rejected["final_metrics"] is None
    assert rejected["policy_version"] == paused["policy_version"]
    assert rejected["execution"] == paused["execution"]
    accepted = run_sync_grpo(
        cfg, tmp_path / "accepted", resume=checkpoint, accept_candidate=paused["candidate_id"]
    )
    applied, reevaluated = events(tmp_path / "accepted/calibration.jsonl")
    new_policy = applied["policy"]
    next_actions = {row["target"]: row["action"] for row in reevaluated["evidence"]}
    assert next_actions[candidate_target] == "refresh"
    assert new_policy["weight_version"] == paused["policy_version"]["weight_version"]
    assert new_policy["recipe_id"] != paused["policy_version"]["recipe_id"]
    for worker in accepted["execution"]:
        assert set(worker["targets"]) == initial_targets | {candidate_target}
        assert worker["identity"] == accepted["policy_version"]["recipe_id"]
    next_batch = events(tmp_path / "accepted/trajectories.jsonl")[0]
    assert all(t["policy_version"] == new_policy for t in next_batch["trajectories"])
    auto = run_sync_grpo(
        replace(cfg, recipe_calibration={**cfg.recipe_calibration, "approval": "auto"}), tmp_path / "auto"
    )
    assert auto["status"] == accepted["status"] == "completed"
    assert events(tmp_path / "auto/trajectories.jsonl")[1] == next_batch
    assert auto["policy_version"] == accepted["policy_version"]
    a, b = [torch.load(Path(run["checkpoint"]) / "learner.pt", weights_only=True) for run in (auto, accepted)]
    for field in ("model", "master_parameters", "optimizer", "optimizer_steps"):
        torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)


def test_external_scale_installed_before_first_sampling_and_restored_without_source_file(tmp_path, cluster):
    cfg = replace(
        TrainConfig.load(CONFIG),
        steps=1,
        recipe_options={"layers": [0]},
        recipe_calibration={"scale_artifact": "scales.json", "refresh_every_optimizer_steps": 0},
    )
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    scales = make_scale_artifact(
        {"model.layers.0.mlp.down_proj.weight": torch.full((32,), 8.0)}, **artifact_context(cfg)
    )
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(json.dumps(scales))
    cfg = TrainConfig.load(config_path)
    initial = run_sync_grpo(cfg, tmp_path / "imported")
    imported = events(tmp_path / "imported/calibration.jsonl")
    assert [e["event"] for e in imported] == ["imported"]
    assert imported[0]["policy"]["recipe_id"].endswith(scales["scale_digest"])
    assert all(
        t["policy_version"] == imported[0]["policy"]
        for t in events(tmp_path / "imported/trajectories.jsonl")[0]["trajectories"]
    )
    scale_path.unlink()
    continued = run_sync_grpo(replace(cfg, steps=2), tmp_path / "continued", resume=initial["checkpoint"])
    assert continued["policy_version"]["recipe_id"] == initial["policy_version"]["recipe_id"]


def test_incomplete_review_or_substituted_checkpoint_is_rejected(tmp_path):
    (tmp_path / "candidate.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete"):
        read_review(tmp_path, {}, None, None, None)
    (tmp_path / "learner.pt").write_bytes(b"other state")
    with pytest.raises(ValueError, match="payload"):
        read_payload(tmp_path, {"checkpoint_sha256": "wrong"})


def test_schedule_uses_completed_updates_at_batch_boundaries():
    assert not crosses_interval(19, 19, 20)
    assert crosses_interval(19, 22, 20)
    assert not crosses_interval(20, 22, 20)
    assert not crosses_interval(19, 22, 0)
