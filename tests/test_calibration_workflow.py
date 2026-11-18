"""Exercise review, resume, and external-scale publication across real Ray processes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import ray
import torch

from qorerl.recipes.calibration import make_scale_artifact
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


def test_reviewed_change_and_auto_apply_yield_the_same_continuation(tmp_path, cluster):
    cfg = replace(TrainConfig.load(CONFIG), steps=2)
    paused = run_sync_grpo(replace(cfg, steps=1), tmp_path / "paused")
    assert paused["status"] == "awaiting_review" and paused["last_step"] == 1
    checkpoint = Path(paused["checkpoint"])
    review = json.loads((checkpoint / "candidate.json").read_text())
    assert [row["status"] for row in review["evidence"]] == ["proposed", "no_gain", "not_selected"]
    assert list(review["scale_artifact"]["scales"]) == ["model.layers.0.mlp.down_proj.weight"]
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
    accepted = run_sync_grpo(
        cfg, tmp_path / "accepted", resume=checkpoint, accept_candidate=paused["candidate_id"]
    )
    new_policy = events(tmp_path / "accepted/calibration.jsonl")[0]["policy"]
    assert new_policy["weight_version"] == paused["policy_version"]["weight_version"]
    assert new_policy["recipe_id"] != paused["policy_version"]["recipe_id"]
    assert all(
        t["policy_version"] == new_policy
        for t in events(tmp_path / "accepted/trajectories.jsonl")[0]["trajectories"]
    )
    auto = run_sync_grpo(
        replace(cfg, recipe_calibration={**cfg.recipe_calibration, "approval": "auto"}), tmp_path / "auto"
    )
    assert auto["status"] == accepted["status"] == "completed"
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
