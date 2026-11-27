"""Exercise review, resume, and external-scale publication across real Ray processes."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import ray
import torch

from qorerl.engines.factory import make_learner_engine
from qorerl.observability.viewer import export_report, load_report
from qorerl.recipes.state import down_proj_name
from qorerl.runtime.calibration import (
    crosses_interval,
    prepare_review,
    read_payload,
    read_review,
    review_candidate,
)
from qorerl.runtime.ray_runtime import run_sync_grpo
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import PolicyVersion
from qorerl.validation import validate_run

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


def test_joint_logits_candidate_review_publication_and_evidence(tmp_path, cluster):
    cfg = TrainConfig.load(CONFIG)
    cfg = replace(
        cfg, steps=1, recipe_calibration={**cfg.recipe_calibration, "evaluation_scope": "model_logits"}
    )
    paused = run_sync_grpo(cfg, tmp_path / "paused")
    assert paused["status"] == "awaiting_review"
    checkpoint = Path(paused["checkpoint"])
    review = json.loads((checkpoint / "candidate.json").read_text())
    comparison = review["comparison"]
    assert comparison["status"] == "proposed" and comparison["scope"] == "model_logits"
    assert comparison["reference_execution"] == "cpu_fixture"
    assert set(comparison["targets"]) == set(review["candidate"]["scales"])
    assert len(comparison["targets"]) == 2
    assert all(row["current_relative_l2"] is None for row in review["evidence"])
    accepted = run_sync_grpo(
        replace(cfg, steps=2),
        tmp_path / "accepted",
        resume=checkpoint,
        accept_candidate=paused["candidate_id"],
    )
    applied = events(tmp_path / "accepted/calibration.jsonl")[0]
    trajectories = events(tmp_path / "accepted/trajectories.jsonl")[0]["trajectories"]
    assert applied["event"] == "applied"
    assert applied["policy"]["weight_version"] == paused["policy_version"]["weight_version"]
    assert applied["policy"]["recipe_id"] != paused["policy_version"]["recipe_id"]
    assert all(t["policy_version"] == applied["policy"] for t in trajectories)
    report = load_report(tmp_path / "accepted", related_runs=[tmp_path / "paused"])
    assert not report["warnings"]
    first = report["calibration"]["evaluations"][0]
    assert first["comparison"] == comparison and first["decision"] == "applied"
    audit = validate_run(tmp_path / "accepted", related_runs=[tmp_path / "paused"], require_calibration=True)
    checks = {item["name"]: item["status"] for item in audit["checks"]}
    assert checks["calibration_apply_and_continue"] == "pass"
    assert checks["native_execution"] == "fail" and audit["result"] == "failed"
    auto = run_sync_grpo(
        replace(cfg, steps=2, recipe_calibration={**cfg.recipe_calibration, "approval": "auto"}),
        tmp_path / "auto",
    )
    if accepted["status"] == "awaiting_review":
        # The second batch may propose another jointly improving combination.
        # Resolve that final-boundary decision to compare with the auto path.
        accepted = run_sync_grpo(
            replace(cfg, steps=2),
            tmp_path / "accepted-final",
            resume=accepted["checkpoint"],
            accept_candidate=accepted["candidate_id"],
        )
    assert auto["status"] == accepted["status"]
    assert auto["policy_version"] == accepted["policy_version"]
    assert events(tmp_path / "auto/trajectories.jsonl")[1]["trajectories"] == trajectories
    for key in ("model", "master_parameters", "optimizer", "optimizer_steps"):
        a, b = [
            torch.load(Path(run["checkpoint"]) / "learner.pt", weights_only=True)[key]
            for run in (auto, accepted)
        ]
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("expand", [False, True], ids=["fixed-scope", "first-activation"])
def test_reviewed_change_and_auto_apply_yield_the_same_continuation(tmp_path, cluster, expand):
    options = (
        {"layers": [0, 1], "smoothquant_layers": [1], "allow_layer_expansion": True}
        if expand
        else {"layers": [0, 1, 2]}
    )
    cfg = replace(
        TrainConfig.load(CONFIG),
        steps=2,
        recipe_options=options,
        worker_placement="collocate" if expand else "separate",
        rollout_workers=1 if expand else 2,
    )
    base = {down_proj_name(index) for index in options["layers"]}
    initial_overlay = {down_proj_name(1)} if expand else base
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
    if expand:
        assert set(evidence) == base
    else:
        assert evidence[down_proj_name(2)]["status"] == "not_selected"
    assert review["schema"] == 2
    assert set(review["candidate"]["scales"]) == {candidate_target}
    assert len(paused["execution"]) == cfg.rollout_workers + 1
    for worker in paused["execution"]:
        assert set(worker["targets"]) == base
        assert set(worker["smoothquant_targets"]) == initial_overlay
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
        assert set(worker["targets"]) == base
        assert set(worker["smoothquant_targets"]) == initial_overlay | {candidate_target}
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
    # Real persisted CPU evidence drives the same viewer used on GPU runs.
    report = load_report(tmp_path / "accepted", related_runs=[tmp_path / "paused"])
    initial = report["calibration"]["evaluations"][0]
    assert initial["id"] == review["evaluation_id"] == applied["evaluation_id"]
    assert initial["decision"] == "applied" and initial["source"] == "cpu_simulation"
    assert not report["warnings"]
    export_report(tmp_path / "accepted", tmp_path / "calibration.html", related_runs=[tmp_path / "paused"])
    audit = validate_run(tmp_path / "accepted", related_runs=[tmp_path / "paused"], require_calibration=True)
    checks = {item["name"]: item["status"] for item in audit["checks"]}
    assert checks["calibration_apply_and_continue"] == "pass"
    assert checks["sampling_after_publication"] == "pass"
    assert checks["native_execution"] == "fail" and audit["result"] == "failed"


def test_external_artifact_installed_before_first_sampling_and_restored_without_source_file(
    tmp_path, cluster, smoothing_artifact
):
    cfg = replace(
        TrainConfig.load(CONFIG),
        steps=1,
        recipe_options={"layers": [0]},
        recipe_calibration={"initial_artifact": "scales.json", "refresh_every_optimizer_steps": 0},
    )
    config_path = tmp_path / "run.json"
    saved_config = cfg.to_dict()
    options = saved_config["recipe_calibration"]
    options["scale_artifact"] = options.pop("initial_artifact")
    config_path.write_text(json.dumps(saved_config))
    artifact = smoothing_artifact(cfg, {down_proj_name(0): torch.full((32,), 8.0)})
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(json.dumps(artifact))
    cfg = TrainConfig.load(config_path)
    assert cfg.calibration.initial_artifact == str(scale_path)
    assert cfg.to_dict()["recipe_calibration"]["initial_artifact"] == str(scale_path)
    initial = run_sync_grpo(cfg, tmp_path / "imported")
    imported = events(tmp_path / "imported/calibration.jsonl")
    assert [e["event"] for e in imported] == ["imported"]
    assert imported[0]["policy"]["recipe_id"].endswith(artifact["scale_digest"])
    assert all(
        t["policy_version"] == imported[0]["policy"]
        for t in events(tmp_path / "imported/trajectories.jsonl")[0]["trajectories"]
    )
    # The original checkpoint metadata stored the absolute path under the legacy key.
    metadata_file = Path(initial["checkpoint"]) / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    saved_options = metadata["config"]["recipe_calibration"]
    saved_options["scale_artifact"] = saved_options.pop("initial_artifact")
    metadata_file.write_text(json.dumps(metadata))
    scale_path.unlink()
    continued = run_sync_grpo(replace(cfg, steps=2), tmp_path / "continued", resume=initial["checkpoint"])
    assert continued["policy_version"]["recipe_id"] == initial["policy_version"]["recipe_id"]


def test_accepts_a_saved_v1_review_with_its_original_confirmation_id(tmp_path, cluster):
    cfg = replace(
        TrainConfig.load(CONFIG),
        steps=1,
        recipe_options={"layers": [0]},
        recipe_calibration={"refresh_every_optimizer_steps": 20},
    )
    checkpoint = tmp_path / "v1"
    checkpoint.mkdir()
    learner = make_learner_engine(cfg)
    payload = learner.checkpoint()
    policy = asdict(learner.policy_version)
    target = down_proj_name(0)
    digest = "294e00f8a72f4aae7424c20f40ead180e11b671ef515d759dc5eee5abfb21c04"
    preset = "mxfp8-smoothquant-mlp-down-v1:df3af9f4b126c793"
    review = {
        "schema": 1,
        "source_policy": policy,
        "optimizer_steps": 0,
        "scale_artifact": {
            "schema": 1,
            "kind": "smoothquant_scales",
            "model": cfg.model,
            "model_revision": cfg.model_revision,
            "recipe_id": preset,
            "scales": {target: [2.0] * 32},
            "scale_digest": digest,
        },
        "evidence": [],
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
    }
    # A reviewer confirms the hash of the persisted envelope, before schema translation.
    encoded = json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
    candidate_id = hashlib.sha256(encoded).hexdigest()
    (checkpoint / "learner.pt").write_bytes(payload)
    (checkpoint / "candidate.json").write_text(json.dumps(review))
    metadata = {
        "schema": 2,
        "step": 0,
        "policy_version": policy,
        "config": cfg.to_dict(),
        "pending_candidate": candidate_id,
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "accepted"
    result = run_sync_grpo(cfg, output, resume=checkpoint, accept_candidate=candidate_id)
    [applied] = events(output / "calibration.jsonl")
    assert result["status"] == "completed"
    assert applied["candidate_id"] == candidate_id
    assert applied["policy"] == {"weight_version": 0, "recipe_id": f"{preset}:{digest}"}
    [batch] = events(output / "trajectories.jsonl")
    assert all(item["policy_version"] == applied["policy"] for item in batch["trajectories"])


def test_review_preserves_an_opaque_recipe_payload():
    artifact = {"kind": "test_recipe", "parameters": {"blocks": [2, 4]}}
    evidence = [{"target": "block.2", "status": "proposed"}]
    policy = PolicyVersion(7, "test-v1")
    review = prepare_review(policy, 5, artifact, evidence)
    assert review == {
        "schema": 2,
        "source_policy": asdict(policy),
        "optimizer_steps": 5,
        "candidate": artifact,
        "evidence": evidence,
    }
    assert review_candidate(json.loads(json.dumps(review))) == artifact


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
