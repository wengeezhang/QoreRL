"""Acceptance checks consume artifacts, never treat CPU tests as native proof."""

import json
from pathlib import Path

import pytest
import torch

from qorerl.training.config import TrainConfig
from qorerl.validation import preflight, validate_run, write_report


def save(path, value):
    path.write_text(json.dumps(value))


def log(path, values):
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


def policy(step):
    return {"weight_version": step, "recipe_id": "bf16-baseline-v1"}


def environment():
    return {
        "cuda_available": True,
        "packages": {"torch": "test"},
        "devices": [{"peak_reserved_bytes": 4096, "capability": [10, 0]}],
    }


def complete_fixture(path):
    """Synthetic native-shaped records test the auditor, not hardware execution."""
    save(
        path / "config.json",
        {
            "learner_backend": "megatron",
            "rollout_backend": "vllm",
            "rollout_workers": 1,
            "worker_placement": "collocate",
            "update_epochs": 1,
        },
    )
    save(path / "workers.json", {"workers": [{"node": "n", "gpu_ids": ["0"]}] * 2})
    save(
        path / "summary.json",
        {
            "native_gpu_executed": True,
            "status": "completed",
            "first_step": 0,
            "last_step": 2,
            "policy_version": policy(2),
            "execution": [{"environment": environment()}, {"workers": [{"environment": environment()}]}],
        },
    )
    metrics = [
        {
            "step": step,
            "loss": 0.1,
            "gradient_norm": 0.5,
            "optimizer_steps": step,
            "pre_update_abs_log_ratio_mean": 0.0001,
            "pre_update_abs_log_ratio_max": 0.001,
            "nonzero_advantage_fraction": 1,
            "skipped_zero_advantage": False,
        }
        for step in (1, 2)
    ]
    log(path / "metrics.jsonl", metrics)
    log(path / "publications.jsonl", [{"policy": policy(s), "acknowledged": [policy(s)]} for s in (0, 1, 2)])
    log(
        path / "trajectories.jsonl",
        [
            {
                "step": s,
                "trajectories": [
                    {
                        "policy_version": policy(s - 1),
                        "response_ids": [3],
                        "behavior_log_probs": [-0.01],
                    }
                ],
            }
            for s in (1, 2)
        ],
    )
    for s in (0, 2):
        cp = path / "checkpoints" / f"step-{s:06d}"
        cp.mkdir(parents=True)
        torch.save(
            {
                "model": {"weight": torch.tensor([float(s)])},
                "optimizer_steps": s,
                "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.1])}}},
            },
            cp / "learner.pt",
        )
    return metrics


def checks(result):
    return {c["name"]: c["status"] for c in result["checks"]}


def test_complete_smoke_evidence_has_narrow_scope_and_explicit_limits(tmp_path):
    complete_fixture(tmp_path)
    result = validate_run(tmp_path, max_logprob_gap=0.01, check_checkpoints=True)
    assert result["result"] == "passed", result
    assert result["scope"] == "native_smoke_acceptance"
    assert "not full GPU validation" in result["interpretation"]
    output = write_report(result, tmp_path / "acceptance")
    assert json.loads((output / "report.json").read_text())["result"] == "passed"
    assert "Not assessed" in (output / "report.md").read_text()
    with pytest.raises(FileExistsError):
        write_report(result, output)


def test_missing_tolerance_and_contents_cannot_pass(tmp_path):
    complete_fixture(tmp_path)
    result = validate_run(tmp_path)
    assert result["result"] == "incomplete"
    assert checks(result)["same_snapshot_logprob_gap"] == "missing"
    assert checks(result)["checkpoint_contents"] == "missing"


def test_disabled_module_extra_state_does_not_count_as_a_parameter(tmp_path):
    complete_fixture(tmp_path)
    for path in tmp_path.glob("checkpoints/*/learner.pt"):
        payload = torch.load(path, weights_only=True)
        payload["model"]["projection._extra_state"] = None
        torch.save(payload, path)
    assert validate_run(tmp_path, max_logprob_gap=0.01, check_checkpoints=True)["result"] == "passed"


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("cpu", "native_execution"),
        ("zero_updates", "optimizer_updates"),
        ("ack", "publication_acknowledgments"),
        ("wrong_samples", "sampling_after_publication"),
        ("tail_gap", "same_snapshot_logprob_gap"),
        ("unchanged_weights", "checkpoint_contents"),
        ("partial", "complete_readable_artifacts"),
        ("incident", "incident"),
    ],
)
def test_false_successes_are_rejected(tmp_path, failure, expected):
    metrics = complete_fixture(tmp_path)
    if failure == "cpu":
        save(tmp_path / "config.json", {"learner_backend": "torch_test", "rollout_backend": "torch_test"})
    elif failure == "zero_updates":
        for row in metrics:
            row.update(optimizer_steps=0, skipped_zero_advantage=True, nonzero_advantage_fraction=0)
        log(tmp_path / "metrics.jsonl", metrics)
    elif failure == "ack":
        log(tmp_path / "publications.jsonl", [{"policy": policy(0), "acknowledged": [policy(3)]}])
    elif failure == "wrong_samples":
        log(tmp_path / "trajectories.jsonl", [{"step": 1, "trajectories": [{"policy_version": policy(3)}]}])
    elif failure == "tail_gap":
        metrics[-1]["pre_update_abs_log_ratio_max"] = 3
        log(tmp_path / "metrics.jsonl", metrics)
    elif failure == "unchanged_weights":
        before = tmp_path / "checkpoints/step-000000/learner.pt"
        after = tmp_path / "checkpoints/step-000002/learner.pt"
        after.write_bytes(before.read_bytes())
    elif failure == "partial":
        with (tmp_path / "metrics.jsonl").open("a") as stream:
            stream.write("{")
    else:
        save(tmp_path / "incident.json", {"message": "publication failed"})
    result = validate_run(tmp_path, max_logprob_gap=0.01, check_checkpoints=True)
    assert result["result"] != "passed"
    assert checks(result)[expected] in ("fail", "missing")


def test_claimed_calibration_needs_source_evidence_and_post_apply_samples(tmp_path):
    complete_fixture(tmp_path)
    log(
        tmp_path / "calibration.jsonl",
        [{"event": "applied", "step": 1, "policy": policy(1), "candidate_id": "unknown"}],
    )
    result = validate_run(tmp_path, max_logprob_gap=0.01, check_checkpoints=True, require_calibration=True)
    assert checks(result)["calibration_apply_and_continue"] == "missing"


def test_preflight_without_cuda_and_invalid_arguments(monkeypatch):
    from qorerl import validation

    monkeypatch.setattr(
        validation,
        "environment_report",
        lambda: {
            "devices": [],
            "cuda_available": False,
            "packages": {},
        },
    )
    cfg = TrainConfig.load(Path(__file__).parents[1] / "configs/grpo_single_gpu.json")
    assert preflight(cfg)["result"] == "failed"
    with pytest.raises(ValueError, match="role"):
        preflight(cfg, role="bad")
    with pytest.raises(ValueError, match="min_updates"):
        validate_run("unused", min_updates=0)
    with pytest.raises(ValueError, match="finite"):
        validate_run("unused", max_logprob_gap=float("nan"))


def test_hopper_native_format_rejection_is_reported(monkeypatch):
    from qorerl import validation
    from qorerl.recipes import te_block_linear

    monkeypatch.setattr(validation, "environment_report", environment)

    def reject(*args):
        raise RuntimeError("requires SM10 or SM11; got SM9")

    monkeypatch.setattr(te_block_linear, "validate_native_format", reject)
    cfg = TrainConfig.load(Path(__file__).parents[1] / "configs/grpo_dynamic_smoothquant.json")
    result = preflight(cfg, role="learner")
    assert checks(result)["native_format"] == "fail"
    assert "SM9" in str(result)
