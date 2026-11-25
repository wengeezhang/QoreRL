"""Runs actual Ray worker processes; no mocks of actor calls or weight transfer."""

import json
from dataclasses import replace

import pytest
import torch

from qorerl.runtime.checkpoint import resume_metadata
from qorerl.runtime.ray_runtime import run_sync_grpo
from qorerl.training.config import TrainConfig


@pytest.mark.parametrize("placement,replicas", [("separate", 2), ("collocate", 1)])
def test_ray_group_training_and_resume(tmp_path, placement, replicas):
    cfg = TrainConfig(
        learner_backend="torch_test",
        rollout_backend="torch_test",
        recipe_id="fp32-test-v1",
        model="cpu-copy",
        steps=3,
        prompts_per_step=4,
        group_size=8,
        max_new_tokens=1,
        max_model_len=16,
        rollout_workers=replicas,
        worker_placement=placement,
        learning_rate=0.003,
        micro_batch_size=7,
        checkpoint_every=2,
        rpc_timeout=60,
    )
    uninterrupted = run_sync_grpo(cfg, tmp_path / "continuous")
    # Exercise migration from separate placement into collocation on resume.
    partial = run_sync_grpo(
        replace(cfg, steps=2, worker_placement="separate", gpu_memory_utilization=0.7), tmp_path / "partial"
    )
    resumed = run_sync_grpo(cfg, tmp_path / "resumed", resume=partial["checkpoint"])
    workers = json.loads((tmp_path / "continuous/workers.json").read_text())
    pids = [w["pid"] for w in workers["workers"]]
    assert len(set(pids)) == replicas + 1 and workers["driver_pid"] not in pids
    assert workers["placement"] == placement
    assert all(worker["gpu_ids"] == [] for worker in workers["workers"])
    if placement == "collocate":
        assert len({worker["node"] for worker in workers["workers"]}) == 1
    rows = [
        json.loads(line) for line in (tmp_path / "continuous/trajectories.jsonl").read_text().splitlines()
    ]
    for step, row in enumerate(rows, start=1):
        assert len(row["trajectories"]) == 32
        assert {t["policy_version"]["weight_version"] for t in row["trajectories"]} == {step - 1}
    assert not uninterrupted["native_gpu_executed"]
    assert uninterrupted["native_gpu_acceptance"] == "not_applicable"
    assert resumed["first_step"] == 2 and resumed["last_step"] == 3
    assert resumed["worker_placement"] == placement
    a = torch.load(uninterrupted["checkpoint"] + "/learner.pt", weights_only=True)
    b = torch.load(resumed["checkpoint"] + "/learner.pt", weights_only=True)
    assert a["optimizer_steps"] == b["optimizer_steps"] > 0
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    for name, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][name][key], atol=0, rtol=0)


def test_resume_rejects_changed_training_semantics(tmp_path):
    cfg = TrainConfig()
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "config": cfg.to_dict(),
                "step": 1,
                "policy": {"weights": 1},
            }
        )
    )
    with pytest.raises(ValueError, match="learning_rate"):
        resume_metadata(tmp_path, replace(cfg, learning_rate=0.01))


@pytest.mark.parametrize("schema", [1, 2])
def test_resume_accepts_legacy_and_current_metadata(tmp_path, schema):
    cfg = TrainConfig()
    policy = (
        {"policy": {"weights": 1, "recipe": cfg.policy_recipe_id}}
        if schema == 1
        else {"policy_version": {"weight_version": 1, "recipe_id": cfg.policy_recipe_id}}
    )
    config = cfg.to_dict()
    # Checkpoints written before placement was configurable retain their defaults.
    del config["worker_placement"]
    metadata = {"schema": schema, "config": config, "step": 1, **policy}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    restored = resume_metadata(tmp_path, cfg)
    assert restored["policy_version"] == {"weight_version": 1, "recipe_id": cfg.policy_recipe_id}
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata


def test_historical_implicit_bf16_checkpoint_keeps_its_original_format(tmp_path):
    cfg = TrainConfig(recipe_id="bf16-baseline-v1")
    config = cfg.to_dict()
    del config["recipe_id"]
    metadata = {"schema": 1, "step": 1, "config": config, "policy": {"weights": 1, "recipe": cfg.recipe_id}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    assert resume_metadata(tmp_path, cfg)["step"] == 1
    with pytest.raises(ValueError, match="recipe_id"):
        resume_metadata(tmp_path, TrainConfig())


def test_resume_rejects_checkpoint_that_could_promote_bf16_layers(tmp_path):
    cfg = TrainConfig(
        recipe_id="mxfp8-smoothquant-mlp-down-v1",
        recipe_options={"layers": [7, 15, 20, 25], "allow_layer_expansion": True},
    )
    # Identity emitted by the pre-scope recipe for this exact configuration.
    old_policy = "mxfp8-smoothquant-mlp-down-v1:18604a217adbe84e:" + "1" * 64
    metadata = {
        "schema": 2,
        "config": cfg.to_dict(),
        "step": 1,
        "policy_version": {"weight_version": 1, "recipe_id": old_policy},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incompatible scope"):
        resume_metadata(tmp_path, cfg)


def test_resume_accepts_equivalent_spelling_of_low_precision_recipe(tmp_path):
    old = TrainConfig(recipe_id="mxfp8-mlp-down-v1", recipe_options={"layers": [7, 15]})
    cfg = replace(old, recipe_id="mlp-down-v1", recipe_options={"layers": [15, 7], "precision": "mxfp8"})
    metadata = {
        "schema": 2,
        "config": old.to_dict(),
        "step": 1,
        "policy_version": {"weight_version": 1, "recipe_id": old.policy_recipe_id},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    assert resume_metadata(tmp_path, cfg)["policy_version"]["recipe_id"] == old.policy_recipe_id


@pytest.mark.parametrize("fault", ["schema", "version", "recipe"])
def test_resume_rejects_inconsistent_metadata(tmp_path, fault):
    cfg = TrainConfig()
    metadata = {
        "schema": 2,
        "config": cfg.to_dict(),
        "step": 1,
        "policy_version": {"weight_version": 1, "recipe_id": cfg.policy_recipe_id},
    }
    if fault == "schema":
        metadata["schema"] = 999
    elif fault == "version":
        metadata["policy_version"]["weight_version"] = 2
    else:
        metadata["policy_version"]["recipe_id"] = "different-recipe"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        resume_metadata(tmp_path, cfg)
