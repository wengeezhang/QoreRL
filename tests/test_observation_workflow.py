"""Report previews and diagnostic settings must not masquerade as training changes."""

import json
from dataclasses import replace

import pytest

from qorerl.observability.demo import write_demo
from qorerl.observability.viewer import export_report, load_report
from qorerl.runtime.checkpoint import resume_metadata
from qorerl.training.config import TrainConfig


def test_demo_export_labels_every_record_and_refuses_overwrite(tmp_path):
    run = write_demo(tmp_path / "demo")
    report = load_report(run)
    assert not report["warnings"]
    assert len(report["records"]) == 24
    assert all(row["source"] == "demo" and row["num_layers"] == 32 for row in report["records"])
    assert report["metadata"]["source"] == "demo"
    page = export_report(run, tmp_path / "report.html")
    html = page.read_text()
    assert "__QORERL_REPORT__" not in html
    assert "recipe_probe" in html and "Demonstration · synthetic data" in html
    original = (run / "recipe_metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        write_demo(run)
    assert (run / "recipe_metrics.jsonl").read_bytes() == original


def test_resume_can_change_observation_but_not_training_recipe(tmp_path):
    previous = TrainConfig(recipe_id="mxfp8-mlp-down-v1", steps=20)
    metadata = {
        "schema": 2,
        "step": 20,
        "policy_version": {"weight_version": 20, "recipe_id": previous.policy_recipe_id},
        "config": previous.to_dict(),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    observed = replace(previous, steps=40, recipe_observation={"enabled": True, "every_steps": 5})
    assert resume_metadata(tmp_path, observed)["step"] == 20
    assert observed.policy_recipe_id == previous.policy_recipe_id
    with pytest.raises(ValueError, match="recipe_options"):
        resume_metadata(tmp_path, replace(observed, recipe_options={"layers": [0]}))
