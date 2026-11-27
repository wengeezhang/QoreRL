"""Calibration presentation reuses persisted evidence, with no model replay."""

import json

import pytest

from qorerl.observability.viewer import _html, export_report, load_report


def log(path, *events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def evaluation(*, identifier="eval-1", step=1):
    return {
        "event": "evaluated",
        "evaluation_id": identifier,
        "step": step,
        "policy": {"weight_version": step, "recipe_id": "before"},
        "sample_policy": {"weight_version": step - 1, "recipe_id": "before"},
        "optimizer_steps": step,
        "evaluation_seconds": 0.25,
        "evidence": [
            {
                "target": "model.layers.7.mlp.down_proj.weight",
                "action": "refresh",
                "status": "proposed",
                "current_relative_l2": 0.08,
                "candidate_relative_l2": 0.04,
                "outlier_score": 20,
                "threshold": 8,
                "fit_rows": 100,
                "heldout_rows": 20,
            },
            {
                "target": "model.layers.15.mlp.down_proj.weight",
                "status": "not_selected",
                "outlier_score": 2,
                "threshold": 8,
            },
        ],
    }


def decision(kind, *, candidate="candidate-1", identifier="eval-1"):
    return {
        "event": kind,
        "evaluation_id": identifier,
        "candidate_id": candidate,
        "step": 1,
        "policy": {"weight_version": 1, "recipe_id": "after" if kind == "applied" else "before"},
    }


def make_run(root, name):
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "learner_backend": "torch_test",
                "rollout_backend": "torch_test",
                "recipe_id": "smoothquant-mlp-down-v1",
            }
        )
    )
    return path


def model_evaluation():
    event = evaluation()
    event["evidence"][0].update(current_relative_l2=None, candidate_relative_l2=None)
    event["comparison"] = {
        "scope": "model_logits",
        "reference": "full_model_bf16",
        "activation_source": "full_model_bf16",
        "reference_execution": "cpu_fixture",
        "status": "proposed",
        "reason": None,
        "fit_rows": 100,
        "heldout_rows": 20,
        "current_relative_l2": 0.08,
        "candidate_relative_l2": 0.04,
        "min_relative_improvement": 0.1,
        "max_logits_relative_l2": None,
        "targets": [event["evidence"][0]["target"]],
    }
    return event


def test_model_comparison_has_one_logits_pair_and_separate_layer_discovery(tmp_path):
    run = make_run(tmp_path, "model")
    log(run / "calibration.jsonl", model_evaluation())
    report = load_report(run)
    [window] = report["calibration"]["evaluations"]
    assert not report["warnings"]
    assert window["comparison"]["relative_improvement"] == 0.5
    assert all(row["relative_improvement"] is None for row in window["evidence"])
    html = _html(report, live=False)
    assert "Final-logits error change" in html
    assert "model_logits" in html and "cpu_fixture" in html


@pytest.mark.parametrize(
    "updates",
    [
        {"candidate_relative_l2": 0.09},
        {"candidate_relative_l2": float("nan")},
        {"max_logits_relative_l2": 0.01},
        {"reference_execution": "unknown"},
        {"heldout_rows": 0},
        {"targets": []},
    ],
)
def test_malformed_or_unqualified_model_evidence_cannot_claim_success(tmp_path, updates):
    run = make_run(tmp_path, "model")
    event = model_evaluation()
    event["comparison"].update(updates)
    log(run / "calibration.jsonl", event)
    report = load_report(run)
    assert report["warnings"]
    assert not report["calibration"]["evaluations"]


def test_manual_resume_links_explicit_prior_runs_and_preserves_pairs(tmp_path):
    paused, accepted = [make_run(tmp_path, name) for name in ("paused", "accepted")]
    log(paused / "calibration.jsonl", evaluation(), decision("awaiting_review"))
    log(accepted / "calibration.jsonl", {**decision("applied"), "apply_publish_seconds": 0.12})
    assert not load_report(accepted)["calibration"]["evaluations"]
    report = load_report(accepted, related_runs=[paused])
    [window] = report["calibration"]["evaluations"]
    assert window["decision"] == "applied"
    assert window["source"] == "cpu_simulation"
    assert window["evidence"][0]["relative_improvement"] == 0.5
    assert window["evidence"][1]["current_relative_l2"] is None
    assert window["decisions"][-1]["apply_publish_seconds"] == 0.12
    assert not report["warnings"]
    # An untrusted checkpoint path in an event is never opened automatically.
    log(accepted / "calibration.jsonl", {**decision("applied"), "source_checkpoint": str(paused)})
    assert not load_report(accepted)["calibration"]["evaluations"]


def test_distinct_windows_and_explicit_resume_branches_are_not_merged(tmp_path):
    paused, accepted, rejected = [make_run(tmp_path, n) for n in ("paused", "accepted", "rejected")]
    log(
        paused / "calibration.jsonl",
        evaluation(),
        decision("awaiting_review"),
        evaluation(identifier="eval-2", step=2),
    )
    log(accepted / "calibration.jsonl", decision("applied"))
    log(rejected / "calibration.jsonl", decision("rejected"))
    windows = load_report(accepted, related_runs=[paused, rejected])["calibration"]["evaluations"]
    assert [w["decision"] for w in windows] == ["multiple_branches", "evaluated"]
    assert windows[0]["policy"]["weight_version"] != windows[1]["policy"]["weight_version"]


def test_old_logs_join_by_candidate_id_without_guessing_across_runs(tmp_path):
    paused, accepted = [make_run(tmp_path, n) for n in ("paused", "accepted")]
    e, awaiting, applied = evaluation(), decision("awaiting_review"), decision("applied")
    awaiting.pop("policy")  # Older awaiting_review events contained no policy.
    for event in (e, awaiting, applied):
        event.pop("evaluation_id")
    log(paused / "calibration.jsonl", e, awaiting)
    log(accepted / "calibration.jsonl", applied)
    assert (
        load_report(accepted, related_runs=[paused])["calibration"]["evaluations"][0]["decision"] == "applied"
    )
    applied["candidate_id"] = "unrelated"
    log(accepted / "calibration.jsonl", applied)
    assert (
        load_report(accepted, related_runs=[paused])["calibration"]["evaluations"][0]["decision"]
        == "awaiting_review"
    )


def test_partial_lines_invalid_values_and_incident_remain_visible(tmp_path):
    run = make_run(tmp_path, "paused")
    bad = evaluation(identifier="bad")
    bad["evidence"][0]["candidate_relative_l2"] = -1
    log(run / "calibration.jsonl", evaluation(), bad)
    with (run / "calibration.jsonl").open("a") as stream:
        stream.write('{"event":')
    (run / "incident.json").write_text("{}")
    report = load_report(run)
    assert len(report["calibration"]["evaluations"]) == 1
    assert any("invalid candidate_relative_l2" in w for w in report["warnings"])
    assert any("incident.json" in w for w in report["warnings"])


def test_export_is_standalone_and_escapes_evidence(tmp_path):
    run = make_run(tmp_path, "paused")
    event = evaluation()
    event["evidence"][0]["reason"] = "</script><script>alert(1)</script>"
    log(run / "calibration.jsonl", event)
    output = tmp_path / "report.html"
    export_report(run, output)
    html = output.read_text()
    assert "Calibration evidence" in html
    assert 'data-live="false"' in html
    assert "CPU SIMULATION · NOT GPU VALIDATION" in html
    assert "\\u003c/script>" in html
    assert "<script>alert(1)</script>" not in html
    assert 'src="data:image/png;base64,' in html
    assert "__QORERL_" not in html


def test_empty_smoothquant_run_uses_calibration_view(tmp_path):
    run = make_run(tmp_path, "starting")
    assert load_report(run)["calibration"] == {"events": [], "evaluations": []}


def test_existing_probe_view_remains_accessible_when_discovery_is_also_enabled():
    report = {
        "metadata": {"recipe_id": "mxfp8-hadamard-mlp-down-v1"},
        "records": [{}],
        "calibration": {"evaluations": [], "events": []},
        "warnings": [],
    }
    assert "Recipe Insights" in _html(report, live=False)
    assert "Calibration evidence" in _html(report, live=False, view="calibration")
    report["metadata"]["recipe_id"] = "smoothquant-mlp-down-v1"
    assert "Calibration evidence" in _html(report, live=False)
    assert "Recipe Insights" in _html(report, live=False, view="probes")
