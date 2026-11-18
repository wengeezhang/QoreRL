"""Scope, held-out evidence, and portable scale artifacts for adaptive SmoothQuant."""

import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from qorerl.recipes.calibration import (
    CalibrationConfig,
    discover,
    make_scale_artifact,
    outlier_report,
    parse_scale_artifact,
    prepare,
)
from qorerl.recipes.state import down_proj_name

NAME = down_proj_name(0)
OTHER = down_proj_name(1)
CONTEXT = {"model": "tiny", "model_revision": "abc123", "recipe_id": "scoped-preset-id"}


def _operands():
    # The final column is unseen in fitting; retain its existing scale.
    x = torch.tensor([[16, 1, 0], [4, 1, 0], [16, 2, 2], [32, 2, 4]], dtype=torch.bfloat16)
    w = torch.ones(1, 3, dtype=torch.bfloat16, requires_grad=True)
    scales = {NAME: torch.tensor([2.0, 1.0, 2.0])}
    return {NAME: w}, {NAME: x}, scales


def test_measures_changed_and_active_scales_on_disjoint_identical_heldout_rows():
    weights, samples, scales = _operands()
    config = CalibrationConfig(discover_outliers=False)
    calls = []

    def clipped(x, w):
        # Explicit deterministic test quantizer; production uses the native FPROP.
        calls.append((x.clone(), w.clone()))
        return F.linear(x.clamp(-4, 4), w)

    overrides, [report] = prepare(weights, samples, scales, config, fprop=clipped)
    new = overrides[NAME]
    torch.testing.assert_close(new, torch.tensor([4.0, 1.0, 2.0]))
    assert report["status"] == "proposed"
    assert report["fit_rows"] == report["heldout_rows"] == 2
    heldout = samples[NAME][2:]
    reference = F.linear(heldout, weights[NAME]).double()
    for (x, w), s, field in zip(
        calls, [scales[NAME], new], ["current_relative_l2", "candidate_relative_l2"], strict=True
    ):
        torch.testing.assert_close(x, (heldout / s).bfloat16())
        torch.testing.assert_close(w, (weights[NAME] * s).bfloat16())
        output = F.linear(x.clamp(-4, 4), w).double()
        assert report[field] == pytest.approx(((output - reference).norm() / reference.norm()).item())
    assert 0 < report["relative_improvement"] < 1
    # Fitting cannot mutate an active scale, weight, or gradient.
    assert scales[NAME].tolist() == [2, 1, 2]
    assert weights[NAME].grad is None
    rejected, [threshold_report] = prepare(
        weights, samples, scales, replace(config, min_relative_improvement=1), fprop=clipped
    )
    assert rejected == {} and threshold_report["status"] == "no_gain"


def test_no_candidate_for_a_tie_at_zero_error_or_worse_output():
    weights, samples, scales = _operands()
    config = CalibrationConfig(discover_outliers=False)
    overrides, [ev] = prepare(weights, samples, scales, config, fprop=F.linear)
    assert overrides == {} and ev["status"] == "no_gain"
    assert ev["current_relative_l2"] == ev["candidate_relative_l2"] == 0
    assert ev["relative_improvement"] is None
    # Make the NEW transform worse even though the fit window contained an outlier.
    overrides, [ev] = prepare(
        weights, samples, scales, config, fprop=lambda x, w: F.linear(x, w.clamp_max(2))
    )
    assert overrides == {} and ev["candidate_relative_l2"] > ev["current_relative_l2"]


def test_outlier_discovery_uses_only_fit_rows_and_can_run_without_quantization():
    weights, samples, scales = _operands()
    weights[OTHER] = weights[NAME].clone()
    samples[OTHER] = torch.ones_like(samples[NAME])
    samples[OTHER][-1, 2] = 256
    scales[OTHER] = torch.ones(3)
    cfg = CalibrationConfig(outlier_threshold=2)
    assert discover(samples, cfg) == {NAME: pytest.approx(16 / (17 / 3))}
    assert set(discover(samples, replace(cfg, discover_outliers=False))) == {NAME, OTHER}
    # Discovery-only examines the entire bounded window, without fitting or replaying.
    discovery = replace(cfg, discovery_only=True, max_rows=2)
    overrides, reports = prepare(weights, samples, scales, discovery, fprop=lambda *_: pytest.fail("FPROP"))
    assert not overrides
    assert [(r["target"], r["status"]) for r in reports] == [(NAME, "discovered"), (OTHER, "not_selected")]
    assert all(
        r["fit_rows"] == 2 and r["heldout_rows"] == 0 and r["candidate_relative_l2"] is None for r in reports
    )


@pytest.mark.parametrize("rows", [0, 1])
def test_short_windows_have_no_evidence_for_acceptance(rows):
    weights, samples, scales = _operands()
    samples[NAME] = samples[NAME][:rows]
    overrides, [report] = prepare(
        weights, samples, scales, CalibrationConfig(), fprop=lambda *_: pytest.fail()
    )
    assert not overrides and report["status"] == "insufficient_rows"


def test_bad_operands_and_backend_outputs_cannot_generate_a_proposal():
    weights, samples, scales = _operands()
    cfg = CalibrationConfig(discover_outliers=False)
    with pytest.raises(ValueError, match="scope"):
        prepare(weights, {}, scales, cfg)
    with pytest.raises(ValueError, match="BF16"):
        prepare(weights, {NAME: samples[NAME].float()}, scales, cfg)
    with pytest.raises(ValueError, match="width"):
        prepare(weights, samples, {NAME: torch.ones(1)}, cfg)
    with pytest.raises(ValueError, match="BF16 output"):
        prepare(weights, samples, scales, cfg, fprop=lambda a, b: a.float())
    with pytest.raises(ValueError, match="Nonfinite held-out"):
        prepare(weights, samples, scales, cfg, fprop=lambda a, b: F.linear(a, b) * float("nan"))
    samples[NAME][2:] = 0
    overrides, [ev] = prepare(weights, samples, scales, cfg, fprop=lambda *_: pytest.fail("zero reference"))
    assert not overrides and ev["reason"] == "zero_heldout_reference"
    samples[NAME][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite calibration"):
        discover(samples, cfg)


@pytest.mark.parametrize(
    "options",
    [
        {"alpha": -0.1},
        {"alpha": True},
        {"min_relative_improvement": 1.1},
        {"outlier_threshold": 0},
        {"max_rows": 1},
        {"refresh_every_optimizer_steps": 0},
        {"refresh_every_optimizer_steps": True},
        {"approval": "silently"},
        {"scale_artifact": ""},
        {"scale_artifact": 1},
        {"unused": 1},
        {"outlier_threshold": float("inf")},
        {"discovery_only": True, "discover_outliers": False},
        {"discovery_only": True, "scale_artifact": "external.json"},
    ],
)
def test_configuration_rejects_invalid_or_incompatible_options(options):
    with pytest.raises(ValueError):
        CalibrationConfig.from_dict(options)


def test_external_final_scale_json_roundtrip_and_subset_scope():
    scales = {NAME: torch.tensor([1.0, 2.0, 0.25])}
    payload = make_scale_artifact(scales, **CONTEXT)
    decoded = parse_scale_artifact(
        json.loads(json.dumps(payload)), **CONTEXT, targets={NAME: (32, 3), OTHER: (32, 3)}
    )
    torch.testing.assert_close(decoded[NAME], scales[NAME], atol=0, rtol=0)
    assert CalibrationConfig(scale_artifact="scales.json", refresh_every_optimizer_steps=0).scale_artifact
    for field in CONTEXT:
        with pytest.raises(ValueError, match=field):
            parse_scale_artifact(payload, **(CONTEXT | {field: "other"}))
    with pytest.raises(ValueError, match="outside installed"):
        parse_scale_artifact(payload, **CONTEXT, targets={OTHER: (32, 3)})
    with pytest.raises(ValueError, match="width mismatch"):
        parse_scale_artifact(payload, **CONTEXT, targets={NAME: (32, 32)})
    with pytest.raises(ValueError, match="digest mismatch"):
        parse_scale_artifact(payload | {"scale_digest": "changed"}, **CONTEXT)


@pytest.mark.parametrize(
    "scales",
    [
        {},
        {NAME: [0]},
        {NAME: [-1]},
        {NAME: [float("nan")]},
        {NAME: [float("inf")]},
        {NAME: [1e-100]},
        {NAME: [1e100]},
        {NAME: [True]},
        {NAME: [[1.0]]},
        {"invalid": [1.0]},
    ],
)
def test_external_artifact_rejects_unusable_values_and_targets(scales):
    payload = make_scale_artifact({NAME: torch.ones(1)}, **CONTEXT)
    with pytest.raises(ValueError):
        parse_scale_artifact(payload | {"scales": scales}, **CONTEXT)


def test_zero_fit_signal_and_invalid_candidate_are_local_rejections():
    weights, samples, scales = _operands()
    cfg = CalibrationConfig(discover_outliers=False)
    weights[OTHER] = weights[NAME]
    samples[OTHER] = torch.zeros_like(samples[NAME])
    scales[OTHER] = scales[NAME]
    assert set(discover(samples, cfg)) == {NAME}
    calls = 0

    def overflow_candidate(x, w):
        nonlocal calls
        calls += 1
        return F.linear(x, w) * (float("inf") if calls == 2 else 1)

    overrides, [invalid, unseen] = prepare(weights, samples, scales, cfg, fprop=overflow_candidate)
    assert not overrides
    assert invalid["status"] == "invalid_candidate" and invalid["current_relative_l2"] == 0.0
    assert invalid["candidate_relative_l2"] is None
    assert unseen["status"] == "insufficient_signal" and unseen["reason"] == "all_zero_fit_window"
    assert calls == 2


def test_candidate_backend_failure_remains_distinguishable_from_unusable_scales():
    weights, samples, scales = _operands()

    def unavailable(x, w):
        if w[0, 0] == 4:
            raise RuntimeError("Backend failure")
        return F.linear(x, w)

    with pytest.raises(RuntimeError, match="Backend failure"):
        prepare(weights, samples, scales, CalibrationConfig(discover_outliers=False), fprop=unavailable)


def test_standalone_discovery_uses_only_named_activations_and_all_bounded_rows():
    samples = {
        "decoder.2.attention": torch.tensor(
            [[1, 1, 1], [1, 1, 1], [16, 1, 1], [1, 1, 1], [0, float("nan"), 0]], dtype=torch.bfloat16
        )
    }
    cfg = CalibrationConfig(discovery_only=True, max_rows=4, outlier_threshold=2)
    [report] = outlier_report(samples, cfg)
    assert report["target"] == "decoder.2.attention" and report["status"] == "discovered"
    assert report["outlier_score"] == pytest.approx(16 / 6)
    assert (report["fit_rows"], report["heldout_rows"]) == (4, 0)
    assert prepare(None, samples, None, cfg) == ({}, [report])
    [before_fit] = outlier_report(samples, replace(cfg, discovery_only=False))
    assert before_fit["status"] == "not_selected" and before_fit["outlier_score"] == 1
    assert (before_fit["fit_rows"], before_fit["heldout_rows"]) == (2, 2)
    [one] = outlier_report(samples, replace(cfg, max_rows=1))
    assert one["fit_rows"] == 1 and one["outlier_score"] == 1


@pytest.mark.parametrize(
    "samples",
    [
        {},
        {"": torch.ones(1, 1, dtype=torch.bfloat16)},
        {"arbitrary": torch.ones(1)},
        {"arbitrary": torch.empty(2, 0, dtype=torch.bfloat16)},
    ],
)
def test_activation_report_validates_its_own_inputs(samples):
    with pytest.raises(ValueError):
        outlier_report(samples, CalibrationConfig(discovery_only=True))


def test_not_selected_weight_is_never_read():
    weights, samples, scales = _operands()
    weights[OTHER] = object()  # A deferred matrix outside the selected set.
    samples[OTHER] = torch.ones_like(samples[NAME])
    scales[OTHER] = scales[NAME]
    overrides, reports = prepare(
        weights, samples, scales, CalibrationConfig(outlier_threshold=2), fprop=F.linear
    )
    assert not overrides and reports[1]["status"] == "not_selected"
