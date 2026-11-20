"""Scope, held-out evidence, and portable scale artifacts for adaptive SmoothQuant."""

import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from qorerl.recipes.calibration import CalibrationConfig, discover, outlier_report
from qorerl.recipes.smoothquant import (
    make_smoothing_artifact,
    parse_smoothing_artifact,
    prepare_smoothquant_candidate,
)
from qorerl.recipes.state import down_proj_name

NAME = down_proj_name(0)
OTHER = down_proj_name(1)
CONTEXT = {"model": "tiny", "model_revision": "abc123", "recipe_id": "scoped-preset-id"}


def _operands():
    # The final column is unseen in fitting; retain its existing scale.
    x = torch.tensor([[16, 1, 0], [4, 1, 0], [16, 2, 2], [32, 2, 4]], dtype=torch.bfloat16)
    w = torch.ones(1, 3, dtype=torch.bfloat16, requires_grad=True)
    factors = {NAME: torch.tensor([2.0, 1.0, 2.0])}
    return {NAME: w}, {NAME: x}, factors


def test_measures_changed_and_active_scales_on_disjoint_identical_heldout_rows():
    weights, samples, factors = _operands()
    config = CalibrationConfig(discover_outliers=False)
    calls = []

    def clipped(x, w):
        # Explicit deterministic test quantizer; production uses the native FPROP.
        calls.append((x.clone(), w.clone()))
        return F.linear(x.clamp(-4, 4), w)

    overrides, [report] = prepare_smoothquant_candidate(weights, samples, factors, config, fprop=clipped)
    new = overrides[NAME]
    torch.testing.assert_close(new, torch.tensor([4.0, 1.0, 2.0]))
    assert report["status"] == "proposed"
    assert report["fit_rows"] == report["heldout_rows"] == 2
    heldout = samples[NAME][2:]
    reference = F.linear(heldout, weights[NAME]).double()
    for (x, w), s, field in zip(
        calls, [factors[NAME], new], ["current_relative_l2", "candidate_relative_l2"], strict=True
    ):
        torch.testing.assert_close(x, (heldout / s).bfloat16())
        torch.testing.assert_close(w, (weights[NAME] * s).bfloat16())
        output = F.linear(x.clamp(-4, 4), w).double()
        assert report[field] == pytest.approx(((output - reference).norm() / reference.norm()).item())
    assert 0 < report["relative_improvement"] < 1
    # Fitting cannot mutate an active scale, weight, or gradient.
    assert factors[NAME].tolist() == [2, 1, 2]
    assert weights[NAME].grad is None
    rejected, [threshold_report] = prepare_smoothquant_candidate(
        weights, samples, factors, replace(config, min_relative_improvement=1), fprop=clipped
    )
    assert rejected == {} and threshold_report["status"] == "no_gain"


def test_no_candidate_for_a_tie_at_zero_error_or_worse_output():
    weights, samples, factors = _operands()
    config = CalibrationConfig(discover_outliers=False)
    overrides, [ev] = prepare_smoothquant_candidate(weights, samples, factors, config, fprop=F.linear)
    assert overrides == {} and ev["status"] == "no_gain"
    assert ev["current_relative_l2"] == ev["candidate_relative_l2"] == 0
    assert ev["relative_improvement"] is None
    # Make the NEW transform worse even though the fit window contained an outlier.
    overrides, [ev] = prepare_smoothquant_candidate(
        weights, samples, factors, config, fprop=lambda x, w: F.linear(x, w.clamp_max(2))
    )
    assert overrides == {} and ev["candidate_relative_l2"] > ev["current_relative_l2"]


def test_outlier_discovery_uses_only_fit_rows_and_can_run_without_quantization():
    weights, samples, factors = _operands()
    weights[OTHER] = weights[NAME].clone()
    samples[OTHER] = torch.ones_like(samples[NAME])
    samples[OTHER][-1, 2] = 256
    factors[OTHER] = torch.ones(3)
    cfg = CalibrationConfig(outlier_threshold=2)
    assert discover(samples, cfg) == {NAME: pytest.approx(16 / (17 / 3))}
    assert set(discover(samples, replace(cfg, discover_outliers=False))) == {NAME, OTHER}
    # Discovery-only examines the entire bounded window, without fitting or replaying.
    discovery = replace(cfg, discovery_only=True, max_rows=2)
    overrides, reports = prepare_smoothquant_candidate(
        weights, samples, factors, discovery, fprop=lambda *_: pytest.fail("FPROP")
    )
    assert not overrides
    assert [(r["target"], r["status"]) for r in reports] == [(NAME, "discovered"), (OTHER, "not_selected")]
    assert all(
        r["fit_rows"] == 2 and r["heldout_rows"] == 0 and r["candidate_relative_l2"] is None for r in reports
    )


@pytest.mark.parametrize("rows", [0, 1])
def test_short_windows_have_no_evidence_for_acceptance(rows):
    weights, samples, factors = _operands()
    samples[NAME] = samples[NAME][:rows]
    overrides, [report] = prepare_smoothquant_candidate(
        weights, samples, factors, CalibrationConfig(), fprop=lambda *_: pytest.fail()
    )
    assert not overrides and report["status"] == "insufficient_rows"


def test_bad_operands_and_backend_outputs_cannot_generate_a_proposal():
    weights, samples, factors = _operands()
    cfg = CalibrationConfig(discover_outliers=False)
    with pytest.raises(ValueError, match="scope"):
        prepare_smoothquant_candidate(weights, {}, factors, cfg)
    with pytest.raises(ValueError, match="BF16"):
        prepare_smoothquant_candidate(weights, {NAME: samples[NAME].float()}, factors, cfg)
    with pytest.raises(ValueError, match="width"):
        prepare_smoothquant_candidate(weights, samples, {NAME: torch.ones(1)}, cfg)
    with pytest.raises(ValueError, match="BF16 output"):
        prepare_smoothquant_candidate(weights, samples, factors, cfg, fprop=lambda a, b: a.float())
    with pytest.raises(ValueError, match="Nonfinite held-out"):
        prepare_smoothquant_candidate(
            weights, samples, factors, cfg, fprop=lambda a, b: F.linear(a, b) * float("nan")
        )
    samples[NAME][2:] = 0
    overrides, [ev] = prepare_smoothquant_candidate(
        weights, samples, factors, cfg, fprop=lambda *_: pytest.fail("zero reference")
    )
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
        {"max_activation_relative_l2": -1e-6},
        {"max_activation_relative_l2": True},
        {"max_activation_relative_l2": float("nan")},
        {"max_activation_relative_l2": float("inf")},
        {"outlier_threshold": 0},
        {"max_rows": 1},
        {"refresh_every_optimizer_steps": 0},
        {"refresh_every_optimizer_steps": True},
        {"approval": "silently"},
        {"scale_artifact": ""},
        {"initial_artifact": 1},
        {"scale_artifact": "one.json", "initial_artifact": "two.json"},
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
    factors = {NAME: torch.tensor([1.0, 2.0, 0.25])}
    payload = make_smoothing_artifact(factors, **CONTEXT)
    # The v1 content digest is also embedded in published policy identities.
    assert payload == {
        "schema": 1,
        "kind": "smoothquant_scales",
        **CONTEXT,
        "scales": {NAME: [1.0, 2.0, 0.25]},
        "scale_digest": "9d6d717167fb0606a669bfe3851123f4d5009c9bd9355e39e2520c198adcbe3d",
    }
    decoded = parse_smoothing_artifact(
        json.loads(json.dumps(payload)), **CONTEXT, targets={NAME: (32, 3), OTHER: (32, 3)}
    )
    torch.testing.assert_close(decoded[NAME], factors[NAME], atol=0, rtol=0)
    assert CalibrationConfig(initial_artifact="scales.json", refresh_every_optimizer_steps=0).initial_artifact
    for field in CONTEXT:
        with pytest.raises(ValueError, match=field):
            parse_smoothing_artifact(payload, **(CONTEXT | {field: "other"}))
    with pytest.raises(ValueError, match="outside installed"):
        parse_smoothing_artifact(payload, **CONTEXT, targets={OTHER: (32, 3)})
    with pytest.raises(ValueError, match="width mismatch"):
        parse_smoothing_artifact(payload, **CONTEXT, targets={NAME: (32, 32)})
    with pytest.raises(ValueError, match="digest mismatch"):
        parse_smoothing_artifact(payload | {"scale_digest": "changed"}, **CONTEXT)


@pytest.mark.parametrize(
    "factors",
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
def test_external_artifact_rejects_unusable_values_and_targets(factors):
    payload = make_smoothing_artifact({NAME: torch.ones(1)}, **CONTEXT)
    with pytest.raises(ValueError):
        parse_smoothing_artifact(payload | {"scales": factors}, **CONTEXT)


def test_zero_fit_signal_and_invalid_candidate_are_local_rejections():
    weights, samples, factors = _operands()
    cfg = CalibrationConfig(discover_outliers=False)
    weights[OTHER] = weights[NAME]
    samples[OTHER] = torch.zeros_like(samples[NAME])
    factors[OTHER] = factors[NAME]
    assert set(discover(samples, cfg)) == {NAME}
    calls = 0

    def overflow_candidate(x, w):
        nonlocal calls
        calls += 1
        return F.linear(x, w) * (float("inf") if calls == 2 else 1)

    overrides, [invalid, unseen] = prepare_smoothquant_candidate(
        weights, samples, factors, cfg, fprop=overflow_candidate
    )
    assert not overrides
    assert invalid["status"] == "invalid_candidate" and invalid["current_relative_l2"] == 0.0
    assert invalid["candidate_relative_l2"] is None
    assert unseen["status"] == "insufficient_signal" and unseen["reason"] == "all_zero_fit_window"
    assert calls == 2


def test_candidate_backend_failure_remains_distinguishable_from_unusable_scales():
    weights, samples, factors = _operands()

    def unavailable(x, w):
        if w[0, 0] == 4:
            raise RuntimeError("Backend failure")
        return F.linear(x, w)

    with pytest.raises(RuntimeError, match="Backend failure"):
        prepare_smoothquant_candidate(
            weights, samples, factors, CalibrationConfig(discover_outliers=False), fprop=unavailable
        )


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
    assert prepare_smoothquant_candidate(None, samples, None, cfg) == ({}, [report])
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
    weights, samples, factors = _operands()
    weights[OTHER] = object()  # A deferred matrix outside the selected set.
    samples[OTHER] = torch.ones_like(samples[NAME])
    factors[OTHER] = factors[NAME]
    overrides, reports = prepare_smoothquant_candidate(
        weights, samples, factors, CalibrationConfig(outlier_threshold=2), fprop=F.linear
    )
    assert not overrides and reports[1]["status"] == "not_selected"


def test_first_activation_measures_both_quantized_paths_against_the_same_heldout_bf16():
    weights, samples, factors = _operands()
    inactive = down_proj_name(23)
    quiet = down_proj_name(24)
    weights.update({quiet: object(), inactive: weights[NAME].detach().clone()})
    samples.update({quiet: torch.ones_like(samples[NAME]), inactive: samples[NAME].clone()})
    samples[quiet][-1, 0] = 1024  # A held-out outlier cannot authorize activation.
    cfg = CalibrationConfig(discover_outliers=False, outlier_threshold=2)
    calls = []

    def clipped(x, w):
        calls.append((x.clone(), w.clone()))
        return F.linear(x.clamp(-4, 4), w)

    overrides, reports = prepare_smoothquant_candidate(
        weights, samples, factors, cfg, fprop=clipped, allow_layer_expansion=True
    )
    assert [r["target"] for r in reports] == sorted(weights)
    refresh, activation, skipped = reports
    assert (refresh["action"], refresh["selection"], refresh["status"]) == ("refresh", "explicit", "proposed")
    assert (skipped["action"], skipped["selection"], skipped["status"]) == (
        "activate",
        "outlier",
        "not_selected",
    )
    assert skipped["direct_low_precision_relative_l2"] is None
    assert (activation["action"], activation["selection"], activation["status"]) == (
        "activate",
        "outlier",
        "proposed",
    )
    assert activation["current_relative_l2"] == activation["direct_low_precision_relative_l2"]
    assert activation["relative_improvement"] == pytest.approx(
        1 - activation["candidate_relative_l2"] / activation["current_relative_l2"]
    )
    assert activation["max_activation_relative_l2"] is None
    assert set(overrides) == {NAME, inactive}
    torch.testing.assert_close(overrides[inactive], torch.tensor([4.0, 1.0, 1.0]))
    assert len(calls) == 4
    heldout = samples[inactive][2:]
    reference = F.linear(heldout, weights[inactive]).double()
    for (x, w), s, field in zip(
        calls[2:],
        [torch.ones(3), overrides[inactive]],
        ["direct_low_precision_relative_l2", "candidate_relative_l2"],
        strict=True,
    ):
        torch.testing.assert_close(x, (heldout / s).bfloat16())
        torch.testing.assert_close(w, (weights[inactive] * s).bfloat16())
        output = F.linear(x.clamp(-4, 4), w).double()
        assert activation[field] == pytest.approx(((output - reference).norm() / reference.norm()).item())
    assert 0 < activation["candidate_relative_l2"] < activation["direct_low_precision_relative_l2"]
    assert list(factors) == [NAME] and factors[NAME].tolist() == [2, 1, 2]


def test_first_activation_uses_optional_inclusive_bf16_error_bound():
    weights, samples, factors = _operands()
    weights[OTHER], samples[OTHER] = weights[NAME], samples[NAME]
    cfg = CalibrationConfig(outlier_threshold=2)

    def evaluate(bound):
        overrides, [_, event] = prepare_smoothquant_candidate(
            weights,
            samples,
            factors,
            replace(cfg, max_activation_relative_l2=bound),
            allow_layer_expansion=True,
            fprop=lambda x, w: F.linear(x.clamp(-4, 4), w),
        )
        return overrides, event

    accepted, first = evaluate(None)
    assert OTHER in accepted and first["direct_low_precision_relative_l2"] > 0
    cost = first["candidate_relative_l2"]
    assert cost > 0
    at_bound, event = evaluate(cost)
    assert OTHER in at_bound and event["max_activation_relative_l2"] == cost
    refused, event = evaluate(0.0)
    assert OTHER not in refused and event["status"] == "activation_cost_exceeded"
    assert event["candidate_relative_l2"] == cost


@pytest.mark.parametrize("adverse", [False, True], ids=["gain_below_threshold", "worse_than_direct"])
def test_expansion_rejects_insufficient_gain_over_current_low_precision(adverse):
    weights, samples, factors = _operands()
    weights[OTHER], samples[OTHER] = weights[NAME], samples[NAME]
    cfg = CalibrationConfig(outlier_threshold=2, min_relative_improvement=1)

    def fprop(x, w):
        if adverse:
            return F.linear(x, w.clamp_max(2))
        return F.linear(x.clamp(-4, 4), w)

    overrides, [_, event] = prepare_smoothquant_candidate(
        weights, samples, factors, cfg, allow_layer_expansion=True, fprop=fprop
    )
    assert OTHER not in overrides and event["status"] == "no_gain"
    if adverse:
        assert event["candidate_relative_l2"] > event["current_relative_l2"]
    else:
        assert 0 < event["relative_improvement"] < 1


def test_identity_scale_never_proposes_a_transform():
    weights, samples, _ = _operands()
    weights[OTHER], samples[OTHER] = weights[NAME], samples[NAME]
    factors = {NAME: torch.ones(3)}
    cfg = CalibrationConfig(outlier_threshold=2, alpha=0, max_activation_relative_l2=0)
    overrides, [refresh, activation] = prepare_smoothquant_candidate(
        weights, samples, factors, cfg, allow_layer_expansion=True, fprop=F.linear
    )
    assert refresh["status"] == activation["status"] == "no_gain"
    assert not overrides


def test_expansion_requires_an_authorized_observation_scope_and_active_subset():
    weights, samples, factors = _operands()
    weights[OTHER], samples[OTHER] = weights[NAME], samples[NAME]
    cfg = CalibrationConfig()
    with pytest.raises(ValueError, match="entire supplied calibration scope"):
        prepare_smoothquant_candidate(weights, samples, factors, cfg)
    with pytest.raises(ValueError, match="subset"):
        prepare_smoothquant_candidate(
            weights, samples, {down_proj_name(23): torch.ones(3)}, cfg, allow_layer_expansion=True
        )
    overrides, reports = prepare_smoothquant_candidate(
        weights, samples, {}, cfg, allow_layer_expansion=True, fprop=F.linear
    )
    assert not overrides and all(event["action"] == "activate" for event in reports)
    with pytest.raises(ValueError, match="same nonempty scope"):
        prepare_smoothquant_candidate(
            weights, {NAME: samples[NAME]}, factors, cfg, allow_layer_expansion=True
        )
    with pytest.raises(ValueError, match="must be boolean"):
        prepare_smoothquant_candidate(weights, samples, factors, cfg, allow_layer_expansion="yes")


@pytest.mark.parametrize("invalid_stage", ["direct", "candidate", "reference"])
def test_inactive_target_requires_valid_reference_and_both_quantized_trials(invalid_stage):
    weights, samples, factors = _operands()
    weights[OTHER], samples[OTHER] = weights[NAME], samples[NAME].clone()
    samples[NAME] = torch.ones_like(samples[NAME])  # Active target is below threshold.
    if invalid_stage == "reference":
        samples[OTHER][2:] = 0

    calls = 0

    def fprop(x, w):
        nonlocal calls
        calls += 1
        if calls == (1 if invalid_stage == "direct" else 2):
            return torch.full((x.shape[0], w.shape[0]), float("nan"), dtype=x.dtype)
        return F.linear(x, w)

    overrides, [_, event] = prepare_smoothquant_candidate(
        weights,
        samples,
        factors,
        CalibrationConfig(outlier_threshold=2),
        allow_layer_expansion=True,
        fprop=fprop,
    )
    assert not overrides and event["action"] == "activate" and event["candidate_relative_l2"] is None
    if invalid_stage == "reference":
        assert calls == 0 and event["reason"] == "zero_heldout_reference"
        assert event["current_relative_l2"] is None
    else:
        assert calls == (1 if invalid_stage == "direct" else 2)
        assert event["status"] == "invalid_candidate"
        assert event["current_relative_l2"] == (None if invalid_stage == "direct" else 0)
