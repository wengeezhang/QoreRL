"""Read calibration evidence without replaying models or opening checkpoint payloads."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024
_TARGET = re.compile(r"model\.layers\.(\d+)\.mlp\.down_proj(?:\.weight)?$")
_EVENTS = {"evaluated", "awaiting_review", "applied", "rejected", "imported"}


def read_json(path, warnings, *, limit=1024 * 1024):
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("file exceeds size limit")
        value = json.loads(raw)
        json.dumps(value, allow_nan=False)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, RecursionError) as exc:
        warnings.append(f"{path}: {type(exc).__name__}: could not read JSON object.")
        return {}


def read_events(path, warnings, *, tail=True):
    """Bound reads and defer the final uncommitted line of a live JSONL file."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            if size > MAX_BYTES and not tail:
                warnings.append(f"{path}: exceeds 16 MiB; full validation requires a smaller run segment.")
                return []
            offset = max(0, size - MAX_BYTES)
            stream.seek(offset)
            if offset:
                stream.readline()
                warnings.append(f"{path}: showing recent tail; earlier evidence may be absent.")
            raw = stream.read(MAX_BYTES)
    except FileNotFoundError:
        return []
    except OSError as exc:
        warnings.append(f"{path}: {type(exc).__name__}.")
        return []
    lines = raw.split(b"\n")
    if lines[-1].strip() and not tail:
        warnings.append(f"{path}: final line is incomplete; validation requires a closed run.")
    result = []
    for index, line in enumerate(lines[:-1], 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            json.dumps(value, allow_nan=False)
            if not isinstance(value, dict):
                raise ValueError("expected object")
            result.append(value)
        except (UnicodeError, ValueError, RecursionError):
            if len(warnings) < 50:
                warnings.append(f"{path}: invalid completed JSON at line {index}.")
    return result


def _number(value, *, integer=False):
    return (
        type(value) in ((int,) if integer else (int, float))
        and 0 <= value <= 2**53 - 1
        and math.isfinite(value)
    )


def _policy(value):
    if not isinstance(value, dict) or not _number(value.get("weight_version"), integer=True):
        raise ValueError("missing policy version")
    recipe = value.get("recipe_id")
    if not isinstance(recipe, str) or not 0 < len(recipe) <= 512:
        raise ValueError("missing recipe identity")
    return {"weight_version": value["weight_version"], "recipe_id": recipe}


def _evidence(row):
    if not isinstance(row, dict):
        raise ValueError("expected layer evidence")
    match = _TARGET.fullmatch(str(row.get("target", "")))
    if match is None or int(match[1]) >= 4096:
        raise ValueError("invalid layer target")
    result = {"target": row["target"], "layer": int(match[1])}
    for key in ("status", "reason", "action", "selection"):
        value = row.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 1000):
            raise ValueError(f"invalid {key}")
        result[key] = value
    if not result["status"]:
        raise ValueError("missing layer status")
    for key in (
        "outlier_score",
        "threshold",
        "alpha",
        "fit_rows",
        "heldout_rows",
        "current_relative_l2",
        "candidate_relative_l2",
        "direct_low_precision_relative_l2",
    ):
        value = row.get(key)
        if value is not None and not _number(value, integer=key.endswith("rows")):
            raise ValueError(f"invalid {key}")
        result[key] = value
    # Compute the comparison from the paired values; never turn missing data into zero.
    current_relative_l2, candidate_relative_l2 = (
        result["current_relative_l2"],
        result["candidate_relative_l2"],
    )
    result["relative_improvement"] = (
        (current_relative_l2 - candidate_relative_l2) / current_relative_l2
        if current_relative_l2 and candidate_relative_l2 is not None
        else None
    )
    return result


def _comparison(value):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("scope") != "model_logits"
        or value.get("reference") != "full_model_bf16"
        or value.get("activation_source") != "full_model_bf16"
        or value.get("reference_execution") not in ("bf16", "cpu_fixture")
    ):
        raise ValueError("invalid full-model comparison scope or reference")
    result = {key: value[key] for key in ("scope", "reference", "activation_source", "reference_execution")}
    for key in ("status", "reason"):
        item = value.get(key)
        if item is not None and (not isinstance(item, str) or len(item) > 1000):
            raise ValueError(f"invalid comparison {key}")
        result[key] = item
    if not result["status"]:
        raise ValueError("missing comparison status")
    for key in (
        "fit_rows",
        "heldout_rows",
        "current_relative_l2",
        "candidate_relative_l2",
        "min_relative_improvement",
        "max_logits_relative_l2",
    ):
        item = value.get(key)
        if item is not None and not _number(item, integer=key.endswith("rows")):
            raise ValueError(f"invalid comparison {key}")
        result[key] = item
    targets = value.get("targets")
    if (
        not isinstance(targets, list)
        or len(targets) > 4096
        or any(not isinstance(t, str) or not _TARGET.fullmatch(t) for t in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("invalid comparison targets")
    result["targets"] = targets
    current, candidate = result["current_relative_l2"], result["candidate_relative_l2"]
    result["relative_improvement"] = (
        (current - candidate) / current if current and candidate is not None else None
    )
    if result["status"] == "proposed" and not (
        targets
        and current is not None
        and candidate is not None
        and candidate < current
        and result["fit_rows"]
        and result["heldout_rows"]
        and result["min_relative_improvement"] is not None
        and result["relative_improvement"] >= result["min_relative_improvement"]
        and (result["max_logits_relative_l2"] is None or candidate <= result["max_logits_relative_l2"])
    ):
        raise ValueError("proposed comparison did not pass the joint logits gate")
    return result


def load_calibration(run_dirs, warnings, *, tail=True):
    """Join only explicitly supplied runs. Never follow paths from their logs."""
    evaluations, events = [], []
    for directory in dict.fromkeys(str(Path(p).resolve()) for p in run_dirs):
        config = read_json(Path(directory) / "config.json", warnings)
        backend = (config.get("learner_backend"), config.get("rollout_backend"))
        source = (
            "cpu_simulation"
            if backend == ("torch_test", "torch_test")
            else "native_gpu"
            if backend == ("megatron", "vllm")
            else "unknown"
        )
        local = []
        for index, raw in enumerate(read_events(Path(directory) / "calibration.jsonl", warnings, tail=tail)):
            try:
                kind = raw.get("event")
                if kind not in _EVENTS or not _number(raw.get("step"), integer=True):
                    raise ValueError("invalid event or step")
                event = {
                    "event": kind,
                    "step": raw["step"],
                    "policy": (
                        None
                        if kind == "awaiting_review" and raw.get("policy") is None
                        else _policy(raw.get("policy"))
                    ),
                    "run": directory,
                    "source": source,
                }
                for key in ("evaluation_id", "candidate_id"):
                    value = raw.get(key)
                    if value is not None:
                        if not isinstance(value, str) or not 0 < len(value) <= 512:
                            raise ValueError(f"invalid {key}")
                        event[key] = value
                duration = raw.get("apply_publish_seconds")
                if duration is not None:
                    if not _number(duration):
                        raise ValueError("invalid apply_publish_seconds")
                    event["apply_publish_seconds"] = duration
                if kind == "evaluated":
                    if not _number(raw.get("optimizer_steps"), integer=True):
                        raise ValueError("invalid optimizer_steps")
                    seconds = raw.get("evaluation_seconds")
                    if seconds is not None and not _number(seconds):
                        raise ValueError("invalid evaluation_seconds")
                    rows = raw.get("evidence")
                    if not isinstance(rows, list) or len(rows) > 4096:
                        raise ValueError("invalid evidence list")
                    rows = [_evidence(row) for row in rows]
                    if len({row["target"] for row in rows}) != len(rows):
                        raise ValueError("duplicate layer evidence")
                    event.update(
                        id=event.get("evaluation_id", f"{directory}:{index}"),
                        optimizer_steps=raw["optimizer_steps"],
                        evaluation_seconds=seconds,
                        sample_policy=_policy(raw["sample_policy"]) if "sample_policy" in raw else None,
                        evidence=rows,
                        comparison=_comparison(raw.get("comparison")),
                        decisions=[],
                    )
                    evaluations.append(event)
                    local.append(event)
                else:
                    # Old logs lack evaluation_id. A decision in the same run and at
                    # the same boundary identifies the preceding evaluation only.
                    candidates = [item for item in local if item["step"] == event["step"]]
                    if "evaluation_id" not in event and candidates:
                        event["evaluation_id"] = candidates[-1]["id"]
                    events.append(event)
            except (ValueError, TypeError, OverflowError) as exc:
                warnings.append(f"{directory}/calibration.jsonl event {index + 1}: {exc}.")
    for evaluation in evaluations:
        direct = [e for e in events if e.get("evaluation_id") == evaluation["id"]]
        ids = {e["candidate_id"] for e in direct if e.get("candidate_id")}
        related = [e for e in events if e in direct or e.get("candidate_id") in ids]
        evaluation["decisions"] = related
        terminal = [e for e in related if e["event"] in ("applied", "rejected")]
        # Explicitly supplied branches must not be flattened into a single decision.
        outcomes = {(e["run"], e["event"]) for e in terminal}
        evaluation["decision"] = (
            "multiple_branches"
            if len(outcomes) > 1
            else terminal[-1]["event"]
            if terminal
            else "awaiting_review"
            if any(e["event"] == "awaiting_review" for e in related)
            else "evaluated"
        )
    return {"evaluations": evaluations, "events": events}
