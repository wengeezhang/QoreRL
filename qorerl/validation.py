"""Native preflight and evidence-based smoke acceptance; never certify learning gains."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import pickle
import platform
from pathlib import Path

from qorerl.observability.calibration import load_calibration, read_events, read_json

_PACKAGES = ("torch", "ray", "transformers", "transformer_engine", "megatron-core", "megatron-bridge", "vllm")


def environment_report():
    """Report this process, including its allocator peaks, without resetting counters."""
    import torch

    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
        "memory_scope": "Per-process PyTorch allocator; excludes non-PyTorch and simultaneous device peaks",
    }
    if result["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": [props.major, props.minor],
                    "total_bytes": props.total_memory,
                    "allocated_bytes": torch.cuda.memory_allocated(index),
                    "reserved_bytes": torch.cuda.memory_reserved(index),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
                }
            )
    return result


def _check(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


def _result(checks):
    return (
        "failed"
        if any(c["status"] == "fail" for c in checks)
        else ("incomplete" if any(c["status"] == "missing" for c in checks) else "passed")
    )


def preflight(cfg, *, role="all"):
    """Check the current environment. This does not load the model or run training."""
    if role not in ("all", "learner", "rollout"):
        raise ValueError("role must be all, learner, or rollout")
    env = environment_report()
    checks = [
        _check(
            "native_backend",
            "fail" if cfg.is_cpu_test else "pass",
            "Expected Megatron + vLLM, not CPU simulation.",
        )
    ]
    devices = env["devices"]
    required = 1 if role != "all" or cfg.worker_placement == "collocate" else cfg.rollout_workers + 1
    checks.append(
        _check(
            "visible_gpus",
            "pass" if len(devices) >= required else "fail",
            f"Visible {len(devices)}; required {required} for role={role}.",
        )
    )
    required_packages = ["torch", "ray", "transformers", "transformer_engine"]
    if role in ("learner", "all"):
        required_packages += ["megatron-core", "megatron-bridge"]
    if role in ("rollout", "all"):
        required_packages += ["vllm"]
    missing = [p for p in required_packages if not env["packages"].get(p)]
    checks.append(
        _check(
            "packages",
            "missing" if missing else "pass",
            f"Missing: {missing}" if missing else "Versions recorded; presence is not ABI compatibility.",
        )
    )
    precision = cfg.recipe.precision if cfg.recipe.is_low_precision else "bf16"
    if devices and cfg.recipe.is_low_precision:
        from qorerl.recipes.te_block_linear import validate_native_format

        try:
            for index in range(len(devices)):
                validate_native_format(precision, f"cuda:{index}")
            checks.append(_check("native_format", "pass", f"{precision} hardware and TE API checks passed."))
        except (RuntimeError, ValueError, ImportError, AttributeError) as exc:
            checks.append(_check("native_format", "fail", f"{type(exc).__name__}: {exc}"))
    elif devices:
        checks.append(
            _check(
                "bf16_hardware",
                "pass" if all(d["capability"][0] >= 8 for d in devices) else "fail",
                "BF16 baseline requires supported NVIDIA hardware.",
            )
        )
    return {
        "schema": 1,
        "scope": "environment_preflight",
        "result": _result(checks),
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "precision": precision,
        "placement": cfg.worker_placement,
        "role": role,
        "environment": env,
        "checks": checks,
        "not_assessed": [
            "Model loading and engine ABI compatibility",
            "Memory coexistence under load",
            "Native forward/backward execution",
            "Numerical correctness and learning quality",
        ],
    }


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _policy(value):
    return (
        isinstance(value, dict)
        and type(value.get("weight_version")) is int
        and value["weight_version"] >= 0
        and isinstance(value.get("recipe_id"), str)
        and bool(value["recipe_id"])
    )


def _environments(value):
    if isinstance(value, dict):
        if isinstance(value.get("environment"), dict):
            yield value["environment"]
        workers = value.get("workers", [])
        if isinstance(workers, list):
            for worker in workers:
                yield from _environments(worker)


def _checkpoint_check(directory, *, min_updates):
    """Explicit opt-in: deserialize only local run checkpoints using weights_only=True."""
    import torch

    paths = sorted((directory / "checkpoints").glob("step-*/learner.pt"))
    if len(paths) < 2:
        return _check("checkpoint_contents", "missing", "Need initial and final checkpoints in this run.")
    try:
        first = torch.load(paths[0], map_location="cpu", weights_only=True)
        last = torch.load(paths[-1], map_location="cpu", weights_only=True)
        before, after = first["model"], last["model"]
        if not before or before.keys() != after.keys():
            raise ValueError("Model tensor names changed")
        changed = 0
        for name, a in before.items():
            b = after[name]
            # PyTorch modules can include disabled TE extra-state entries in
            # state_dict. They are not parameters and must not imply an update.
            if name.endswith("_extra_state") and a is None and b is None:
                continue
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise ValueError("Model state contains non-tensor entries")
            if (
                a.shape != b.shape
                or a.dtype != b.dtype
                or not torch.isfinite(a).all()
                or not torch.isfinite(b).all()
            ):
                raise ValueError("Invalid model tensors")
            changed += not torch.equal(a, b)
        updates = last["optimizer_steps"] - first["optimizer_steps"]
        valid = changed > 0 and updates >= min_updates and bool(last["optimizer"]["state"])
        return _check(
            "checkpoint_contents",
            "pass" if valid else "fail",
            f"{changed} model tensors changed; {updates} actual updates; optimizer state present. "
            "This is not a resumed-vs-uninterrupted equivalence test.",
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError, EOFError) as exc:
        return _check("checkpoint_contents", "fail", f"{type(exc).__name__}: {exc}")


def validate_run(
    run_dir,
    *,
    related_runs=(),
    min_updates=2,
    max_logprob_gap=None,
    check_checkpoints=False,
    require_calibration=False,
):
    """Audit one run segment. Prior runs supply calibration provenance, not extra updates."""
    if type(min_updates) is not int or min_updates < 2:
        raise ValueError("min_updates must be an integer >= 2")
    if max_logprob_gap is not None and (not _finite_number(max_logprob_gap) or max_logprob_gap < 0):
        raise ValueError("max_logprob_gap must be a finite nonnegative experiment-specific bound")
    directory = Path(run_dir).resolve()
    warnings = []
    config = read_json(directory / "config.json", warnings)
    summary = read_json(directory / "summary.json", warnings)
    workers = read_json(directory / "workers.json", warnings).get("workers", [])
    metrics = read_events(directory / "metrics.jsonl", warnings, tail=False)
    publications = read_events(directory / "publications.jsonl", warnings, tail=False)
    trajectories = read_events(directory / "trajectories.jsonl", warnings, tail=False)
    calibration = load_calibration([*related_runs, directory], warnings, tail=False)
    checks = []
    native = (config.get("learner_backend"), config.get("rollout_backend")) == ("megatron", "vllm")
    checks.append(
        _check(
            "native_execution",
            "pass" if native and summary.get("native_gpu_executed") is True else "fail",
            "Requires native backends and an execution summary; CPU fixtures cannot pass.",
        )
    )
    checks.append(
        _check(
            "completed",
            "pass" if summary.get("status") == "completed" else "missing",
            f"Run status: {summary.get('status', 'not recorded')}.",
        )
    )
    checks.append(
        _check(
            "incident",
            "fail" if (directory / "incident.json").exists() else "pass",
            "An incident.json means this segment stopped on an error.",
        )
    )
    rollout_workers = config.get("rollout_workers")
    expected_workers = rollout_workers + 1 if type(rollout_workers) is int and rollout_workers > 0 else 0
    worker_evidence = expected_workers > 1 and isinstance(workers, list) and len(workers) == expected_workers
    if worker_evidence:
        worker_evidence = all(
            isinstance(w, dict)
            and isinstance(w.get("node"), str)
            and w["node"]
            and isinstance(w.get("gpu_ids"), list)
            and w["gpu_ids"]
            for w in workers
        )
    if worker_evidence and config.get("worker_placement") == "collocate":
        worker_evidence = (
            len(workers) == 2
            and len({w["node"] for w in workers}) == 1
            and all(len(w["gpu_ids"]) == 1 for w in workers)
            and workers[0]["gpu_ids"] == workers[1]["gpu_ids"]
        )
    checks.append(
        _check(
            "placement",
            "pass" if worker_evidence else "missing",
            "Worker node and assigned GPU IDs must match the requested placement.",
        )
    )
    execution = summary.get("execution", [])
    environments = (
        [env for item in execution for env in _environments(item)] if isinstance(execution, list) else []
    )
    measured = (
        expected_workers > 1
        and len(environments) >= expected_workers
        and all(
            env.get("cuda_available") is True
            and isinstance(env.get("devices"), list)
            and env["devices"]
            and env.get("packages")
            and all(
                isinstance(d, dict) and _finite_number(d.get("peak_reserved_bytes")) for d in env["devices"]
            )
            for env in environments
        )
    )
    checks.append(
        _check(
            "environment_and_allocator_peaks",
            "pass" if measured else "missing",
            "Need CUDA/device/package and allocator-peak evidence from both executing engines. "
            "Separate process peaks must not be summed into a simultaneous device peak.",
        )
    )
    required_metrics = (
        "loss",
        "gradient_norm",
        "optimizer_steps",
        "pre_update_abs_log_ratio_mean",
        "pre_update_abs_log_ratio_max",
        "nonzero_advantage_fraction",
    )
    finite = bool(metrics) and all(
        all(_finite_number(row.get(k)) for k in required_metrics) for row in metrics
    )
    checks.append(
        _check(
            "finite_training_metrics",
            "pass" if finite else "missing",
            "Need loss, gradients, actual update counts and same-snapshot mean/max log-prob gaps.",
        )
    )
    epochs = config.get("update_epochs", 1)
    active_updates = 0
    consistent = finite and type(epochs) is int and epochs > 0
    previous = None
    for row in metrics:
        count, skipped = row.get("optimizer_steps"), row.get("skipped_zero_advantage")
        if type(count) is not int or count < 0 or type(skipped) is not bool:
            consistent = False
            break
        increment = 0 if skipped else epochs
        if previous is not None and count != previous + increment:
            consistent = False
        if not skipped and (
            not _finite_number(row.get("nonzero_advantage_fraction"))
            or row["nonzero_advantage_fraction"] <= 0
        ):
            consistent = False
        if count < increment:
            consistent = False
        active_updates += increment
        previous = count
    checks.append(
        _check(
            "optimizer_updates",
            "pass" if consistent and active_updates >= min_updates else "fail",
            f"{active_updates} updates in this segment; require {min_updates} with consistent counters.",
        )
    )
    gaps = [
        row["pre_update_abs_log_ratio_max"]
        for row in metrics
        if _finite_number(row.get("pre_update_abs_log_ratio_max"))
    ]
    checks.append(
        _check(
            "same_snapshot_logprob_gap",
            "missing"
            if max_logprob_gap is None or len(gaps) != len(metrics) or not gaps
            else "pass"
            if max(gaps) <= max_logprob_gap
            else "fail",
            f"Maximum absolute token log-prob gap: {max(gaps) if gaps else None}; "
            f"user-selected bound: {max_logprob_gap}. No universal default tolerance.",
        )
    )
    valid_pubs = bool(publications) and all(
        _policy(p.get("policy"))
        and isinstance(p.get("acknowledged"), list)
        and len(p["acknowledged"]) == expected_workers - 1
        and all(a == p["policy"] for a in p["acknowledged"])
        for p in publications
    )
    checks.append(
        _check(
            "publication_acknowledgments",
            "pass" if valid_pubs else "missing",
            "Every recorded publication requires matching acknowledgments from all replicas.",
        )
    )
    versions = {p["policy"]["weight_version"]: p["policy"] for p in publications if _policy(p.get("policy"))}
    valid_samples = bool(trajectories)
    first, last = summary.get("first_step"), summary.get("last_step")
    if type(first) is not int or type(last) is not int or first < 0 or last < first:
        valid_samples = False
    else:
        expected_steps = list(range(first + 1, last + 1))
        valid_samples = (
            [r.get("step") for r in metrics] == expected_steps
            and [r.get("step") for r in trajectories] == expected_steps
            and bool(expected_steps)
        )
        valid_samples &= versions.get(last) == summary.get("policy_version") and first in versions
    for row in trajectories:
        step, samples = row.get("step"), row.get("trajectories")
        if type(step) is not int or not isinstance(samples, list) or not samples:
            valid_samples = False
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                valid_samples = False
                continue
            response, logps = sample.get("response_ids"), sample.get("behavior_log_probs")
            valid_samples &= (
                sample.get("policy_version") == versions.get(step - 1)
                and (step - 1) in versions
                and isinstance(response, list)
                and bool(response)
                and isinstance(logps, list)
                and len(response) == len(logps)
                and all(_finite_number(p) and p <= 1e-5 for p in logps)
            )
    checks.append(
        _check(
            "sampling_after_publication",
            "pass" if valid_samples and valid_pubs else "fail",
            "Each batch must use the last acknowledged preceding policy and finite per-token log-probs; "
            "batch steps must be complete. This does not independently verify loaded tensor contents.",
        )
    )
    checks.append(
        _checkpoint_check(directory, min_updates=min_updates)
        if check_checkpoints
        else _check(
            "checkpoint_contents",
            "missing",
            "Use --check-checkpoints to compare local model tensors "
            "and optimizer state (CPU RAM roughly two checkpoints plus deserialization overhead).",
        )
    )
    if require_calibration:
        applied = [
            event
            for event in calibration["events"]
            if event["event"] == "applied" and event["run"] == str(directory)
        ]
        verified = False
        for event in applied:
            evaluation = next((e for e in calibration["evaluations"] if event in e["decisions"]), None)
            if evaluation is None:
                continue
            proposed = [r for r in evaluation["evidence"] if r["status"] == "proposed"]
            compared = bool(proposed) and all(
                _finite_number(r["current_relative_l2"])
                and _finite_number(r["candidate_relative_l2"])
                and r["candidate_relative_l2"] < r["current_relative_l2"]
                and r["fit_rows"]
                and r["heldout_rows"]
                for r in proposed
            )
            comparison = evaluation.get("comparison")
            if comparison is not None:
                compared = (
                    comparison["status"] == "proposed"
                    and set(comparison["targets"]) == {r["target"] for r in proposed}
                    and (evaluation["source"] != "native_gpu" or comparison["reference_execution"] == "bf16")
                )
            changed = (
                event["policy"]["weight_version"] == evaluation["policy"]["weight_version"]
                and event["policy"]["recipe_id"] != evaluation["policy"]["recipe_id"]
            )
            sampled = any(
                t.get("policy_version") == event["policy"]
                for batch in trajectories
                if type(batch.get("step")) is int and batch["step"] > event["step"]
                for t in batch.get("trajectories", [])
                if isinstance(t, dict)
            )
            verified |= bool(compared and changed and sampled)
        checks.append(
            _check(
                "calibration_apply_and_continue",
                "pass" if verified else "missing",
                "Require paired candidate evidence, a changed recipe at unchanged weight version, "
                "and a subsequent rollout. Supply --related-run for the proposal's source run.",
            )
        )
    checks.append(
        _check(
            "complete_readable_artifacts",
            "fail" if warnings else "pass",
            "\n".join(warnings) or "All available artifacts parsed within the validation read budget.",
        )
    )
    return {
        "schema": 1,
        "scope": "native_smoke_acceptance",
        "result": _result(checks),
        "run": str(directory),
        "checks": checks,
        "environment": environments,
        "not_assessed": [
            "Resumed-vs-uninterrupted optimizer and output equivalence",
            "Injected publication failure and fail-closed serving",
            "Simultaneous device-wide peak memory and non-PyTorch allocations",
            "Gradient normalization across workloads; independent post-refit weight/output checks",
            "Held-out task quality, multiple seeds and end-to-end cost benefit",
        ],
        "interpretation": (
            "Passing this smoke audit is not full GPU validation or evidence that SmoothQuant improves RL."
        ),
    }


def write_report(report, output):
    """Keep acceptance artifacts separate from the training run and its status."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [f"# QoreRL {report['scope']}", "", f"Result: **{report['result']}**", ""]
    for check in report["checks"]:
        lines.append(f"- **{check['status']} · {check['name']}**: {check['detail']}")
    lines += ["", "## Not assessed", "", *[f"- {item}" for item in report["not_assessed"]]]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return directory
