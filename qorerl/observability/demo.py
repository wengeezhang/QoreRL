"""Deterministic, prominently labelled fixture data; never training evidence."""

import json
import math
from pathlib import Path


def write_demo(output):
    """Create a new preview directory, refusing to overwrite any existing run."""
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": "Illustrative 32-layer model (not a measured workload)",
        "recipe_id": "mxfp8-hadamard-mlp-down-v1",
        "recipe_options": {"block_size": 128, "seed": 0},
        "source": "demo",
        "description": "Synthetic UI preview only. No GPU was run; no performance or quality claim.",
    }
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    records = []
    for layer in range(24):
        latency = round(0.65 + 0.012 * layer + 0.11 * (1 + math.sin(layer * 0.8)) / 2, 4)
        error = 0.016 + 0.022 * (1 + math.cos(layer * 1.7)) / 2
        if layer in (6, 17, 21):
            error += 0.05
        improvement = 0.38 + 0.08 * math.sin(layer * 0.4)
        if layer == 15:
            improvement = -0.12
        records.append(
            {
                "schema": 1,
                "kind": "recipe_probe",
                "source": "demo",
                "status": "ok",
                "step": 260 + (layer // 2) * 20,
                "weight_version": 259 + (layer // 2) * 20,
                "optimizer_steps": 259 + (layer // 2) * 20,
                "recipe_id": "mxfp8-hadamard-mlp-down-v1:demo",
                "layer": layer,
                "num_layers": 32,
                "module": f"model.layers.{layer}.mlp.down_proj",
                "phase": "learner.fprop",
                "shape_mnk": [128, 1024, 4096],
                "original_rows": 512,
                "repeats": 5,
                "variants": {
                    "bf16": {"latency_ms": round(latency * 1.4, 4), "output_relative_l2": 0.0},
                    "mxfp8": {"latency_ms": latency, "output_relative_l2": round(error, 6)},
                    "mxfp8_hadamard": {
                        "latency_ms": round(latency * (1.06 + 0.01 * math.sin(layer)), 4),
                        "output_relative_l2": round(error * (1 - improvement), 6),
                    },
                },
                "probe_seconds": 0.0,
                "timing_scope": "synthetic_preview_not_measured",
                "reference": "synthetic_preview_not_measured",
            }
        )
    (directory / "recipe_metrics.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8"
    )
    return directory
