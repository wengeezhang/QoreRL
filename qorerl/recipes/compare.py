"""GPU operator probe for BF16, scoped MXFP8, and MXFP8 + fixed Hadamard.

This evaluates a local linear/MSE problem, not an RL learning benchmark. It never
falls back to CPU or fake quantization. Model-level evidence comes from the three
GRPO configurations; this probe is an earlier native-kernel acceptance step.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from qorerl.recipes.hadamard import PreparedHadamard, make_signs
from qorerl.recipes.te_mxfp8 import native_mxfp8_linear, validate_native_mxfp8


def relative_error(value, reference):
    return ((value.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)).item()


def cosine(value, reference):
    a, b = value.float().flatten(), reference.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def run_probe(rows=33, in_features=256, out_features=128, block_size=128, seed=42, repeats=5):
    if min(rows, in_features, out_features, repeats) < 1:
        raise ValueError("Dimensions and repeats must be positive")
    if in_features % 32 or out_features % 32:
        raise ValueError("Input/output dimensions must be multiples of 32")
    backend = validate_native_mxfp8()
    transform = PreparedHadamard(make_signs(in_features, seed), block_size)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16, generator=generator)
    w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16, generator=generator)
    w = (w.float() / in_features**0.5).to(torch.bfloat16)
    target = torch.randn(rows, out_features, device="cuda", generator=generator)
    modes = {
        "bf16": lambda a, b: F.linear(a, b),
        "mxfp8": native_mxfp8_linear,
        "mxfp8_hadamard": lambda a, b: native_mxfp8_linear(transform.apply(a), transform.apply(b)),
    }
    report = {
        "kind": "native_operator_probe_not_rl_evidence",
        "backend": backend,
        "shape_mnk": [rows, out_features, in_features],
        "seed": seed,
        "block_size": block_size,
        "objective": "mean squared error on one fixed synthetic target",
        "memory_measurement": "warmed_up_torch_allocated_peak_above_iteration_start_not_total_gpu_memory",
        "modes": {},
    }
    baseline = None
    for name, operation in modes.items():
        timings, peaks = [], []
        for iteration in range(repeats + 1):
            a, b = x.detach().requires_grad_(), w.detach().requires_grad_()
            torch.cuda.synchronize()
            start_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            y = operation(a, b)
            loss = F.mse_loss(y.float(), target)
            dx, dw = torch.autograd.grad(loss, (a, b))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not all(torch.isfinite(t).all().item() for t in (y, loss, dx, dw)):
                raise FloatingPointError(f"Nonfinite values in {name}")
            if iteration:
                timings.append(elapsed)
                peaks.append(torch.cuda.max_memory_allocated() - start_allocated)
            result = (y.detach().cpu(), dx.detach().cpu(), dw.detach().cpu())
            loss_value = loss.item()
            del a, b, y, dx, dw, loss
        if baseline is None:
            baseline = result
        report["modes"][name] = {
            "loss": loss_value,
            "median_forward_backward_seconds": statistics.median(timings),
            "peak_extra_allocated_bytes": max(peaks),
            "output_relative_error": relative_error(result[0], baseline[0]),
            "dgrad_relative_error": relative_error(result[1], baseline[1]),
            "wgrad_relative_error": relative_error(result[2], baseline[2]),
            "wgrad_cosine": cosine(result[2], baseline[2]),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = vars(parser.parse_args())
    output = Path(args.pop("out"))
    report = run_probe(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
