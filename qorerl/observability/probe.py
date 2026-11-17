"""Bounded, paired FPROP replays of actual pre-update learner operands.

This reports local output error against BF16, not policy or training quality.
There is deliberately no CPU/native-emulation production backend.
"""

import math
import random
import statistics
import sys
import time
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from qorerl.recipes.hadamard import PreparedHadamard, make_signs
from qorerl.recipes.state import down_proj_name


def _fatal_cuda_error(exc):
    """A poisoned CUDA context cannot safely continue the training update."""
    message = str(exc).lower()
    return (
        isinstance(exc, RuntimeError)
        and not isinstance(exc, torch.cuda.OutOfMemoryError)
        and any(
            marker in message
            for marker in (
                "cuda error:",
                "device-side assert",
                "illegal memory access",
                "unspecified launch failure",
                "misaligned address",
                "context is destroyed",
                "cublas_status_execution_failed",
            )
        )
    )


@contextmanager
def _preserve_rng(device):
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


def _relative_l2(output, reference):
    delta = torch.linalg.vector_norm(output.float() - reference.float()).item()
    scale = torch.linalg.vector_norm(reference.float()).item()
    if not math.isfinite(delta) or not math.isfinite(scale):
        raise FloatingPointError("Probe output contains non-finite values")
    if scale == 0:
        if delta == 0:
            return 0.0
        raise FloatingPointError("Relative L2 is undefined for a zero BF16 reference")
    return delta / scale


@torch.no_grad()
def native_replay(x, weight, *, repeats, block_size, seed):
    """Time full variant calls, including transforms, quantization and padding.

    One warmup per variant precedes equal, interleaved repetitions. CUDA events
    measure device elapsed time; host capture plus replay wall time is reported
    apart. That wall time may include queued training work at synchronization;
    it is not isolated instrumentation overhead.
    Neither canonical weights nor transformed/quantized weights are cached.
    """
    from qorerl.recipes.te_mxfp8 import native_mxfp8_linear, validate_native_mxfp8

    environment = validate_native_mxfp8(x.device)
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Recipe probes require canonical BF16 operands")
    transform = PreparedHadamard(make_signs(weight.shape[1], seed), block_size)
    variants = {
        "bf16": lambda: F.linear(x, weight),
        "mxfp8": lambda: native_mxfp8_linear(x, weight),
        "mxfp8_hadamard": lambda: native_mxfp8_linear(transform.apply(x), transform.apply(weight)),
    }
    timings = {name: [] for name in variants}
    outputs = {}
    with torch.cuda.device(x.device), torch.autocast("cuda", enabled=False):
        for operation in variants.values():
            operation()
        torch.cuda.current_stream(x.device).synchronize()
        names = list(variants)
        for repeat in range(repeats):
            # Rotate ordering to reduce systematic first/last-variant bias.
            order = names[repeat % len(names) :] + names[: repeat % len(names)]
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                outputs[name] = variants[name]()
                end.record()
                end.synchronize()
                milliseconds = start.elapsed_time(end)
                if not math.isfinite(milliseconds) or milliseconds < 0:
                    raise FloatingPointError("Invalid CUDA event latency")
                timings[name].append(milliseconds)
    reference = outputs["bf16"]
    result = {}
    for name, output in outputs.items():
        latency = statistics.median(timings[name])
        error = _relative_l2(output, reference)
        result[name] = {
            "latency_ms": latency,
            "latency_min_ms": min(timings[name]),
            "latency_max_ms": max(timings[name]),
            "output_relative_l2": error,
        }
    return result, environment


def _bounded_rows(x, rows):
    """Copy only prefix rows even for a noncontiguous, higher-rank activation."""
    if x.ndim == 2:
        return x.detach()[:rows].clone(memory_format=torch.contiguous_format)
    # reshape/contiguous on the full tensor could silently retain/copy all
    # activations. Advanced indexing instead materializes only sampled rows.
    flattened = torch.arange(rows, device=x.device)
    indices = []
    for size in reversed(x.shape[:-1]):
        indices.append(flattened % size)
        flattened = flattened // size
    return x.detach()[tuple(reversed(indices))].contiguous()


class RecipeProbe:
    """Capture once per selected layer in the first microbatch of a window.

    The stateless round-robin schedule derives from the coordinator step, so
    checkpoint restore neither repeats a window nor needs extra model state.
    ``replay`` injection is for explicit unit-test stubs, never configuration.

    ``probe_seconds`` sums host capture scheduling and replay wall time. Capture
    does not synchronize asynchronous GPU copies, and replay's warmup sync may
    wait for queued training work. It is not isolated instrumentation overhead.
    """

    def __init__(self, config, recipe, targets, num_layers, *, replay=None):
        self.config, self.recipe, self.num_layers = config, recipe, num_layers
        self.targets = dict(targets)
        self.layers = {down_proj_name(index): index for index in recipe.select_layers(num_layers)}
        if not self.targets or set(self.targets) != set(self.layers):
            raise ValueError("Recipe probe targets must match the installed recipe")
        self.replay = native_replay if replay is None else replay
        self._selected = {}
        self._snapshots = {}
        self._events = []
        self._snapshot_bytes = 0
        self._capture_seconds = 0.0

    def begin(self, *, step, weight_version, optimizer_steps):
        self._selected.clear()
        self._snapshots.clear()
        self._events.clear()
        self._snapshot_bytes = 0
        self._capture_seconds = 0.0
        if not self.config.enabled or step % self.config.every_steps:
            return
        names = list(self.targets)
        count = min(self.config.layers_per_window, len(names))
        offset = ((step // self.config.every_steps - 1) * count) % len(names)
        for position in range(count):
            name = names[(offset + position) % len(names)]
            n, k = self.targets[name]
            block_size = self.recipe.hadamard_block_size or min(128, k & -k)
            self._selected[name] = {
                "schema": 1,
                "kind": "recipe_probe",
                "source": "measured",
                "status": "skipped",
                "step": step,
                "weight_version": weight_version,
                "optimizer_steps": optimizer_steps,
                "recipe_id": self.recipe.identity,
                "layer": self.layers[name],
                "num_layers": self.num_layers,
                "module": name.removesuffix(".weight"),
                "phase": "learner.fprop",
                "shape_mnk": [0, n, k],
                "original_rows": 0,
                "repeats": self.config.repeats,
                "variants": {},
                "probe_seconds": 0.0,
                "probe_seconds_scope": "host_capture_plus_replay_wall_may_include_queued_training",
                "hadamard_block_size": block_size,
                "hadamard_seed": self.recipe.seed + self.layers[name],
                "sampling": "prefix_rows_first_microbatch_first_epoch",
                "timing": "cuda_event_median_full_variant_call",
                "warmup_iterations": 1,
            }

    def capture(self, name, x, weight):
        event = self._selected.pop(name, None)
        if event is None:
            return
        started = time.perf_counter()
        try:
            if x.ndim < 2 or tuple(weight.shape) != tuple(self.targets[name]):
                raise ValueError("Probe operand shape disagrees with the installed projection")
            if x.shape[-1] != weight.shape[1] or not x.numel():
                raise ValueError("Probe requires nonempty matching input channels")
            original_rows = x.numel() // x.shape[-1]
            rows = min(original_rows, self.config.max_rows)
            event.update(original_rows=original_rows, shape_mnk=[rows, *weight.shape])
            snapshot_bytes = rows * x.shape[-1] * x.element_size() + weight.numel() * weight.element_size()
            event["snapshot_bytes"] = snapshot_bytes
            if self._snapshot_bytes + snapshot_bytes > self.config.max_snapshot_bytes:
                event["error"] = "Window snapshot byte cap exceeded; no operands copied for this layer"
                self._events.append(event)
                return
            with torch.no_grad():
                sampled = _bounded_rows(x, rows)
                saved_weight = weight.detach().clone(memory_format=torch.contiguous_format)
            self._snapshots[name] = (event, sampled, saved_weight)
            self._snapshot_bytes += snapshot_bytes
        except Exception as exc:
            if _fatal_cuda_error(exc):
                raise
            event.update(status="error", error=f"{type(exc).__name__}: {exc}")
            self._events.append(event)
        finally:
            elapsed = time.perf_counter() - started
            event["probe_seconds"] += elapsed
            self._capture_seconds += elapsed

    def finish(self):
        """Replay before the update; return rows and host capture + replay wall time.

        The latter may include queued training work and does not isolate the
        asynchronous snapshot copy time. No extra training synchronization is
        introduced to turn it into a measurement of instrumentation overhead.
        """
        for event in self._selected.values():
            event["error"] = "Selected layer was not executed in the first microbatch"
            self._events.append(event)
        self._selected.clear()  # Later microbatches/epochs must never capture.
        elapsed = self._capture_seconds
        try:
            while self._snapshots:
                name = next(iter(self._snapshots))
                event, x, weight = self._snapshots.pop(name)
                started = time.perf_counter()
                try:
                    with _preserve_rng(x.device), torch.no_grad():
                        variants, environment = self.replay(
                            x,
                            weight,
                            repeats=self.config.repeats,
                            block_size=event["hadamard_block_size"],
                            seed=event["hadamard_seed"],
                        )
                    _validate_variants(variants)
                    event.update(status="ok", variants=variants, environment=environment)
                except Exception as exc:
                    if _fatal_cuda_error(exc):
                        raise
                    event.update(status="error", error=f"{type(exc).__name__}: {exc}")
                finally:
                    seconds = time.perf_counter() - started
                    event["probe_seconds"] += seconds
                    elapsed += seconds
                self._events.append(event)
                del x, weight
            return list(self._events), elapsed
        finally:
            self._snapshots.clear()
            self._events.clear()
            self._snapshot_bytes = 0
            self._capture_seconds = 0.0


def _validate_variants(variants):
    if set(variants) != {"bf16", "mxfp8", "mxfp8_hadamard"}:
        raise ValueError("Probe must return all three paired variants")
    for values in variants.values():
        for key in ("latency_ms", "output_relative_l2"):
            value = values[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise FloatingPointError(f"Invalid probe {key}")
    if variants["bf16"]["output_relative_l2"] != 0:
        raise ValueError("BF16 reference must have zero relative error")
