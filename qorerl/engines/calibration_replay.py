"""Bounded, provisional full-model execution for recipe candidate evaluation.

Recipes choose factors and acceptance rules. This module executes the same
recorded contexts with canonical, active or provisional recipe state. It never
updates optimizer state or publishes a policy.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch

from qorerl.engines.single_rank_learner import _calibration_prefixes, _CalibrationInputs
from qorerl.recipes.calibration import NonfiniteLogits


class CalibrationReplay:
    def __init__(self, learner, trajectories, max_rows):
        self.learner = learner
        self.sequences = list(_calibration_prefixes(trajectories, max_rows))
        self.total_rows = sum(len(tokens) - shared for tokens, shared in self.sequences)
        self.fit_rows = self.total_rows // 2
        self.heldout_rows = self.total_rows - self.fit_rows
        self.reference_execution = "cpu_fixture" if learner.cfg.is_cpu_test else "bf16"

    @contextmanager
    def _execution(self, *, reference=False, state=None, capture=None):
        installation = self.learner.recipe_installation
        if installation is None or not hasattr(installation, "reference_mode"):
            raise ValueError("This learner installation cannot replay a canonical model reference")
        original = {
            key: getattr(installation, key) for key in ("state", "reference_mode", "calibrator", "observer")
        }
        calls = dict(installation.calls) if hasattr(installation, "calls") else None
        modes = [(module, module.training) for module in self.learner.model.modules()]
        device = torch.device(self.learner.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        try:
            # Isolate transient device caches as well as candidate factors from live state.
            installation.state = deepcopy(state if state is not None else original["state"])
            installation.reference_mode = reference
            installation.calibrator = capture
            installation.observer = None
            self.learner.model.eval()
            with (
                torch.random.fork_rng(devices=devices),
                torch.no_grad(),
                torch.autocast(device.type, enabled=False),
            ):
                yield
        finally:
            for key, value in original.items():
                setattr(installation, key, value)
            if calls is not None:
                installation.calls.clear()
                installation.calls.update(calls)
            for module, training in modes:
                module.training = training

    def run(self, *, reference=False, state=None, collect_inputs=False):
        """Return bounded projection inputs and held-out logits on CPU.

        Each full forward generates its own activations. Only the unique held-out
        token positions contribute logits; fitting positions never enter the metric.
        """
        targets = tuple(self.learner.recipe_state.observation_targets)
        capture = _CalibrationInputs(targets) if collect_inputs else None
        logits_rows = []
        offset = 0
        with self._execution(reference=reference, state=state, capture=capture):
            for tokens, shared in self.sequences:
                if capture is not None:
                    capture.begin_sequence(len(tokens), shared)
                logits = self.learner.logits(torch.tensor([tokens], device=self.learner.device))
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 3
                    or logits.shape[:2] != (1, len(tokens))
                    or not logits.shape[-1]
                    or not logits.is_floating_point()
                ):
                    raise ValueError("Calibration requires [1, sequence, vocabulary] floating-point logits")
                if not torch.isfinite(logits).all().item():
                    raise NonfiniteLogits("Nonfinite full-model logits")
                if capture is not None and capture.seen != capture.rows.keys():
                    raise ValueError("Calibration replay did not execute every scoped target")
                new_rows = len(tokens) - shared
                start = shared + max(0, self.fit_rows - offset)
                if start < len(tokens):
                    logits_rows.append(logits[0, start:].detach().float().cpu().clone())
                offset += new_rows
                del logits
        inputs = {name: torch.cat(parts) for name, parts in capture.rows.items()} if capture else {}
        return inputs, torch.cat(logits_rows) if logits_rows else torch.empty(0, 0)
