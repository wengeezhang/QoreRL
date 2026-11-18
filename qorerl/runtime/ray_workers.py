"""Stateful Ray workers. Algorithms and model execution belong to the engines."""

import os

import ray

from qorerl.engines.base import LearnerEngine, RolloutEngine
from qorerl.engines.factory import make_learner_engine, make_rollout_engine


@ray.remote(max_restarts=0, max_task_retries=0)
class RayLearnerWorker:
    def __init__(self, cfg):
        self.engine: LearnerEngine = make_learner_engine(cfg)

    def ready(self):
        return {
            "role": "learner",
            "pid": os.getpid(),
            "node": ray.get_runtime_context().get_node_id(),
            "execution": self.execution_report(),
        }

    def execution_report(self):
        report = getattr(self.engine, "execution_report", None)
        return report() if report is not None else {"backend": "torch_test"}

    def update(self, batch):
        return self.engine.update(batch)

    def current_policy(self):
        return self.engine.policy_version

    def completed_updates(self):
        return self.engine.optimizer_steps

    def prepare_calibration(self, batch):
        return self.engine.prepare_calibration(batch)

    def apply_calibration(self, scales):
        return self.engine.apply_calibration(scales)

    def export_weights(self):
        return self.engine.export_weights()

    def checkpoint(self):
        return self.engine.checkpoint()

    def restore(self, payload):
        self.engine.restore(payload)

    def close(self):
        self.engine.close()


@ray.remote(max_restarts=0, max_task_retries=0)
class RayRolloutWorker:
    def __init__(self, cfg):
        self.engine: RolloutEngine = make_rollout_engine(cfg)

    def ready(self):
        return {
            "role": "rollout",
            "pid": os.getpid(),
            "node": ray.get_runtime_context().get_node_id(),
            "execution": self.execution_report(),
        }

    def execution_report(self):
        report = getattr(self.engine, "execution_report", None)
        return report() if report is not None else {"backend": "torch_test"}

    def install_weights(self, snapshot):
        return self.engine.install_weights(snapshot)

    def generate(self, prompts, policy_version, seed):
        return self.engine.generate(prompts, policy_version, seed)

    def close(self):
        self.engine.close()
