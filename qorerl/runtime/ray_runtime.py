"""Synchronous GRPO over isolated Ray actors and versioned weight publication."""

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from qorerl.training.config import TrainConfig
from qorerl.training.contracts import assemble_batch
from qorerl.training.tasks import ArithmeticTask, CopyTask


def make_engine(role, cfg):
    if cfg.is_cpu_test:
        from qorerl.engines.torch_test import TestLearner, TestRollout

        return (TestLearner if role == "learner" else TestRollout)(cfg)
    if role == "learner":
        from qorerl.engines.megatron import MegatronLearner

        return MegatronLearner(cfg)
    from qorerl.engines.vllm import VLLMRollout

    return VLLMRollout(cfg)


@ray.remote(max_restarts=0, max_task_retries=0)
class LearnerActor:
    def __init__(self, cfg):
        self.engine = make_engine("learner", cfg)

    def ready(self):
        return {"role": "learner", "pid": os.getpid(), "node": ray.get_runtime_context().get_node_id()}

    def update(self, batch):
        return self.engine.update(batch)

    def export_weights(self):
        return self.engine.export_weights()

    def checkpoint(self):
        return self.engine.checkpoint()

    def restore(self, payload):
        self.engine.restore(payload)

    def close(self):
        self.engine.close()


@ray.remote(max_restarts=0, max_task_retries=0)
class RolloutActor:
    def __init__(self, cfg):
        self.engine = make_engine("rollout", cfg)

    def ready(self):
        return {"role": "rollout", "pid": os.getpid(), "node": ray.get_runtime_context().get_node_id()}

    def install_weights(self, snapshot):
        return self.engine.install_weights(snapshot)

    def generate(self, prompts, policy, seed):
        return self.engine.generate(prompts, policy, seed)

    def close(self):
        self.engine.close()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def save_checkpoint(directory, learner, cfg, step, policy):
    """Publish a checkpoint directory only after its payload and metadata exist."""
    root = directory / "checkpoints"
    root.mkdir(exist_ok=True)
    temp = root / f".step-{step:06d}.pending"
    target = root / f"step-{step:06d}"
    temp.mkdir()
    (temp / "learner.pt").write_bytes(ray.get(learner.checkpoint.remote(), timeout=cfg.rpc_timeout))
    write_json(
        temp / "metadata.json",
        {
            "schema": 1,
            "step": step,
            "policy": asdict(policy),
            "config": cfg.to_dict(),
            "boundary": "all_rollouts_acknowledged_no_inflight_samples",
        },
    )
    temp.rename(target)
    return str(target.resolve())


def resume_metadata(path, cfg):
    metadata = json.loads((Path(path) / "metadata.json").read_text())
    if metadata["schema"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    previous = TrainConfig(**metadata["config"])
    allowed = {"steps", "checkpoint_every", "rpc_timeout", "learner_runtime_env", "rollout_runtime_env"}
    changed = [k for k, v in cfg.to_dict().items() if k not in allowed and previous.to_dict()[k] != v]
    if changed:
        raise ValueError(f"Resume currently requires identical training semantics; changed: {changed}")
    if cfg.steps <= metadata["step"] or metadata["policy"]["weights"] != metadata["step"]:
        raise ValueError("Resume requires a consistent checkpoint and a larger total steps value")
    return metadata


def run(cfg, output, *, address=None, resume=None):
    """One learner rank and independently placed rollout replicas.

    State machine: publish/ack -> sample -> assemble -> update -> publish/ack.
    No actor method is retried after a failure: update/refit are mutations.
    """
    if not cfg.is_cpu_test:
        from transformers import AutoConfig

        # Resolve a mutable HF revision once, before either engine starts.
        if resume and cfg.model_revision == "main":
            saved = json.loads((Path(resume) / "metadata.json").read_text())
            cfg = replace(cfg, model_revision=saved["config"]["model_revision"])
        hf = AutoConfig.from_pretrained(cfg.model, revision=cfg.model_revision)
        if getattr(hf, "_commit_hash", None):
            cfg = replace(cfg, model_revision=hf._commit_hash)
    metadata = resume_metadata(resume, cfg) if resume else None
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", cfg.to_dict())
    task = CopyTask() if cfg.is_cpu_test else ArithmeticTask(cfg)
    owned_ray = not ray.is_initialized()
    actors, pg = [], None
    last_checkpoint = None
    started = time.perf_counter()
    try:
        if owned_ray:
            options = {"address": address} if address else {"num_cpus": cfg.rollout_workers + 1}
            ray.init(**options, include_dashboard=False, log_to_driver=False)
        gpus = 0 if cfg.is_cpu_test else 1
        resources = ray.cluster_resources()
        required = cfg.rollout_workers + 1
        if resources.get("CPU", 0) < required or resources.get("GPU", 0) < gpus * required:
            raise RuntimeError(f"Need {required} CPUs and {gpus * required} GPUs; found {resources}")
        bundles = [{"CPU": 1, **({"GPU": 1} if gpus else {})} for _ in range(required)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready(), timeout=cfg.rpc_timeout)

        def actor_options(index, env):
            return dict(
                num_cpus=1,
                num_gpus=gpus,
                runtime_env=env,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=index,
                    placement_group_capture_child_tasks=False,
                ),
            )

        learner = LearnerActor.options(**actor_options(0, cfg.learner_runtime_env)).remote(cfg)
        actors.append(learner)
        workers = []
        for i in range(cfg.rollout_workers):
            worker = RolloutActor.options(**actor_options(i + 1, cfg.rollout_runtime_env)).remote(cfg)
            workers.append(worker)
            actors.append(worker)
        info = ray.get([actor.ready.remote() for actor in actors], timeout=cfg.rpc_timeout)
        write_json(directory / "workers.json", {"driver_pid": os.getpid(), "workers": info})
        if resume:
            ray.get(
                learner.restore.remote((Path(resume) / "learner.pt").read_bytes()), timeout=cfg.rpc_timeout
            )

        def publish():
            # Ray passes this object reference directly to actors. The driver
            # never fetches full weights. Replace this transport later with NCCL/IPC.
            snapshot = learner.export_weights.remote()
            versions = ray.get(
                [worker.install_weights.remote(snapshot) for worker in workers], timeout=cfg.rpc_timeout
            )
            if any(v != versions[0] for v in versions):
                raise RuntimeError("Rollout replicas acknowledged different policies")
            return versions[0]

        policy = publish()
        first_step = metadata["step"] if metadata else 0
        if policy.weights != first_step:
            raise RuntimeError("Published policy and coordinator progress disagree")
        last_checkpoint = save_checkpoint(directory, learner, cfg, first_step, policy)
        for step in range(first_step + 1, cfg.steps + 1):
            tick = time.perf_counter()
            prompts = task.batch(step, cfg.prompts_per_step, cfg.seed)
            # Every group stays intact on one rollout replica; arrival order is irrelevant.
            shards = [prompts[i :: cfg.rollout_workers] for i in range(cfg.rollout_workers)]
            replies = ray.get(
                [
                    worker.generate.remote(shard, policy, cfg.seed + step * 1000003)
                    for worker, shard in zip(workers, shards, strict=True)
                ],
                timeout=cfg.rpc_timeout,
            )
            rollout_seconds = time.perf_counter() - tick
            batch = assemble_batch(prompts, [t for reply in replies for t in reply], cfg.group_size, policy)
            update_start = time.perf_counter()
            metrics = ray.get(learner.update.remote(ray.put(batch)), timeout=cfg.rpc_timeout)
            train_seconds = time.perf_counter() - update_start
            sync_start = time.perf_counter()
            policy = publish()
            if policy.weights != step or metrics["policy_version"] != step:
                raise RuntimeError("Publication did not commit the expected training version")
            metrics.update(
                step=step,
                reward_mean=batch.rewards.mean().item(),
                nonzero_advantage_fraction=(batch.advantages != 0).float().mean().item(),
                response_length_mean=sum(len(t.response_ids) for t in batch.trajectories)
                / len(batch.trajectories),
                truncated_fraction=sum(t.finish_reason == "length" for t in batch.trajectories)
                / len(batch.trajectories),
                rollout_seconds=rollout_seconds,
                train_seconds=train_seconds,
                sync_seconds=time.perf_counter() - sync_start,
            )
            append_json(directory / "metrics.jsonl", metrics)
            append_json(
                directory / "trajectories.jsonl",
                {
                    "step": step,
                    "trajectories": [asdict(t) for t in batch.trajectories],
                    "rewards": batch.rewards.tolist(),
                    "advantages": batch.advantages.tolist(),
                },
            )
            if step % cfg.checkpoint_every == 0 or step == cfg.steps:
                last_checkpoint = save_checkpoint(directory, learner, cfg, step, policy)
            print(
                f"step={step} reward={metrics['reward_mean']:.3f} "
                f"ratio={metrics['ratio_mean']:.4f} version={policy.weights}",
                flush=True,
            )
        summary = {
            "status": "completed",
            "backend": f"{cfg.learner_backend}/{cfg.rollout_backend}",
            "first_step": first_step,
            "last_step": cfg.steps,
            "final_metrics": metrics,
            "checkpoint": last_checkpoint,
            "seconds": time.perf_counter() - started,
            "native_gpu_validated_by_this_run": not cfg.is_cpu_test,
        }
        write_json(directory / "summary.json", summary)
        return summary
    except BaseException as exc:
        write_json(
            directory / "incident.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "last_recovery_checkpoint": last_checkpoint,
                "action": "stop; no automatic retry of optimizer updates or partial weight publication",
            },
        )
        raise
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            except Exception:
                pass
            finally:
                ray.kill(actor, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        if owned_ray:
            ray.shutdown()
