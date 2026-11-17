"""Synchronous GRPO over isolated Ray actors and versioned weight publication."""

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from qorerl.runtime.checkpoint import resume_metadata, save_checkpoint, write_json
from qorerl.runtime.ray_workers import RayLearnerWorker, RayRolloutWorker
from qorerl.training.contracts import PolicyVersion, TrainingBatch, validate_and_order_trajectories
from qorerl.training.objective import compute_group_advantages
from qorerl.training.tasks import ArithmeticTask, CopyTask, exact_match_rewards


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def run_sync_grpo(cfg, output, *, address=None, resume=None):
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
    write_json(directory / "recipe.json", {**cfg.recipe.to_dict(), "identity": cfg.policy_recipe_id})
    task = CopyTask() if cfg.is_cpu_test else ArithmeticTask(cfg)
    owned_ray = not ray.is_initialized()
    worker_handles, resource_group = [], None
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
        resource_group = placement_group(bundles, strategy="PACK")
        ray.get(resource_group.ready(), timeout=cfg.rpc_timeout)

        def actor_options(index, env):
            return dict(
                num_cpus=1,
                num_gpus=gpus,
                runtime_env=env,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=resource_group,
                    placement_group_bundle_index=index,
                    placement_group_capture_child_tasks=False,
                ),
            )

        learner = RayLearnerWorker.options(**actor_options(0, cfg.learner_runtime_env)).remote(cfg)
        worker_handles.append(learner)
        rollout_workers = []
        for i in range(cfg.rollout_workers):
            worker = RayRolloutWorker.options(**actor_options(i + 1, cfg.rollout_runtime_env)).remote(cfg)
            rollout_workers.append(worker)
            worker_handles.append(worker)
        info = ray.get([worker.ready.remote() for worker in worker_handles], timeout=cfg.rpc_timeout)
        write_json(directory / "workers.json", {"driver_pid": os.getpid(), "workers": info})
        if resume:
            ray.get(
                learner.restore.remote((Path(resume) / "learner.pt").read_bytes()), timeout=cfg.rpc_timeout
            )

        def publish():
            # Ray passes this object reference directly to actors. The driver
            # never fetches full weights. Sharded/GPU transports will need a new
            # payload interface; the current snapshot is a full CPU tensor mapping.
            snapshot = learner.export_weights.remote()
            versions = ray.get(
                [worker.install_weights.remote(snapshot) for worker in rollout_workers],
                timeout=cfg.rpc_timeout,
            )
            if any(v != versions[0] for v in versions):
                raise RuntimeError("Rollout replicas acknowledged different policies")
            return versions[0]

        policy_version = publish()
        first_step = metadata["step"] if metadata else 0
        if policy_version.weight_version != first_step or (
            metadata and policy_version != PolicyVersion.from_dict(metadata["policy_version"])
        ):
            raise RuntimeError("Published policy and coordinator progress disagree")
        last_checkpoint = save_checkpoint(directory, learner, cfg, first_step, policy_version)
        for step in range(first_step + 1, cfg.steps + 1):
            tick = time.perf_counter()
            prompts = task.batch(step, cfg.prompts_per_step, cfg.seed)
            # Every group stays intact on one rollout replica; arrival order is irrelevant.
            prompt_batches = [prompts[i :: cfg.rollout_workers] for i in range(cfg.rollout_workers)]
            replies = ray.get(
                [
                    worker.generate.remote(prompt_batch, policy_version, cfg.seed + step * 1000003)
                    for worker, prompt_batch in zip(rollout_workers, prompt_batches, strict=True)
                ],
                timeout=cfg.rpc_timeout,
            )
            rollout_seconds = time.perf_counter() - tick
            trajectories = validate_and_order_trajectories(
                prompts, [t for reply in replies for t in reply], cfg.group_size, policy_version
            )
            rewards = exact_match_rewards(prompts, trajectories)
            advantages = compute_group_advantages(rewards, cfg.group_size)
            batch = TrainingBatch(trajectories, rewards, advantages, policy_version)
            update_start = time.perf_counter()
            metrics = ray.get(learner.update.remote(ray.put(batch)), timeout=cfg.rpc_timeout)
            observations = metrics.pop("recipe_observations", [])
            for observation in observations:
                append_json(directory / "recipe_metrics.jsonl", observation)
            train_seconds = time.perf_counter() - update_start
            sync_start = time.perf_counter()
            policy_version = publish()
            if policy_version.weight_version != step or metrics["weight_version"] != step:
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
                    "schema": 2,
                    "step": step,
                    "trajectories": [asdict(t) for t in batch.trajectories],
                    "rewards": batch.rewards.tolist(),
                    "advantages": batch.advantages.tolist(),
                },
            )
            if step % cfg.checkpoint_every == 0 or step == cfg.steps:
                last_checkpoint = save_checkpoint(directory, learner, cfg, step, policy_version)
            print(
                f"step={step} reward={metrics['reward_mean']:.3f} "
                f"ratio={metrics['ratio_mean']:.4f} version={policy_version.weight_version}",
                flush=True,
            )
        summary = {
            "schema": 2,
            "status": "completed",
            "backend": f"{cfg.learner_backend}/{cfg.rollout_backend}",
            "first_step": first_step,
            "last_step": cfg.steps,
            "final_metrics": metrics,
            "checkpoint": last_checkpoint,
            "seconds": time.perf_counter() - started,
            "native_gpu_executed": not cfg.is_cpu_test,
            "native_gpu_acceptance": "not_evaluated" if not cfg.is_cpu_test else "not_applicable",
            "execution": ray.get(
                [worker.execution_report.remote() for worker in worker_handles], timeout=cfg.rpc_timeout
            ),
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
        for worker in worker_handles:
            try:
                ray.get(worker.close.remote(), timeout=10)
            except Exception:
                pass
            finally:
                ray.kill(worker, no_restart=True)
        if resource_group is not None:
            remove_placement_group(resource_group)
        if owned_ray:
            ray.shutdown()
