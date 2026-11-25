"""Real Ray placement with logical GPUs; no CUDA or native execution is claimed."""

import os

import pytest
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from qorerl.runtime.placement import WorkerPlacement
from qorerl.training.config import TrainConfig


@ray.remote
class PlacementProbe:
    def report(self):
        context = ray.get_runtime_context()
        return {
            "node": context.get_node_id(),
            "gpu_ids": context.get_accelerator_ids().get("GPU", []),
            "pid": os.getpid(),
        }


@pytest.fixture(scope="module")
def logical_gpu_cluster():
    # Ray GPU resources are logical. These actors inspect scheduling only and
    # never initialize CUDA, so this also runs on a CPU development machine.
    ray.init(num_cpus=2, num_gpus=2, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


@pytest.mark.parametrize("mode", ["separate", "collocate"])
def test_real_ray_assigns_the_requested_gpu_topology(logical_gpu_cluster, mode):
    plan = WorkerPlacement.from_config(TrainConfig(worker_placement=mode))
    group = placement_group(plan.bundles, strategy="PACK")
    workers = []
    try:
        ray.get(group.ready(), timeout=30)
        for index in range(2):
            workers.append(
                PlacementProbe.options(
                    num_cpus=1,
                    num_gpus=plan.gpus_per_worker,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=group, placement_group_bundle_index=plan.bundle_index(index)
                    ),
                ).remote()
            )
        reports = ray.get([worker.report.remote() for worker in workers], timeout=30)
        assert reports[0]["pid"] != reports[1]["pid"]
        assert all(len(report["gpu_ids"]) == 1 for report in reports)
        assert (reports[0]["gpu_ids"] == reports[1]["gpu_ids"]) == (mode == "collocate")
        plan.check_workers(reports)
    finally:
        for worker in workers:
            ray.kill(worker)
        remove_placement_group(group)


def test_one_gpu_satisfies_only_collocated_native_placement():
    resources = {"CPU": 2, "GPU": 1}
    WorkerPlacement.from_config(TrainConfig(worker_placement="collocate")).check_resources(resources)
    with pytest.raises(RuntimeError, match="2 GPUs"):
        WorkerPlacement.from_config(TrainConfig()).check_resources(resources)
    with pytest.raises(RuntimeError, match="2 CPUs"):
        WorkerPlacement.from_config(TrainConfig(worker_placement="collocate")).check_resources({"GPU": 1})


@pytest.mark.parametrize(
    "other", [{"node": "b", "gpu_ids": ["0"]}, {"node": "a", "gpu_ids": ["1"]}, {"node": "a", "gpu_ids": []}]
)
def test_collocate_rejects_wrong_node_or_gpu(other):
    plan = WorkerPlacement.from_config(TrainConfig(worker_placement="collocate"))
    with pytest.raises(RuntimeError, match="Collocated workers"):
        plan.check_workers([{"node": "a", "gpu_ids": ["0"]}, other])


def test_placement_resolves_and_persists_the_vllm_memory_budget():
    assert TrainConfig().gpu_memory_utilization == 0.7
    cfg = TrainConfig(worker_placement="collocate")
    assert cfg.gpu_memory_utilization == 0.2
    assert TrainConfig(**cfg.to_dict()) == cfg
    assert TrainConfig(worker_placement="collocate", gpu_memory_utilization=0.3).gpu_memory_utilization == 0.3


@pytest.mark.parametrize(
    "options",
    [
        {"worker_placement": "unknown"},
        {"worker_placement": "collocate", "rollout_workers": 2},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 1},
        {"gpu_memory_utilization": True},
        {"gpu_memory_utilization": float("nan")},
    ],
)
def test_invalid_placement_and_memory_settings_fail_early(options):
    with pytest.raises(ValueError):
        TrainConfig(**options)
