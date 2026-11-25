"""Ray resource placement, independent of the training and recipe algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerPlacement:
    """Separate devices, or two resident engines sharing one GPU sequentially.

    Fractional GPU requests are scheduling tokens, not memory partitions. A
    single bundle pins the collocated workers to the same node and GPU.
    CPU test engines exercise the same grouping without requesting a GPU.
    """

    mode: str
    worker_count: int
    uses_gpu: bool

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.worker_placement, cfg.rollout_workers + 1, not cfg.is_cpu_test)

    @property
    def collocate(self):
        return self.mode == "collocate"

    @property
    def bundles(self):
        if self.collocate:
            return [{"CPU": self.worker_count, **({"GPU": 1} if self.uses_gpu else {})}]
        return [{"CPU": 1, **({"GPU": 1} if self.uses_gpu else {})} for _ in range(self.worker_count)]

    @property
    def gpus_per_worker(self):
        if not self.uses_gpu:
            return 0
        return 0.5 if self.collocate else 1

    def bundle_index(self, worker_index):
        return 0 if self.collocate else worker_index

    def check_resources(self, resources):
        required_gpus = sum(bundle.get("GPU", 0) for bundle in self.bundles)
        if resources.get("CPU", 0) < self.worker_count or resources.get("GPU", 0) < required_gpus:
            raise RuntimeError(
                f"{self.mode} placement needs {self.worker_count} CPUs and {required_gpus} GPUs; "
                f"found {resources}"
            )

    def check_workers(self, reports):
        if not self.collocate:
            return
        if len(reports) != self.worker_count or len({r["node"] for r in reports}) != 1:
            raise RuntimeError("Collocated workers must run on the same Ray node")
        if self.uses_gpu:
            assignments = [tuple(r["gpu_ids"]) for r in reports]
            if any(len(ids) != 1 for ids in assignments) or len(set(assignments)) != 1:
                raise RuntimeError("Collocated workers must be assigned the same single GPU")
