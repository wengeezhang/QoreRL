"""Native vLLM worker loaded only inside a recipe-enabled GPU engine.

Install before vLLM profiles memory and allocates its KV cache. vLLM 0.32
measures persistent consumption from the change in free device memory, so the
derived weights and cached GPU signs are counted without adjusting its earlier
model_memory_usage snapshot. The inference profile measures activation scratch;
refit scratch, overlapping caches and carriers for subsequently promoted layers
still require headroom.
"""

from vllm.v1.worker.gpu_worker import Worker

from qorerl.engines.vllm import WeightLoaderExtension
from qorerl.recipes.spec import resolve_recipe
from qorerl.recipes.vllm_mxfp8 import install


class QoreRLRecipeWorker(WeightLoaderExtension, Worker):
    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise ValueError("QoreRL recipes require canonical HF weights, not dummy weights")
        super().load_model(load_dummy_weights=load_dummy_weights)
        config = self.vllm_config.additional_config["qorerl_recipe"]
        recipe = resolve_recipe(config["name"], config["options"])
        self.qorerl_recipe_installation = install(self.model_runner.model, recipe)
