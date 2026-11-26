"""vLLM rollout adapter; canonical-weight refits and explicit recipe rebuilds."""

import torch

from qorerl.engines.base import validate_install
from qorerl.training.contracts import Trajectory


class WeightLoaderExtension:
    """Runs inside vLLM's worker, where the executable model actually lives."""

    def qorerl_load_weights(self, tensors, recipe_state=None, expected_recipe_id=None):
        model = self.model_runner.model
        installation = getattr(self, "qorerl_recipe_installation", None)
        if installation is not None:
            incoming_identity = installation.state.validate(recipe_state)
            if expected_recipe_id is not None or installation.state.spec.uses_smoothquant:
                if expected_recipe_id != incoming_identity:
                    raise ValueError("Published policy identity disagrees with the worker's recipe state")
            installation.state.validate_weights(tensors)
            installation.invalidate()
        elif recipe_state is not None:
            raise ValueError("Recipe weights cannot be installed on a BF16 worker")
        try:
            loaded = model.load_weights((name, value.to("cuda")) for name, value in tensors.items())
            if not loaded:
                raise RuntimeError("vLLM did not report any loaded parameters")
            # HF names can fuse into qkv/gate_up names. Check coverage against the
            # executable model's names, not by comparing HF and packed names directly.
            expected = set(dict(model.named_parameters()))
            missing = expected - set(loaded)
            if missing:
                raise RuntimeError(f"Incomplete vLLM refit: {sorted(missing)}")
            if installation is not None:
                if installation.state.spec.uses_smoothquant:
                    installation.state.install_smoothing_factors(recipe_state["scales"])
                installation.refresh()
            torch.cuda.synchronize()
        except BaseException:
            # refresh() publishes its carriers before asynchronous CUDA work is
            # synchronized. A later failure must also disable those new caches.
            if installation is not None:
                installation.invalidate()
            raise
        if installation is not None and installation.state.spec.uses_smoothquant:
            return installation.state.active_identity
        return len(loaded)

    def qorerl_execution_report(self):
        from qorerl.validation import environment_report

        installation = getattr(self, "qorerl_recipe_installation", None)
        report = installation.report() if installation is not None else {"precision": "bf16"}
        report["environment"] = environment_report()
        return report


class VLLMRollout:
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM requires a Ray-assigned NVIDIA GPU; no CPU fallback")
        from vllm import LLM

        self.cfg, self.policy_version = cfg, None
        worker_options = {"worker_extension_cls": "qorerl.engines.vllm.WeightLoaderExtension"}
        if cfg.recipe.is_low_precision:
            worker_options = {
                "worker_cls": "qorerl.engines.vllm_worker.QoreRLRecipeWorker",
                "additional_config": {
                    "qorerl_recipe": {"name": cfg.recipe_id, "options": cfg.recipe_options}
                },
            }
        self.llm = LLM(
            model=cfg.model,
            revision=cfg.model_revision,
            tokenizer_revision=cfg.model_revision,
            dtype="bfloat16",
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_prefix_caching=False,
            enforce_eager=True,
            generation_config="vllm",  # Do not inherit HF sampling penalties/filters.
            seed=cfg.seed,
            **worker_options,
        )

    def install_weights(self, snapshot):
        validate_install(snapshot, self.policy_version, self.cfg)
        # generate() is blocking; no in-flight requests exist at publication.
        # Prefix caching and CUDA graphs are disabled in this initial refit path.
        self.policy_version = None
        acknowledgements = self.llm.collective_rpc(
            "qorerl_load_weights",
            args=(snapshot.tensors, snapshot.recipe_state, snapshot.policy_version.recipe_id),
            timeout=self.cfg.rpc_timeout,
        )
        if self.cfg.recipe.uses_smoothquant and acknowledgements != [snapshot.policy_version.recipe_id]:
            raise RuntimeError("vLLM worker failed to acknowledge the active SmoothQuant transform identity")
        self.policy_version = snapshot.policy_version
        return self.policy_version

    def execution_report(self):
        return {
            "identity": self.policy_version.recipe_id if self.policy_version is not None else None,
            "workers": self.llm.collective_rpc("qorerl_execution_report", timeout=self.cfg.rpc_timeout),
        }

    def generate(self, prompts, policy_version, seed):
        if self.policy_version != policy_version:
            raise ValueError("Rollout must acknowledge the exact requested policy first")
        from vllm import SamplingParams

        params = [
            SamplingParams(
                n=self.cfg.group_size,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=self.cfg.max_new_tokens,
                logprobs=0,
                seed=seed + p.group_id,
            )
            for p in prompts
        ]
        requests = self.llm.generate(
            [{"prompt_token_ids": list(p.token_ids)} for p in prompts], params, use_tqdm=False
        )
        if len(requests) != len(prompts):
            raise RuntimeError("vLLM returned an incomplete prompt batch")
        result = []
        for prompt, request in zip(prompts, requests, strict=True):
            if tuple(request.prompt_token_ids) != prompt.token_ids:
                raise RuntimeError("vLLM output prompt identity mismatch")
            for output in request.outputs:
                if output.logprobs is None or len(output.logprobs) != len(output.token_ids):
                    raise RuntimeError("vLLM omitted sampled token log probabilities")
                result.append(
                    Trajectory(
                        prompt.group_id,
                        output.index,
                        prompt.token_ids,
                        tuple(output.token_ids),
                        tuple(
                            lp[token].logprob
                            for lp, token in zip(output.logprobs, output.token_ids, strict=True)
                        ),
                        output.text,
                        output.finish_reason,
                        policy_version,
                    )
                )
        return result

    def close(self):
        # Ray terminates this dedicated actor (and its single-process executor).
        del self.llm
