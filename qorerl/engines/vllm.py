"""vLLM rollout adapter; synchronous requests and acknowledged BF16 refits."""

import torch

from qorerl.engines.base import validate_install
from qorerl.training.contracts import Trajectory


class WeightLoaderExtension:
    """Runs inside vLLM's worker, where the executable model actually lives."""

    def qorerl_load_weights(self, tensors):
        model = self.model_runner.model
        loaded = model.load_weights((name, value.to("cuda")) for name, value in tensors.items())
        if not loaded:
            raise RuntimeError("vLLM did not report any loaded parameters")
        # HF names can fuse into qkv/gate_up names. Check coverage against the
        # executable model's names, not by comparing HF and packed names directly.
        expected = set(dict(model.named_parameters()))
        missing = expected - set(loaded)
        if missing:
            raise RuntimeError(f"Incomplete vLLM refit: {sorted(missing)}")
        torch.cuda.synchronize()
        return len(loaded)


class VLLMRollout:
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM requires a Ray-assigned NVIDIA GPU; no CPU fallback")
        from vllm import LLM

        self.cfg, self.policy = cfg, None
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
            worker_extension_cls="qorerl.engines.vllm.WeightLoaderExtension",
        )

    def install_weights(self, snapshot):
        validate_install(snapshot, self.policy, self.cfg)
        # generate() is blocking; no in-flight requests exist at publication.
        # Prefix caching and CUDA graphs are disabled in this initial refit path.
        self.policy = None
        self.llm.collective_rpc("qorerl_load_weights", args=(snapshot.tensors,), timeout=self.cfg.rpc_timeout)
        self.policy = snapshot.policy
        return self.policy

    def generate(self, prompts, policy, seed):
        if self.policy != policy:
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
                        policy,
                    )
                )
        return result

    def close(self):
        # Ray terminates this dedicated actor (and its single-process executor).
        del self.llm
