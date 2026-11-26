"""Single-GPU Megatron Core learner via NVIDIA's model-conversion Bridge.

API target: Megatron Bridge 0.6.2 / Core 0.19.2. Native validation is pending.
No verl code or custom HF-to-Megatron weight mapping is embedded here.
"""

import os
import tempfile
from datetime import timedelta

import torch

from qorerl.engines.single_rank_learner import SingleRankGRPOLearner
from qorerl.recipes.state import down_proj_name
from qorerl.training.contracts import WeightSnapshot


class MegatronLearner(SingleRankGRPOLearner):
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError("Megatron requires a Ray-assigned NVIDIA GPU; no CPU fallback")
        from megatron.bridge import AutoBridge
        from transformers import AutoConfig, AutoTokenizer

        torch.cuda.set_device(0)  # Ray exposes this actor's assigned GPU as cuda:0.
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        hf_config = AutoConfig.from_pretrained(cfg.model, revision=cfg.model_revision)
        if hf_config.model_type != "qwen3" or getattr(hf_config, "quantization_config", None):
            raise ValueError("First native baseline supports unquantized dense Qwen3 only")
        self.vocab_size = hf_config.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # Each single-rank learner has a private rendezvous, including concurrent runs.
        self.rendezvous = tempfile.TemporaryDirectory(prefix="qorerl-mcore-")
        os.environ["LOCAL_RANK"] = "0"
        torch.distributed.init_process_group(
            "nccl",
            init_method=f"file://{self.rendezvous.name}/store",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=cfg.rpc_timeout),
        )
        self.bridge = AutoBridge.from_hf_pretrained(
            cfg.model, revision=cfg.model_revision, trust_remote_code=False
        )
        provider = self.bridge.to_megatron_provider(load_weights=False)
        provider.apply_overrides_and_finalize(
            dtype=torch.bfloat16,
            overrides={
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "sequence_parallel": False,
                "gradient_accumulation_fusion": False,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "perform_initialization": True,
            },
        )
        self.modules = provider.provide_distributed_model(wrap_with_ddp=False)
        if len(self.modules) != 1:
            raise ValueError("This adapter implements one Megatron rank, not pipeline parallel training")
        self.bridge.load_hf_weights(self.modules)
        self.recipe_installation = None
        if cfg.recipe.is_low_precision:
            from qorerl.recipes.megatron_mxfp8 import install

            self.recipe_installation = install(self.modules[0], cfg.recipe, hf_config.num_hidden_layers)
            self.recipe_state = self.recipe_installation.state
        self.setup_optimizer(cfg, self.modules[0], pad_id, "cuda")
        if cfg.observation.enabled:
            from qorerl.observability.probe import RecipeProbe

            self.recipe_probe = RecipeProbe(
                cfg.observation, cfg.recipe, self.recipe_state.targets, hf_config.num_hidden_layers
            )
            self.recipe_installation.observer = self.recipe_probe

    def logits(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        # Right padding cannot affect preceding real tokens under a causal mask.
        logits = self.model(
            input_ids=input_ids, position_ids=positions, attention_mask=None, runtime_gather_output=True
        )
        return logits[..., : self.vocab_size]

    def _discovery_projections(self):
        from megatron.core.transformer.module import Float16Module

        model = self.model.module if isinstance(self.model, Float16Module) else self.model
        layers = model.decoder.layers
        if not len(layers):
            raise ValueError("BF16 discovery requires dense decoder MLP down projections")
        return {down_proj_name(i): layer.mlp.linear_fc2 for i, layer in enumerate(layers)}

    def export_weights(self):
        tensors = {
            name: value.detach().cpu().clone()
            for name, value in self.bridge.export_hf_weights(self.modules, cpu=True, show_progress=False)
        }
        if self.recipe_installation is not None:
            if self.policy_version.recipe_id != self.recipe_state.active_identity:
                raise ValueError("Learner policy version disagrees with its active recipe state")
            self.recipe_state.validate_weights(tensors)
        snapshot = WeightSnapshot(
            self.policy_version,
            self.cfg.model,
            tensors,
            self.recipe_state.to_dict() if self.recipe_installation is not None else None,
        )
        snapshot.validate()
        return snapshot

    def execution_report(self):
        from qorerl.validation import environment_report

        if self.recipe_installation is not None:
            report = self.recipe_installation.report()
            if self.cfg.observation.enabled:
                report["recipe_observation"] = {
                    "enabled": True,
                    "artifact": "recipe_metrics.jsonl",
                    "phase": "learner.fprop",
                    "interpretation": "local paired output error against BF16; not training quality",
                }
        else:
            report = {"identity": self.cfg.policy_recipe_id, "precision": "bf16", "validation": "pending"}
        report["environment"] = environment_report()
        return report

    def close(self):
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self.rendezvous.cleanup()
