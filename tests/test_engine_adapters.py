"""CPU contract tests for adapter glue; these do not validate native GPU execution."""

import sys
from types import SimpleNamespace

import pytest
import torch

from qorerl.engines.megatron import MegatronLearner
from qorerl.engines.vllm import VLLMRollout, WeightLoaderExtension
from qorerl.training.config import TrainConfig
from qorerl.training.contracts import PolicyVersion, Prompt, WeightSnapshot


@pytest.mark.parametrize("placement,budget", [("separate", 0.7), ("collocate", 0.2)])
def test_vllm_receives_resolved_memory_budget(monkeypatch, placement, budget):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **kw: calls.append(kw)))
    VLLMRollout(TrainConfig(recipe_id="bf16-baseline-v1", worker_placement=placement))
    assert calls[0]["gpu_memory_utilization"] == budget
    assert calls[0]["distributed_executor_backend"] == "uni"
    assert calls[0]["enforce_eager"] is True


def test_megatron_output_keeps_batch_sequence_and_removes_vocab_padding():
    learner = MegatronLearner.__new__(MegatronLearner)
    learner.vocab_size = 10
    expected = torch.randn(2, 5, 16)

    def model(**kwargs):
        assert kwargs["runtime_gather_output"]
        assert kwargs["attention_mask"] is None
        assert kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 4]] * 2
        return expected

    learner.model = model
    torch.testing.assert_close(learner.logits(torch.ones(2, 5, dtype=torch.long)), expected[..., :10])


def test_vllm_publication_failure_does_not_acknowledge_version():
    rollout = VLLMRollout.__new__(VLLMRollout)
    rollout.cfg = TrainConfig(recipe_id="bf16-baseline-v1")
    rollout.policy_version = PolicyVersion(0, rollout.cfg.recipe_id)

    def fail(*args, **kwargs):
        raise RuntimeError("injected incomplete refit")

    rollout.llm = SimpleNamespace(collective_rpc=fail)
    snapshot = WeightSnapshot(
        PolicyVersion(1, rollout.cfg.recipe_id), rollout.cfg.model, {"w": torch.ones(2)}
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        rollout.install_weights(snapshot)
    assert rollout.policy_version is None


def test_vllm_worker_checks_fused_model_parameter_coverage(monkeypatch):
    extension = WeightLoaderExtension()
    model = torch.nn.Linear(2, 2, bias=False)
    # No real CUDA transfer here: test the coverage check after a loader returns.
    model.load_weights = lambda stream: {"other.weight"}
    extension.model_runner = SimpleNamespace(model=model)
    with pytest.raises(RuntimeError, match="Incomplete"):
        extension.qorerl_load_weights({"hf.weight": torch.ones(2, 2)})
    model.load_weights = lambda stream: {"weight"}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    assert extension.qorerl_load_weights({"hf.weight": torch.ones(2, 2)}) == 1


def test_vllm_uses_selected_token_logprob_and_stores_eos(monkeypatch):
    rollout = VLLMRollout.__new__(VLLMRollout)
    rollout.cfg = TrainConfig(group_size=2)
    rollout.policy_version = PolicyVersion(0, rollout.cfg.recipe_id)
    prompt = Prompt(4, (1, 3), "2")

    def generate(prompts, params, **kwargs):
        assert params[0].temperature == 1 and params[0].top_p == 1 and params[0].top_k == -1
        assert params[0].n == 2 and params[0].logprobs == 0
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 3],
                outputs=[
                    SimpleNamespace(
                        index=i,
                        token_ids=[6, 2],
                        text="2",
                        finish_reason="stop",
                        logprobs=[{6: SimpleNamespace(logprob=-0.2)}, {2: SimpleNamespace(logprob=-0.3)}],
                    )
                    for i in range(2)
                ],
            )
        ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))
    rollout.llm = SimpleNamespace(generate=generate)
    results = rollout.generate([prompt], rollout.policy_version, 42)
    assert [r.sample_id for r in results] == [0, 1]
    assert results[0].response_ids == (6, 2) and results[0].behavior_log_probs == (-0.2, -0.3)
