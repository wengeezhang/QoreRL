"""CPU integration fixture. Real sampling/backprop, NOT Megatron or vLLM."""

import torch
from torch import nn

from qorerl.engines.base import validate_install
from qorerl.engines.single_rank_learner import SingleRankGRPOLearner, cpu_tree
from qorerl.training.contracts import Trajectory, WeightSnapshot


class TestLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(14, 32)
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, 14)

    def forward(self, ids):
        return self.head(self.rnn(self.embedding(ids))[0])


class TestLearner(SingleRankGRPOLearner):
    def __init__(self, cfg):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.setup_optimizer(cfg, TestLM(), 0, "cpu")

    def logits(self, input_ids):
        return self.model(input_ids)

    def export_weights(self):
        return WeightSnapshot(self.policy_version, self.cfg.model, cpu_tree(self.model.state_dict()))


class TestRollout:
    def __init__(self, cfg):
        torch.set_num_threads(1)
        self.cfg, self.model, self.policy_version = cfg, TestLM().eval(), None

    def install_weights(self, snapshot):
        validate_install(snapshot, self.policy_version, self.cfg)
        self.model.load_state_dict(snapshot.tensors, strict=True)
        self.policy_version = snapshot.policy_version
        return self.policy_version

    @torch.no_grad()
    def generate(self, prompts, policy_version, seed):
        if self.policy_version != policy_version:
            raise ValueError("Rollout must acknowledge the exact requested policy first")
        result = []
        for prompt in prompts:
            for sample_id in range(self.cfg.group_size):
                rng = torch.Generator().manual_seed(seed + prompt.group_id * self.cfg.group_size + sample_id)
                tokens = list(prompt.token_ids)
                response, behavior_log_probs = [], []
                for _ in range(self.cfg.max_new_tokens):
                    probs = self.model(torch.tensor([tokens]))[0, -1].log_softmax(-1)
                    token = torch.multinomial(probs.exp(), 1, generator=rng).item()
                    response.append(token)
                    behavior_log_probs.append(probs[token].item())
                    tokens.append(token)
                    if token == 2:
                        break
                text = " ".join(str(t - 4) if t >= 4 else "?" for t in response if t != 2)
                result.append(
                    Trajectory(
                        prompt.group_id,
                        sample_id,
                        prompt.token_ids,
                        tuple(response),
                        tuple(behavior_log_probs),
                        text,
                        "stop" if response[-1] == 2 else "length",
                        policy_version,
                    )
                )
        return result

    def close(self):
        pass
