"""CPU integration fixture. Real sampling/backprop, NOT Megatron or vLLM."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from qorerl.engines.base import validate_install
from qorerl.engines.single_rank_learner import SingleRankGRPOLearner, cpu_tree
from qorerl.recipes.state import RecipeState, down_proj_name
from qorerl.training.contracts import Trajectory, WeightSnapshot


def _emulated_mxfp8_operand(value):
    """Rowwise E4M3 rounding with a shared power-of-two scale for each group of 32."""
    with torch.no_grad():
        blocks = value.float().reshape(-1, 32)
        maximum = blocks.abs().amax(dim=-1, keepdim=True)
        exponent = (torch.floor(maximum.log2()) - 8).clamp(-127, 127)
        scale = torch.exp2(exponent)
        rounded = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        rounded = (rounded * scale).reshape_as(value).to(value.dtype)
    # An explicit straight-through derivative exercises canonical optimizer updates.
    return rounded + (value - value.detach())


def _emulated_mxfp8_linear(x, weight):
    return F.linear(_emulated_mxfp8_operand(x), _emulated_mxfp8_operand(weight))


class TestProjection(nn.Module):
    """Known per-channel imbalance makes held-out selection measurable on the CPU fixture."""

    def __init__(self, index):
        super().__init__()
        self.down_proj = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        gain = torch.ones(32, dtype=torch.bfloat16)
        gain[index] = (65536, 2048, 1)[index]
        self.register_buffer("gain", gain, persistent=False)
        with torch.no_grad():
            self.down_proj.weight.div_(gain)

    def forward(self, x, name, installation):
        x = x.to(torch.bfloat16) * self.gain
        if installation.calibrator is not None and name in installation.weights:
            installation.calibrator.capture(name, x)
        state = installation.state
        if not state.is_active(name):
            return self.down_proj(x)
        return _emulated_mxfp8_linear(
            state.transform_input(x, name), state.transform_weight(self.down_proj.weight, name)
        )


class TestLM(nn.Module):
    def __init__(self, recipe=None):
        super().__init__()
        self.embedding = nn.Embedding(14, 32)
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, 14)
        self.installation = None
        if recipe is not None and recipe.uses_smoothquant:
            # Preserve the canonical HF names used by the production payload.
            self.model = nn.Module()
            self.model.layers = nn.ModuleList()
            state, weights = RecipeState(recipe), {}
            selected = recipe.select_layers(3)
            eligible = recipe.eligible_layers(3)
            for index in range(3):
                layer = nn.Module()
                layer.mlp = TestProjection(index)
                self.model.layers.append(layer)
                if index in eligible:
                    weight = layer.mlp.down_proj.weight
                    weights[state.add_target(index, weight, active=index in selected)] = weight
            self.installation = SimpleNamespace(state=state, weights=weights, calibrator=None)

    def forward(self, ids):
        x = self.rnn(self.embedding(ids))[0]
        if self.installation is not None:
            for index, layer in enumerate(self.model.layers):
                x = x + layer.mlp(x, down_proj_name(index), self.installation).float().tanh()
        return self.head(x)

    def execution_report(self):
        report = {"backend": "torch_test"}
        if self.installation is not None:
            state = self.installation.state
            report.update(
                identity=state.active_identity,
                scale_digest=state.scale_digest,
                targets={name: shape for name, shape in state.targets.items() if state.is_active(name)},
                fprop="cpu_emulated_rowwise_block32_e4m3_bf16_linear",
                backward="straight_through_fixture",
            )
            if state.spec.allow_layer_expansion:
                report["eligible_targets"] = sorted(state.targets)
        return report


class TestLearner(SingleRankGRPOLearner):
    calibration_fprop = staticmethod(_emulated_mxfp8_linear)

    def __init__(self, cfg):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        model = TestLM(cfg.recipe)
        self.recipe_installation = model.installation
        if self.recipe_installation is not None:
            self.recipe_state = self.recipe_installation.state
        self.setup_optimizer(cfg, model, 0, "cpu")

    def logits(self, input_ids):
        return self.model(input_ids)

    def export_weights(self):
        tensors = cpu_tree(self.model.state_dict())
        state = None
        if self.recipe_installation is not None:
            if self.policy_version.recipe_id != self.recipe_state.active_identity:
                raise ValueError("Export policy does not match active SmoothQuant scales")
            self.recipe_state.validate_weights(tensors)
            state = self.recipe_state.to_dict()
        return WeightSnapshot(self.policy_version, self.cfg.model, tensors, state)

    def execution_report(self):
        return self.model.execution_report()


class TestRollout:
    def __init__(self, cfg):
        torch.set_num_threads(1)
        self.cfg, self.model, self.policy_version = cfg, TestLM(cfg.recipe).eval(), None

    def install_weights(self, snapshot):
        validate_install(snapshot, self.policy_version, self.cfg)
        installation = self.model.installation
        if installation is not None:
            installation.state.validate(snapshot.recipe_state)
            installation.state.validate_weights(snapshot.tensors)
        elif snapshot.recipe_state is not None:
            raise ValueError("CPU baseline cannot install a transformed recipe")
        self.policy_version = None
        self.model.load_state_dict(snapshot.tensors, strict=True)
        if installation is not None:
            installation.state.load(snapshot.recipe_state)
        self.policy_version = snapshot.policy_version
        return self.policy_version

    def execution_report(self):
        return self.model.execution_report()

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
