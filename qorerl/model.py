"""A small causal Transformer with genuinely sparse top-2 SwiGLU experts."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from qorerl.precision import PrecisionLinear, bf16_values


@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden: int = 32
    intermediate: int = 64
    layers: int = 2
    heads: int = 4
    experts: int = 4
    top_k: int = 2
    max_length: int = 16

    def __post_init__(self):
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if not 1 <= self.top_k <= self.experts:
            raise ValueError("top_k must be in [1, experts]")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return bf16_values(normalized * bf16_values(self.weight))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.qkv = PrecisionLinear(cfg.hidden, 3 * cfg.hidden)
        self.out = PrecisionLinear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)]
        scores = (q @ k.transpose(-1, -2)) * (d // self.heads) ** -0.5
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        probs = scores.masked_fill(mask, -torch.inf).softmax(-1)
        y = bf16_values(probs @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y)


class Expert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up = PrecisionLinear(cfg.hidden, 2 * cfg.intermediate)
        self.down = PrecisionLinear(cfg.intermediate, cfg.hidden)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(bf16_values(F.silu(gate) * up))


class SparseMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.router = PrecisionLinear(cfg.hidden, cfg.experts, quantizable=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.experts)])
        self.top_k = cfg.top_k
        self.last_counts = []

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        scores, indices = self.router(flat).topk(self.top_k, dim=-1)
        mixing = scores.softmax(-1)
        result = torch.zeros_like(flat)
        self.last_counts = []
        for i, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == i)
            self.last_counts.append(tokens.numel())
            if tokens.numel():
                values = expert(flat[tokens]) * mixing[tokens, slots, None]
                result = result.index_add(0, tokens, values)
        return bf16_values(result.reshape_as(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden)
        self.moe = SparseMoE(cfg)

    def forward(self, x):
        x = bf16_values(x + self.attn(self.norm1(x)))
        return bf16_values(x + self.moe(self.norm2(x)))


class TinyMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.position = nn.Embedding(cfg.max_length, cfg.hidden)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.hidden)
        self.lm_head = PrecisionLinear(cfg.hidden, cfg.vocab_size, quantizable=False)

    def forward(self, tokens):
        if tokens.shape[1] > self.cfg.max_length:
            raise ValueError("Sequence exceeds max_length")
        position = torch.arange(tokens.shape[1], device=tokens.device)
        x = bf16_values(self.embedding(tokens) + self.position(position))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def expert_counts(self):
        return {f"blocks.{i}.moe": block.moe.last_counts for i, block in enumerate(self.blocks)}
