"""On-policy group-relative RL on a two-token copy task (no external model)."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class RolloutBatch:
    tokens: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    prompt_length: int = 4


def prompts(count, generator, device="cpu", vocab_size=16):
    # CPU RNG is separate from the sampling RNG and shared across experiment arms.
    content = torch.randint(4, vocab_size, (count, 2), generator=generator)
    prefix = torch.cat(
        [torch.ones(count, 1, dtype=torch.long), content, torch.full((count, 1), 2, dtype=torch.long)], dim=1
    )
    return prefix.to(device), content.to(device)


def group_advantages(rewards, group_size):
    grouped = rewards.reshape(-1, group_size)
    return (
        (grouped - grouped.mean(-1, keepdim=True)) / (grouped.std(-1, keepdim=True, unbiased=False) + 1e-6)
    ).reshape(-1)


@torch.no_grad()
def rollout(model, prefix, target, group_size, generator):
    tokens = prefix.repeat_interleave(group_size, dim=0)
    old = []
    for _ in range(2):
        log_probs = model(tokens)[:, -1].log_softmax(-1)
        chosen = torch.multinomial(log_probs.exp().cpu(), 1, generator=generator).to(tokens.device)
        old.append(log_probs.gather(-1, chosen).squeeze(-1))
        tokens = torch.cat([tokens, chosen], dim=1)
    rewards = (tokens[:, -2:] == target.repeat_interleave(group_size, dim=0)).float().mean(-1)
    return RolloutBatch(tokens, torch.stack(old, dim=-1), group_advantages(rewards, group_size), rewards)


def policy_loss(model, batch, clip=0.2):
    logits = model(batch.tokens[:, :-1])[:, batch.prompt_length - 1 :]
    chosen = batch.tokens[:, batch.prompt_length :]
    log_probs = logits.log_softmax(-1).gather(-1, chosen[..., None]).squeeze(-1)
    ratio = (log_probs - batch.old_log_probs).exp()
    adv = batch.advantages[:, None]
    loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    if not torch.isfinite(loss) or not torch.isfinite(log_probs).all():
        raise FloatingPointError("Non-finite policy loss/log probabilities")
    return (
        loss,
        {
            "loss": loss.detach().item(),
            "ratio_mean": ratio.detach().mean().item(),
            "clip_fraction": ((ratio.detach() - 1).abs() > clip).float().mean().item(),
            "entropy": (-(logits.log_softmax(-1) * logits.softmax(-1)).sum(-1)).detach().mean().item(),
        },
        logits,
    )


def supervised_loss(model, prefix, target):
    tokens = torch.cat([prefix, target], dim=1)
    logits = model(tokens[:, :-1])[:, prefix.shape[1] - 1 :]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


@torch.no_grad()
def evaluate(model, prefix, target):
    tokens = prefix
    for _ in range(2):
        chosen = model(tokens)[:, -1].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, chosen], dim=-1)
    correct = tokens[:, -2:] == target
    return {
        "token_accuracy": correct.float().mean().item(),
        "exact_match": correct.all(-1).float().mean().item(),
    }
