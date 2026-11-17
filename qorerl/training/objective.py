"""GRPO: group-standardized advantages and response-sequence-mean PPO surrogate.

Uses saved rollout log probabilities as the behavior denominator. beta=0 (no
reference KL). Sampling is fixed to temperature=1, top_p=1, no top_k filtering.
"""

import torch
from torch.nn import functional as F


def grpo_loss(logits, batch, clip_epsilon):
    if logits.shape[:2] != batch.labels.shape:
        raise ValueError("Engine must return [batch, sequence, vocabulary] logits")
    mask = batch.response_mask
    # Normalize over the real vocabulary, not a Megatron padded vocabulary.
    log_probs = -F.cross_entropy(logits.float().transpose(1, 2), batch.labels, reduction="none")
    delta = torch.where(mask, log_probs - batch.behavior_log_probs, 0)
    ratio = delta.exp()
    if not torch.isfinite(log_probs[mask]).all() or not torch.isfinite(ratio[mask]).all():
        raise FloatingPointError("Non-finite response log probability/importance ratio")
    advantages = batch.advantages[:, None]
    raw = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    terms = -torch.minimum(raw, clipped)
    lengths = mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("Empty response mask")
    loss = (torch.where(mask, terms, 0).sum(-1) / lengths).mean()
    metrics = {
        "loss": loss.detach().item(),
        "ratio_sum": ratio.detach()[mask].sum().item(),
        "abs_log_ratio_sum": delta.detach()[mask].abs().sum().item(),
        "clipped_tokens": ((raw > clipped) & mask).sum().item(),
        "tokens": lengths.sum().item(),
    }
    return loss, metrics
