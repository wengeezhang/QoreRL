"""Device-side microbatch construction: right padding, not sequence packing."""

from dataclasses import dataclass

import torch


@dataclass
class PaddedMicrobatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    behavior_log_probs: torch.Tensor
    advantages: torch.Tensor


def collate_microbatch(batch, indices, pad_id, device):
    """Right padding, causal shift, and response-only mask (EOS included)."""
    items = [batch.trajectories[i] for i in indices]
    if not items:
        raise ValueError("Cannot collate an empty microbatch")
    width = max(len(t.prompt_ids) + len(t.response_ids) - 1 for t in items)
    tokens = torch.full((len(items), width + 1), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(items), width), dtype=torch.bool, device=device)
    behavior_log_probs = torch.zeros((len(items), width), dtype=torch.float32, device=device)
    for row, item in enumerate(items):
        seq = item.prompt_ids + item.response_ids
        tokens[row, : len(seq)] = torch.tensor(seq, device=device)
        start = len(item.prompt_ids) - 1
        stop = start + len(item.response_ids)
        mask[row, start:stop] = True
        behavior_log_probs[row, start:stop] = torch.tensor(item.behavior_log_probs, device=device)
    return PaddedMicrobatch(
        tokens[:, :-1], tokens[:, 1:], mask, behavior_log_probs, batch.advantages[indices].to(device)
    )
