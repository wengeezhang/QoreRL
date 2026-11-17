"""Self-contained, verifiable tasks; no downloaded training dataset required."""

import random

from qorerl.training.contracts import Prompt


class CopyTask:
    """CPU fixture vocabulary: PAD=0, BOS=1, EOS=2, SEP=3, digits=4..13."""

    def batch(self, step, count, seed):
        rng = random.Random(seed + step)
        return [
            Prompt(step * count + i, (1, (digit := rng.randrange(10)) + 4, 3), str(digit))
            for i in range(count)
        ]


class ArithmeticTask:
    def __init__(self, cfg):
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)

    def batch(self, step, count, seed):
        rng = random.Random(seed + step)
        result = []
        for i in range(count):
            a, b = rng.randrange(100), rng.randrange(100)
            text = f"Calculate {a} + {b}. Reply with only the integer answer, no explanation."
            tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tokens) + self.cfg.max_new_tokens > self.cfg.max_model_len:
                raise ValueError("Prompt plus response budget exceeds model context")
            result.append(Prompt(step * count + i, tuple(tokens), str(a + b)))
        return result
