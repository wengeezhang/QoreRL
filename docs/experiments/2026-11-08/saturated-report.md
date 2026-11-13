# QoreRL: Wgrad-only BF16 activation experiment

Backend: **CPU/CUDA numerical emulation, not native MXFP8 Tensor Core execution**.

Each seed shares one BF16 SFT warm-start checkpoint across all arms. Each arm then samples from its own current policy. Fixed evaluation prompts are shared, but are not a held-out generalization benchmark.

| Seed | Arm | Initial eval token acc. | Final eval token acc. | Last 20 reward | Opposite events | Direction recovered | Active layers | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | bf16 | 1.0000 | 1.0000 | 0.9801 | 0 | 0 | 0 | 0.8 |
| 0 | mxfp8 | 1.0000 | 1.0000 | 0.9789 | 0 | 0 | 0 | 1.9 |
| 0 | adaptive | 1.0000 | 1.0000 | 0.9789 | 1 | 0 | 1 | 2.0 |
| 1 | bf16 | 1.0000 | 1.0000 | 0.9711 | 0 | 0 | 0 | 0.9 |
| 1 | mxfp8 | 1.0000 | 1.0000 | 0.9707 | 0 | 0 | 0 | 1.9 |
| 1 | adaptive | 1.0000 | 1.0000 | 0.9707 | 0 | 0 | 0 | 1.9 |
| 2 | bf16 | 1.0000 | 1.0000 | 0.9691 | 0 | 0 | 0 | 0.9 |
| 2 | mxfp8 | 1.0000 | 1.0000 | 0.9680 | 0 | 0 | 0 | 1.9 |
| 2 | adaptive | 1.0000 | 1.0000 | 0.9680 | 0 | 0 | 0 | 2.0 |

A direction recovery is a cosine-sign result, not a proof of better RL learning.
If no opposite gradients occur, the natural run has not tested intervention efficacy. A separate unit test exercises the trigger and intervention branch without injecting a fault into these training results.

Calibration files include paired BF16 reference, pure MXFP8 baseline, and MXFP8 with BF16 Wgrad activations; shadow AdamW steps evaluate both surrogate objectives without mutating the live model/optimizer. Overrides are applied on negative cosine and retained; candidate outcomes are measured, not filtered to make the intervention look successful.
