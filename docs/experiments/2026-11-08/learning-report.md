# QoreRL: Wgrad-only BF16 activation experiment

Backend: **CPU/CUDA numerical emulation, not native MXFP8 Tensor Core execution**.

Each seed shares one BF16 SFT warm-start checkpoint across all arms. Each arm then samples from its own current policy. Fixed evaluation prompts are shared, but are not a held-out generalization benchmark.

| Seed | Arm | Initial eval token acc. | Final eval token acc. | Last 20 reward | Opposite events | Direction recovered | Active layers | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | bf16 | 0.2383 | 0.9102 | 0.6469 | 0 | 0 | 0 | 1.5 |
| 0 | mxfp8 | 0.2266 | 0.8789 | 0.6391 | 0 | 0 | 0 | 3.2 |
| 0 | adaptive | 0.2266 | 0.8789 | 0.6391 | 0 | 0 | 0 | 3.4 |
| 1 | bf16 | 0.1289 | 0.6602 | 0.5559 | 0 | 0 | 0 | 1.4 |
| 1 | mxfp8 | 0.1328 | 0.6484 | 0.5516 | 0 | 0 | 0 | 3.3 |
| 1 | adaptive | 0.1328 | 0.6484 | 0.5516 | 0 | 0 | 0 | 3.3 |
| 2 | bf16 | 0.1367 | 0.9648 | 0.7168 | 0 | 0 | 0 | 1.4 |
| 2 | mxfp8 | 0.1641 | 0.9570 | 0.6836 | 0 | 0 | 0 | 3.3 |
| 2 | adaptive | 0.1641 | 0.9570 | 0.6836 | 0 | 0 | 0 | 3.4 |

A direction recovery is a cosine-sign result, not a proof of better RL learning.
If no opposite gradients occur, the natural run has not tested intervention efficacy. A separate unit test exercises the trigger and intervention branch without injecting a fault into these training results.

Calibration files include paired BF16 reference, pure MXFP8 baseline, and MXFP8 with BF16 Wgrad activations; shadow AdamW steps evaluate both surrogate objectives without mutating the live model/optimizer. Overrides are applied on negative cosine and retained; candidate outcomes are measured, not filtered to make the intervention look successful.
