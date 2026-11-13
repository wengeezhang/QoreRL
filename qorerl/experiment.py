"""Run: uv run qorerl --steps 60 --seeds 0,1,2 --out results/first-study"""

import argparse
import copy
import json
import platform
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from qorerl.calibration import assign_gradients, calibrate, capture
from qorerl.model import ModelConfig, TinyMoE
from qorerl.precision import precision
from qorerl.rl import evaluate, prompts, rollout, supervised_loss


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def append_json(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def warm_start(cfg, args, seed, directory):
    torch.manual_seed(seed)
    random.seed(seed)
    model = TinyMoE(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.warmup_lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 1000)
    for step in range(args.warmup_steps):
        prefix, target = prompts(32, generator, args.device, cfg.vocab_size)
        optimizer.zero_grad(set_to_none=True)
        with precision(model, "bf16"):
            loss = supervised_loss(model, prefix, target)
            loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite warmup loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        append_json(directory / "warmup.jsonl", {"step": step + 1, "loss": loss.item()})
    return copy.deepcopy(model.state_dict())


def run_arm(cfg, args, seed, arm, state, directory):
    directory.mkdir()
    model = TinyMoE(cfg).to(args.device)
    model.load_state_dict(state)
    model.train()  # No dropout, capacity dropping, or mutable running statistics.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    prompt_rng = torch.Generator().manual_seed(seed + 2000)
    sample_rng = torch.Generator().manual_seed(seed + 3000)
    eval_prompt, eval_target = prompts(128, torch.Generator().manual_seed(999), args.device, cfg.vocab_size)
    mode = "bf16" if arm == "bf16" else "mxfp8"
    overrides, rewards, calibrations = set(), [], []
    with precision(model, mode):
        initial_eval = evaluate(model, eval_prompt, eval_target)
    start = time.perf_counter()
    calibration_seconds = 0.0
    for step in range(1, args.steps + 1):
        prefix, target = prompts(args.prompts, prompt_rng, args.device, cfg.vocab_size)
        with precision(model, mode, overrides):
            batch = rollout(model, prefix, target, args.group_size, sample_rng)
        probe_start = time.perf_counter()
        is_calibration = arm == "adaptive" and step % args.calibration_interval == 0
        if is_calibration:
            overrides, probe, report = calibrate(
                model,
                optimizer,
                batch,
                step,
                overrides,
                args.cosine_threshold,
                args.min_grad_norm,
                args.max_grad_norm,
                not args.no_trial_updates,
            )
            elapsed = time.perf_counter() - probe_start
            calibration_seconds += elapsed
            report["seconds"] = elapsed
            append_json(directory / "calibrations.jsonl", report)
            calibrations.append(report)
            print(
                f"seed={seed} {arm} calibration={step} opposite={len(report['triggered'])} "
                f"new={len(report['new_overrides'])} active={len(overrides)}",
                flush=True,
            )
        else:
            probe = capture(model, batch, mode, overrides)
        assign_gradients(model, probe.gradients)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()  # Exactly one real update per train batch; probes did not step.
        reward = batch.rewards.mean().item()
        rewards.append(reward)
        metrics = {
            "step": step,
            "reward": reward,
            "exact_match": (batch.rewards == 1).float().mean().item(),
            "nonzero_advantage_fraction": (batch.advantages != 0).float().mean().item(),
            "grad_norm": grad_norm,
            "active_overrides": sorted(overrides),
            "calibration": is_calibration,
            **probe.metrics,
        }
        if step % args.calibration_interval == 0 or step == args.steps:
            with precision(model, mode, overrides):
                metrics["eval"] = evaluate(model, eval_prompt, eval_target)
        append_json(directory / "steps.jsonl", metrics)
        if step % 10 == 0 or step == args.steps:
            print(
                f"seed={seed} {arm} step={step}/{args.steps} reward={reward:.4f} "
                f"loss={probe.metrics['loss']:.6f}",
                flush=True,
            )
    recovered = sum(
        r["triggered"] and r["after"]["cosine"] is not None and r["after"]["cosine"] >= 0
        for c in calibrations
        for r in c["layers"]
    )
    triggers = sum(len(c["triggered"]) for c in calibrations)
    summary = {
        "seed": seed,
        "arm": arm,
        "backend": "mxfp8_emulated_fp32_gemm",
        "steps": args.steps,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_eval": initial_eval,
        "final_eval": metrics["eval"],
        "mean_reward": sum(rewards) / len(rewards),
        "last_20_reward": sum(rewards[-20:]) / len(rewards[-20:]),
        "calibration_steps": [c["step"] for c in calibrations],
        "trigger_events": triggers,
        "recovered_direction_events": recovered,
        "active_overrides": sorted(overrides),
        "seconds": time.perf_counter() - start,
        "calibration_seconds": calibration_seconds,
    }
    write_json(directory / "summary.json", summary)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.steps,
            "overrides": sorted(overrides),
            "model_config": asdict(cfg),
            "run_config": vars(args),
            "prompt_rng": prompt_rng.get_state(),
            "sample_rng": sample_rng.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        directory / "final.pt",
    )
    return summary


def report_markdown(summaries):
    lines = [
        "# QoreRL: Wgrad-only BF16 activation experiment",
        "",
        "Backend: **CPU/CUDA numerical emulation, not native MXFP8 Tensor Core execution**.",
        "",
        "Each seed shares one BF16 SFT warm-start checkpoint across all arms. "
        "Each arm then samples from its own current policy. Fixed evaluation prompts "
        "are shared, but are not a held-out generalization benchmark.",
        "",
        "| Seed | Arm | Initial eval token acc. | Final eval token acc. | Last 20 reward | "
        "Opposite events | Direction recovered | Active layers | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['seed']} | {s['arm']} | {s['initial_eval']['token_accuracy']:.4f} | "
            f"{s['final_eval']['token_accuracy']:.4f} | {s['last_20_reward']:.4f} | "
            f"{s['trigger_events']} | {s['recovered_direction_events']} | "
            f"{len(s['active_overrides'])} | {s['seconds']:.1f} |"
        )
    lines += [
        "",
        "A direction recovery is a cosine-sign result, not a proof of better RL learning.",
        "If no opposite gradients occur, the natural run has not tested intervention efficacy. "
        "A separate unit test exercises the trigger and intervention branch without "
        "injecting a fault into these training results.",
        "",
        "Calibration files include paired BF16 reference, pure MXFP8 baseline, "
        "and MXFP8 with BF16 Wgrad activations; shadow AdamW steps evaluate both "
        "surrogate objectives without mutating the live model/optimizer. "
        "Overrides are applied on negative cosine and retained; candidate outcomes "
        "are measured, not filtered to make the intervention look successful.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--calibration-interval", type=int, default=20)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--arms", default="bf16,mxfp8,adaptive")
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup-lr", type=float, default=0.003)
    parser.add_argument("--cosine-threshold", type=float, default=0.0)
    parser.add_argument("--min-grad-norm", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-trial-updates", action="store_true")
    parser.add_argument("--out", default="results/wgrad-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if min(args.steps, args.calibration_interval, args.prompts, args.threads, args.layers) <= 0:
        parser.error("steps, interval, prompts, threads and layers must be positive")
    if args.group_size < 2 or args.warmup_steps < 0 or not -1 <= args.cosine_threshold <= 0:
        parser.error("group_size >= 2, warmup_steps >= 0 and cosine_threshold in [-1,0] required")
    if args.min_grad_norm < 0 or min(args.lr, args.warmup_lr, args.max_grad_norm) <= 0:
        parser.error("Invalid learning rate or gradient norm threshold")
    arms = args.arms.split(",")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if set(arms) - {"bf16", "mxfp8", "adaptive"} or len(set(arms)) != len(arms):
        parser.error("arms must be unique entries from bf16,mxfp8,adaptive")
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for numerical emulation")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = ModelConfig(
        hidden=args.hidden, intermediate=2 * args.hidden, layers=args.layers, experts=args.experts
    )
    write_json(
        directory / "config.json",
        {
            "args": vars(args),
            "model": asdict(cfg),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "backend": "mxfp8_emulated_fp32_gemm",
        },
    )
    summaries = []
    for seed in seeds:
        seed_dir = directory / f"seed_{seed}"
        seed_dir.mkdir()
        state = warm_start(cfg, args, seed, seed_dir)
        torch.save(state, seed_dir / "initial.pt")
        for arm in arms:
            summaries.append(run_arm(cfg, args, seed, arm, state, seed_dir / arm))
            write_json(directory / "summary.json", summaries)
            (directory / "report.md").write_text(report_markdown(summaries))
    print(f"Report: {(directory / 'report.md').resolve()}", flush=True)


if __name__ == "__main__":
    main()
