"""QoreRL: train, inspect recipe evidence, and export reports."""

import argparse

from qorerl.training.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run synchronous GRPO through Ray workers")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--address", help="Existing Ray cluster address; omit to start local Ray")
    train.add_argument("--resume", help="Completed checkpoint directory; output must be a new directory")
    decision = train.add_mutually_exclusive_group()
    decision.add_argument(
        "--accept-candidate", metavar="ID", help="Resume and install this reviewed candidate"
    )
    decision.add_argument("--reject-candidate", metavar="ID", help="Resume with the current recipe settings")
    inspect = commands.add_parser("inspect", help="View live layer comparisons from a training run")
    inspect.add_argument(
        "--run", required=True, help="Run directory containing probe or calibration evidence"
    )
    inspect.add_argument("--port", type=int, default=8765)
    inspect.add_argument("--related-run", action="append", default=[], help="Explicit prior calibration run")
    inspect.add_argument("--view", choices=("auto", "probes", "calibration"), default="auto")
    report = commands.add_parser("report", help="Export a standalone recipe comparison HTML report")
    report.add_argument("--run", required=True)
    report.add_argument("--out", required=True, help="Output .html file")
    report.add_argument("--related-run", action="append", default=[], help="Explicit prior calibration run")
    report.add_argument("--view", choices=("auto", "probes", "calibration"), default="auto")
    demo = commands.add_parser("demo-report", help="Create explicitly synthetic data to preview the viewer")
    demo.add_argument("--out", required=True, help="New demo directory; never overwrites a training run")
    preflight = commands.add_parser("gpu-preflight", help="Check native GPU prerequisites without training")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--role", choices=("all", "learner", "rollout"), default="all")
    preflight.add_argument("--out", required=True, help="New directory for JSON and Markdown results")
    validate = commands.add_parser("validate-run", help="Audit recorded GPU smoke-test evidence")
    validate.add_argument("--run", required=True)
    validate.add_argument("--related-run", action="append", default=[])
    validate.add_argument("--out", required=True, help="New directory for JSON and Markdown results")
    validate.add_argument("--min-updates", type=int, default=2)
    validate.add_argument(
        "--max-logprob-gap", type=float, help="Experiment-specific maximum absolute token log-prob gap"
    )
    validate.add_argument(
        "--check-checkpoints", action="store_true", help="Compare local initial/final model tensors"
    )
    validate.add_argument(
        "--require-calibration",
        action="store_true",
        help="Require an applied candidate and subsequent rollout",
    )
    args = parser.parse_args()
    if args.command in ("gpu-preflight", "validate-run"):
        from qorerl.validation import preflight, validate_run, write_report

        if args.command == "gpu-preflight":
            result = preflight(TrainConfig.load(args.config), role=args.role)
        else:
            result = validate_run(
                args.run,
                related_runs=args.related_run,
                min_updates=args.min_updates,
                max_logprob_gap=args.max_logprob_gap,
                check_checkpoints=args.check_checkpoints,
                require_calibration=args.require_calibration,
            )
        print(f"{result['result']}: {write_report(result, args.out)}")
        raise SystemExit({"passed": 0, "failed": 1, "incomplete": 2}[result["result"]])
    if args.command == "inspect":
        from qorerl.observability.viewer import serve

        serve(args.run, port=args.port, related_runs=args.related_run, view=args.view)
        return
    if args.command == "report":
        from qorerl.observability.viewer import export_report

        print(export_report(args.run, args.out, related_runs=args.related_run, view=args.view))
        return
    if args.command == "demo-report":
        from qorerl.observability.demo import write_demo

        print(write_demo(args.out))
        return
    from qorerl.runtime.ray_runtime import run_sync_grpo

    result = run_sync_grpo(
        TrainConfig.load(args.config),
        args.out,
        address=args.address,
        resume=args.resume,
        accept_candidate=args.accept_candidate,
        reject_candidate=args.reject_candidate,
    )
    if result["status"] == "awaiting_review":
        print(f"Review {result['checkpoint']}/candidate.json")
        print(
            f"Resume with --resume {result['checkpoint']} and "
            f"--accept-candidate {result['candidate_id']} or --reject-candidate {result['candidate_id']}"
        )


if __name__ == "__main__":
    main()
