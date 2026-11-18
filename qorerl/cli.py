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
    decision.add_argument("--reject-candidate", metavar="ID", help="Resume keeping the previous scales")
    inspect = commands.add_parser("inspect", help="View live layer comparisons from a training run")
    inspect.add_argument("--run", required=True, help="Run directory containing recipe_metrics.jsonl")
    inspect.add_argument("--port", type=int, default=8765)
    report = commands.add_parser("report", help="Export a standalone recipe comparison HTML report")
    report.add_argument("--run", required=True)
    report.add_argument("--out", required=True, help="Output .html file")
    demo = commands.add_parser("demo-report", help="Create explicitly synthetic data to preview the viewer")
    demo.add_argument("--out", required=True, help="New demo directory; never overwrites a training run")
    args = parser.parse_args()
    if args.command == "inspect":
        from qorerl.observability.viewer import serve

        serve(args.run, port=args.port)
        return
    if args.command == "report":
        from qorerl.observability.viewer import export_report

        print(export_report(args.run, args.out))
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
