"""QoreRL: run the synchronous training foundation."""

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
    args = parser.parse_args()
    from qorerl.runtime.ray_runtime import run_sync_grpo

    run_sync_grpo(TrainConfig.load(args.config), args.out, address=args.address, resume=args.resume)


if __name__ == "__main__":
    main()
