"""QoreRL entry points: native training and isolated numerical experiments."""

import argparse
import sys

from qorerl.training.config import TrainConfig


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "research":
        sys.argv.pop(1)
        from qorerl.experiment import main as research_main

        return research_main()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("research", help="Run the independent tiny-MoE numerical experiment")
    train = commands.add_parser("train", help="Run synchronous GRPO through Ray actors")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--address", help="Existing Ray cluster address; omit to start local Ray")
    train.add_argument("--resume", help="Completed checkpoint directory; output must be a new directory")
    args = parser.parse_args()
    from qorerl.runtime.ray_runtime import run

    run(TrainConfig.load(args.config), args.out, address=args.address, resume=args.resume)


if __name__ == "__main__":
    main()
