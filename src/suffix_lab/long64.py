"""Ordinary fresh draft64 -> verify SD; shared audited engine, no anchor reuse."""
import argparse
import json
from .assets import ROOT
from .long60 import execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--resume-root")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    config = json.loads((ROOT/"configs/long64.json").read_text())
    config.update(prompts_per_dataset=args.count, max_new_tokens=args.max_new_tokens)
    execute(args.output, args.revision, config, args.resume_root)


if __name__ == "__main__":
    main()
