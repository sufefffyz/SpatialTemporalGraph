#!/usr/bin/env python3

import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a shared random seed set for LargeST multi-seed runs."
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        required=True,
        help="Number of unique seeds to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to profile_logs/shared_seeds/largest_shared_seeds_<num>.txt",
    )
    parser.add_argument(
        "--low",
        type=int,
        default=1,
        help="Inclusive lower bound of the random seed range.",
    )
    parser.add_argument(
        "--high",
        type=int,
        default=10000,
        help="Inclusive upper bound of the random seed range.",
    )
    parser.add_argument(
        "--generator-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible seed-set generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_seeds <= 0:
        raise SystemExit("--num-seeds must be positive")
    if args.low > args.high:
        raise SystemExit("--low must be <= --high")

    population = args.high - args.low + 1
    if args.num_seeds > population:
        raise SystemExit("Requested more unique seeds than the available range")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            repo_root
            / "profile_logs"
            / "shared_seeds"
            / f"largest_shared_seeds_{args.num_seeds}.txt"
        )
    )

    rng = random.Random(args.generator_seed)
    seeds = sorted(rng.sample(range(args.low, args.high + 1), args.num_seeds))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(seed) for seed in seeds) + "\n")

    print(f"Saved {len(seeds)} shared seeds to {output_path}")
    print("Seeds:")
    print(" ".join(str(seed) for seed in seeds))


if __name__ == "__main__":
    main()
