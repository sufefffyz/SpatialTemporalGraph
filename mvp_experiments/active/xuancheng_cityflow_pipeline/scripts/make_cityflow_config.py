#!/usr/bin/env python3
"""Build a CityFlow config for one Xuancheng daily flow file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def daily_name(date_text: str) -> str:
    return f"data_{date_text.replace('-', '_')}_type_filtered.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root with raw/ downloaded files.")
    parser.add_argument("--date", required=True, help="Date such as 2023-04-03.")
    parser.add_argument("--output", required=True, help="Config JSON path to write.")
    parser.add_argument("--template", help="Optional official config to patch.")
    parser.add_argument(
        "--tl-mode",
        choices=["fixed_time", "official_rl"],
        default="fixed_time",
        help="fixed_time sets rlTrafficLight=false; official_rl preserves the official RL setting.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-time", type=int, default=10)
    parser.add_argument("--all-red-time", type=int, default=3)
    parser.add_argument("--save-replay", action="store_true", help="Enable bulky CityFlow replay logs.")
    parser.add_argument("--no-validate", action="store_true", help="Do not check input files exist.")
    return parser.parse_args()


def load_template(path_text: str | None) -> dict:
    if not path_text:
        return {
            "interval": 1.0,
            "seed": 0,
            "dir": "./",
            "roadnetFile": "",
            "flowFile": "",
            "rlTrafficLight": False,
            "laneChange": False,
            "saveReplay": False,
            "step_time": 10,
            "all_red_time": 3,
        }
    with Path(path_text).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw"
    roadnet = raw_dir / "roadnet_xuancheng250319.json"
    flow = raw_dir / daily_name(args.date)
    output = Path(args.output).expanduser().resolve()

    if not args.no_validate:
        missing = [str(path) for path in (roadnet, flow) if not path.exists()]
        if missing:
            raise SystemExit("missing input files:\n  " + "\n  ".join(missing))

    cfg = load_template(args.template)
    cfg.update(
        {
            "interval": float(cfg.get("interval", 1.0)),
            "seed": args.seed,
            "dir": str(raw_dir) + "/",
            "roadnetFile": roadnet.name,
            "flowFile": flow.name,
            "rlTrafficLight": args.tl_mode == "official_rl",
            "laneChange": False,
            "saveReplay": bool(args.save_replay),
            "step_time": args.step_time,
            "all_red_time": args.all_red_time,
        }
    )

    if args.save_replay:
        replay_dir = data_root / "replay_logs"
        replay_dir.mkdir(parents=True, exist_ok=True)
        cfg["roadnetLogFile"] = str(replay_dir / f"roadnet_{args.date}.json")
        cfg["replayLogFile"] = str(replay_dir / f"replay_{args.date}.log")
    else:
        cfg.pop("roadnetLogFile", None)
        cfg.pop("replayLogFile", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"wrote {output}")
    print(f"roadnetFile={cfg['roadnetFile']}")
    print(f"flowFile={cfg['flowFile']}")
    print(f"rlTrafficLight={cfg['rlTrafficLight']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
