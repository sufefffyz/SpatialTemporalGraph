#!/usr/bin/env python3
"""Official-first Xuancheng CityFlow reproduction diagnostics.

This runner does not repair, filter, expand, or rewrite flow routes. It is for
checking what the released official code/config can do before any downstream
road-aggregation pipeline is allowed to modify inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path


DEFAULT_REPO = "/home/yuzhang_fei/code/Hierarchical_traffic_control_platform_official"
DEFAULT_CONFIG = "./cfg/xuancheng/config_xuancheng_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", default=DEFAULT_REPO, help="Official code checkout.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config path, resolved from --official-repo.")
    parser.add_argument("--mode", choices=["direct-engine", "mpagent-launcher"], default="direct-engine")
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--progress-interval", type=int, default=600)
    parser.add_argument(
        "--allow-bypass-agent-init",
        action="store_true",
        help=(
            "Required for mpagent-launcher. The official agent/__init__.py imports "
            "unneeded baseline agents and currently fails before MPAgent is reached."
        ),
    )
    return parser.parse_args()


def progress_marks(duration: int, interval: int) -> set[int]:
    marks = {1, 10, 60, duration}
    if interval > 0:
        marks.update(range(interval, duration + 1, interval))
    return {m for m in marks if 0 < m <= duration}


def resolve_config(repo: Path, config_arg: str) -> str:
    config = Path(config_arg).expanduser()
    if config.is_absolute():
        return str(config)
    return str(config)


def load_mpagent_without_package_init(repo: Path):
    """Load official MPAgent source while bypassing the broken package __init__."""
    package = types.ModuleType("agent")
    package.__path__ = [str(repo / "agent")]
    sys.modules["agent"] = package

    for name in ("base_agent", "mp_agent"):
        path = repo / "agent" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"agent.{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load official agent module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"agent.{name}"] = module
        spec.loader.exec_module(module)

    return sys.modules["agent.mp_agent"].MPAgent


def run_direct_engine(args: argparse.Namespace, repo: Path, config: str) -> None:
    import cityflow  # type: ignore

    print("mode direct-engine", flush=True)
    print("official_repo", repo, flush=True)
    print("config", config, flush=True)

    eng = cityflow.Engine(config, thread_num=args.thread_num)
    started = time.time()
    marks = progress_marks(args.duration, args.progress_interval)
    for step in range(1, args.duration + 1):
        eng.next_step()
        if step in marks:
            print("step", step, "vehicles", eng.get_vehicle_count(), flush=True)

    print(
        "done",
        args.duration,
        "elapsed",
        round(time.time() - started, 2),
        "avg_travel_time",
        eng.get_average_travel_time(),
        flush=True,
    )


def run_mpagent_launcher(args: argparse.Namespace, repo: Path, config: str) -> None:
    if not args.allow_bypass_agent_init:
        raise SystemExit("mpagent-launcher requires --allow-bypass-agent-init")

    sys.path.insert(0, str(repo))
    from cityflow_env import CityFlowEnv  # type: ignore

    MPAgent = load_mpagent_without_package_init(repo)

    print("mode mpagent-launcher", flush=True)
    print("official_repo", repo, flush=True)
    print("config", config, flush=True)
    print(
        "official_deviation",
        "bypass agent/__init__.py only; official base_agent.py and mp_agent.py are unchanged",
        flush=True,
    )

    env = CityFlowEnv(config, thread_num=args.thread_num, simu_log=["net_accum", "trip_completion"])
    agent = MPAgent(env, args.decision_interval)
    env.reset()

    dyna = {}
    started = time.time()
    marks = progress_marks(args.duration, args.progress_interval)
    for current in range(args.decision_interval, args.duration + 1, args.decision_interval):
        state = agent.observe(env, dyna=dyna)
        actions = agent.act(state)
        dyna = env.step(actions, agent.decision_interval)
        if current in marks:
            net_accum = env.simu_log["net_accum"][-1] if env.simu_log["net_accum"] else None
            print("step", current, "vehicles", env.get_network_veh_num(), "net_accum_last", net_accum, flush=True)

    print(
        "done",
        args.duration,
        "elapsed",
        round(time.time() - started, 2),
        "avg_travel_time",
        env.get_avg_travel_time(),
        "net_accum_points",
        len(env.simu_log["net_accum"]),
        "trip_completion_points",
        len(env.simu_log["trip_completion"]),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.decision_interval <= 0:
        raise SystemExit("--decision-interval must be positive")

    repo = Path(args.official_repo).expanduser().resolve()
    if not repo.exists():
        raise SystemExit(f"official repo not found: {repo}")
    config = resolve_config(repo, args.config)

    old_cwd = Path.cwd()
    try:
        # Official config files use "dir": "./", so run from the official repo root.
        import os

        os.chdir(repo)
        if args.mode == "direct-engine":
            run_direct_engine(args, repo, config)
        else:
            run_mpagent_launcher(args, repo, config)
    finally:
        import os

        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
