#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append Chinese progress updates to Obsidian project notes."
    )
    parser.add_argument("--repo-root", type=Path, required=True, help="Absolute repository root.")
    parser.add_argument("--vault", type=Path, required=True, help="Absolute Obsidian vault path.")
    parser.add_argument(
        "--project",
        default=None,
        help="Project name. Defaults to the repository directory name.",
    )
    parser.add_argument("--done", action="append", default=[], help="Completed work item.")
    parser.add_argument("--blocker", action="append", default=[], help="Blocker item.")
    parser.add_argument("--next", action="append", default=[], dest="next_steps", help="Next-step item.")
    parser.add_argument("--decision", action="append", default=[], help="Decision item.")
    parser.add_argument("--backlog", action="append", default=[], help="Backlog item.")
    parser.add_argument("--dry-run", action="store_true", help="Print updates without writing files.")
    return parser.parse_args()


def normalize_items(items: list[str], fallback: str) -> list[str]:
    normalized = [item.strip() for item in items if item and item.strip()]
    return normalized or [fallback]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_text(path: Path, block: str) -> None:
    existing = read_text(path)
    separator = "\n" if existing.endswith("\n") else "\n\n"
    updated = f"{existing}{separator}{block}" if existing else block
    write_text(path, updated)


def ensure_progress_file(path: Path, project: str) -> None:
    if path.exists():
        return
    content = "\n".join(
        [
            "---",
            f"project: {project}",
            "type: progress-log",
            "status: active",
            "---",
            "",
            f"# {project} progress",
            "",
            "## Overview",
            "- Goal:",
            "- Repo:",
            "- Owner:",
        ]
    )
    write_text(path, content)


def ensure_decisions_file(path: Path, project: str) -> None:
    if path.exists():
        return
    content = "\n".join(
        [
            "---",
            f"project: {project}",
            "type: decisions",
            "status: active",
            "---",
            "",
            "# Decisions",
        ]
    )
    write_text(path, content)


def ensure_backlog_file(path: Path, project: str) -> None:
    if path.exists():
        return
    content = "\n".join(
        [
            "---",
            f"project: {project}",
            "type: backlog",
            "status: active",
            "---",
            "",
            "# Backlog",
        ]
    )
    write_text(path, content)


def build_progress_session_block(now: datetime, done: list[str], blockers: list[str], next_steps: list[str]) -> str:
    time_label = now.strftime("%H:%M")
    return "\n".join(
        [
            f"### {time_label}",
            "",
            "#### Done",
            *(f"- {item}" for item in done),
            "",
            "#### Blockers",
            *(f"- {item}" for item in blockers),
            "",
            "#### Next",
            *(f"- {item}" for item in next_steps),
        ]
    )


def build_backlog_block(items: list[str]) -> str:
    return "\n".join(f"- [ ] {item}" for item in items)


def append_under_date(path: Path, date_label: str, block: str) -> None:
    date_heading = f"## {date_label}"
    existing = read_text(path)
    if existing and date_heading in existing:
        append_text(path, block)
        return
    dated_block = "\n".join([date_heading, "", block])
    append_text(path, dated_block)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    vault_root = args.vault.expanduser().resolve()
    project = args.project or repo_root.name
    project_dir = vault_root / "Projects" / project

    progress_path = project_dir / "progress.md"
    decisions_path = project_dir / "decisions.md"
    backlog_path = project_dir / "backlog.md"

    done_items = normalize_items(args.done, "无")
    blocker_items = normalize_items(args.blocker, "无")
    next_items = normalize_items(args.next_steps, "无")
    decision_items = [item.strip() for item in args.decision if item and item.strip()]
    backlog_items = [item.strip() for item in args.backlog if item and item.strip()]

    now = datetime.now().astimezone()
    date_label = now.strftime("%Y-%m-%d")
    progress_block = build_progress_session_block(now, done_items, blocker_items, next_items)
    decision_block = "\n".join(f"- {item}" for item in decision_items) if decision_items else ""
    backlog_block = build_backlog_block(backlog_items) if backlog_items else ""

    if args.dry_run:
        print(f"[progress] {progress_path}")
        print("\n".join([f"## {date_label}", "", progress_block]))
        if decision_block:
            print()
            print(f"[decisions] {decisions_path}")
            print("\n".join([f"## {date_label}", decision_block]))
        if backlog_block:
            print()
            print(f"[backlog] {backlog_path}")
            print(backlog_block)
        return

    ensure_progress_file(progress_path, project)
    ensure_decisions_file(decisions_path, project)
    ensure_backlog_file(backlog_path, project)

    append_under_date(progress_path, date_label, progress_block)
    print(f"Updated {progress_path}")

    if decision_block:
        append_under_date(decisions_path, date_label, decision_block)
        print(f"Updated {decisions_path}")

    if backlog_block:
        append_text(backlog_path, backlog_block)
        print(f"Updated {backlog_path}")


if __name__ == "__main__":
    main()
