#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = REPO_ROOT.name


@dataclass
class GitSnapshot:
    branch: str | None
    commit: str | None
    status_lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a project progress entry to an Obsidian vault note."
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Absolute path to the Obsidian vault. Falls back to OBSIDIAN_VAULT_PATH.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"Project name used in the note title. Default: {DEFAULT_PROJECT}",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Relative note path inside the vault. Default: Projects/<project>/Progress.md",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="One-line progress summary. If omitted, stdin is used when piped.",
    )
    parser.add_argument(
        "--status",
        default="In progress",
        help="High-level status label for this entry.",
    )
    parser.add_argument(
        "--next-step",
        action="append",
        default=[],
        dest="next_steps",
        help="Next step item. Repeat the flag for multiple items.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        dest="files",
        help="Related file path. Repeat the flag for multiple items.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"Repository root used for git metadata. Default: {REPO_ROOT}",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Do not include git branch and worktree status.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the note path and entry content without writing the file.",
    )
    return parser.parse_args()


def resolve_vault_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.expanduser().resolve()

    env_value = os.environ.get("OBSIDIAN_VAULT_PATH")
    env_path = Path(env_value) if env_value else None

    if env_path is None:
        raise SystemExit(
            "Missing vault path. Pass --vault or set OBSIDIAN_VAULT_PATH."
        )

    return env_path.expanduser().resolve()


def read_summary(args: argparse.Namespace) -> str:
    if args.summary:
        return args.summary.strip()

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped

    return "Project progress update."


def run_git(repo_root: Path, *git_args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *git_args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def collect_git_snapshot(repo_root: Path) -> GitSnapshot:
    branch = run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = run_git(repo_root, "rev-parse", "--short", "HEAD")
    status_output = run_git(repo_root, "status", "--short")
    status_lines = status_output.splitlines() if status_output else []
    return GitSnapshot(branch=branch, commit=commit, status_lines=status_lines)


def build_note_path(vault_path: Path, project: str, note_override: str | None) -> Path:
    if note_override:
        relative = Path(note_override)
        if relative.is_absolute():
            raise SystemExit("--note must be a path relative to the vault root.")
    else:
        relative = Path("Projects") / project / "Progress.md"
    return vault_path / relative


def build_entry(
    summary: str,
    status: str,
    files: list[str],
    next_steps: list[str],
    git_snapshot: GitSnapshot | None,
) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [f"## {timestamp}", "", f"- Summary: {summary}", f"- Status: {status}"]

    if git_snapshot and git_snapshot.branch:
        branch_line = f"- Branch: `{git_snapshot.branch}`"
        if git_snapshot.commit:
            branch_line += f" at `{git_snapshot.commit}`"
        lines.append(branch_line)
    elif git_snapshot and git_snapshot.commit:
        lines.append(f"- Commit: `{git_snapshot.commit}`")

    if files:
        lines.append("- Related files:")
        for item in files:
            lines.append(f"  - `{item}`")

    if git_snapshot and git_snapshot.status_lines:
        lines.append("- Git status snapshot:")
        for item in git_snapshot.status_lines[:20]:
            lines.append(f"  - `{item}`")
        if len(git_snapshot.status_lines) > 20:
            lines.append(
                f"  - `... {len(git_snapshot.status_lines) - 20} more entries omitted`"
            )

    if next_steps:
        lines.append("- Next steps:")
        for item in next_steps:
            lines.append(f"  - {item}")

    lines.append("")
    return "\n".join(lines)


def ensure_note_header(note_path: Path, project: str, repo_root: Path) -> None:
    if note_path.exists():
        return

    header = "\n".join(
        [
            f"# {project} Progress",
            "",
            "Auto-generated project progress log for Obsidian.",
            "",
            f"- Repository: `{repo_root}`",
            f"- Created: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
            "",
        ]
    )
    note_path.write_text(header, encoding="utf-8")


def append_entry(note_path: Path, entry: str) -> None:
    existing = note_path.read_text(encoding="utf-8")
    separator = "\n" if existing.endswith("\n") else "\n\n"
    note_path.write_text(existing + separator + entry, encoding="utf-8")


def main() -> None:
    args = parse_args()
    vault_path = resolve_vault_path(args.vault)
    if not vault_path.exists():
        raise SystemExit(f"Vault path does not exist: {vault_path}")
    if not vault_path.is_dir():
        raise SystemExit(f"Vault path is not a directory: {vault_path}")

    note_path = build_note_path(vault_path, args.project, args.note)
    git_snapshot = None if args.skip_git else collect_git_snapshot(args.repo_root)
    summary = read_summary(args)
    entry = build_entry(
        summary=summary,
        status=args.status,
        files=args.files,
        next_steps=args.next_steps,
        git_snapshot=git_snapshot,
    )

    if args.dry_run:
        print(f"Note path: {note_path}")
        print()
        print(entry)
        return

    note_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_note_header(note_path, args.project, args.repo_root)
    append_entry(note_path, entry)
    print(f"Updated Obsidian note: {note_path}")


if __name__ == "__main__":
    main()
