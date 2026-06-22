#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/push_and_trigger_remote.sh [options] -- <command> [args...]

Options:
  --config PATH          Optional env file. Defaults to .remote-run.env at repo root when present.
  --host HOST            Remote SSH target, e.g. user@server.
  --remote-repo-dir DIR  Absolute repo path on the remote machine.
  --project NAME         Subproject directory to run in on the remote machine.
  --conda-env NAME       Conda environment on the remote machine.
  --ref REF              Branch to push and run. Defaults to the current local branch.
  --logs-dir PATH        Override remote logs directory.
  --name NAME            Optional run label.
  --mode fg|bg           Run remotely in foreground or background. Default: bg.
  --allow-dirty-remote   Allow a dirty git worktree on the remote machine.
  --no-push             Skip git push and only trigger the remote side.
  --dry-run             Print the generated SSH command without executing it.
  -h, --help            Show this help message.

The config file can define:
  STG_REMOTE_HOST
  STG_REMOTE_REPO_DIR
  STG_REMOTE_PROJECT
  STG_REMOTE_CONDA_ENV
  STG_REMOTE_LOGS_DIR
  STG_REMOTE_MODE

Example:
  bash scripts/push_and_trigger_remote.sh \
    --host user@server \
    --remote-repo-dir /home/user/code/SpatialTemporalGraph \
    --project urban-traffic-benchmark \
    --conda-env graph_ml \
    --mode bg \
    -- bash reproduce_experiments_models_dataset_path.sh
EOF
}

quote_args() {
  local parts=()
  local arg
  for arg in "$@"; do
    parts+=("$(printf '%q' "$arg")")
  done
  printf '%s' "${parts[*]}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG="$REPO_DIR/.remote-run.env"

CONFIG_FILE="$DEFAULT_CONFIG"
if [[ ! -f "$CONFIG_FILE" ]]; then
  CONFIG_FILE=""
fi

if [[ -n "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

HOST="${STG_REMOTE_HOST:-}"
REMOTE_REPO_DIR="${STG_REMOTE_REPO_DIR:-}"
PROJECT="${STG_REMOTE_PROJECT:-}"
CONDA_ENV="${STG_REMOTE_CONDA_ENV:-}"
LOGS_DIR="${STG_REMOTE_LOGS_DIR:-}"
MODE="${STG_REMOTE_MODE:-bg}"
RUN_NAME=""
ALLOW_DIRTY_REMOTE=0
NO_PUSH=0
DRY_RUN=0
REF="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      # shellcheck disable=SC1090
      source "$CONFIG_FILE"
      HOST="${STG_REMOTE_HOST:-$HOST}"
      REMOTE_REPO_DIR="${STG_REMOTE_REPO_DIR:-$REMOTE_REPO_DIR}"
      PROJECT="${STG_REMOTE_PROJECT:-$PROJECT}"
      CONDA_ENV="${STG_REMOTE_CONDA_ENV:-$CONDA_ENV}"
      LOGS_DIR="${STG_REMOTE_LOGS_DIR:-$LOGS_DIR}"
      MODE="${STG_REMOTE_MODE:-$MODE}"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --remote-repo-dir)
      REMOTE_REPO_DIR="$2"
      shift 2
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --logs-dir)
      LOGS_DIR="$2"
      shift 2
      ;;
    --name)
      RUN_NAME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --allow-dirty-remote)
      ALLOW_DIRTY_REMOTE=1
      shift
      ;;
    --no-push)
      NO_PUSH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "A remote command must be provided after --." >&2
  usage >&2
  exit 1
fi

if [[ "$REF" == "HEAD" ]]; then
  echo "Detached HEAD detected. Pass --ref explicitly." >&2
  exit 1
fi

if [[ "$MODE" != "fg" && "$MODE" != "bg" ]]; then
  echo "--mode must be either fg or bg." >&2
  exit 1
fi

if [[ -z "$HOST" || -z "$REMOTE_REPO_DIR" || -z "$PROJECT" ]]; then
  echo "--host, --remote-repo-dir, and --project are required." >&2
  usage >&2
  exit 1
fi

if [[ "$DRY_RUN" -ne 1 ]]; then
  if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    echo "Local repo has uncommitted changes. Commit or stash them before pushing to remote." >&2
    exit 1
  fi
fi

COMMAND=( "$@" )

if [[ "$NO_PUSH" -ne 1 && "$DRY_RUN" -ne 1 ]]; then
  git -C "$REPO_DIR" push origin "$REF"
fi

REMOTE_SCRIPT_ARGS=(
  --repo-dir "$REMOTE_REPO_DIR"
  --ref "$REF"
  --project "$PROJECT"
  --mode "$MODE"
  --skip-pull
)

if [[ -n "$CONDA_ENV" ]]; then
  REMOTE_SCRIPT_ARGS+=(--conda-env "$CONDA_ENV")
fi

if [[ -n "$LOGS_DIR" ]]; then
  REMOTE_SCRIPT_ARGS+=(--logs-dir "$LOGS_DIR")
fi

if [[ -n "$RUN_NAME" ]]; then
  REMOTE_SCRIPT_ARGS+=(--name "$RUN_NAME")
fi

if [[ "$ALLOW_DIRTY_REMOTE" -eq 1 ]]; then
  REMOTE_SCRIPT_ARGS+=(--allow-dirty)
fi

REMOTE_SCRIPT_ARGS+=(-- "${COMMAND[@]}")

REMOTE_SCRIPT_CALL="bash $(printf '%q' "$REMOTE_REPO_DIR/scripts/remote_pull_and_run.sh") $(quote_args "${REMOTE_SCRIPT_ARGS[@]}")"

REMOTE_BOOTSTRAP=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$REMOTE_REPO_DIR")
git fetch origin
if git show-ref --verify --quiet $(printf '%q' "refs/heads/$REF"); then
  git switch $(printf '%q' "$REF") >/dev/null
else
  git switch --track -c $(printf '%q' "$REF") $(printf '%q' "origin/$REF") >/dev/null
fi
git pull --ff-only origin $(printf '%q' "$REF")
$REMOTE_SCRIPT_CALL
EOF
)

SSH_COMMAND=(ssh "$HOST" "bash -lc $(printf '%q' "$REMOTE_BOOTSTRAP")")

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%s\n' "$(quote_args "${SSH_COMMAND[@]}")"
  exit 0
fi

"${SSH_COMMAND[@]}"
