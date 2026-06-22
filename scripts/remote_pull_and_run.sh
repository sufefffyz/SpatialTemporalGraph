#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/remote_pull_and_run.sh [options] -- <command> [args...]

Options:
  --repo-dir PATH        Repo root on the remote machine. Defaults to the parent of this script.
  --ref REF              Git branch to pull before running. Defaults to the current branch.
  --project NAME         Subproject directory to run in, e.g. BasicTS or urban-traffic-benchmark.
  --conda-env NAME       Conda environment to run the command in.
  --logs-dir PATH        Directory used to store logs and metadata.
  --name NAME            Optional label included in the run directory name.
  --mode fg|bg           Run in foreground or background. Default: fg.
  --allow-dirty          Allow a dirty git worktree on the remote machine.
  --skip-pull            Skip git fetch/switch/pull and only run the command.
  -h, --help             Show this help message.

Examples:
  bash scripts/remote_pull_and_run.sh \
    --project urban-traffic-benchmark \
    --conda-env graph_ml \
    --ref 0.5.8 \
    -- bash reproduce_experiments_models_dataset_path.sh

  bash scripts/remote_pull_and_run.sh \
    --project urban-traffic-benchmark \
    --conda-env graph_ml \
    --mode bg \
    -- python run_single_experiment.py --name debug --dataset city_traffic_m_speed --metric MAE
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
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REPO_DIR="$DEFAULT_REPO_DIR"
REF=""
PROJECT=""
CONDA_ENV=""
LOGS_DIR="${HOME}/.spatial-temporal-graph-runs"
RUN_NAME=""
MODE="fg"
ALLOW_DIRTY=0
SKIP_PULL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
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
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --skip-pull)
      SKIP_PULL=1
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
  echo "A command must be provided after --." >&2
  usage >&2
  exit 1
fi

if [[ -z "$PROJECT" ]]; then
  echo "--project is required." >&2
  usage >&2
  exit 1
fi

if [[ "$MODE" != "fg" && "$MODE" != "bg" ]]; then
  echo "--mode must be either fg or bg." >&2
  exit 1
fi

COMMAND=( "$@" )

git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1

if [[ -z "$REF" ]]; then
  REF="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
fi

if [[ "$ALLOW_DIRTY" -ne 1 ]]; then
  if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    echo "Remote repo has uncommitted changes. Commit/stash them or use --allow-dirty." >&2
    exit 1
  fi
fi

if [[ "$SKIP_PULL" -ne 1 ]]; then
  git -C "$REPO_DIR" fetch origin
  if git -C "$REPO_DIR" show-ref --verify --quiet "refs/heads/$REF"; then
    git -C "$REPO_DIR" switch "$REF" >/dev/null
  else
    git -C "$REPO_DIR" switch --track -c "$REF" "origin/$REF" >/dev/null
  fi
  git -C "$REPO_DIR" pull --ff-only origin "$REF"
fi

WORKDIR="$REPO_DIR/$PROJECT"
if [[ ! -d "$WORKDIR" ]]; then
  echo "Project directory not found: $WORKDIR" >&2
  exit 1
fi

mkdir -p "$LOGS_DIR"

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
SAFE_PROJECT="${PROJECT//\//_}"
SAFE_NAME=""
if [[ -n "$RUN_NAME" ]]; then
  SAFE_NAME="_${RUN_NAME// /_}"
fi
RUN_DIR="$LOGS_DIR/${TIMESTAMP}_${SAFE_PROJECT}${SAFE_NAME}"
mkdir -p "$RUN_DIR"

STDOUT_LOG="$RUN_DIR/stdout.log"
COMMAND_TXT="$RUN_DIR/command.txt"
META_TXT="$RUN_DIR/meta.txt"

COMMAND_STRING="$(quote_args "${COMMAND[@]}")"
INNER_COMMAND="cd $(printf '%q' "$WORKDIR") && exec ${COMMAND_STRING}"

if [[ -n "$CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available in PATH on the remote machine." >&2
    exit 1
  fi
  LAUNCHER=(conda run --no-capture-output -n "$CONDA_ENV" bash -lc "$INNER_COMMAND")
else
  LAUNCHER=(bash -lc "$INNER_COMMAND")
fi

{
  echo "timestamp=$TIMESTAMP"
  echo "repo_dir=$REPO_DIR"
  echo "project=$PROJECT"
  echo "ref=$REF"
  echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "conda_env=$CONDA_ENV"
  echo "mode=$MODE"
  echo "workdir=$WORKDIR"
} >"$META_TXT"

printf '%s\n' "$COMMAND_STRING" >"$COMMAND_TXT"

echo "Run directory: $RUN_DIR"
echo "Command: $COMMAND_STRING"

if [[ "$MODE" == "fg" ]]; then
  "${LAUNCHER[@]}" 2>&1 | tee "$STDOUT_LOG"
else
  nohup "${LAUNCHER[@]}" >"$STDOUT_LOG" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" >"$RUN_DIR/pid"
  echo "Started background job with PID $PID"
  echo "Logs: $STDOUT_LOG"
  echo "Tail with: tail -f $(printf '%q' "$STDOUT_LOG")"
fi
