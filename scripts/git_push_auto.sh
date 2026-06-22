#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE_NAME="${1:-origin}"
BRANCH_NAME="${2:-$(git symbolic-ref --quiet --short HEAD)}"

if [[ -z "${BRANCH_NAME}" ]]; then
  echo "Unable to infer current branch. Pass it explicitly: scripts/git_push_auto.sh origin <branch>" >&2
  exit 1
fi

ORIGIN_URL="$(git remote get-url "$REMOTE_NAME")"

extract_slug() {
  local url="$1"
  if [[ "$url" =~ github\.com[:/]([^/]+/[^/]+)(\.git)?$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

if ! REPO_SLUG="$(extract_slug "$ORIGIN_URL")"; then
  echo "Unsupported remote URL for auto push: $ORIGIN_URL" >&2
  exit 1
fi

SSH_KEY="${HOME}/.ssh/id_ed25519"
SSH_COMMON_OPTS=(-o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)
if [[ -f "$SSH_KEY" ]]; then
  SSH_COMMON_OPTS+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
fi

attempt_push() {
  local label="$1"
  shift
  echo
  echo "==> Trying ${label}"
  if "$@"; then
    echo "==> Success via ${label}"
    return 0
  fi
  echo "==> Failed via ${label}" >&2
  return 1
}

SSH443_URL="ssh://git@ssh.github.com:443/${REPO_SLUG}.git"
SSH22_URL="git@github.com:${REPO_SLUG}.git"
HTTPS_URL="https://github.com/${REPO_SLUG}.git"

if attempt_push \
  "SSH over 443" \
  env GIT_SSH_COMMAND="ssh ${SSH_COMMON_OPTS[*]}" \
  git push "$SSH443_URL" "HEAD:${BRANCH_NAME}"; then
  exit 0
fi

if attempt_push \
  "SSH over 22" \
  env GIT_SSH_COMMAND="ssh ${SSH_COMMON_OPTS[*]}" \
  git push "$SSH22_URL" "HEAD:${BRANCH_NAME}"; then
  exit 0
fi

if attempt_push \
  "HTTPS (HTTP/1.1)" \
  git -c http.version=HTTP/1.1 push "$HTTPS_URL" "HEAD:${BRANCH_NAME}"; then
  exit 0
fi

echo
echo "All push methods failed for ${REPO_SLUG} on branch ${BRANCH_NAME}." >&2
exit 1
