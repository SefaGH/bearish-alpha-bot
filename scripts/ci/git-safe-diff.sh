#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BRANCH="main"

git fetch --no-tags origin "$DEFAULT_BRANCH" || true

if git show-ref --verify --quiet "refs/remotes/origin/$DEFAULT_BRANCH"; then
  echo "Diff against origin/$DEFAULT_BRANCH (top 50):"
  git diff "origin/$DEFAULT_BRANCH" HEAD --name-status | head -n 50 || true
else
  echo "origin/$DEFAULT_BRANCH not found. Attempting remote HEAD detect."
  git remote set-head origin -a || true
  ORIGIN_HEAD=$(git symbolic-ref --quiet refs/remotes/origin/HEAD || echo "")
  if [ -n "$ORIGIN_HEAD" ]; then
    HEAD_BRANCH="${ORIGIN_HEAD##*/}"
    git fetch --no-tags origin "$HEAD_BRANCH" || true
    echo "Diff against origin/$HEAD_BRANCH (top 50):"
    git diff "origin/$HEAD_BRANCH" HEAD --name-status | head -n 50 || true
  else
    echo "No remote HEAD could be detected; skipping diff."
  fi
fi
