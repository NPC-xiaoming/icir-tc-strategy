#!/usr/bin/env bash
set -euo pipefail
branch="${DEFAULT_BRANCH:-$(git symbolic-ref --short HEAD)}"
git check-ref-format --branch "$branch" >/dev/null
paths=()
if [[ "${CACHE_ONLY:-0}" == "1" ]]; then
  for path in state/market_history.parquet state/market_calendar.parquet; do
    if [[ -f "$path" ]]; then paths+=("$path"); fi
  done
else
  for path in state reports; do
    if [[ -d "$path" ]]; then paths+=("$path"); fi
  done
fi
if [[ ${#paths[@]} -eq 0 ]]; then
  echo 'No state or reports to save.'
  exit 0
fi
git add -- "${paths[@]}"
if git diff --cached --quiet; then
  echo 'No changed state or reports.'
  exit 0
fi
git config user.name 'github-actions[bot]'
git config user.email 'github-actions[bot]@users.noreply.github.com'
git commit -m "run: save paper ledger and reports $(date -u +%Y-%m-%d)"
# Never force-push or merge conflicting portfolio states automatically.
if ! git push origin "HEAD:$branch"; then
  echo 'Persistence failed. Keep the recovery artifact; reconcile against the latest remote ledger before retrying.' >&2
  exit 1
fi
