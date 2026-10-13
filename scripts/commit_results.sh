#!/usr/bin/env bash
set -euo pipefail
branch="${DEFAULT_BRANCH:-$(git symbolic-ref --short HEAD)}"
git check-ref-format --branch "$branch" >/dev/null
paths=()
for path in state reports; do
  if [[ -d "$path" ]]; then paths+=("$path"); fi
done
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
