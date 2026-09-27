#!/usr/bin/env bash
# Reproduce the paper from the two inputs configured in configs/paths.env.
#   scripts/reproduce.sh all        # everything
#   scripts/reproduce.sh extract    # one stage (see `python -m rbb.pipeline -h`)
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f configs/paths.env ]]; then
  set -a
  source configs/paths.env
  set +a
else
  echo "configs/paths.env not found; copy configs/paths.env.example and edit it." >&2
  exit 1
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python -m rbb.pipeline "${@:-all}"
