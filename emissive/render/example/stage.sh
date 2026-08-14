#!/bin/bash
# Copy the render code to a path the cluster can actually see.
#
# WHY THIS EXISTS. The repository working tree lives on `/local-scratch2`, which
# is a LOCAL disk on the workstation. Compute nodes cannot see it. A job that
# points sbatch at the repo path fails with
#
#   can't open file '/local-scratch2/.../render_emissive.py': [Errno 2] No such file
#
# and that is the first thing anyone running this on solar will hit. Everything
# else in the pipeline reads from `/project`, which is shared, so the code is the
# only piece that has to be moved.
#
# Usage:  bash stage.sh /project/3dlg-hcvc/omages/<somewhere>/render
set -euo pipefail

DEST="${1:?usage: bash stage.sh <cluster-visible destination>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$DEST" in
  /project/*|/cs/*) : ;;
  *) echo "refusing to stage to '$DEST': it must be under /project or /cs, or" >&2
     echo "the compute nodes will not see it either" >&2
     exit 1 ;;
esac

mkdir -p "$DEST"
# merge-copy, never delete: a destination may hold a colleague's outputs
cp -f "$SRC"/*.py "$DEST"/
mkdir -p "$DEST/cameras"
cp -f "$SRC"/cameras/*.json "$DEST/cameras/" 2>/dev/null || true
cp -f "$SRC"/manifest12.json "$DEST"/ 2>/dev/null || true

echo "staged to $DEST"
ls -1 "$DEST"/*.py
