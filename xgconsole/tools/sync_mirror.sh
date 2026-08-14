#!/bin/bash
# Durability mirror: copy the PRECIOUS, hard-to-reproduce lightgen files to
# shared NFS so a failure of this workstation (cs-3dlg-25) can't lose work.
#
# WHY: the working tree lives on /local-scratch2 (local ext4 -> dies with the
# machine). /project/3dlg-hcvc is NFS on cs-saturn (survives). This is a
# SNAPSHOT, not live sync -- re-run after substantial changes.
#
# Excluded = reproducible bulk (venvs, renders, decoded meshes, eval dumps,
# pycache). Rebuild those with `uv sync` or re-running decode/eval jobs.
# The heavy datasets + checkpoints already live on the cluster
# (/3dlg-jupiter-project/lightgen) -- not this machine, not this mirror.
#
# Usage: bash tools/sync_mirror.sh            # mirror working tree + .claude
#        bash tools/sync_mirror.sh --dry      # preview only
set -euo pipefail

SRC=/local-scratch2/xya120/studio/misc/lightgen
DEST=/project/3dlg-hcvc/omages/lightgen
CLAUDE_SRC=/localhome/xya120/.claude
CLAUDE_DEST="$DEST/_claude_backup"

DRY=""
[ "${1:-}" = "--dry" ] && DRY="-n"

echo "=== mirror working tree -> $DEST ==="
rsync -a $DRY --delete-excluded --info=stats2 \
  --exclude='.venv/' --exclude='.venv_console/' \
  --exclude='**/__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  --exclude='vis_data/' --exclude='pred_glb/' \
  --exclude='dump_w1ema/' --exclude='dump_w5ema/' \
  "$SRC/" "$DEST/" | grep -E 'Number of files:|Total file size|transferred' || true

echo "=== mirror ~/.claude authored bits -> $CLAUDE_DEST (NO secrets/transcripts) ==="
rsync -a $DRY --info=stats2 \
  --exclude='.credentials.json' \
  --exclude='projects/**/*.jsonl' --exclude='history.jsonl' \
  --exclude='mcpb/' --exclude='file-history/' --exclude='plugins/' \
  --exclude='cache/' --exclude='paste-cache/' --exclude='shell-snapshots/' \
  --exclude='session-env/' --exclude='telemetry/' --exclude='backups/' \
  --exclude='tasks/' --exclude='jobs/' --exclude='teams/' \
  --exclude='usage_cache.json' --exclude='**/__pycache__/' \
  "$CLAUDE_SRC/" "$CLAUDE_DEST/" | grep -E 'Number of files:|Total file size|transferred' || true

echo "=== done. mirror size: $(du -sh "$DEST" 2>/dev/null | cut -f1) ==="
