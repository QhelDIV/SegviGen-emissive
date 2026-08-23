#!/bin/bash
# Record WHICH trainer a run actually executed.
#
# Why: SLURM copies the sbatch at submit time, but the trainer is an ordinary .py
# read from the shared deploy at job START, so a queued job runs whatever is
# committed when it begins rather than when it was submitted. Two trainer commits
# landed under one queued run on 2026-08-23 before this was noticed. Git holds the
# bytes; what needs pinning is the FACT of which bytes ran.
#
# The deploy is NOT a git checkout, so this does not assume deploy == canonical.
# It hashes the deployed file, then resolves that blob against the canonical
# repo's history and says plainly whether it matched a commit or not.
#
# usage: capture_trainer_version.sh <out_dir> <job_id> [label]
set -euo pipefail
OUT_DIR=$1
JOB_ID=$2
LABEL=${3:-run}
DEPLOY=/cs/3dlg-jupiter-project/lightgen/segvigen_emissive
CANON=/cs/3dlg-falas/project/omages/lightgen/segvigen_emissive
REL=emissive/train/train_emissive.py
F=$DEPLOY/$REL
OUT=$OUT_DIR/TRAINER_AT_START.txt

{
  echo "# Trainer actually executed by this run"
  echo "captured_at:   $(date -Is)"
  echo "job_id:        $JOB_ID"
  echo "label:         $LABEL"
  echo "file:          $F"
  echo "md5:           $(md5sum "$F" | awk '{print $1}')"
  echo "lines:         $(wc -l < "$F")"
  echo "mtime:         $(stat -c %y "$F")"
  BLOB=$(cd "$CANON" && git hash-object "$F")
  echo "git_blob:      $BLOB"
  MATCH=""
  for c in $(cd "$CANON" && git log --format=%H --all -- "$REL"); do
    if [ "$(cd "$CANON" && git rev-parse "$c:$REL" 2>/dev/null)" = "$BLOB" ]; then
      MATCH=$c
      break
    fi
  done
  if [ -n "$MATCH" ]; then
    echo "git_commit:    $(cd "$CANON" && git log --format='%h %ad %s' --date=iso -1 "$MATCH")"
    echo "resolution:    the deployed file is byte-identical to this commit's version"
  else
    echo "git_commit:    UNRESOLVED"
    echo "resolution:    WARNING the deployed file matches NO commit in the canonical"
    echo "               repo for this path. It is uncommitted or locally modified."
  fi
  echo "canonical_head: $(cd "$CANON" && git log --format='%h %s' -1)"
  echo
  echo "# Caveat: captured shortly AFTER the job started, not atomically at start."
  echo "# Accurate only while trainer commits to the deploy are frozen, which they"
  echo "# were for this run by the team lead's instruction. The self-pinning pattern"
  echo "# in README_RUN_PINNING.md removes the need for this caveat on future runs."
} > "$OUT"

echo "TRAINER_CAPTURED $LABEL job=$JOB_ID -> $OUT"
grep -E "^(md5|git_commit|resolution):" "$OUT"
