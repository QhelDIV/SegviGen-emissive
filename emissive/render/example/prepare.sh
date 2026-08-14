#!/bin/bash
# Build a self-contained working directory for the worked example.
#
# CLEAN-ROOM BY CONSTRUCTION: every input is either read from the shared TexVerse
# dataset or written by this script. Nothing reads anyone's personal scratch, so
# the example runs for any account with TexVerse read access.
#
# TWO FILESYSTEM FACTS THIS SCRIPT EXISTS TO HANDLE. Both were found by running
# the example on a compute node rather than on the workstation, and both produce
# a confusing failure rather than a clear one.
#
#   1. TexVerse lives on /cs/3dlg-falas, which is mounted on the WORKSTATION
#      ONLY. Compute nodes cannot see it, so the GLB has to be COPIED to shared
#      storage, not symlinked. A symlink resolves fine where you prepare it and
#      dangles where the render runs, and Blender reports it as
#      "Error: Please select a file", which reads like a bad argument.
#
#   2. The destination therefore has to be under /project or /cs/3dlg-project,
#      which the nodes do see. A working directory on /local-scratch2 or in a
#      home directory fails the same way.
#
# Run this on the WORKSTATION, where the dataset is mounted.
#
# Usage:  bash prepare.sh /project/3dlg-hcvc/omages/<you>/ladder_example [sid]
set -euo pipefail

WORK="${1:?usage: bash prepare.sh <workdir under /project> [sid]}"
SID="${2:-48af42db48c44cd9bfab32bbb057a39c}"
TEXVERSE="${TEXVERSE:-/cs/3dlg-falas/datasets/TexVerse-1K/glbs/glbs_1k}"

case "$WORK" in
  /project/*|/cs/3dlg-project/*) : ;;
  *) echo "refusing to prepare '$WORK': the render runs on a compute node," >&2
     echo "which only sees /project and /cs/3dlg-project. A workdir anywhere" >&2
     echo "else will fail at load time with 'Error: Please select a file'." >&2
     exit 1 ;;
esac

if [ ! -d "$TEXVERSE" ]; then
  echo "TexVerse not found at $TEXVERSE" >&2
  echo "It is mounted on the workstation, not on compute nodes: run this there," >&2
  echo "or set TEXVERSE=<path> if the dataset has moved." >&2
  exit 1
fi

mkdir -p "$WORK/glb" "$WORK/out"

# TexVerse shards its GLBs into subdirectories and names them "<sid>_1024.glb";
# the renderer wants "<glb_dir>/<sid>.glb".
SRC="$(find "$TEXVERSE" -name "${SID}_1024.glb" -print -quit)"
if [ -z "$SRC" ]; then
  echo "could not find ${SID}_1024.glb under $TEXVERSE" >&2
  exit 1
fi
cp -f "$SRC" "$WORK/glb/${SID}.glb"     # copy, not link: see fact 1 above

# The manifest is a list of records; only "sid" is required.
printf '[\n  {"sid": "%s"}\n]\n' "$SID" > "$WORK/manifest.json"

echo "prepared $WORK"
echo "  glb      $WORK/glb/${SID}.glb  (copied from $SRC)"
echo "  manifest $WORK/manifest.json"
echo "  out      $WORK/out"
