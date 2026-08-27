#!/usr/bin/env python3
"""Publish web/_preview/live_eval_mock/ (the live_eval redesign MOCKUP) to
the aspis web root by merge-copy. Mirrors live_eval/publish.py but targets
its own destination directory -- this is review material, not the live page,
and must never collide with the every-ten-minutes loop rebuilding the real
one.

Live URL: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/live_eval_mock/

Run: /cs/3dlg-falas/project/omages/lightgen/segvigen_emissive/xgconsole/.venv_console/bin/python publish.py
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
WWW = "/project/3dlg-hcvc/omages/www/yanxg/lightgen"
DEST = os.path.join(WWW, "_preview", "live_eval_mock")

SKIP_DIRS = {"__pycache__", "_snapshot"}
SKIP_EXT = (".pyc", ".py", ".sh", ".sbatch", ".log", ".md")
KEEP_FILES = {"index.html"}


def main():
    os.makedirs(DEST, exist_ok=True)
    n = 0
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, HERE)
        target = DEST if rel == "." else os.path.join(DEST, rel)
        os.makedirs(target, exist_ok=True)
        for f in files:
            if f not in KEEP_FILES and f.endswith(SKIP_EXT):
                continue
            src, dst = os.path.join(root, f), os.path.join(target, f)
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                continue
            shutil.copy2(src, dst)
            n += 1
    shutil.copytree(os.path.join(WEB, "assets"), os.path.join(WWW, "assets"), dirs_exist_ok=True)
    print(f"published {n} file(s) -> {DEST}")


if __name__ == "__main__":
    main()
