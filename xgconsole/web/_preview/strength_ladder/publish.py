#!/usr/bin/env python3
"""Publish web/_preview/strength_ladder/ to the aspis web root by MERGE-COPY.

Never rmtree: the web root's assets/ is shared with other generators, and a
sibling page's directory must survive a republish of this one.

Live URL: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/strength_ladder/

Run: /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/strength_ladder/publish.py
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
WWW = "/project/3dlg-hcvc/omages/www/yanxg/lightgen"
DEST = os.path.join(WWW, "_preview", "strength_ladder")

SKIP = {"__pycache__", "build.py", "publish.py", "prepare_assets.py",
        "measurements.json"}


def main():
    os.makedirs(DEST, exist_ok=True)
    n = 0
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in SKIP]
        rel = os.path.relpath(root, HERE)
        target = DEST if rel == "." else os.path.join(DEST, rel)
        os.makedirs(target, exist_ok=True)
        for f in files:
            if f in SKIP or f.endswith(".pyc"):
                continue
            shutil.copy2(os.path.join(root, f), os.path.join(target, f))
            n += 1

    # the shared assets directory: merge-copy, never replace
    shutil.copytree(os.path.join(WEB, "assets"), os.path.join(WWW, "assets"),
                    dirs_exist_ok=True)
    print(f"published {n} files -> {DEST}")


if __name__ == "__main__":
    main()
