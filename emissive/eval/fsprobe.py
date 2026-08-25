"""Ask the filesystem whether a file exists, and confirm a "no" before believing it.

Why this exists. On 2026-08-23 training smoke job 248420 died 46 seconds in with
"cond.pth missing" for a file that was there all along: a cold scan of a large
freshly built symlink farm took an NFS lookup miss and the scanning node cached
the negative. The next day the same thing killed eval job 248718 fourteen minutes
in, mid-generation, after five draws of real GPU work. Separately, a scan of
dataset_direct/excluded_v2 reported 41 of 286 shapes missing cond.pth, and those
41 really are missing. A bare os.path.exists cannot tell the two apart, and
neither can a retry that only re-stats: it spins on the real gap and then fails
anyway, turning a defect into a slower defect.

The discriminator is the parent's listing. A cached negative dentry suppresses a
LOOKUP; it cannot remove a name from a readdir. So:

  absent from the listing   -> really missing. No retry. The caller's original
                               behavior stands.
  present in the listing    -> the lookup was wrong. Re-listing the parent has
                               already dropped the cached negative, so one
                               re-check settles it. Logged, never swallowed.
  present but still failing -> fits neither story. Gets its own name rather than
                               hiding in either bucket, and falls back to the
                               caller's original behavior.

Placement note. This lives beside eval_emissive.py rather than in a package of
its own because train_emissive.py and every consumer of eval_emissive already
put this directory on sys.path in order to import eval_emissive at all, so
`from fsprobe import probe_exists` needs no new path plumbing anywhere. It
deliberately imports nothing beyond the standard library: that keeps it
importable by a test on a machine with no CUDA and no trellis2, so the function
under test is the real one rather than a stub.
"""
import os


def new_stats():
    """A fresh accumulator for one scan. Pass it to every probe_exists in that
    scan, then hand it to summary_line()."""
    return {"recovered": [], "lookup_stuck": [], "absent": []}


def probe_exists(path, stats=None):
    """os.path.exists, except a negative is CONFIRMED against the parent listing.

    stats is optional: without it the per-file lines still print, there is just
    nothing to summarize at the end."""
    if os.path.exists(path):
        return True
    if stats is None:
        stats = new_stats()
    parent, name = os.path.split(path)
    try:
        names = os.listdir(parent)      # also drops any cached negative for parent
    except OSError:
        stats["absent"].append(path)
        return False
    if name not in names:
        stats["absent"].append(path)
        return False
    if os.path.exists(path):
        stats["recovered"].append(path)
        print(f"[fs] RECOVERED {path}: the first existence check said missing, but the "
              f"name is in its parent's listing and re-checking after re-listing found "
              f"it. Transient lookup miss.", flush=True)
        return True
    stats["lookup_stuck"].append(path)
    print(f"[fs] LOOKUP_STUCK {path}: the name IS in its parent's listing but the "
          f"existence check keeps failing after a re-list. Treating it as missing.",
          flush=True)
    return False


def summary_line(stats, what):
    """One line per scan, printed even when nothing went wrong, so a clean scan
    says so instead of staying silent."""
    return (f"[fs] {what}: {len(stats['recovered'])} transient lookup misses recovered, "
            f"{len(stats['lookup_stuck'])} lookups stuck, "
            f"{len(stats['absent'])} files absent from their parent listing")
