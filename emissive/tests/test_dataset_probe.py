"""Tests for EmisDataset's negative-confirming file probe.

Background. On 2026-08-23 smoke job 248420 died 46 seconds in with "cond.pth
missing" for a file that exists: a cold scan of a large freshly built symlink
farm took an NFS lookup miss, and the scanning node cached the negative. A
multi-day job lost its scheduler slot to a file that was there all along. On the
same day a scan of dataset_direct/excluded_v2 reported 41 of 286 shapes missing
cond.pth and those 41 are genuinely absent. Both look identical to
os.path.exists, so the scan has to be able to tell them apart.

The discriminator: a cached negative dentry suppresses a LOOKUP, it does not
remove a name from a readdir. So the parent's listing is ground truth. A name
absent from the listing is really absent; a name present in the listing whose
lookup fails is the cache artifact.

What must NOT change is the behavior on a real gap, which is what these tests
mostly pin down: missing core slat file skips the sample silently as before,
missing cond.pth or emis_mask.pth raises as before, and the admitted set and its
order are untouched.

Run it (no GPU, no trellis2 needed, the heavy imports are stubbed):
  python emissive/tests/test_dataset_probe.py

Against a real split on the cluster, to check a known real gap end to end:
  python emissive/tests/test_dataset_probe.py \
      --real_split /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct excluded_v2
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def load_trainer_module():
    """Import train_emissive with the heavy, GPU-bound imports stubbed.

    EmisDataset.__init__ touches only os and json; everything the stubs stand in
    for (the sparse transformer, the eval decoders, the HF hub) is used elsewhere
    in the file. Stubbing lets this run on a workstation with no CUDA in about a
    second, which is the difference between a test that gets run and one that
    does not."""
    for name in ("trellis2", "trellis2.modules", "trellis2.modules.sparse",
                 "inference_full", "eval_emissive", "huggingface_hub"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["trellis2"].models = types.SimpleNamespace(from_pretrained=None)
    sys.modules["inference_full"].Gen3DSeg = object
    sys.modules["eval_emissive"].load_eval_models = None
    sys.modules["eval_emissive"].evaluate_split = None
    sys.modules["eval_emissive"].THRS = ()
    sys.modules["huggingface_hub"].hf_hub_download = None
    # EMIS_TRAINER_DIR lets this run against a STAGED copy of the trainer, so the
    # change can be checked against real cluster splits before it is deployed.
    trainer_dir = os.environ.get("EMIS_TRAINER_DIR", os.path.join(REPO, "emissive", "train"))
    sys.path.insert(0, trainer_dir)
    print(f"[test] trainer under test: {os.path.join(trainer_dir, 'train_emissive.py')}")
    import importlib
    return importlib.import_module("train_emissive")


CORE = ["shape_slat.pth", "input_tex_slat.pth", "output_tex_slat.pth"]
ALL_FILES = CORE + ["cond.pth", "emis_mask.pth"]


def make_split(root, sids, omit=None, frac=0.25):
    """A split dir of empty files. omit maps sid -> list of filenames to leave out."""
    omit = omit or {}
    sdir = os.path.join(root, "split")
    os.makedirs(sdir, exist_ok=True)
    for sid in sids:
        d = os.path.join(sdir, sid)
        os.makedirs(d, exist_ok=True)
        for f in ALL_FILES:
            if f in omit.get(sid, []):
                continue
            open(os.path.join(d, f), "w").close()
        if "meta.json" not in omit.get(sid, []):
            json.dump({"emissive_frac": frac}, open(os.path.join(d, "meta.json"), "w"))
    return root


class FailNTimes:
    """os.path.exists that reports a chosen path missing for its first N calls.

    This is the transient lookup miss: the file IS on disk and IS in the parent
    listing, only the stat says otherwise, and only for a while."""

    def __init__(self, target_suffix, n):
        self.target_suffix = target_suffix
        self.n = n
        self.real = os.path.exists
        self.hits = 0

    def __call__(self, path):
        if path.endswith(self.target_suffix):
            self.hits += 1
            if self.hits <= self.n:
                return False
        return self.real(path)


@contextlib.contextmanager
def patched_exists(fake):
    real = os.path.exists
    os.path.exists = fake
    try:
        yield
    finally:
        os.path.exists = real


@contextlib.contextmanager
def captured():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail and not cond else ''}")


def test_transient_cond_miss_recovers(M):
    """A lookup miss on cond.pth whose name IS in the listing must recover, admit
    the sample, and say so out loud."""
    with tempfile.TemporaryDirectory() as tmp:
        make_split(tmp, ["aaa", "bbb"])
        fake = FailNTimes("aaa/cond.pth", 1)
        with patched_exists(fake), captured() as buf:
            ds = M.EmisDataset(tmp, "split", cond_mode="real", require_mask=True)
        out = buf.getvalue()
        check("transient cond.pth miss still admits the sample", len(ds.dirs) == 2,
              f"admitted {len(ds.dirs)}")
        check("recovery is logged loudly", "RECOVERED" in out and "aaa" in out, out[:200])
        check("end-of-scan summary is printed", "scan of 'split'" in out, out[:200])
        check("recovered sample keeps its emissive_frac", ds.fracs == [0.25, 0.25],
              str(ds.fracs))


def test_real_gap_cond_raises(M):
    """cond.pth genuinely absent: the original hard error, and no pretence of a retry."""
    with tempfile.TemporaryDirectory() as tmp:
        make_split(tmp, ["aaa"], omit={"aaa": ["cond.pth"]})
        err = None
        with captured() as buf:
            try:
                M.EmisDataset(tmp, "split", cond_mode="real", require_mask=True)
            except RuntimeError as e:
                err = str(e)
        out = buf.getvalue()
        check("absent cond.pth still raises RuntimeError", err is not None)
        check("error still names the file and the fix",
              err is not None and "cond.pth" in err and "build_dataset.py" in err, str(err))
        check("absent is reported as absent, not as a recovery",
              "RECOVERED" not in out, out[:200])


def test_real_gap_core_skips(M):
    """A missing core slat file is skipped silently, exactly as before."""
    with tempfile.TemporaryDirectory() as tmp:
        make_split(tmp, ["aaa", "bbb", "ccc"], omit={"bbb": ["shape_slat.pth"]})
        with captured():
            ds = M.EmisDataset(tmp, "split", cond_mode="real", require_mask=True)
        check("missing core slat file skips that sample only", len(ds.dirs) == 2,
              f"admitted {len(ds.dirs)}")
        check("skip does not disturb order",
              [os.path.basename(d) for d in ds.dirs] == ["aaa", "ccc"],
              str([os.path.basename(d) for d in ds.dirs]))


def test_lookup_stuck_falls_back_to_original(M):
    """Name in the listing but the lookup never succeeds. Neither story fits, so it
    must not be silently admitted: original behavior stands, reported distinctly."""
    with tempfile.TemporaryDirectory() as tmp:
        make_split(tmp, ["aaa"])
        fake = FailNTimes("aaa/cond.pth", 99)
        err = None
        with patched_exists(fake), captured() as buf:
            try:
                M.EmisDataset(tmp, "split", cond_mode="real", require_mask=True)
            except RuntimeError as e:
                err = str(e)
        out = buf.getvalue()
        check("stuck lookup still raises rather than admitting", err is not None)
        check("stuck lookup is reported as its own outcome", "LOOKUP_STUCK" in out, out[:300])
        check("stuck lookup is not counted as recovered", "RECOVERED" not in out, out[:300])


def test_transient_core_miss_recovers(M):
    """The same recovery on a core slat file: the sample must not be dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        make_split(tmp, ["aaa", "bbb"])
        fake = FailNTimes("bbb/shape_slat.pth", 1)
        with patched_exists(fake), captured() as buf:
            ds = M.EmisDataset(tmp, "split", cond_mode="real", require_mask=True)
        check("transient core miss does not drop the sample", len(ds.dirs) == 2,
              f"admitted {len(ds.dirs)}")
        check("core recovery is logged", "RECOVERED" in buf.getvalue())


def test_transient_meta_miss_keeps_frac(M):
    """meta.json falls back to frac 0.0 when absent, which feeds --emis_oversample
    weights. A transient miss there would silently mis-weight the sample."""
    with tempfile.TemporaryDirectory() as tmp:
        make_split(tmp, ["aaa"], frac=0.75)
        fake = FailNTimes("aaa/meta.json", 1)
        with patched_exists(fake), captured():
            ds = M.EmisDataset(tmp, "split", cond_mode="real", require_mask=True)
        check("transient meta.json miss does not zero the oversample weight",
              ds.fracs == [0.75], str(ds.fracs))


def test_clean_split_unchanged(M):
    """The ordinary case: nothing logged as trouble, everything admitted in order."""
    with tempfile.TemporaryDirectory() as tmp:
        sids = [f"s{i:03d}" for i in range(12)]
        make_split(tmp, sids)
        with captured() as buf:
            ds = M.EmisDataset(tmp, "split", cond_mode="real", require_mask=True)
        out = buf.getvalue()
        check("clean split admits every sample in sorted order",
              [os.path.basename(d) for d in ds.dirs] == sorted(sids))
        check("clean split reports no recoveries and no gaps",
              "RECOVERED" not in out and "LOOKUP_STUCK" not in out, out[:200])


def test_require_mask_and_zero_cond_unchanged(M):
    """The two flags that gate the extra probes still gate them."""
    with tempfile.TemporaryDirectory() as tmp:
        make_split(tmp, ["aaa"], omit={"aaa": ["cond.pth", "emis_mask.pth"]})
        with captured():
            ds = M.EmisDataset(tmp, "split", cond_mode="zero", require_mask=False)
        check("cond=zero with require_mask=False admits a dir lacking both",
              len(ds.dirs) == 1, f"admitted {len(ds.dirs)}")
        err = None
        with captured():
            try:
                M.EmisDataset(tmp, "split", cond_mode="zero", require_mask=True)
            except RuntimeError as e:
                err = str(e)
        check("require_mask=True still raises on an absent emis_mask.pth",
              err is not None and "emis_mask.pth" in err, str(err))


def run_real_split(M, root, split):
    """Point the real scan at a real split. excluded_v2 is the known real-gap case:
    41 of 286 shapes have no cond.pth, absent from their parent listings."""
    print(f"\n=== real split: {root}/{split} ===")
    err = None
    try:
        ds = M.EmisDataset(root, split, cond_mode="real", require_mask=False)
        print(f"  admitted {len(ds.dirs)} samples, no error")
    except RuntimeError as e:
        err = str(e)
        print(f"  RuntimeError: {err[:300]}")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_split", nargs=2, metavar=("DATASET_ROOT", "SPLIT"), default=None)
    args = ap.parse_args()

    M = load_trainer_module()
    print("EmisDataset probe tests")
    for fn in (test_transient_cond_miss_recovers, test_real_gap_cond_raises,
               test_real_gap_core_skips, test_lookup_stuck_falls_back_to_original,
               test_transient_core_miss_recovers, test_transient_meta_miss_keeps_frac,
               test_clean_split_unchanged, test_require_mask_and_zero_cond_unchanged):
        print(f"\n{fn.__name__}:")
        fn(M)

    if args.real_split:
        run_real_split(M, args.real_split[0], args.real_split[1])

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
