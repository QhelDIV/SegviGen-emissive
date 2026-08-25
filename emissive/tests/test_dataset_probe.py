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

Run it. No GPU and no trellis2 needed; those imports are stubbed. Real torch is
used when the interpreter has it, and stubbed with a notice when it does not, so
plain `python3` works too:
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
# fsprobe imports nothing but the standard library, on purpose, so these tests
# exercise the REAL probe rather than a stub of it
sys.path.insert(0, os.path.join(REPO, "emissive", "eval"))
import fsprobe


def stub_torch_if_missing():
    """Use the real torch when the interpreter has it; stub it when it does not.

    train_emissive.py imports torch at module scope, before any of the stubs above
    can matter, so on an interpreter without torch this suite died at line 37
    rather than running. EmisDataset.__init__ touches only os and json, and torch
    appears in the part under test solely as the base class, so a stub is enough
    to exercise the scan. Preferring the real torch when it is importable keeps
    the usual run honest; the stub only rescues the case that would otherwise be
    no run at all, and it says so rather than pretending."""
    try:
        import torch  # noqa: F401
        return False
    except ImportError:
        pass
    print("[test] NOTE: torch is not importable here, so it is being stubbed. "
          "EmisDataset's scan does not use torch, but this run does NOT check "
          "anything torch-dependent. Run under an interpreter with torch for that.")
    torch_mod = types.ModuleType("torch")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = object
    utils_mod = types.ModuleType("torch.utils")
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object          # FlowStep subclasses it at module scope
    parallel_mod = types.ModuleType("torch.nn.parallel")
    parallel_mod.DistributedDataParallel = object
    nn_mod.parallel = parallel_mod
    torch_mod.nn = nn_mod
    dist_mod = types.ModuleType("torch.distributed")
    torch_mod.distributed = dist_mod
    for name, mod in (("torch", torch_mod), ("torch.utils", utils_mod),
                      ("torch.utils.data", data_mod), ("torch.nn", nn_mod),
                      ("torch.nn.parallel", parallel_mod),
                      ("torch.distributed", dist_mod)):
        sys.modules[name] = mod
    return True


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
    stub_torch_if_missing()
    sys.modules["trellis2"].models = types.SimpleNamespace(from_pretrained=None)
    sys.modules["inference_full"].Gen3DSeg = object
    sys.modules["inference_full"].Sampler = object
    sys.modules["inference_full"].slat_to_glb = None
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




# ---------------------------------------------------------------- fsprobe itself
def test_probe_absent_from_listing(M):
    """The real-gap case: no retry, and it is classified as absent."""
    with tempfile.TemporaryDirectory() as tmp:
        st = fsprobe.new_stats()
        with captured() as buf:
            ok = fsprobe.probe_exists(os.path.join(tmp, "nope.pth"), st)
        check("absent file probes False", ok is False)
        check("absent file is classified absent", len(st["absent"]) == 1 and not st["recovered"])
        check("absent file logs nothing noisy", buf.getvalue() == "", buf.getvalue()[:120])


def test_probe_recovers_transient(M):
    """Name IS in the listing, first stat lies: recovered, and said out loud."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "there.pth")
        open(target, "w").close()
        st = fsprobe.new_stats()
        with patched_exists(FailNTimes("there.pth", 1)), captured() as buf:
            ok = fsprobe.probe_exists(target, st)
        check("transient miss probes True", ok is True)
        check("transient miss is classified recovered",
              len(st["recovered"]) == 1 and not st["absent"])
        check("transient miss prints RECOVERED", "RECOVERED" in buf.getvalue())


def test_probe_lookup_stuck(M):
    """Name in the listing, stat never recovers: its own class, treated as missing."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "there.pth")
        open(target, "w").close()
        st = fsprobe.new_stats()
        with patched_exists(FailNTimes("there.pth", 99)), captured() as buf:
            ok = fsprobe.probe_exists(target, st)
        check("stuck lookup probes False", ok is False)
        check("stuck lookup is its own class",
              len(st["lookup_stuck"]) == 1 and not st["absent"] and not st["recovered"])
        check("stuck lookup prints LOOKUP_STUCK", "LOOKUP_STUCK" in buf.getvalue())


def test_probe_missing_parent(M):
    """A directory that is not there at all must not raise out of the probe."""
    st = fsprobe.new_stats()
    with captured():
        ok = fsprobe.probe_exists("/definitely/not/a/real/path/x.pth", st)
    check("probe on a missing parent returns False rather than raising", ok is False)
    check("probe on a missing parent counts as absent", len(st["absent"]) == 1)


def test_probe_works_without_stats(M):
    """stats is optional; callers that do not aggregate still get the loud line."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "there.pth")
        open(target, "w").close()
        with patched_exists(FailNTimes("there.pth", 1)), captured() as buf:
            ok = fsprobe.probe_exists(target)
        check("probe without stats still recovers", ok is True)
        check("probe without stats still logs", "RECOVERED" in buf.getvalue())


# ------------------------------------------------------------------- eval path
def test_eval_path_has_no_bare_exists(M):
    """Regression guard. Every one of these four sites cost a job or corrupted a
    number: cond.pth (killed eval job 248718 fourteen minutes in, mid-generation),
    output_tex_slat.pth (silently drops a sample and changes the IoU denominator),
    emis_mask.pth and meta.json (silently mis-stratify). A new bare os.path.exists
    in this file is how the next one gets in."""
    src = open(os.path.join(REPO, "emissive", "eval", "eval_emissive.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    check("eval_emissive.py has no bare os.path.exists left",
          "os.path.exists" not in code,
          "found: " + str([l.strip() for l in code.splitlines() if "os.path.exists" in l]))
    check("eval_emissive.py imports the shared probe", "from fsprobe import probe_exists" in src)


def test_trainer_uses_shared_probe(M):
    """The trainer must not carry its own copy; one definition, one behavior."""
    src = open(os.path.join(REPO, "emissive", "train", "train_emissive.py")).read()
    check("trainer imports probe from fsprobe", "from fsprobe import probe_exists" in src)
    check("trainer defines no second copy of probe_exists",
          "def probe_exists" not in src)
    # strip comments first: the scan carries a comment that names os.path.exists to
    # explain why it is gone, and a prose mention is not a call site
    scan = src.split("class EmisDataset")[1].split("def __len__")[0]
    code = "\n".join(l.split("#")[0] for l in scan.splitlines())
    check("trainer scan has no bare os.path.exists call", "os.path.exists" not in code,
          "found: " + str([l.strip() for l in code.splitlines() if "os.path.exists" in l]))





def load_eval_module():
    """Import eval_emissive with the same stubs. This is the test that would have
    caught a broken sibling import: `from fsprobe import ...` inside eval_emissive
    resolves only because every consumer puts emissive/eval on sys.path, and that
    is an assumption worth pinning rather than reasoning about."""
    load_trainer_module()          # installs the stubs (incl. an eval_emissive stub)
    import importlib
    # drop the stub so the REAL eval_emissive is imported. train_emissive is already
    # cached with the names it needed, so replacing the entry now affects nothing else.
    sys.modules.pop("eval_emissive", None)
    return importlib.import_module("eval_emissive")


def test_eval_module_imports_with_shared_probe(M):
    """eval_emissive must import, and must be using the shared probe."""
    try:
        EE = load_eval_module()
    except Exception as e:
        check("eval_emissive imports", False, f"{type(e).__name__}: {e}")
        return
    check("eval_emissive imports", True)
    check("eval_emissive got probe_exists from fsprobe",
          getattr(EE, "probe_exists", None) is fsprobe.probe_exists)
    import inspect
    check("eval_sample takes an optional stats kwarg",
          inspect.signature(EE.eval_sample).parameters.get("stats") is not None
          and inspect.signature(EE.eval_sample).parameters["stats"].default is None)
    check("bucket_frac_for takes an optional stats kwarg",
          inspect.signature(EE.bucket_frac_for).parameters.get("stats") is not None)


def test_eval_bucket_frac_recovers_transient_meta(M):
    """An eval-path site exercised for real: a transient miss on meta.json used to
    give frac 0.0 and silently mis-stratify the sample."""
    try:
        EE = load_eval_module()
    except Exception as e:
        check("eval_emissive imports for bucket_frac test", False, str(e))
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "aaa")
        os.makedirs(d)
        json.dump({"emissive_frac": 0.6}, open(os.path.join(d, "meta.json"), "w"))
        with patched_exists(FailNTimes("aaa/meta.json", 1)), captured() as buf:
            frac, fallback = EE.bucket_frac_for(d, "face")
        check("transient meta.json miss no longer zeroes the eval bucket frac",
              abs(frac - 0.6) < 1e-9, f"got {frac}")
        check("eval-side recovery is logged", "RECOVERED" in buf.getvalue())

    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "bbb")
        os.makedirs(d)
        with captured():
            frac, fallback = EE.bucket_frac_for(d, "face")
        check("genuinely absent meta.json still falls back to 0.0",
              frac == 0.0, f"got {frac}")



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
               test_clean_split_unchanged, test_require_mask_and_zero_cond_unchanged,
               test_probe_absent_from_listing, test_probe_recovers_transient,
               test_probe_lookup_stuck, test_probe_missing_parent,
               test_probe_works_without_stats,
               test_eval_path_has_no_bare_exists, test_trainer_uses_shared_probe,
               test_eval_module_imports_with_shared_probe,
               test_eval_bucket_frac_recovers_transient_meta):
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
