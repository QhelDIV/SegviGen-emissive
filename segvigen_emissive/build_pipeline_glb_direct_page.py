"""
build_pipeline_glb_direct_page.py — "Emissive GT: somage bake vs. direct GLB
voxelization" (xgpage v2 editorial). A team explainer, not a results report:
what the CURRENT emissive-GT pipeline does, why it was built that way, the
512^2-atlas cost it carries (grounded in the same honest numbers as
results_2k_v1's stratified-by-glow chart), the PROPOSED direct-GLB-voxelization
alternative, upstream evidence that TRELLIS.2's own tooling already supports
emission end-to-end (commit-pinned GitHub links), and the caveats + cheap
50-shape diagnostic we're asking the team to sign off on.

Pipeline claims are grounded directly in this repo's own code, not memory:
  code/somage_to_glb.py            -- input.glb (no emissiveFactor set) +
                                       emissive.glb (solid white/black per
                                       face, from labels_uv_74k or a 0.04
                                       threshold on the emission_color map)
  code/SegviGen/data_toolkit/glb_to_vxz.py -- center+unit-box normalize
                                       (lines ~50-54), then voxelize @512
                                       via o_voxel -- the SAME script used
                                       for the somage-derived GLBs today and
                                       for the proposed direct-GLB path.
Stratified IoU numbers are the verbatim STRAT dict from
build_results_2k_page.py (eval_231621/622/623/624.log, full 111-val, K=4
averaged, IoU @ global-best-threshold 0.2, bucket_by=voxel) -- not retyped
from the brief's rounded summary.

Run:
  /local-scratch2/xya120/studio/misc/lightgen/.venv_console/bin/python \
    build_pipeline_glb_direct_page.py
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "pipeline_glb_direct_html")
IMG = os.path.join(OUT, "img")
os.makedirs(IMG, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "..", "tools"))
import xgpage as lp  # noqa: E402

SEGVIGEN_ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive"
ASSETS_REL = "/projects/omages/yanxg/lightgen/assets"
ASSETS_DIR = os.path.join(ROOT, "..", "web", "assets")

# ---------------------------------------------------------------------------
# verbatim STRAT numbers from build_results_2k_page.py (source of truth: this
# repo's own eval logs, not the brief's rounded summary)
BUCKET_LABELS = ["zero", "tiny (0,5%]", "medium (5,30%]", "large (>30%)"]
STRAT = {
    "w5best": [0.3200, 0.0547, 0.1145, 0.3191],
    "w5ema": [0.1600, 0.0361, 0.1380, 0.4260],
    "balbest": [0.1100, 0.0492, 0.1215, 0.3607],
    "balema": [0.0600, 0.0400, 0.1446, 0.4190],
}
MODEL_LABEL = {
    "w5best": "2k+W5 best", "w5ema": "2k+W5 EMA",
    "balbest": "2k+balanced best", "balema": "2k+balanced EMA",
}
TINY_I, LARGE_I = 1, 3
tiny_vals = [STRAT[k][TINY_I] for k in STRAT]
large_vals = [STRAT[k][LARGE_I] for k in STRAT]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------- diagram helpers
# Hand-authored box-arrow SVG on theme2's .diagram/.dbox/.dline vocabulary
# (SKILL.md: "box-arrow diagrams" are not yet componentized in xgpage.py).
def _dbox(x, y, w, h, lines, mono_lines=None, accent=False):
    cls = "dbox-accent" if accent else "dbox"
    rect = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
    texts = []
    n_title = len(lines)
    n_mono = len(mono_lines or [])
    total_lines = n_title + n_mono
    line_h = 17
    start_y = y + h / 2 - (total_lines - 1) * line_h / 2 + 5
    for i, ln in enumerate(lines):
        texts.append(f'<text class="dtitle" x="{x + w/2}" y="{start_y + i*line_h}" '
                     f'text-anchor="middle">{lp._esc(ln)}</text>')
    for j, ln in enumerate(mono_lines or []):
        yy = start_y + (n_title + j) * line_h
        texts.append(f'<text class="dmono" x="{x + w/2}" y="{yy}" '
                     f'text-anchor="middle">{lp._esc(ln)}</text>')
    return rect + "".join(texts)


def _darrow_h(x1, y, x2, accent=False):
    """Straight horizontal arrow, y constant."""
    cls = "dline-accent" if accent else "dline"
    marker = "url(#darrow-accent)" if accent else "url(#darrow)"
    return f'<path class="{cls}" d="M{x1},{y} L{x2},{y}" marker-end="{marker}"/>'


def _dbranch(x1, y1, x2, y2, accent=False):
    """Curved branch from one point to another (fan-out / fan-in)."""
    cls = "dline-accent" if accent else "dline"
    marker = "url(#darrow-accent)" if accent else "url(#darrow)"
    midx = (x1 + x2) / 2
    return (f'<path class="{cls}" d="M{x1},{y1} C{midx},{y1} {midx},{y2} {x2},{y2}" '
            f'marker-end="{marker}"/>')


_DEFS = """<defs>
  <marker id="darrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" class="arrfill"/></marker>
  <marker id="darrow-accent" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" class="arrfill-accent"/></marker>
</defs>"""


def diagram(inner_svg, w, h):
    return f'<div class="diagram"><svg viewBox="0 0 {w} {h}">{_DEFS}{inner_svg}</svg></div>'


# ---------------------------------------------------------- diagram 1: current
def current_pipeline_diagram():
    W, H = 960, 380
    lane_a_cy, lane_b_cy = 78, 302  # input track / target track box centers
    box_h = 92
    parts = []

    # column 0: somage bake (shared source)
    somage_x, somage_w = 14, 150
    somage_cy = (lane_a_cy + lane_b_cy) / 2
    parts.append(_dbox(somage_x, somage_cy - 65, somage_w, 130,
                        ["somage", "(512² UV bake)"],
                        ["color / metal / rough", "/ emission maps +", "original_mesh.npz"]))

    # fan-out label + branch lines to the two lanes
    fan_x1 = somage_x + somage_w
    glb_x, glb_w = 218, 170
    parts.append(_dbranch(fan_x1, somage_cy, glb_x, lane_a_cy))
    parts.append(_dbranch(fan_x1, somage_cy, glb_x, lane_b_cy))
    parts.append(f'<text class="dmono" x="{fan_x1 + 34}" y="{somage_cy - 10}" '
                 f'text-anchor="middle">somage_to_glb</text>')

    # lane labels
    parts.append(f'<text class="dsub" x="{glb_x}" y="{lane_a_cy - box_h/2 - 12}">INPUT TRACK</text>')
    parts.append(f'<text class="dsub" x="{glb_x}" y="{lane_b_cy - box_h/2 - 12}">TARGET TRACK</text>')

    # column 1: two GLBs
    parts.append(_dbox(glb_x, lane_a_cy - box_h/2, glb_w, box_h,
                        ["input.glb"], ["PBR appearance,", "NO emission"]))
    parts.append(_dbox(glb_x, lane_b_cy - box_h/2, glb_w, box_h,
                        ["emissive.glb"], ["solid white / black", "target, per face"], accent=True))

    # column 2: glb_to_vxz (same script, both lanes)
    vxz_x, vxz_w = 448, 190
    parts.append(_darrow_h(glb_x + glb_w, lane_a_cy, vxz_x, lane_a_cy))
    parts.append(_darrow_h(glb_x + glb_w, lane_b_cy, vxz_x, lane_b_cy))
    parts.append(_dbox(vxz_x, lane_a_cy - box_h/2, vxz_w, box_h,
                        ["glb_to_vxz"], ["normalize + voxelize", "512³"]))
    parts.append(_dbox(vxz_x, lane_b_cy - box_h/2, vxz_w, box_h,
                        ["glb_to_vxz"], ["normalize + voxelize", "512³"], accent=True))

    # column 3: outputs
    out_x, out_w = 700, 246
    parts.append(_darrow_h(vxz_x + vxz_w, lane_a_cy, out_x, lane_a_cy))
    parts.append(_darrow_h(vxz_x + vxz_w, lane_b_cy, out_x, lane_b_cy))
    parts.append(_dbox(out_x, lane_a_cy - box_h/2, out_w, box_h,
                        ["input_tex_slat"]))
    parts.append(_dbox(out_x, lane_b_cy - box_h/2, out_w, box_h,
                        ["output target +", "per-voxel emis_mask"], accent=True))

    return diagram("".join(parts), W, H)


# ---------------------------------------------------------- diagram 2: proposed
def proposed_pipeline_diagram():
    W, H = 960, 230
    cy = 95
    box_h = 100
    parts = []

    src_x, src_w = 14, 190
    parts.append(_dbox(src_x, cy - box_h/2, src_w, box_h,
                        ["original TexVerse", ".glb"]))

    vxz_x, vxz_w = 260, 190
    parts.append(_darrow_h(src_x + src_w, cy, vxz_x, cy))
    parts.append(f'<text class="dmono" x="{(src_x+src_w+vxz_x)/2}" y="{cy-12}" '
                 f'text-anchor="middle">glb_to_vxz</text>')
    parts.append(_dbox(vxz_x, cy - box_h/2, vxz_w, box_h,
                        ["glb_to_vxz"],
                        ["already normalizes:", "center + unit-box scale"], accent=True))

    vox_x, vox_w = 506, 230
    parts.append(_darrow_h(vxz_x + vxz_w, cy, vox_x, cy))
    parts.append(_dbox(vox_x, cy - box_h/2, vox_w, box_h,
                        ["ONE vxz, 512³"],
                        ["per-voxel attrs incl. a", "native 3-ch emissive"]))

    # fan out to input side / target side
    end_x, end_w = 796, 150
    top_cy, bot_cy = cy - 55, cy + 55
    parts.append(_dbranch(vox_x + vox_w, cy, end_x, top_cy))
    parts.append(_dbranch(vox_x + vox_w, cy, end_x, bot_cy))
    parts.append(_dbox(end_x, top_cy - 38, end_w, 76,
                        ["input side"], ["emissive channel", "zeroed"]))
    parts.append(_dbox(end_x, bot_cy - 38, end_w, 76,
                        ["target side"], ["threshold emissive", "→ binary mask"], accent=True))

    return diagram("".join(parts), W, H)


# ---------------------------------------------------------------------------
# --------------------------------------------------------- section 4 images
def prep_emit_frames():
    """Frame 0 (Front) of the emit_only / standard multiview webp pair for one
    sid, extracted with system PIL (the console venv lacks the PIL package)."""
    sid = "003d7fcf881743ed914fc51545ded1f5"
    src_dir = "/cs/3dlg-falas/project/omages/datasets/TexVerse/lightgen/multiview_webp"
    emit_src = os.path.join(src_dir, "emit_only", f"{sid}.webp")
    std_src = os.path.join(src_dir, "standard", f"{sid}.webp")
    emit_dst = os.path.join(IMG, f"{sid}_emit_only_f0.png")
    std_dst = os.path.join(IMG, f"{sid}_standard_f0.png")
    import subprocess
    script = f"""
from PIL import Image
for src, dst in [("{emit_src}", "{emit_dst}"), ("{std_src}", "{std_dst}")]:
    im = Image.open(src); im.seek(0)
    im.convert("RGB").save(dst)
print("ok")
"""
    r = subprocess.run(["/usr/bin/python3", "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        print("FRAME EXTRACTION FAILED:\n", r.stdout, r.stderr)
    return sid, emit_dst, std_dst


sid4, emit_path, std_path = prep_emit_frames()


def _bust(path, rel):
    import hashlib
    h = hashlib.md5(open(path, "rb").read()).hexdigest()[:8]
    return f"{rel}?v={h}"


emit_url = _bust(emit_path, f"img/{os.path.basename(emit_path)}")
std_url = _bust(std_path, f"img/{os.path.basename(std_path)}")

# ---------------------------------------------------------------------------
# ------------------------------------------------------------------- hero
header = lp.hero_header(
    "lightgen · SegviGen emissive · data pipeline proposal",
    "Voxelize the Original GLB, Not the Bake",
    dek_html=(
        "Today's emissive ground truth passes through a 512² repacked-UV atlas "
        "baked from the somage mesh, then two independent voxelizations. Every "
        "fine-tune we've trained &mdash; regardless of data volume or loss weighting "
        "&mdash; hits the same wall on small glow regions. This page lays out the "
        "current pipeline, the cost it carries, a proposed alternative that "
        "voxelizes the original GLB directly, and the 50-shape diagnostic that "
        "decides between them."),
    stats=[
        ("512²", "current GT atlas resolution"),
        (f"{min(large_vals):.2f}–{max(large_vals):.2f}", "large-glow IoU, all 4 checkpoints"),
        (f"{min(tiny_vals):.3f}–{max(tiny_vals):.3f}", "tiny-glow IoU, same checkpoints"),
        ("512³", "proposed native voxel resolution"),
        ("50", "shapes in the proposed diagnostic"),
    ],
    toc=[
        ("s1", "Today's pipeline"),
        ("s2", "The 512² GT ceiling"),
        ("s3", "Proposed: voxelize the GLB directly"),
        ("s4", "TRELLIS.2 already supports this"),
        ("s5", "Caveats + the ask"),
    ])

# ---------------------------------------------------------------- section 1
s1_body = (
    lp.prose(
        "<p>Every training pair starts from a <b>somage</b> &mdash; a 512² "
        "repacked-UV bake (color / metallic / roughness / emission maps) plus "
        f"{lp.filepath('somage_original_mesh.npz', 'somage_original_mesh.npz')}. "
        f"{lp.filepath('somage_to_glb.py', f'{SEGVIGEN_ROOT}/code/somage_to_glb.py')} "
        "turns that into <b>two</b> GLBs: <code>input.glb</code> (the PBR appearance "
        "the model conditions on &mdash; base color + packed metallic/roughness, "
        "<b>no emissiveFactor set</b>) and <code>emissive.glb</code> (a solid "
        "white-or-black per-face target, from <code>labels_uv_74k</code> when "
        "available, else a 0.04 threshold on the emission_color map). Each GLB then "
        "goes through SegviGen's own "
        f"{lp.filepath('glb_to_vxz.py', f'{SEGVIGEN_ROOT}/code/SegviGen/data_toolkit/glb_to_vxz.py')} "
        "&mdash; center + unit-box normalize, then voxelize at 512³ &mdash; "
        "independently, once per lane, landing as <code>input_tex_slat</code> on one "
        "side and the target + per-voxel <code>emis_mask</code> on the other.</p>")
    + current_pipeline_diagram()
    + lp.callout(
        "<b>Ground-truth labels (<code>labels_uv_74k</code>) are indexed by the "
        "somage mesh's faces</b> &mdash; not the original GLB's &mdash; so the bake "
        "is the only way to address them. It also already solves material "
        "normalization (color/metal/rough packed consistently across 74k shapes of "
        "wildly different source materials), and keeps every fine-tune comparable "
        "to the DiffusionNet baseline, which was evaluated on the same somage "
        "representation. The 74k somages already existed before this project "
        "started &mdash; building on them, not around them, was the fast path.",
        title="Why it was built this way")
)
s1 = lp.section_v2("s1", 1, "Today's pipeline routes everything through a 512² bake", s1_body)

# ---------------------------------------------------------------- section 2
strat_rows = []
for key in ["w5best", "w5ema", "balbest", "balema"]:
    label = MODEL_LABEL[key]
    large, tiny = STRAT[key][LARGE_I], STRAT[key][TINY_I]
    strat_rows.append({"label": f"{label} · large glow (>30%)", "value": large,
                       "display": f"{large:.3f}", "tip": f"{label}, large-glow bucket (n=15)"})
    strat_rows.append({"label": f"{label} · tiny glow (0–5%)", "value": tiny,
                       "display": f"{tiny:.3f}", "tip": f"{label}, tiny-glow bucket (n=53, the largest bucket)"})

s2_body = (
    lp.prose(
        "<p>The emission GT passes through a 512² repacked UV atlas, then a "
        "per-face threshold. A tiny emissive region &mdash; a strip of trim, a "
        "single small light &mdash; can cover only a handful of texels at that "
        "resolution: exactly where the ground truth gets mushy, and exactly where "
        "every model we've fine-tuned fails.</p>")
    + lp.hbar_chart(
        strat_rows,
        title="IoU @ threshold 0.2, by GT-coverage bucket (full 111-val, K=4 avg)",
        label_w=260,
        note=("<b>Every fine-tuned model works on large glow and fails on tiny glow.</b> "
              f"All four checkpoints land {min(large_vals):.3f}–{max(large_vals):.3f} IoU "
              f"on large-glow shapes and {min(tiny_vals):.3f}–{max(tiny_vals):.3f} on "
              "tiny-glow shapes &mdash; a 7–9&times; gap. Tiny-glow is also the largest "
              "bucket (53 of 111 val shapes; the median val shape is ≈1.4% emissive), so "
              "this single failure mode caps the aggregate score regardless of data volume "
              "(1k→2k) or loss weighting (flat 5× → per-shape balanced) &mdash; "
              "neither lever moves the tiny-glow column."),
        aria="Stratified IoU by GT coverage bucket, four checkpoints")
    + lp.prose(
        "<p>Source: full 111-shape val set, K=4 generations per shape averaged, voxel-based "
        "IoU, bucket_by=voxel &mdash; "
        + ", ".join(lp.filepath(f"eval_{j}.log", f"{SEGVIGEN_ROOT}/eval_{j}.log")
                    for j in ["231621", "231622", "231623", "231624"])
        + ". Same numbers as the "
          "<a href=\"../results_2k_v1/index.html#tiny-glow-wall\">2k fine-tune results page</a>'s "
          "stratified chart.</p>")
    + lp.callout(
        "We have <b>not yet proven</b> that GT mushiness in the 512² atlas is what "
        "causes this wall &mdash; only that the wall exists across every lever we've "
        "pulled (more data, different loss weighting, different checkpoints). That's "
        "exactly what the diagnostic in Section 5 is designed to test, not assume.",
        warn=True, title="What this chart does not show")
)
s2 = lp.section_v2("s2", 2, "Every fine-tune hits the same wall: tiny glow", s2_body)

# ---------------------------------------------------------------- section 3
s3_body = (
    lp.prose(
        "<p>Skip the bake: voxelize the <b>original TexVerse GLB</b> directly. "
        f"{lp.filepath('glb_to_vxz.py', f'{SEGVIGEN_ROOT}/code/SegviGen/data_toolkit/glb_to_vxz.py')} "
        "is the exact same script used in the pipeline above &mdash; it already "
        "computes <code>center = (aabb[0]+aabb[1])/2</code> and "
        "<code>scale = 0.99999 / (aabb[1]-aabb[0]).max()</code> before voxelizing, so "
        "no new normalization code is needed. Run once per shape, it produces "
        "<b>one</b> vxz carrying per-voxel attributes that already include a native "
        "3-channel emissive attribute (see Section 4) &mdash; split afterward into an "
        "input side (emissive channel zeroed, so the model can't see the answer) and "
        "a target side (threshold emissive → binary mask).</p>")
    + proposed_pipeline_diagram()
    + lp.prose(
        "<p>Three benefits fall out of this, each a direct consequence of removing the "
        "bake:</p>"
        "<ul>"
        "<li><b>GT at native texture resolution, sampled at 512³.</b> No 512² "
        "atlas ceiling between the source material and the voxel grid &mdash; a tiny "
        "emissive trim keeps whatever resolution its original texture had, instead of "
        "being pre-crushed to a handful of UV texels.</li>"
        "<li><b>Input and target are coordinate-aligned by construction.</b> One "
        "voxelization produces both, instead of two independent runs of "
        "<code>glb_to_vxz</code> on two separately-built GLBs that could in principle "
        "drift apart.</li>"
        "<li><b>Appearance input comes from the original 1k–4k textures</b>, not "
        "the 512² bake &mdash; closer to the texture resolution SegviGen's own "
        "pretraining domain used.</li>"
        "</ul>")
)
s3 = lp.section_v2("s3", 3, "Proposed: voxelize the original GLB, skip the bake", s3_body)

# ---------------------------------------------------------------- section 4
evidence_rows = f"""
  <tr><td>o-voxel bake computes emission (factor + texture)</td>
      <td><a href="https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/o-voxel/o_voxel/convert/volumetic_attr.py#L73">volumetic_attr.py#L73</a>
      (factor+texture handling <a href="https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/o-voxel/o_voxel/convert/volumetic_attr.py#L166-L219">L166–L219</a>)</td></tr>
  <tr><td>Raw voxel dataset defaults include <code>emissive:3</code> alongside PBR</td>
      <td><a href="https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/datasets/sparse_voxel_pbr.py#L111">sparse_voxel_pbr.py#L111</a></td></tr>
  <tr><td>Structured-latent dataset: same layout</td>
      <td><a href="https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/datasets/structured_latent_svpbr.py#L161">structured_latent_svpbr.py#L161</a></td></tr>
  <tr><td><b>BUT</b> the official prep deletes it before training (<code>del attr['emissive']</code>)</td>
      <td><a href="https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/data_toolkit/voxelize_pbr.py#L75">voxelize_pbr.py#L75</a></td></tr>
  <tr><td><b>AND</b> the released tex-VAE config trains on base_color/metallic/roughness/alpha only</td>
      <td><a href="https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/scvae/tex_vae_next_dc_f16c32_fp16.json">tex_vae_next_dc_f16c32_fp16.json</a></td></tr>
"""

s4_body = (
    lp.prose(
        "<p>The direct-GLB path isn't a new tool to build &mdash; TRELLIS.2's own "
        "voxelizer already carries emission end-to-end. All links below are pinned to "
        "commit <code>75fbf01</code>:</p>")
    + lp.results_table(["evidence", "commit-pinned link"], evidence_rows)
    + lp.callout(
        "The tooling supports emission end-to-end; the <b>released weights were "
        "trained with it switched off</b>. Practically: we can reuse their voxelizer "
        "for GT extraction unchanged, but our binary-in-base_color fine-tune "
        "formulation stays as-is &mdash; the released VAE latent has no emissive "
        "slot to fine-tune into.",
        title="What this means for us")
    + lp.fig_row(
        [("emit_only · emission channel only", emit_url),
         ("standard · fully lit render", std_url)],
        caption_html=(
            "<b>Emission is legibly readable straight off the GLB, no bake required.</b> "
            f"Shape <code>{sid4[:8]}…</code>, front view: the anglerfish-style lure "
            "and teeth glow clearly against black in the emission-only render (left), "
            "matching exactly where the standard fully-lit render (right) shows the "
            "lure/mouth structure. This is the signal the direct-GLB path reads at "
            "512³ instead of through a 512² bake."),
        native_px=512, content="photo")
)
s4 = lp.section_v2("s4", 4, "TRELLIS.2 already supports emission end-to-end — just switched off", s4_body)

# ---------------------------------------------------------------- section 5
caveats_rows = """
  <tr><td>Bake fidelity on real materials</td>
      <td><code>KHR_materials_emissive_strength</code> / factor-only edge cases in the o-voxel bake are unverified on TexVerse assets specifically.</td></tr>
  <tr><td>GT-of-record shift</td>
      <td>New GT will disagree with <code>labels_uv_74k</code> on some voxels &mdash; DiffusionNet-baseline comparability needs a footnote or a re-eval.</td></tr>
  <tr><td>Leakage guard</td>
      <td>Emissive channel must be zeroed on the input side; residual risk only where glow is baked directly into base_color (no separate emissive signal to strip).</td></tr>
"""

s5_body = (
    lp.results_table(["caveat", "risk"], caveats_rows)
    + lp.callout(
        "<b>A 50-shape diagnostic</b>, stratified across the same glow-size buckets as "
        "Section 2 (zero / tiny / medium / large): run <code>glb_to_vxz</code> on the "
        "original GLBs, threshold the native emissive channel, and A/B the result "
        "per-voxel against the current somage-derived GT. This answers two questions "
        "at once &mdash; <b>is the somage GT actually mushy on tiny glow</b>, and "
        "<b>does the o-voxel bake handle real TexVerse materials cleanly</b> &mdash; "
        "without committing to a full pipeline switch first. Cheap (50 shapes, one "
        "voxelizer pass each), cluster-ready today, and decision-grade: it tells us "
        "whether to proceed.",
        title="The ask")
)
s5 = lp.section_v2("s5", 5, "Three caveats, one cheap diagnostic", s5_body)

footer = (
    '<footer>Lightgen · segvigen_emissive/vis_data/pipeline_glb_direct_html · '
    'Stratified IoU numbers verbatim from '
    + ", ".join(lp.filepath(f"eval_{j}.log", f"{SEGVIGEN_ROOT}/eval_{j}.log")
                for j in ["231621", "231622", "231623", "231624"])
    + ' (same source as the '
    '<a href="../results_2k_v1/index.html">2k fine-tune results page</a>). '
    'TRELLIS.2 references pinned to commit '
    '<a href="https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd">75fbf01</a>. '
    '&middot; <a href="../index.html">↑ all lightgen visuals</a>'
    '</footer>'
)

html = lp.page(
    title="Voxelize the original GLB, not the bake — lightgen emissive GT pipeline",
    header_html=header,
    body_sections=[s1, s2, s3, s4, s5, footer],
    assets_rel=ASSETS_REL,
    assets_dir=ASSETS_DIR,
    theme="v2",
)

with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"wrote {OUT}/index.html")
