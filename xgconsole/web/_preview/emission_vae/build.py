#!/usr/bin/env python3
"""Build the emission-VAE explainer page (xgpage v2 editorial).

Content source: segvigen_emissive/FACTSHEET_emission_vae.md (assembled by the
master session 2026-08-06 from Dongchen's configs, checkpoint filenames and
his own visualization PNGs). Every number on this page comes from that file.

Figure legibility is handled by make_figs.py, which crops each sparse
projection panel to its content bbox, keeps all panels of a sample registered
to the same box, and emits a native-exposure panel next to a cube-root
tone-curve panel plus a two-color leak map. Run make_figs.py first.

Run: /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/emission_vae/build.py
"""
import hashlib
import math
import os

import xgpage as lp
from xgpage.publish import publish_assets

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = "/local-scratch2/xya120/studio/misc/lightgen/web"
SITE_ASSETS = "/projects/omages/yanxg/lightgen/assets"
SNAP = "trellis2_bw/code_snapshot"


def img(name):
    """Image src with a content hash, so a republished same-name file is not
    served from a stale browser cache."""
    p = os.path.join(HERE, "img", name)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return f"img/{name}?v={h}"


def code(t):
    return f"<code>{t}</code>"


def asset(rel):
    """Content-hashed src for a non-image asset next to the page (preview GLBs)."""
    p = os.path.join(HERE, rel)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return f"{rel}?v={h}"


# --------------------------------------------------------------- vocabulary --
COLS6 = ["OBJECT", "GT", "RECON", "GT BOOSTED", "RECON BOOSTED", "LEAK MAP"]
COLS5 = ["GT", "RECON", "GT BOOSTED", "RECON BOOSTED", "LEAK MAP"]

# What each shape is, read off its TexVerse render, and which part of it the
# ground-truth emission corresponds to. Descriptions are of the thumbnail
# itself; a part is named only where the correspondence is unambiguous.
OBJECTS = {
    "s01": ("birthday cake", "the candle flames"),
    "s02": ("television-headed character", "the screen, and the orb on its chest"),
    "s03": ("armored soldier", "the visor"),
    "s04": ("chandelier", "the lamp shades, seen from above"),
    "s05": ("armored character with glowing trim", None),
    "s06": ("toy gun", "the bulb"),
    "s07": ("sports car", "two interior screens"),
    "s08": ("robot", "the face screen"),
    "s09": ("glowing orb", "the whole sphere"),
    "s10": ("figure behind a metal panel", "the panel"),
}

TONE = ("each 8-bit display value <i>v</i> is replaced by "
        "255&middot;(<i>v</i>/255)<sup>1/3</sup>, so a value of 1/255 renders "
        "at 40/255 and a value of 8/255 at 80/255")

LEAK = ("white where the ground truth is nonzero, accent where the "
        "reconstruction is nonzero and the ground truth is exactly zero, "
        "black where both are zero")


def _sw(color, label):
    """A legend swatch with a hairline border, so the near-white and the
    near-black entries both stay visible in either theme."""
    return (f'<span class="lg-sw" style="background:{color};'
            f'border:1px solid var(--line)"></span>{label}')


def leak_legend():
    return lp.legend([
        _sw("#ECECE8", "leak map: ground-truth emission"),
        _sw("#D66A3F", "reconstruction emission where the ground truth is zero"),
        _sw("#0A0000", "both zero"),
    ])


def sample_row(label, slug):
    """One matrix row: the object render first, then the five emission panels."""
    what = OBJECTS[slug][0]
    return (label, [
        {"img": img(f"{slug}_obj.png"), "badge": "3D", "alt": what},
        img(f"{slug}_gt.png"), img(f"{slug}_rec.png"),
        img(f"{slug}_gt_boost.png"), img(f"{slug}_rec_boost.png"),
        img(f"{slug}_leak.png"),
    ])


def with_viewers(matrix_html, slugs):
    """Make each row's OBJECT tile open its preview GLB in the 3D lightbox.

    method_matrix emits plain <img> cells, so the click target is added here by
    rewriting exactly the tags this builder itself produced: the .fig-px class
    (pixelated rendering, right for the voxel projections) is dropped from the
    object render, which is a photograph-like image, and the .v3d hooks ui.js
    binds to are added in its place. A shape whose preview GLB is missing keeps
    a plain thumbnail and loses its badge, rather than opening an empty viewer.
    """
    for slug in slugs:
        src = img(f"{slug}_obj.png")
        old = (f'<img loading="lazy" class="fig-px" src="{src}" '
               f'alt="{OBJECTS[slug][0]}">')
        if old not in matrix_html:
            raise RuntimeError(f"object cell markup for {slug} not found")
        glb = os.path.join(HERE, "glb", f"{slug}.glb")
        if not os.path.exists(glb):
            matrix_html = matrix_html.replace(
                old, old.replace(' class="fig-px"', "")).replace(
                f'<span class="mm-badge">3D</span>', "", 1)
            continue
        title = f"{SHA[slug][:8]} · {OBJECTS[slug][0]}"
        new = (f'<img loading="lazy" class="v3d" src="{src}" '
               f'alt="{OBJECTS[slug][0]}" '
               f'data-glb="{asset(f"glb/{slug}.glb")}" data-title="{title}">')
        matrix_html = matrix_html.replace(old, new)
    return matrix_html


SHA = {
    "s01": "0007deb6d96c4474b80faa5aa3888760",
    "s02": "000b9fd47d6d4f7db7b2f5022d1ae9aa",
    "s03": "000bc33c1a0d4b36acab1e18de6617e1",
    "s04": "00192b5a4a3249c79141c8dccaad2947",
    "s05": "001a188012214d9ca9b8b22087296558",
    "s06": "001b64d2ec45496792f4edcf036bbaaf",
    "s07": "001c79293c3e4f938798026a79f2d26a",
    "s08": "001dd28130354d36b8f04ffe59c30abe",
    "s09": "002342e8d06042d69aed2919731d4d5f",
    "s10": "002af43e7cab4e5b9490b59534636173",
}


def object_figure(slug, caption_html):
    """The object render beside its ground-truth emission, as a two-panel row.

    Used where a whole section is about one shape, so a per-row object column
    would repeat the same tile three times.
    """
    what = OBJECTS[slug][0]
    html = lp.fig_row([
        (f"object &middot; {SHA[slug][:8]}", img(f"{slug}_obj.png"), what),
        ("ground-truth emission, XY projection", img(f"{slug}_gt.png")),
    ], caption_html=caption_html, native_px=440, content="pixel-map")
    src = img(f"{slug}_obj.png")
    old = (f'<img loading="lazy" class="fig-px" src="{src}" alt="{what}">')
    if old not in html:
        raise RuntimeError(f"object panel markup for {slug} not found")
    title = f"{SHA[slug][:8]} · {what}"
    return html.replace(old, (
        f'<img loading="lazy" class="v3d" src="{src}" alt="{what}" '
        f'data-glb="{asset(f"glb/{slug}.glb")}" data-title="{title}">'))


def sample_matrix(rows, caption_html):
    """A method_matrix over sample_row()s, with the 3D lightbox wired in."""
    slugs = [s for _, s in rows]
    html = lp.method_matrix(
        COLS6, [sample_row(lbl, slug) for lbl, slug in rows],
        caption_html=caption_html, native_px=440, content="pixel-map")
    return with_viewers(html, slugs)


def proj_row(label, prefix, view):
    return (label, [
        img(f"{prefix}_{view}_gt.png"), img(f"{prefix}_{view}_rec.png"),
        img(f"{prefix}_{view}_gt_boost.png"), img(f"{prefix}_{view}_rec_boost.png"),
        img(f"{prefix}_{view}_leak.png"),
    ])


HOWTOREAD = 'Columns and transforms as in <a href="#roundtrip">&sect;01</a>.'

READING_KEY = (
    "<b>Six columns, the same six everywhere below.</b> "
    "<b>OBJECT</b> is the shape's own TexVerse render, cropped to the object, "
    "so what is glowing is visible before the glow is judged; click it to open "
    "the shape in a 3D viewer. "
    "<b>GT</b> and <b>RECON</b> are the ground truth and the reconstruction at "
    "the source figure's own exposure, the pair to read first. <b>GT "
    "BOOSTED</b> and <b>RECON BOOSTED</b> are the same two panels under one "
    "fixed tone curve: " + TONE +
    ". The boosted ground-truth column is the control: wherever it stays black, "
    "the ground truth was exactly zero, so anything visible in the boosted "
    "reconstruction next to it is emission the model invented. <b>LEAK MAP</b> "
    "reduces that comparison to " + LEAK + ". "
    "Every panel of a sample is cropped to one shared box, computed from the "
    "ground-truth content unioned with the bright part of the reconstruction so "
    "a reconstruction that puts its content elsewhere stays in frame, then "
    "upscaled nearest-neighbor, so one source pixel is one visible block. "
    "The object render is not cropped to that box: it is a separate camera view "
    "of the whole shape, cropped to the object and shown at its own scale, so "
    "it orients the reader rather than registering pixel for pixel with the "
    "projections. "
    "On a narrow screen each figure keeps its tiles at full size and scrolls "
    "sideways within its own frame; all six columns are there."
)


LIGHTBOX_CSS = """
/* Click-to-load 3D lightbox, page-local so it can follow the v2 palette in
   both themes (the shared v1 stylesheet's copy is dark-chrome only). */
img.v3d { cursor: zoom-in; }
.mm-cell:has(img.v3d):hover { outline: 2px solid var(--accent); outline-offset: -1px; }
.mv3d-modal { position: fixed; inset: 0; z-index: 90; display: none;
  background: color-mix(in srgb, var(--bg2) 88%, transparent);
  backdrop-filter: blur(4px);
  flex-direction: column; align-items: center; justify-content: center;
  padding: 2vh 2vw; }
.mv3d-modal.open { display: flex; }
.mv3d-modal model-viewer { width: min(92vw, 860px); height: min(80vh, 860px);
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
.mv3d-bar { display: flex; gap: 1rem; align-items: center; color: var(--ink);
  font-size: .84rem; margin-bottom: .6rem; width: min(92vw, 860px); }
.mv3d-bar #mv3d-title { font-family: ui-monospace, Menlo, monospace;
  font-size: .76rem; color: var(--ink-2); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.mv3d-dl { color: var(--accent-ink); font-size: .76rem; white-space: nowrap; }
.mv3d-close { background: var(--accent); color: #fff; border: 0; border-radius: 6px;
  padding: .35rem .8rem; cursor: pointer; font-size: .78rem; white-space: nowrap; }
@media (max-width: 760px) {
  .mv3d-modal model-viewer { width: 96vw; height: 74vh; }
}
"""

LIGHTBOX_JS = """
/* Click a .v3d thumbnail -> open its data-glb in the #mv3d model-viewer modal.
   The src is set on open and removed on close, so no GLB downloads until a
   thumbnail is clicked and only one WebGL context is ever live. */
(function () {
  function init() {
    var modal = document.getElementById('mv3d');
    if (!modal) return;
    var mv = document.getElementById('mv3d-viewer');
    var titleEl = document.getElementById('mv3d-title');
    var dl = document.getElementById('mv3d-dl');
    function open(glb, ttl) {
      if (!glb) return;
      mv.setAttribute('src', glb);
      titleEl.textContent = ttl || '';
      dl.setAttribute('href', glb);
      modal.classList.add('open');
    }
    function close() {
      modal.classList.remove('open');
      mv.removeAttribute('src');
    }
    document.querySelectorAll('img.v3d').forEach(function (im) {
      im.addEventListener('click', function () {
        open(im.dataset.glb, im.dataset.title);
      });
    });
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.getElementById('mv3d-close').addEventListener('click', close);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.classList.contains('open')) close();
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
"""


def lightbox_head():
    return (f'<script type="module" src="{SITE_ASSETS}/model-viewer.min.js">'
            f'</script><style>{LIGHTBOX_CSS}</style>')


def lightbox_modal():
    return (
        '<div id="mv3d" class="mv3d-modal">'
        '<div class="mv3d-bar"><span id="mv3d-title"></span>'
        '<a id="mv3d-dl" class="mv3d-dl" href="#" download>download GLB</a>'
        '<button id="mv3d-close" class="mv3d-close" type="button">'
        '&#10005; close</button></div>'
        '<model-viewer id="mv3d-viewer" camera-controls auto-rotate '
        'rotation-per-second="18deg" interaction-prompt="none" exposure="1.1" '
        'tone-mapping="neutral" shadow-intensity="0.3" shadow-softness="1">'
        '</model-viewer></div>'
        f'<script>{LIGHTBOX_JS}</script>')


# ----------------------------------------------------------- design diagram --
def design_diagram():
    """Two fine-tune designs as one schematic: same trunk, different in/out
    width and different initialization."""
    W, H = 972, 486
    colw, gap = 428, 52
    xs = [32, 32 + colw + gap]
    designs = [
        dict(name="albedo2emission", sub="channel substitution",
             io="6 channels in, 6 channels out",
             enc="init: TRELLIS.2 tex_enc_next_dc_f16c32_fp16",
             dec="init: TRELLIS.2 tex_dec_next_dc_f16c32_fp16",
             foot=["EmissionPbrFinetuneTrainer &middot; max_steps 35,800",
                   "CosineAnnealingLR, T_max 10,000, eta_min 1e-6",
                   "step0034800-0.0056.ckpt &middot; 13,258,637,734 bytes"],
             accent=True),
        dict(name="pbr2emission", sub="emission only",
             io="3 channels in, 3 channels out",
             enc="init: pretrained_pbr_encoder (TRELLIS.2 tex encoder)",
             dec="no decoder init listed in the config",
             foot=["EmissionVaeTrainer &middot; max_steps 160,000",
                   "no lr_scheduler in the config",
                   "step0154600-0.0088.ckpt &middot; 2,058,711,934 bytes"],
             accent=False),
    ]
    p = []
    for x, d in zip(xs, designs):
        cls = "dbox dbox-accent" if d["accent"] else "dbox"
        p.append(f'<text x="{x}" y="20" class="dtitle">{d["name"]}</text>')
        p.append(f'<text x="{x}" y="40" class="dsub">{d["sub"]}</text>')
        stack = [
            (58, 38, d["io"].split(",")[0].strip(), ""),
            (116, 52, "SparseUnetVaeEncoder", d["enc"]),
            (188, 38, "latent, 32 channels", ""),
            (246, 52, "SparseUnetVaeDecoder", d["dec"]),
            (318, 38, d["io"].split(",")[1].strip() + " out"
             if False else d["io"].split(",")[-1].strip(), ""),
        ]
        for i, (y, h, t1, t2) in enumerate(stack):
            p.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{colw}" '
                     f'height="{h}" rx="7"/>')
            ty = y + (21 if t2 else h / 2 + 4)
            p.append(f'<text x="{x + 14}" y="{ty}" class="dsub">{t1}</text>')
            if t2:
                p.append(f'<text x="{x + 14}" y="{y + 39}" class="dmono">{t2}</text>')
            if i < len(stack) - 1:
                y0 = y + h
                y1 = stack[i + 1][0]
                cx = x + colw / 2
                p.append(f'<path class="dline" d="M{cx} {y0} L{cx} {y1 - 7}"/>')
                p.append(f'<path class="arrfill" d="M{cx - 5} {y1 - 8} '
                         f'L{cx + 5} {y1 - 8} L{cx} {y1} Z"/>')
        for j, line in enumerate(d["foot"]):
            p.append(f'<text x="{x}" y="{382 + j * 20}" class="dmono">{line}</text>')
    y = 448
    p.append(f'<path class="dline" d="M32 {y - 18} L{W - 32} {y - 18}"/>')
    p.append(f'<text x="32" y="{y}" class="dmono">shared: AdamW lr 1e-4 '
             '&middot; weight_decay 0.0 &middot; l1 loss &middot; lambda_kl '
             '1.0e-07 &middot; ema_rate 0.9999 &middot; fp16_mode inflat_all'
             '</text>')
    p.append(f'<text x="32" y="{y + 20}" class="dmono">AdaptiveGradClipper '
             'max_norm 1.0, clip_percentile 95 &middot; batch_size_per_gpu 32 '
             '&middot; use_balanced_sampler true &middot; resolution 256</text>')
    return (f'<div class="diagram"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="the two emission VAE fine-tune designs side by side">'
            f'{"".join(p)}</svg></div>')


# ------------------------------------------------------------- L1 log chart --
L1_ROWS = [
    ("0007deb6", 0.0066, "ring of orange dots, clean", "clean"),
    ("001c7929", 0.0082, "blue speck, vanishes", "vanish"),
    ("000bc33c", 0.0092, "red ellipse, vanishes", "vanish"),
    ("001dd281", 0.0153, "white bar, vanishes", "vanish"),
    ("001a1880", 0.0154, "blue shape, wrong place", "collapse"),
    ("000b9fd4", 0.0277, "white bar, vanishes", "vanish"),
    ("002af43e", 0.0477, "grey rectangle, vanishes", "vanish"),
    ("001b64d2", 0.1247, "yellow blob, dims", "collapse"),
    ("00192b5a", 0.5031, "flower, starburst", "collapse"),
    ("002342e8", 1.3856, "white disc, worst", "worst"),
]
CKPT_REFS = [(0.0056, "albedo2emission"), (0.0088, "pbr2emission")]


def l1_log_chart():
    W = 760
    lab_w, right = 250, 74
    x0, x1 = lab_w, W - right
    lo, hi = 0.004, 2.2
    top, rowh = 46, 27
    H = top + len(L1_ROWS) * rowh + 46

    def X(v):
        return x0 + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * (x1 - x0)

    p = [f'<text x="{x0}" y="14" class="axislabel">per-sample L1, log scale '
         f'(reported on each panel of the ten-sample figure)</text>']
    for t, lbl in [(0.005, "0.005"), (0.01, "0.01"), (0.05, "0.05"),
                   (0.1, "0.1"), (0.5, "0.5"), (1.0, "1.0"), (2.0, "2.0")]:
        p.append(f'<path class="grid" d="M{X(t):.1f} {top - 12} '
                 f'L{X(t):.1f} {top + len(L1_ROWS) * rowh}"/>')
        p.append(f'<text x="{X(t):.1f}" y="{top - 18}" class="ticktext" '
                 f'text-anchor="middle">{lbl}</text>')
    base = top + len(L1_ROWS) * rowh
    for j, (v, name) in enumerate(CKPT_REFS):
        p.append(f'<path class="zeroline" d="M{X(v):.1f} {top - 12} '
                 f'L{X(v):.1f} {base + 8 + j * 16}"/>')
        p.append(f'<text x="{X(v) + 6:.1f}" y="{base + 12 + j * 16}" '
                 f'class="ticktext">{v:.4f}, from the {name} '
                 f'checkpoint filename</text>')
    for i, (sha, v, desc, kind) in enumerate(L1_ROWS):
        y = top + i * rowh + 13
        fill = "var(--accent)" if kind == "vanish" else "var(--ink-3)"
        p.append(f'<text x="{x0 - 12}" y="{y + 4}" text-anchor="end" '
                 f'class="barlabel">{sha} &#183; {desc}</text>')
        p.append(f'<path class="grid" d="M{x0} {y} L{X(v):.1f} {y}"/>')
        p.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="5.5" fill="{fill}" '
                 f'data-tip="{sha}: L1 {v:.4f}"/>')
        p.append(f'<text x="{x1 + 8}" y="{y + 4}" class="barval">{v:.4f}</text>')
    return (f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="per-sample L1 on a log scale">{"".join(p)}</svg></div>')


# ------------------------------------------------------------------- build ---
def build():
    assets_dir = os.path.join(WEB, "assets")

    hero = lp.hero_header(
        "lightgen \u00b7 SegviGen \u00b7 emission VAE fine-tunes \u00b7 2026-08-06",
        "The emission VAE round trip does not return emission",
        dek_html=(
            "Our emissive-segmentation eval decodes ground truth through "
            "TRELLIS.2's pretrained tex decoder. Dongchen separately fine-tuned "
            "two VAEs for emission. If those fine-tuned VAEs still cannot "
            "reconstruct emission, then no model trained in that latent space "
            "can produce it, and the ~0.1 IoU seen across four unrelated "
            "architectures has a common cause upstream of every one of them. "
            "This page reads the two configs and Dongchen's own reconstruction "
            "figures, re-exposed so the failures are visible and shown beside a "
            "render of the object each one belongs to."),
        stats=[
            ("2", "fine-tuned VAEs"),
            ("5 of 10", "emissive regions vanish"),
            ("1.3856", "worst per-sample L1"),
            ("0.0097", "single-sample overfit L1"),
            ("+16%", "recon emission mean above GT"),
        ],
        toc=[
            ("roundtrip", "Where the VAE sits"),
            ("designs", "Two designs"),
            ("clean", "One clean case"),
            ("vanish", "Five vanish"),
            ("collapse", "Three collapse"),
            ("worst", "The worst case"),
            ("overfit", "The overfit leak"),
            ("loss", "Low L1 is what black scores"),
            ("resolution", "Latent resolution"),
            ("openq", "Not verified"),
        ],
    )

    S = []

    # ------------------------------------------------------------- 01 ------
    S.append(lp.section_v2("roundtrip", "01",
        "A round trip that cannot output zero caps every model downstream", "".join([
        lp.prose(
            "Emissive segmentation in SegviGen is evaluated by decoding ground "
            "truth through TRELLIS.2's pretrained tex decoder. Everything a "
            "model can express is therefore bounded by what that decoder can "
            "put back. Dongchen fine-tuned two VAEs specifically for emission, "
            "which is the direct test of that bound: encode a shape's emission "
            "voxels, decode them, and compare."),
        lp.callout(
            "The ~0.1 IoU seen across four unrelated architectures (TEXGen, two "
            "TRELLIS.2 variants, SegviGen) would have a common cause upstream of "
            "all of them if the round trip itself destroys sparse emission. The "
            "figures below are Dongchen's, re-cropped and re-exposed; the "
            "conclusion they support is stated in "
            "<a href=\"#openq\">&sect;10</a> as interpretation, not measurement.",
            title="What is at stake"),
        lp.prose(
            "The problem with reading these figures as published is that both "
            "failure modes live in the dark range. A region that reconstructs to "
            "nothing and a region that reconstructs to a dim wash both look like "
            "black at native exposure, and the emissive region itself occupies a "
            "small fraction of a 256-voxel projection. Every figure on this page "
            "is therefore cropped to its content and shown twice, once at native "
            "exposure and once under a fixed tone curve, next to a map of where "
            "the reconstruction emits and the ground truth does not."),
        lp.prose(READING_KEY),
        leak_legend(),
    ])))

    # ------------------------------------------------------------- 02 ------
    S.append(lp.section_v2("designs", "02",
        "Two fine-tunes on one trunk, differing in width and in what was inherited",
        "".join([
        design_diagram(),
        lp.chartnote(
            "<b>Both designs keep TRELLIS.2's sparse UNet trunk and its 32-channel "
            "latent; they differ in tensor width and in how much of the pretrained "
            "model they start from.</b> "
            "<code>albedo2emission</code> puts emission RGB in the albedo slot and "
            "pins the other material slots, so input and output shapes match the "
            "pretrained model exactly and both encoder and decoder are initialized "
            "from it. <code>pbr2emission</code> is emission-only at 3 channels in "
            "and out; its config lists an encoder initialization and no decoder "
            "initialization. Both use "
            "<code>SparseUnetVaeEncoder</code>/<code>SparseUnetVaeDecoder</code>, "
            "latent_channels 32, model_channels [32, 64, 128, 256, 512], "
            "<code>SparseConvNeXtBlock3d</code>, and the shared training settings "
            "on the bottom rule. Both live under "
            f"<code>{SNAP.replace('code_snapshot', 'latents_v2')}/vae_ckpts/</code>."),
        lp.prose(
            "The dataset is <code>LightGenSLatEmission</code> over "
            "<code>data/lightgen_74k_newbake</code> at resolution 256, with "
            "pbr_attrs [base_color, metallic, roughness, alpha] and the split "
            "<code>data_splits_emissive_74k_stratified_newbake_vae.json</code>. "
            "Neither figure on this page can be attributed to one of these two "
            "checkpoints; see <a href=\"#openq\">&sect;10</a>."),
    ])))

    # ------------------------------------------------------------- 03 ------
    S.append(lp.section_v2("clean", "03",
        "One of the ten validation samples comes back clean", "".join([
        sample_matrix([("L1 0.0066 · 0007deb6", "s01")],
            "<b>The one clean reconstruction in the set: the candle flames of a "
            "birthday cake return as the same ring of orange dots, and the boosted "
            "panel shows the reconstruction also spreads a dim wash over the rest "
            "of the cake.</b> " + HOWTOREAD + " The object is a birthday cake seen "
            "from the side; the emission projection here looks down the cake's axis, "
            "so the ring is the row of candle flames and everything inside it is the "
            "unlit cake body. The leak map is accent over that whole body, so even "
            "the best case in the set puts nonzero emission where the ground truth "
            "has none. This is the same shape as the single-sample overfit in "
            "<a href=\"#overfit\">&sect;07</a>."),
        leak_legend(),
        lp.prose(
            "Per-sample L1 for this row is 0.0066, the lowest in the set. It is "
            "the only row of the ten where the reconstruction reproduces the "
            "emissive geometry and the color."),
    ])))

    # ------------------------------------------------------------- 04 ------
    S.append(lp.section_v2("vanish", "04",
        "Five of the ten emissive regions vanish, and a dim wash arrives instead",
        "".join([
        sample_matrix([
            ("L1 0.0082 · 001c7929", "s07"),
            ("L1 0.0092 · 000bc33c", "s03"),
            ("L1 0.0153 · 001dd281", "s08"),
            ("L1 0.0277 · 000b9fd4", "s02"),
            ("L1 0.0477 · 002af43e", "s10"),
        ], "<b>In half the set the emissive region does not reconstruct at all: "
            "the reconstruction column is black at native exposure, and under the "
            "tone curve what is there instead is a low-amplitude wash spread over "
            "the object.</b> " + HOWTOREAD + " Rows in ascending L1, object then "
            "what emits on it: a sports car, two interior screens; an armored "
            "soldier, its visor; a robot, its face screen; a television-headed "
            "character, the screen and the orb on its chest; a figure behind a "
            "metal panel, the panel. In every row the ground-truth column shows a "
            "compact bright region, the boosted ground-truth column shows that "
            "everything around it is exactly zero, and the leak map is accent "
            "everywhere the reconstruction put emission the ground truth does not "
            "have. These five occupy five of the seven lowest L1 positions in the "
            "set of ten; see <a href=\"#loss\">&sect;08</a>. Two rows also show why "
            "an emission channel need not be visible in a render: the sports car's "
            "only emitters are interior screens, hidden behind its bodywork from "
            "this camera, and the last row's metal panel carries its own surface "
            "texture in the emission channel yet renders as ordinary metal with no "
            "glow at all. In both, the render says what the object is without "
            "showing the emission, which is why it sits beside the emission "
            "projection rather than replacing it."),
        leak_legend(),
        lp.prose(
            "The crops here are the tightest on the page. In the source figure "
            "these five rows are frames that are almost entirely black, with the "
            "emissive region occupying a few dozen pixels; the boxes above are "
            "56 to 156 source pixels on a side, upscaled nearest-neighbor."),
    ])))

    # ------------------------------------------------------------- 05 ------
    S.append(lp.section_v2("collapse", "05",
        "Three keep some structure and collapse toward orange", "".join([
        sample_matrix([
            ("L1 0.0154 · 001a1880", "s05"),
            ("L1 0.1247 · 001b64d2", "s06"),
            ("L1 0.5031 · 00192b5a", "s04"),
        ], (
            "<b>Where emission does survive the round trip it arrives orange, and "
            "in one row it also arrives in the wrong place.</b> " + HOWTOREAD +
            " Top: an armored character with glowing trim, whose pale blue blade "
            "shapes become a small orange blob elsewhere in the frame (L1 0.0154). "
            "Middle: a toy gun, whose yellow bulb becomes a dim orange one "
            "(0.1247). Bottom: a chandelier, whose lamp shades seen from above form "
            "the flower and come back as an orange starburst (0.5031); its leak map "
            "is white "
            "throughout because the reconstruction stays inside the "
            "ground truth's own support, so the failure there is chromatic rather "
            "than spatial.")),
        leak_legend(),
    ])))

    # ------------------------------------------------------------- 06 ------
    S.append(lp.section_v2("worst", "06",
        "The brightest sample is the worst reconstruction in the set", "".join([
        sample_matrix([("L1 1.3856 · 002342e8", "s09")],
            "<b>A glowing orb, emissive over its whole surface, returns as an "
            "orange-brown disc with concentric rings, at L1 1.3856, the largest L1 "
            "in the set by a wide margin.</b> " + HOWTOREAD + " The geometry is "
            "preserved exactly and the color is not; the leak map is white "
            "throughout because the ground truth is emissive over the whole disc, "
            "so there is no region left for the reconstruction to leak into. The "
            "concentric ringing is visible in the native reconstruction panel and "
            "is not an artifact of the tone curve. The render shows the orb cyan "
            "while the ground-truth emission projection is saturated white in all "
            "three channels; the two come from different places (the render is "
            "TexVerse's own render of the GLB, the projection is of the baked "
            "emission voxels) and this page does not establish which step "
            "introduces the difference."),
        leak_legend(),
        lp.prose(
            "This row and the five in <a href=\"#vanish\">&sect;04</a> are the two "
            "ends of the same behavior. A large, bright, uniform emissive region "
            "is the case a reconstruction cannot hide by predicting black, and it "
            "is where the loss becomes large."),
    ])))

    # ------------------------------------------------------------- 07 ------
    S.append(lp.section_v2("overfit", "07",
        "Memorizing a single sample for 500 steps still leaks emission over the whole object",
        "".join([
        object_figure("s01",
            "<b>The subject of both figures below is a birthday cake with lit "
            "candles: the flames are the only part of it that emits.</b> Left: the "
            "shape's TexVerse render, cropped to the object; click it to open the "
            "shape in a 3D viewer. Right: the ground-truth emission for the same "
            "shape, projected down the cake's axis, which is why the flames form a "
            "ring and the cake body is empty. Every projection panel in this "
            "section is of this one shape, so the object column is shown once here "
            "rather than repeated on each row."),
        lp.method_matrix(COLS5, [
            proj_row("XY projection", "ovf", "xy"),
            proj_row("XZ projection", "ovf", "xz"),
            proj_row("YZ projection", "ovf", "yz"),
        ], caption_html=(
            "<b>The candle flames reconstruct, and the cake body, which is exactly "
            "zero in the ground truth, comes back nonzero across its entire "
            "silhouette.</b> Columns as above; rows are the three axis-aligned "
            "projections. Read the third and fourth columns against each other: "
            "under the same tone curve the ground-truth body stays black and the "
            "reconstructed body fills in, and the leak map turns that difference "
            "into a single accent region covering the whole cake. Sample "
            "0007deb6, 500 steps at lr 0.001, final L1 0.0097; the panel text "
            "reports 303,573 voxels, ground-truth emission mean 0.0506 and "
            "reconstructed emission mean 0.0588, a reconstruction 16% above ground "
            "truth, consistent with the leak. These panels are a matplotlib "
            "<code>hot</code> colormap of a max projection of the per-voxel "
            "emission magnitude, with vmin 0 and one shared vmax per row, so they "
            "were inverted through that colormap back to the scalar field before "
            "the tone curve was applied."),
            native_px=440, content="pixel-map"),
        leak_legend(),
        lp.prose(
            "This is the easiest setting the architecture can be given: one "
            "sample, memorized, with generalization removed from the picture. The "
            "VAE still cannot output exact zeros. The same sample is the clean "
            "row in <a href=\"#clean\">&sect;03</a>."),
        lp.method_matrix(COLS5, [
            proj_row("XY projection", "e2e", "xy"),
            proj_row("XZ projection", "e2e", "xz"),
            proj_row("YZ projection", "e2e", "yz"),
        ], caption_html=(
            "<b>An independent end-to-end test of the same shape, run through a "
            "different encoder path, leaks in the same place.</b> " + HOWTOREAD +
            " These panels are direct RGB (max projection keeping color), not a "
            "colormap, so the tone curve is applied per channel. The reported L1 "
            "is 0.006644, with an emission latent of torch.Size([1009, 32]) at "
            "res ~16 and a PBR latent of the same shape. The source figure's own "
            "|Error| panel shows error across the whole object silhouette rather "
            "than only at the emissive parts, which the leak map here makes "
            "explicit: the accent band covers the cake body in both side views."),
            native_px=440, content="pixel-map"),
        leak_legend(),
    ])))

    # ------------------------------------------------------------- 08 ------
    S.append(lp.section_v2("loss", "08",
        "A low L1 is what predicting black earns on a signal that is mostly black",
        "".join([
        l1_log_chart(),
        lp.chartnote(
            "<b>The five rows whose emissive region vanishes (accent) sit at the "
            "low end of the L1 range, not the high end.</b> Log scale, because the "
            "set spans two orders of magnitude. Dashed lines mark 0.0056 and "
            "0.0088, the losses in the two checkpoint filenames. The five "
            "vanishing rows occupy five of the seven lowest positions, and one of "
            "them, 0.0082, is below the 0.0088 checkpoint-name loss. The worst "
            "reconstruction in the set, at 1.3856, is the one sample a black "
            "prediction could not have produced."),
        lp.prose(
            "Emission is roughly 97.5% black by voxel count (our own sample: "
            "median emissive_frac 0.025). A reconstruction that outputs black "
            "everywhere therefore earns a very low L1, and the ordering above is "
            "the evidence for that internal to Dongchen's own figure: the samples "
            "that reconstruct to nothing are scored better than the samples that "
            "reconstruct to something wrong. The mean L1 of a validation set is "
            "not a measure of whether emission survives the round trip."),
        lp.callout(
            "This is interpretation, not measurement. What is measured: the "
            "per-sample L1 values, and the reconstructions shown above. What is "
            "inferred: that the sparsity of emission is what makes those L1 "
            "values low, and that the failure is representational rather than an "
            "optimization shortfall (the single-sample overfit in "
            "<a href=\"#overfit\">&sect;07</a> removes generalization from the "
            "picture). The further inference, that this caps every downstream "
            "model in the same latent space, is plausible and is not established "
            "on this page.", title="Where measurement ends"),
    ])))

    # ------------------------------------------------------------- 09 ------
    S.append(lp.section_v2("resolution", "09",
        "The latent grid is the input resolution divided by 16, which settles our 16\u00b3 versus 32\u00b3 disagreement",
        "".join([
        lp.fig_row([
            ("input voxel grid, resolution 256", img("res_input_xy.png")),
            ("emission latent, res ~16", img("res_latent_xy.png")),
        ], caption_html=(
            "<b>Both readings of the SLAT latent size were right, for different "
            "pipelines: the latent grid is the input resolution divided by 16.</b> "
            "Left: the XY max projection of the input emission voxels at "
            "resolution 256. Right: the emission latent for the same shape, mean "
            "over its 32 channels, drawn at the latent grid's own resolution, so "
            "each block is one latent cell. Dongchen's VAEs run at resolution 256 "
            "and their latent is annotated \"at res ~16\" in the end-to-end "
            "figure. Our direct-ovoxel pipeline runs at 512, so ours is 32&sup3;. "
            "The discrepancy in our notes was two pipelines, not an error. The "
            "right panel is Dongchen's own RdBu_r rendering, cropped and upscaled "
            "nearest-neighbor; its colors encode the signed latent mean and are "
            "not comparable to the emission panels elsewhere on this page."),
            native_px=440, content="pixel-map"),
    ])))

    # ------------------------------------------------------------- 10 ------
    S.append(lp.section_v2("openq", "10",
        "What is not established, including which VAE produced any of this",
        "".join([
        lp.callout(
            "<b>Neither figure on this page can be attributed to "
            "<code>albedo2emission</code> or <code>pbr2emission</code>.</b> The "
            "filenames do not say and neither do the panel titles. The generator "
            "of the overfit figure, "
            f"<code>{SNAP}/data_toolkit/vis_emission_vae.py</code>, loads its "
            "checkpoint from <code>/tmp/emission_vae_overfit/ckpts/last.ckpt</code>, "
            "and the generator of the end-to-end figure, "
            f"<code>{SNAP}/data_toolkit/test_emission_vae_e2e.py</code>, loads "
            "<code>/tmp/emission_vae_optB_overfit/ckpts/stepstep=0001000.ckpt</code>. "
            "Both are separate single-sample overfit runs, not the two stored "
            "checkpoints, and their configs are not in the code snapshot. Read "
            "these figures as evidence about the architecture and the latent "
            "space, not about either stored checkpoint.", warn=True),
        lp.prose("Also not established:"),
        lp.prose(
            "<b>Whether the ten-sample figure comes from a stored checkpoint.</b> "
            "Most per-sample L1 values in it exceed both checkpoint-name losses "
            "(0.0056 and 0.0088), so the panels may predate the saved "
            "checkpoints. The script at that figure's output path in the code "
            f"snapshot, <code>{SNAP}/vis/vis_emission_vae.py</code>, writes a "
            "six-column <code>hot</code>-colormap layout, while the figure has "
            "four columns in color, so the revision that produced it is not in "
            "the snapshot. Its projection reduction and color space are "
            "therefore unverified, and the tone curve applied to it here is a "
            "display transform on its 8-bit pixels, not on the underlying "
            "emission values."),
        lp.prose(
            "<b>Whether our own eval could use either fine-tuned decoder in place "
            "of the pretrained one.</b> This is the question the page ends on and "
            "does not answer. Both fine-tuned decoders exist and are loadable; "
            "what has not been tried is decoding our ground truth through one of "
            "them and re-running the emissive IoU."),
        lp.prose(
            "Resolved while building this page, against the generating code "
            "rather than the fact sheet: the overfit figure's projections are max "
            "projections of the per-voxel emission magnitude rendered with "
            "matplotlib <code>hot</code> at vmin 0 and a per-row shared vmax, its "
            "|Error| panel is <code>plasma</code> at vmin 0 (dark blue is zero "
            "error, not high), and its overlay panel is green for ground truth "
            "and red for the reconstruction. The end-to-end figure's GT and "
            "reconstruction panels are direct RGB max projections with no tone "
            "mapping. No tone mapping is applied by either generator; the tone "
            "curves on this page are ours and are stated in every caption."),
    ])))

    apx = lp.appendix("Provenance", [
        "Configs and checkpoints: "
        "<code>trellis2_bw/latents_v2/vae_ckpts/albedo2emission/{config.yaml, "
        "step0034800-0.0056.ckpt}</code> (13,258,637,734 bytes) and "
        "<code>trellis2_bw/latents_v2/vae_ckpts/pbr2emission/{config.yaml, "
        "step0154600-0.0088.ckpt}</code> (2,058,711,934 bytes).",
        f"Figures, read-only from <code>{SNAP}/</code>: "
        "<code>emission_vae_10sample_vis.png</code> (1523&times;3345, "
        "&sect;03&ndash;&sect;06 and &sect;08), "
        "<code>emission_vae_overfit_vis.png</code> (1988&times;1485, &sect;07), "
        "<code>emission_vae_optB_test.png</code> (2699&times;1776, &sect;07 and "
        "&sect;09). Also present and not used here: "
        "<code>emission_vae_e2e_test.png</code>, <code>vae_channel_vis.png</code>.",
        f"Generating code, read-only under <code>{SNAP}/</code>: "
        "<code>data_toolkit/vis_emission_vae.py</code> (overfit figure), "
        "<code>data_toolkit/test_emission_vae_e2e.py</code> (end-to-end figure), "
        "<code>vis/vis_emission_vae.py</code> (a later revision of the "
        "ten-sample figure's generator). Related: "
        "<code>data_toolkit/test_pbr_vae.py</code>, "
        "<code>data_toolkit/encode_emission_pbrfinetune_10samples.py</code>, "
        "<code>data_toolkit/predict_emission_voxels_twostream.py</code>.",
        "Object renders, read-only from "
        "<code>/cs/3dlg-falas/datasets/TexVerse/thumbnails/thumbnails_batch/"
        "batch_00000/&lt;sid&gt;.png</code> (1920&times;1080 each), cropped to the "
        "object's content bbox and squared by <code>make_context.py</code>. "
        "Preview GLBs derive from "
        "<code>/cs/3dlg-falas/datasets/TexVerse-1K/glbs/glbs_1k/000-000/"
        "&lt;sid&gt;_1024.glb</code> (0.9 to 63 MB each) via gltf-transform 4.4.1 "
        "<code>optimize</code>: mesh quantization (KHR_mesh_quantization, decoded "
        "natively by model-viewer, no external decoder), WebP textures at 512 or "
        "384 px, and mesh simplification. The served previews run 0.05 to 11.6 MB "
        "and are click-to-load, so none of them downloads until a thumbnail is "
        "clicked. The 3D viewer is self-hosted "
        "<code>@google/model-viewer</code> 3.5.0.",
        "Every emission panel on this page is a pixel crop of one of those three source "
        "PNGs, produced by <code>make_figs.py</code> next to this page's builder. "
        "Panel grids were located by scanning the source for its axes boxes; "
        "each sample's panels are cropped to one shared box so they stay "
        "registered; crops are upscaled nearest-neighbor. The tone curve and "
        "the leak map are display transforms defined in that script and stated "
        "in every caption. The sources are never modified.",
        "Full-resolution source figures, as published: "
        f'<a href="{img("src_emission_vae_10sample_vis.png")}">ten-sample</a>, '
        f'<a href="{img("src_emission_vae_overfit_vis.png")}">overfit</a>, '
        f'<a href="{img("src_emission_vae_optB_test.png")}">end-to-end</a> '
        "(downscaled copies of the originals, no other change).",
        "Numbers: <code>segvigen_emissive/FACTSHEET_emission_vae.md</code>, "
        "assembled 2026-08-06 from the configs, checkpoint filenames and the "
        "source PNGs.",
    ])

    html = lp.page(
        title="The emission VAE round trip does not return emission (lightgen)",
        header_html=hero,
        body_sections=S + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v2",
        needs_katex=False,
        extra_head=lightbox_head(),
        extra_body_end=lightbox_modal(),
    )
    out = os.path.join(HERE, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, len(html), "bytes")
    publish_assets(assets_dir)
    print("assets published ->", assets_dir)


if __name__ == "__main__":
    build()
