"""
build_training_inputs_page.py — "What SegviGen training needs, and what we have"
xgpage v2 editorial. A single explanatory page answering one confused question:
what inputs does SegviGen fine-tuning need for training, what do we have now, and
what's missing. Communication artifact — every claim carries a diagram, no prose-only
findings. All diagrams are self-contained inline SVG (no external assets, no CDN).

Findings established 2026-07-24 (per the brief; not re-derived here):
  - EmisDataset / train_emissive.py define the per-shape contract (4 tensors read;
    cond.pth unused, zero-cond).
  - The Dongchen ovoxel dataset (out_uv_voxel_74k, 72,481 shapes) covers input_tex_slat
    and output_tex_slat; it does not cover shape_slat (never computed — voxelize.py
    only calls textured_mesh_to_volumetric_attr, never mesh_to_flexible_dual_grid).
  - Both candidate ways to get shape_slat load the source GLB; the open question is
    how to get input/output_tex_slat at native 512, not whether to load the GLB.

Run:
  /local-scratch2/xya120/studio/misc/lightgen/.venv_console/bin/python3 \
    build_training_inputs_page.py
"""
import html
import math
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "training_inputs_html")
os.makedirs(OUT, exist_ok=True)
import xgpage as lp  # the installed package (uv pip install -e ~/studio/xgpage)

ASSETS_REL = "/projects/omages/yanxg/lightgen/assets"
ASSETS_DIR = os.path.join(ROOT, "..", "web", "assets")
UPDATED = "2026-07-24"

# =============================================================================
# ---------------------------------------------------------- diagram vocabulary
# Hand-authored box-arrow SVG on theme2's .diagram/.dbox/.dline vocabulary (see
# build_pipeline_glb_direct_page.py for the precedent), extended here with a
# "warn/missing" variant (dbox-warn / dline-warn / arrfill-warn, on --accent2,
# the same token callout(warn=True) already uses — no new color introduced).

def _dbox(x, y, w, h, lines, mono_lines=None, cls="dbox", title_cls="dtitle", mono_cls="dmono"):
    rect = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
    texts = []
    n_title = len(lines)
    n_mono = len(mono_lines or [])
    total_lines = n_title + n_mono
    line_h = 17
    start_y = y + h / 2 - (total_lines - 1) * line_h / 2 + 5
    for i, ln in enumerate(lines):
        texts.append(f'<text class="{title_cls}" x="{x + w/2}" y="{start_y + i*line_h}" '
                     f'text-anchor="middle">{html.escape(ln)}</text>')
    for j, ln in enumerate(mono_lines or []):
        yy = start_y + (n_title + j) * line_h
        texts.append(f'<text class="{mono_cls}" x="{x + w/2}" y="{yy}" '
                     f'text-anchor="middle">{html.escape(ln)}</text>')
    return rect + "".join(texts)


def _darrow_h(x1, y, x2, cls="dline", marker="darrow", dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<path class="{cls}" d="M{x1},{y} L{x2},{y}" marker-end="url(#{marker})"{dash}/>'


def _dbranch(x1, y1, x2, y2, cls="dline", marker="darrow", dashed=False):
    midx = (x1 + x2) / 2
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (f'<path class="{cls}" d="M{x1},{y1} C{midx},{y1} {midx},{y2} {x2},{y2}" '
            f'marker-end="url(#{marker})"{dash}/>')


_DEFS = """<defs>
  <marker id="darrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" class="arrfill"/></marker>
  <marker id="darrow-accent" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" class="arrfill-accent"/></marker>
  <marker id="darrow-warn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" class="arrfill-warn"/></marker>
</defs>"""


def diagram(inner_svg, w, h, cls=""):
    c = f' diagram-{cls}' if cls else ""
    return f'<div class="diagram{c}"><svg viewBox="0 0 {w} {h}">{_DEFS}{inner_svg}</svg></div>'


# =============================================================================
# ------------------------------------------------ diagram 1: the four tensors
def contract_diagram():
    W, H = 980, 340
    p = []
    # left column: conditioning inputs + bookkeeping
    p.append(f'<text class="dsub" x="135" y="16">CONDITIONING (fed to the model)</text>')
    p.append(_dbox(20, 24, 230, 82, ["shape_slat.pth"], ["WHERE the surface is"]))
    p.append(_dbox(20, 128, 230, 98, ["input_tex_slat.pth"],
                    ["HOW it reflects light", "base_color · metallic", "roughness · alpha"]))
    p.append(f'<text class="dsub" x="135" y="250">READ, NOT FED TO THE MODEL</text>')
    p.append(_dbox(20, 258, 230, 62, ["meta.json"], ["emissive_frac"]))

    # center: the model
    model_x, model_w = 330, 190
    model_y, model_h = 72, 200
    p.append(_dbox(model_x, model_y, model_w, model_h,
                    ["full_seg fine-tune"], ["zero-cond", "(image cond. = zeros)"]))

    # arrows into the model from the two conditioning boxes
    p.append(_dbranch(250, 65, model_x, model_y + 55))
    p.append(_dbranch(250, 177, model_x, model_y + 145))

    # right column: target
    p.append(f'<text class="dsub" x="715" y="16">TARGET (the loss compares against this)</text>')
    out_x, out_w = 590, 260
    p.append(_darrow_h(model_x + model_w, model_y + 100, out_x, cls="dline-accent", marker="darrow-accent"))
    p.append(f'<text class="dmono" x="{(model_x+model_w+out_x)/2}" y="{model_y+90}" text-anchor="middle">predicts</text>')
    p.append(_dbox(out_x, model_y + 20, out_w, 90, ["output_tex_slat.pth"],
                    ["WHAT it emits", "binary emissive mask"], cls="dbox-accent"))

    # unused, disconnected: cond.pth
    p.append(f'<text class="dsub-warn" x="715" y="250">IN THE CONTRACT — NOT WIRED IN</text>')
    p.append(_dbox(out_x, 258, out_w, 62, ["cond.pth"], ["image conditioning — unused"],
                    cls="dbox-unused", title_cls="dtitle-dim", mono_cls="dmono-dim"))
    return diagram("".join(p), W, H)


# =============================================================================
# ------------------------------------------------- diagram 2: dataset coverage
def coverage_diagram():
    W, H = 980, 340
    p = []
    left_x, left_w = 20, 300
    right_x, right_w = 620, 260
    lane_h = 78
    lanes_cy = [58, 168, 282]

    # lane 1: pbr_voxels -> input_tex_slat (covered)
    p.append(_dbox(left_x, lanes_cy[0] - lane_h/2, left_w, lane_h,
                    ["pbr_voxels_256/{sha}.vxz"], ["base_color · metallic · roughness · alpha"]))
    p.append(_darrow_h(left_x + left_w, lanes_cy[0], right_x, cls="dline-accent", marker="darrow-accent"))
    p.append(_dbox(right_x, lanes_cy[0] - lane_h/2, right_w, lane_h,
                    ["input_tex_slat.pth"], [], cls="dbox-accent"))
    p.append(f'<text class="dcheck" x="{right_x+right_w+18}" y="{lanes_cy[0]+7}">✓</text>')

    # lane 2: emission_voxels -> binarize -> output_tex_slat (covered)
    p.append(_dbox(left_x, lanes_cy[1] - lane_h/2, left_w, lane_h,
                    ["emission_voxels_256/{sha}.vxz"], ["emissive RGB"]))
    p.append(_darrow_h(left_x + left_w, lanes_cy[1], right_x, cls="dline-accent", marker="darrow-accent"))
    p.append(f'<text class="dmono" x="{left_x+left_w+ (right_x-left_x-left_w)/2}" y="{lanes_cy[1]-9}" '
             f'text-anchor="middle">binarize &gt; 0 (any nonzero)</text>')
    p.append(_dbox(right_x, lanes_cy[1] - lane_h/2, right_w, lane_h,
                    ["output_tex_slat.pth"], [], cls="dbox-accent"))
    p.append(f'<text class="dcheck" x="{right_x+right_w+18}" y="{lanes_cy[1]+7}">✓</text>')

    # lane 3: missing -> shape_slat
    lane3_h = 92
    p.append(_dbox(left_x, lanes_cy[2] - lane3_h/2, left_w, lane3_h,
                    ["— nothing in the dataset —"], ["never computed"],
                    cls="dbox-warn", title_cls="dtitle-warn", mono_cls="dmono-warn"))
    p.append(_darrow_h(left_x + left_w, lanes_cy[2], right_x,
                        cls="dline-warn", marker="darrow-warn", dashed=True))
    p.append(_dbox(right_x, lanes_cy[2] - lane3_h/2, right_w, lane3_h,
                    ["shape_slat.pth"], [], cls="dbox-warn", title_cls="dtitle-warn"))
    p.append(f'<text class="dx" x="{right_x+right_w+16}" y="{lanes_cy[2]+8}">✗ MISSING</text>')

    return diagram("".join(p), W, H)


# =============================================================================
# --------------------------------------- diagram 3: occupancy vs. dual grid
# Fully computed, not hand-drawn (D16b/c). Corrected 2026-07-24 (owner catch):
# the true surface is drawn PIECEWISE LINEAR, not a smooth curve — dual
# contouring reconstructs a straight segment between the dual vertices of
# each pair of cells sharing a sign-changed edge (a polyline, never a curve),
# and mesh_to_flexible_dual_grid(vertices, faces, ...) takes a TRIANGLE MESH,
# whose cross-section is itself a polyline. A smooth input implies an
# implicit/SDF surface (the classical Ju et al. 2002 presentation), which is
# not what this pipeline feeds it. Both panels below derive cell membership,
# edge crossings, and vertex placement from the SAME fine input polyline, so
# the two panels stay consistent with each other.
def _hill_y(t, GH):
    """Underlying silhouette used only to PLACE the fine polyline's few
    vertices below — never rendered directly, never sampled densely."""
    return (GH * 0.52
            + GH * 0.16 * math.sin(2 * math.pi * t * 1.35 + 0.4)
            + GH * 0.07 * math.sin(2 * math.pi * t * 3.2 + 1.0))


def _fine_input_polyline(GW, GH, n_vertices=13, seed=7):
    """The 'mesh': a FEW SHORT STRAIGHT SEGMENTS, vertices deliberately not
    grid-aligned (irregular x spacing) — this is the true input, drawn as
    straight segments end to end, exactly what a triangle mesh's
    cross-section is."""
    rng = random.Random(seed)
    ts = sorted(rng.uniform(0.03, 0.97) for _ in range(n_vertices - 2))
    ts = [0.0] + ts + [1.0]
    return [(GW * t, _hill_y(t, GH)) for t in ts]


def _densify_polyline(poly, substeps=40):
    """Dense sub-sampling ALONG the fine polyline's own straight segments —
    an implementation detail for robust cell/crossing detection, never
    rendered itself (the rendered 'true input' is the sparse _fine_input_
    polyline above, connected straight)."""
    pts = []
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        for s in range(substeps):
            f = s / substeps
            pts.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    pts.append(poly[-1])
    return pts


def _cell_membership(pts, GW, GH, NX, NY):
    CSX, CSY = GW / NX, GH / NY

    def cell_of(x, y):
        cx = min(NX - 1, max(0, int(x // CSX)))
        cy = min(NY - 1, max(0, int(y // CSY)))
        return (cx, cy)

    cells = {}
    prev = pts[0]
    prev_c = cell_of(*prev)
    cells.setdefault(prev_c, {"pts": [], "cross": []})["pts"].append(prev)
    for pt in pts[1:]:
        c = cell_of(*pt)
        cells.setdefault(c, {"pts": [], "cross": []})["pts"].append(pt)
        if c != prev_c:
            crossing = ((prev[0] + pt[0]) / 2, (prev[1] + pt[1]) / 2)
            cells[prev_c]["cross"].append(crossing)
            cells[c]["cross"].append(crossing)
        prev, prev_c = pt, c
    return cells, CSX, CSY


def _grid_lines(GW, GH, NX, NY):
    lines = []
    for i in range(NX + 1):
        x = GW * i / NX
        lines.append(f'<line class="dgrid" x1="{x}" y1="0" x2="{x}" y2="{GH}"/>')
    for j in range(NY + 1):
        y = GH * j / NY
        lines.append(f'<line class="dgrid" x1="0" y1="{y}" x2="{GW}" y2="{y}"/>')
    return "".join(lines)


def _poly_path(pts):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return d


def crosssection_panels():
    GW, GH = 460, 260
    NX, NY = 9, 6
    fine = _fine_input_polyline(GW, GH)
    dense = _densify_polyline(fine)
    cells, CSX, CSY = _cell_membership(dense, GW, GH, NX, NY)
    fine_path = _poly_path(fine)

    # ---- panel 1: occupancy only (blocky attributes on voxels) ----
    p1 = [_grid_lines(GW, GH, NX, NY)]
    for (cx, cy) in cells:
        p1.append(f'<rect class="dcell-fill" x="{cx*CSX:.1f}" y="{cy*CSY:.1f}" '
                   f'width="{CSX:.1f}" height="{CSY:.1f}"/>')
    # faint true-surface reference (the fine input polyline), to show what
    # occupancy CANNOT see
    p1.append(f'<path class="dcurve-ghost" d="{fine_path}"/>')
    svg1 = f'<svg viewBox="0 0 {GW} {GH}"><rect class="dcanvas" x="0" y="0" width="{GW}" height="{GH}"/>{"".join(p1)}</svg>'

    # ---- panel 2: dual grid (sub-voxel surface position) ----
    # reconstruction: exactly ONE dual vertex per crossed cell, straight
    # segments between consecutive vertices — a COARSER polyline than the
    # input; the gap between it and the faint true-input line IS the
    # reconstruction error.
    dual_verts = []
    for c in cells.values():
        cxp, cyp = c["pts"][len(c["pts"]) // 2]
        dual_verts.append((cxp, cyp))
    dual_verts.sort(key=lambda v: v[0])
    recon_path = _poly_path(dual_verts)

    p2 = [_grid_lines(GW, GH, NX, NY)]
    for (cx, cy) in cells:
        p2.append(f'<rect class="dcell-touched" x="{cx*CSX:.1f}" y="{cy*CSY:.1f}" '
                   f'width="{CSX:.1f}" height="{CSY:.1f}"/>')
    p2.append(f'<path class="dcurve-ghost" d="{fine_path}"/>')  # true input, same style as panel 1
    p2.append(f'<path class="dcurve" d="{recon_path}"/>')       # reconstruction
    for c in cells.values():
        for (ex, ey) in c["cross"]:
            p2.append(f'<circle class="dtick" cx="{ex:.1f}" cy="{ey:.1f}" r="2.6"/>')
    for (vx, vy) in dual_verts:
        p2.append(f'<circle class="dvert" cx="{vx:.1f}" cy="{vy:.1f}" r="4"/>')
    svg2 = f'<svg viewBox="0 0 {GW} {GH}"><rect class="dcanvas" x="0" y="0" width="{GW}" height="{GH}"/>{"".join(p2)}</svg>'

    legend = (
        '<div class="cross-legend">'
        '<span class="cl-item"><span class="cl-swatch cl-ghost"></span>'
        'true input &mdash; the mesh&rsquo;s cross-section (piecewise linear)</span>'
        '<span class="cl-item"><span class="cl-swatch cl-recon"></span>'
        'dual-grid reconstruction &mdash; one vertex per crossed cell</span>'
        '</div>'
    )

    return (
        '<div class="fig-pair" style="max-width:820px;margin-left:auto;margin-right:auto">'
        '<div><div class="panel-label">Attributes on voxels &middot; occupancy only</div>'
        f'<div class="diagram diagram-cross">{svg1}</div></div>'
        '<div><div class="panel-label">Dual grid &middot; sub-voxel surface position</div>'
        f'<div class="diagram diagram-cross">{svg2}</div></div>'
        '</div>'
        + legend
    )


# =============================================================================
# ------------------------------------------------- diagram 4: shared prefix
def shared_paths_diagram():
    W, H = 980, 440
    p = []
    # shared band background
    p.append(f'<rect class="dband" x="6" y="6" width="{W-12}" height="150" rx="12"/>')
    p.append(f'<text class="dsub-accent" x="22" y="28">SHARED &mdash; both paths load the GLB</text>')

    y0, h0 = 44, 82
    boxes = [
        ("source GLB", ["TexVerse-1K"], 20, 150),
        ("load_and_merge()", [], 194, 168),
        ("_tight_scene()", [], 386, 150),
        ("mesh_to_flexible_dual_grid(512)", [], 560, 230),
        ("shape_slat", [], 814, 150),
    ]
    xs = []
    for i, (title, mono, x, w) in enumerate(boxes):
        accent = (i == 3 or i == 4)
        p.append(_dbox(x, y0, w, h0, [title], mono, cls="dbox-accent" if accent else "dbox"))
        xs.append((x, w))
    for i in range(len(boxes) - 1):
        x1 = xs[i][0] + xs[i][1]
        x2 = xs[i + 1][0]
        p.append(_darrow_h(x1, y0 + h0/2, x2))

    # branch down from _tight_scene() to Option B (same loaded mesh, reused) —
    # routed as an orthogonal polyline through the LEFT MARGIN (x=10, clear
    # of Option A's box1 which starts at x=20) so it never crosses over Option A's
    # row; a straight vertical drop through the middle collided with Option A's
    # second box (found live, screenshot QA at 1400px desktop).
    tsx, tsw = boxes[2][2], boxes[2][3]
    branch_x = tsx + tsw / 2
    top_y = 172   # clears Option A's row top (214) and its "OPTION A" label (200)
    pathB_cy = 372  # Option B row vertical center (by=330, bh=84)
    p.append(f'<path class="dline" d="M{branch_x},{y0+h0} L{branch_x},{top_y} '
             f'L10,{top_y} L10,{pathB_cy} L276,{pathB_cy}" marker-end="url(#darrow)" fill="none"/>')
    p.append(f'<text class="dmono" x="70" y="{top_y-8}">same loaded mesh</text>')

    # Option A row (independent source, NOT the shared GLB load)
    p.append(f'<text class="dsub" x="20" y="200">OPTION A</text>')
    ay, ah = 214, 78
    a_boxes = [
        ("pbr / emission_voxels_256", [], 20, 230),
        ("2× replicate-upsample to 512", [], 274, 220),
        ("intersect w/ native-512 shape coords", [], 518, 250),
        ("input / output_tex_slat", [], 792, 168),
    ]
    axs = []
    for title, mono, x, w in a_boxes:
        p.append(_dbox(x, ay, w, ah, [title], mono))
        axs.append((x, w))
    for i in range(len(a_boxes) - 1):
        x1 = axs[i][0] + axs[i][1]
        x2 = axs[i + 1][0]
        p.append(_darrow_h(x1, ay + ah/2, x2))

    # Option B row (branches off the shared load)
    p.append(f'<text class="dsub" x="20" y="322">OPTION B</text>')
    by, bh = 330, 84
    p.append(_dbox(276, by, 300, bh, ["textured_mesh_to_volumetric_attr(512)"], [], cls="dbox-accent"))
    p.append(_darrow_h(276 + 300, by + bh/2, 616, cls="dline-accent", marker="darrow-accent"))
    p.append(_dbox(616, by, 200, bh, ["input / output_tex_slat"], [], cls="dbox-accent"))

    return diagram("".join(p), W, H)


# =============================================================================
# ================================================================ hero
hero = lp.hero_header(
    f"lightgen · explainer · SegviGen training inputs &nbsp;·&nbsp; {UPDATED}",
    "What SegviGen training needs, and what we have",
    dek_html=(
        "Dongchen&rsquo;s ovoxel dataset supplies two of the three tensors training needs; "
        "the third is <b>geometry conditioning</b>, which exists only in the source mesh "
        "&mdash; so every shape must load its GLB regardless of which path we take."),
    stats=[
        ("3", "tensors required per shape"),
        ("2 of 3", "covered by the ovoxel dataset"),
        ("1", "missing &mdash; shape_slat"),
        ("72,481", "shapes in the dataset"),
        ("256³ vs 512³", "dataset vs. model resolution"),
    ],
    toc=[
        ("contract", "Three tensors, three jobs"),
        ("coverage", "The dataset covers two"),
        ("crux", "Why the geometry isn't there"),
        ("shared", "The GLB loads either way"),
        ("measured", "The upsample loses half the voxels"),
    ],
)

# ================================================================ 01 contract
s_contract = lp.section_v2("contract", "01",
    "Three tensors, three different jobs",
    contract_diagram()
    + lp.prose(
        "<b>Three tensors, three different jobs.</b> <code>EmisDataset</code> "
        "(<code>train_emissive.py</code>) reads four things from a shape's flat "
        "per-shape directory. The model is conditioned on <b>where the surface is</b> "
        "(<code>shape_slat.pth</code>) and <b>how it reflects light</b> "
        "(<code>input_tex_slat.pth</code>: base_color, metallic, roughness, alpha), and "
        "predicts <b>what it emits</b> &mdash; a binary emissive mask, compared against "
        "<code>output_tex_slat.pth</code> via the loss. The split follows the rendering "
        "equation's own outgoing-radiance terms: <code>input_tex_slat</code> parametrizes "
        "the <b>reflected</b> term (plus opacity, which isn't strictly reflectance), and "
        "<code>output_tex_slat</code> is the <b>emitted</b> term. <code>meta.json</code> "
        "carries <code>emissive_frac</code> for bookkeeping, not as a model input. A fifth "
        "file, <code>cond.pth</code> (image conditioning), is part of the contract but "
        "unused: we train zero-cond.")
)

# ================================================================ 02 coverage
s_coverage = lp.section_v2("coverage", "02",
    "The dataset covers two of the three tensors",
    coverage_diagram()
    + lp.prose(
        "<b>Input and target are both already there; geometry is not.</b> "
        "<code>pbr_voxels_256/{sha}.vxz</code> is <code>input_tex_slat</code> directly. "
        "<code>emission_voxels_256/{sha}.vxz</code>, binarized at any nonzero emission "
        "(&gt;0), is <code>output_tex_slat</code>. Nothing in the dataset is "
        "<code>shape_slat</code> &mdash; it was never computed for these shapes. "
        "Also present in the dataset but not used for this: "
        + lp.pill("atlas.npz") + " " + lp.pill("{sha}.coords.npz") + ". "
        "One more gap, quieter than the missing tensor: the dataset is voxelized at "
        "<b>256³</b>, while <code>full_seg</code> trains natively at <b>512³</b> "
        "(f16 downsample to a 32³ latent) &mdash; whatever supplies "
        "<code>shape_slat</code> has to resolve that mismatch too.")
)

# ================================================================ 03 crux
s_crux = lp.section_v2("crux", "03",
    "Occupancy is blocky; the dual grid is not",
    crosssection_panels()
    + lp.prose(
        "<b>These are two different extractions of the same mesh, and only one was ever run.</b> "
        "<code>textured_mesh_to_volumetric_attr</code> writes an attribute value onto each "
        "occupied voxel &mdash; a cube is either on or off, so the surface's true position "
        "inside that cube is lost. This is the one Dongchen's <code>voxelize.py</code> runs. "
        "<code>mesh_to_flexible_dual_grid</code> instead records, for every cell the surface "
        "crosses, the <b>sub-voxel point</b> where the surface actually passes through "
        "(<code>dual_vertices</code>) and which cell edges it crosses "
        "(<code>intersected</code>) &mdash; the right panel above, computed from the same "
        "grid as the left. Both the true surface (thin dashed line) and the reconstruction "
        "(solid line through the dual vertices) are drawn <b>piecewise linear, not smooth</b>: "
        "<code>mesh_to_flexible_dual_grid</code> takes a triangle mesh, whose cross-section is "
        "a polyline, and dual contouring connects dual vertices with straight segments, never "
        "a curve. <b>The visible gap between the two lines between vertices is the "
        "reconstruction error</b> &mdash; still far tighter than occupancy's whole-cell "
        "snapping, but not exact. This extraction <b>was never run</b>, so it is not on disk "
        "anywhere in the dataset. That is why the missing tensor cannot be recovered from "
        "the voxels by any amount of processing: it was never computed, and occupancy alone "
        "is too coarse to reconstruct it.")
)

# ================================================================ 04 shared
s_shared = lp.section_v2("shared", "04",
    "The GLB load is shared; only the attribute step differs",
    shared_paths_diagram()
    + lp.callout(
        "<b>Option A/B is about where the ATTRIBUTES come from &mdash; neither is "
        "&ldquo;Path A,&rdquo; the rejected somage &rarr; recolor GLB &rarr; voxelize "
        "round-trip</b> that the <code>pipeline_design</code> page already covers and that "
        "stays superseded. There is no somage, no recoloring, no emission-label round-trip "
        "on this page. Emission labels come from the authored emission in both Option A and "
        "Option B; the GLB is read only to compute the geometry channel.",
        warn=True, title="What this is not")
    + lp.prose(
        "<b>Loading the GLB is common to both candidates; only what happens after differs.</b> "
        "Both options run the same source-GLB load, merge, and tight-scene fit, then compute "
        "<code>shape_slat</code> from <code>mesh_to_flexible_dual_grid(512)</code> on the "
        "result. Option A pulls <code>input</code>/<code>output_tex_slat</code> from the "
        "already-preprocessed ovoxels (2× replicate-upsampled to 512, then intersected "
        "with the native-512 shape coordinates). Option B reuses the <em>same already-loaded "
        "mesh</em> and runs <code>textured_mesh_to_volumetric_attr</code> on it directly at "
        "512. So using the preprocessed ovoxels only saves the attribute-extraction call, "
        "not the GLB load.")
)

# ================================================================ 05 measured
retention_chart = lp.hbar_chart(
    [
        {"label": "small", "value": 50.0, "display": "50.0%",
         "tip": "small shape: Option A retains 50.0% of attribute voxels after intersection"},
        {"label": "median", "value": 50.5, "display": "50.5%",
         "tip": "median shape: Option A retains 50.5%"},
        {"label": "p90", "value": 50.9, "display": "50.9%",
         "tip": "p90 shape: Option A retains 50.9%"},
        {"label": "p95", "value": 56.2, "display": "56.2%",
         "tip": "p95 shape: Option A retains 56.2%"},
        {"label": "large", "value": 60.8, "display": "60.8%",
         "tip": "large shape: Option A retains 60.8%"},
    ],
    title="Option A attribute-coordinate retention, by shape (% of voxels kept)",
    note=(
        "<b>Option A discards roughly half its attribute voxels on every shape; Option B "
        "retains 100% on all five.</b> The "
        "2×-replicated &ldquo;thick shell&rdquo; only half-overlaps the &ldquo;thin&rdquo; "
        "native-512 surface. Option B's shape-coord and attr-coord sets are <b>identical in "
        "count</b> on all five shapes (e.g. 88,064&nbsp;==&nbsp;88,064; "
        "2,072,200&nbsp;==&nbsp;2,072,200) &mdash; same mesh, same resolution, so the same "
        "voxels by construction. Shape-coord retention is ~100% in both paths, so the "
        "geometry channel is never the bottleneck &mdash; this is specifically an "
        "attribute-channel problem."),
)

compare_html = (
    '<div class="cmp-wrap">'
    '<div class="cmp-card">'
    '<div class="cmp-h">Option A &mdash; keep Dongchen&rsquo;s attrs</div>'
    '<p>Reuses the ovoxel attributes as-is; the extra work is the replicate-upsample and '
    'the intersection with the native-512 shape surface.</p>'
    '<div class="cmp-risk"><b>Measured:</b> discards ~50&ndash;61% of attribute voxels on '
    'every one of the 5 shapes tested (job 236204). Emissive fraction stays within ~10% '
    'relative of native on 4 of 5 shapes, but inflates by <b>62% relative</b> on the '
    'sparsest emitter.</div>'
    '</div>'
    '<div class="cmp-card">'
    '<div class="cmp-h">Option B &mdash; one native-512 extraction</div>'
    '<p>All three channels (<code>shape_slat</code>, <code>input_tex_slat</code>, '
    '<code>output_tex_slat</code>) come from one mesh, one extraction, one resolution.</p>'
    '<div class="cmp-risk cmp-good"><b>Measured:</b> removes the replication artifact and '
    'the alignment risk entirely (100% retention, exact coord-count match), at ~1.5&times; '
    "Option A&rsquo;s per-shape cost, not 2&times; &mdash; see the timing note below.</div>"
    '</div>'
    '</div>'
)

s_open = lp.section_v2("measured", "05",
    "The upsample loses half the attribute voxels",
    retention_chart
    + compare_html
    + lp.callout(
        "<b>Sparse, localized emitters take the largest hit.</b> Option A does not zero out "
        "emissive labels outright &mdash; it drops emissive and non-emissive voxels at "
        "similar rates, so the emissive fraction of the final target stays close to native "
        "on most shapes. But on the sparsest emitter measured (p95), it inflates the "
        "emissive fraction by <b>62% relative</b> (0.267% under A vs. 0.165% true native "
        "under B). Sparse, localized emitters are exactly the shapes the filtering work "
        "exists to identify &mdash; so Option A's error is aimed at the signal that matters "
        "most.",
        warn=True, title="The subtler finding: label survival on sparse emitters")
    + lp.prose(
        "<b>The shared load dominates; Option B is ~1.5&times; Option A per shape, not "
        "2&times;.</b> <code>glb_load + tight_scene + dual_grid_512</code> is the shared "
        "cost from &sect;04 and dominates every shape. Option B's only extra step "
        "(<code>native_attr_512</code>, ~1.0&ndash;6.0s) is the same order as the mandatory "
        "dual-grid call (~1.1&ndash;8.4s) it already had to pay. Option A's upsample is "
        "essentially free (&lt;0.02s), but that saving is irrelevant next to the shared "
        "cost both paths already carry.")
    + lp.callout(
        "The measurement settles the mechanism, not the decision: <b>the A-vs-B choice is "
        "still the project owner's call</b> &mdash; this page presents the evidence and "
        "does not declare a winner. One nuance worth keeping in view: choosing Option B means "
        "we would not literally train on Dongchen's ovoxel <em>files</em> for the input/target "
        "channels &mdash; his exact functions (<code>get_common_coords</code>, "
        "<code>_tight_scene</code>), frame, and authored-emission semantics are preserved "
        "unmodified (still no somage, no recoloring), and his dataset still supplies shape "
        "selection and cross-checks; only the per-voxel attribute values would be "
        "re-derived from the same GLB at 512 instead of read from the 256³ files.",
        title="What the measurement does not decide")
)

# ================================================================ provenance
provenance = lp.expandable(
    "Provenance",
    '<ul class="prov-list">'
    '<li><code>train_emissive.py</code> / <code>EmisDataset</code> define the per-shape '
    'contract in &sect;01.</li>'
    '<li>Training job: <code>train_emissive_v5.sbatch</code>, conda env <code>trellis2</code>, '
    'L40S, <code>--init_ckpt full_seg</code> &rarr; '
    '<code>hf_hub_download("fenghora/SegviGen", "full_seg.ckpt")</code>.</li>'
    '<li>Dataset: <code>/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k/</code> '
    '&mdash; 72,481 shape dirs, mounted directly on solar.</li>'
    '<li>Dongchen&rsquo;s <code>voxelize.py</code> calls only '
    '<code>textured_mesh_to_volumetric_attr</code>.</li>'
    '<li><code>vxz_to_slat.py</code> shows '
    '<code>shape_slat = shape_encoder(dual_vertices, intersected)</code>.</li>'
    '<li>Source GLBs: <code>/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K</code>.</li>'
    '<li>The &sect;05 A-vs-B measurement: job 236204, <code>cs-venus-16</code> L40S, 5 shapes '
    'spanning occupancy (22k&ndash;635k occupied voxels at 256³) and emissive fraction '
    '(0.21%&ndash;99.1%). Script + <code>results.json</code> at '
    '<code>/3dlg-jupiter-project/lightgen/segvigen_emissive/ovotrain_ab_prototype/</code>, '
    'reusing <code>get_common_coords</code>, <code>loader.load_and_merge</code>, and '
    '<code>voxelize._tight_scene</code> unmodified.</li>'
    '</ul>',
    open=False,
)

# =============================================================================
# ================================================================ css
extra_css = """
/* warn/missing SVG variants (--accent2, the same token callout(warn=True) uses) */
.xg2 .dbox-warn { fill: color-mix(in srgb, var(--accent2) 14%, var(--surface));
  stroke: var(--accent2); stroke-width: 1.6; }
.xg2 .dbox-unused { fill: var(--surface); stroke: var(--ink-3); stroke-dasharray: 5 4; opacity: .65; }
.xg2 .dline-warn { stroke: var(--accent2); fill: none; stroke-width: 1.8; }
.xg2 .arrfill-warn { fill: var(--accent2); }
.xg2 .dsub-warn { fill: var(--accent2); font-size: 12.5px; font-weight: 700;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.xg2 .dsub-accent { fill: var(--accent-ink); font-size: 12.5px; font-weight: 700;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.xg2 .dtitle-dim { fill: var(--ink-3); font-size: 15px; font-weight: 600;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.xg2 .dtitle-warn { fill: var(--accent2); font-size: 15px; font-weight: 700;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.xg2 .dmono-dim { fill: var(--ink-3); font-size: 12px; opacity: .8;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.xg2 .dmono-warn { fill: var(--accent2); font-size: 12px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.xg2 .dcheck { fill: var(--accent); font-size: 20px; font-weight: 700; text-anchor: start; }
.xg2 .dx { fill: var(--accent2); font-size: 15px; font-weight: 800; text-anchor: start;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.xg2 .dband { fill: color-mix(in srgb, var(--accent) 7%, transparent);
  stroke: color-mix(in srgb, var(--accent) 30%, transparent); stroke-width: 1; }

/* cross-section panels (diagram 3) */
.xg2 .diagram-cross { margin: 10px 0 0; }
.xg2 .diagram-cross svg { min-width: 0; }
.xg2 .dcanvas { fill: var(--surface); }
.xg2 .dgrid { stroke: var(--line); stroke-width: 1; }
.xg2 .dcell-fill { fill: color-mix(in srgb, var(--ink) 24%, transparent); stroke: none; }
.xg2 .dcell-touched { fill: color-mix(in srgb, var(--accent) 10%, transparent); stroke: none; }
.xg2 .dcurve { stroke: var(--accent-ink); stroke-width: 2.2; fill: none; }
.xg2 .dcurve-ghost { stroke: var(--ink-3); stroke-width: 1.4; stroke-dasharray: 3 4; fill: none; opacity: .55; }
.xg2 .dtick { fill: var(--ink); opacity: .55; }
.xg2 .dvert { fill: var(--accent); stroke: var(--surface); stroke-width: 1.2; }
.cross-legend { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;
  margin: 10px auto 0; max-width: 820px; font-size: .78rem; color: var(--ink-2); }
.cl-item { display: inline-flex; align-items: center; gap: 6px; }
.cl-swatch { display: inline-block; width: 16px; height: 2px; border-radius: 1px; }
.cl-ghost { background: var(--ink-3); opacity: .7; }
.cl-recon { background: var(--accent-ink); }

/* section-05 comparison cards */
.cmp-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 8px auto 4px;
  max-width: var(--breakout-max, 972px); }
@media (max-width: 760px) { .cmp-wrap { grid-template-columns: 1fr; } }
.cmp-card { border-radius: 12px; padding: 16px 18px; background: color-mix(in srgb, var(--ink) 4%, transparent);
  border: 1px solid color-mix(in srgb, var(--ink) 12%, transparent); }
.cmp-h { font-weight: 700; font-size: .98rem; margin-bottom: 8px; }
.cmp-card p { font-size: .92rem; line-height: 1.55; opacity: .9; margin: 0 0 10px; }
.cmp-risk { font-size: .86rem; line-height: 1.5; border-left: 3px solid var(--accent2);
  padding: 6px 0 6px 12px; opacity: .92; }
.cmp-risk.cmp-good { border-left-color: var(--accent); }

/* provenance list */
.prov-list { font-size: .88rem; line-height: 1.6; padding-left: 1.1em; opacity: .92; }
.prov-list li { margin-bottom: 5px; }
"""

html_out = lp.page(
    title="What SegviGen training needs, and what we have — lightgen",
    header_html=hero,
    body_sections=[s_contract, s_coverage, s_crux, s_shared, s_open, provenance],
    theme="v2", assets_rel=ASSETS_REL, assets_dir=ASSETS_DIR,
    extra_head=f"<style>{extra_css}</style>",
)
out_path = os.path.join(OUT, "index.html")
with open(out_path, "w") as f:
    f.write(html_out)
print("wrote", out_path, f"({len(html_out)} bytes)")
