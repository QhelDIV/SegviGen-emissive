"""Computed-geometry inline SVG diagrams for the lightgen emissive data
pipeline explainer (GLB -> ovoxels -> latent).

Both diagrams are schematic box-and-arrow / grid layouts (xgpage D16a: schematic
SVG for schematics needs no render-and-look escalation). Every coordinate comes
from a small column/row/grid table in code, not a hand-placed pixel guess.
Colors reference theme2.css tokens so figures re-skin in light/dark.
"""

ACCENT = "var(--accent)"
INK = "var(--ink)"
INK2 = "var(--ink-2)"
INK3 = "var(--ink-3)"
LINE = "var(--line)"
BLUE = "var(--blue)"
VIOLET = "var(--violet)"
GOOD = "var(--good)"
TILE = "var(--tile)"


def svg_figure(inner_svg, viewbox, caption_html, width_px=940, aspect=None, id=None):
    """Wrap raw inline SVG in a figure/figcaption using the .diagram class
    (theme2.css)."""
    width_px = max(width_px, 640)
    vb_parts = viewbox.split()
    vw, vh = float(vb_parts[2]), float(vb_parts[3])
    if aspect is None:
        aspect = vh / vw
    id_attr = f' id="{id}"' if id else ""
    cap = f'<figcaption>{caption_html}</figcaption>' if caption_html else ""
    return (
        f'<figure class="diagram" style="max-width:{width_px}px;margin-left:auto;'
        f'margin-right:auto"{id_attr}>'
        f'<svg viewBox="{viewbox}" style="aspect-ratio:{vw}/{vh}" role="img">{inner_svg}</svg>'
        f'{cap}</figure>'
    )


# ---------------------------------------------------------------------------
# low-level SVG emitters
def _rect(x, y, w, h, fill="none", stroke=INK, sw=1.5, rx=9, opacity=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{d}/>')


def _line(x1, y1, x2, y2, color=INK, width=1.5, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{d}/>')


def _text(x, y, s, size=13, color=INK, anchor="middle", weight="400", family=None, opacity=1):
    fam = f' font-family="{family}"' if family else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" opacity="{opacity}"{fam}>{s}</text>')


def _mtext(x, y, lines, size=12.5, color=INK2, anchor="middle", lh=16, weight="400", family=None):
    parts = []
    for i, s in enumerate(lines):
        parts.append(_text(x, y + i * lh, s, size, color, anchor, weight, family))
    return "".join(parts)


def _arrow(x1, y1, x2, y2, color=ACCENT, width=2, head=8, dash=None):
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - head * math.cos(ang - 0.4)
    hy1 = y2 - head * math.sin(ang - 0.4)
    hx2 = x2 - head * math.cos(ang + 0.4)
    hy2 = y2 - head * math.sin(ang + 0.4)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}" opacity="0.85"{d}/>'
        f'<polygon points="{x2:.2f},{y2:.2f} {hx1:.2f},{hy1:.2f} {hx2:.2f},{hy2:.2f}" '
        f'fill="{color}" opacity="0.85"/>'
    )


def _box(x, y, w, h, title_lines, sub_lines=None, fill=None, stroke=INK3, title_color=None,
          title_size=13.5, sub_size=11.3, sub_color=INK3):
    fill = fill or TILE
    title_color = title_color or INK
    parts = [_rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.3)]
    n_title = len(title_lines)
    n_sub = len(sub_lines) if sub_lines else 0
    total_lh = n_title * 16.5 + (6 if n_sub else 0) + n_sub * 14.5
    ty = y + h / 2 - total_lh / 2 + 12
    for i, s in enumerate(title_lines):
        parts.append(_text(x + w / 2, ty + i * 16.5, s, title_size, title_color, "middle", "650"))
    if sub_lines:
        sy = ty + n_title * 16.5 + 6
        for i, s in enumerate(sub_lines):
            parts.append(_text(x + w / 2, sy + i * 14.5, s, sub_size, sub_color, "middle", "400"))
    return "".join(parts)


def _lane_band(x, y, w, h, color, tag, label):
    parts = [_rect(x, y, w, h, fill=color, stroke="none", sw=0, rx=0, opacity=0.07)]
    parts.append(_rect(x, y, 4, h, fill=color, stroke="none", sw=0, rx=0, opacity=0.55))
    parts.append(_text(x + 16, y + 22, tag, 15.5, color, "start", "750"))
    parts.append(_text(x + 16, y + 40, label, 11.8, INK3, "start", "400"))
    return "".join(parts)


def _filecard(x, y, w, h, header, rows):
    parts = [_rect(x, y, w, h, fill=TILE, stroke=LINE, sw=1.3)]
    parts.append(_text(x + 14, y + 20, header, 11.3, INK3, "start", "650"))
    ly = y + 40
    for fn, sub in rows:
        parts.append(_text(x + 14, ly, fn, 12.3, INK, "start", "600", family="ui-monospace,monospace"))
        parts.append(_text(x + 14, ly + 15, sub, 10.8, INK3, "start", "400"))
        ly += 34
    return "".join(parts)


# ---------------------------------------------------------------------------
# Diagram 1: end-to-end flow, resolution lanes as horizontal bands
def diagram_pipeline_flow():
    W, H = 1300, 950
    parts = []

    # GLB sits ABOVE all three resolution lanes: it belongs to none of them.
    GLB_Y, GLB_H = 20, 110
    C1 = 40
    W1 = 170
    glb_cx = C1 + W1 / 2
    parts.append(_box(C1, GLB_Y, W1, GLB_H, ["GLB"], ["source mesh", "+ textures"],
                       fill=TILE, stroke=INK2, title_size=15))
    glb_bottom = GLB_Y + GLB_H

    # lane bands: (y0, h); each reserves a 48px label zone at its top,
    # content starts at y0+LZ
    LZ = 48
    LANE256 = (150, 260)
    LANE512 = (430, 300)
    LANE32 = (750, 180)
    parts.append(_lane_band(0, LANE256[0], W, LANE256[1], BLUE, "256³", "attribute VALUES come from here (Dongchen's bake)"))
    parts.append(_lane_band(0, LANE512[0], W, LANE512[1], VIOLET, "512³", "encoder's native grid; geometry extracted here"))
    parts.append(_lane_band(0, LANE32[0], W, LANE32[1], ACCENT, "32³", "the latent the model actually trains on"))

    for (y0, h) in (LANE256, LANE512, LANE32):
        parts.append(_line(0, y0 + h, W, y0 + h, LINE, 1, opacity=0.6))

    # column x positions
    C2, C3, C4, C5 = 250, 520, 800, 1080
    W2, W3, W4, W5 = 230, 260, 240, 190

    # lane 256: extraction call
    y256 = LANE256[0] + LZ + 12
    b256h = LANE256[1] - LZ - 24
    b256_proc = _box(C2, y256, W2, b256h, ["textured_mesh_to_", "volumetric_attr()"],
                      ["grid_size=256", "voxelize.py:218", "Dongchen · uv_voxel_pipeline"],
                      fill=TILE, stroke=BLUE)
    parts.append(b256_proc)
    parts.append(_arrow(glb_cx - 20, glb_bottom + 4, C2 + W2 * 0.28, y256 - 6, BLUE))

    fcard = _filecard(C3, y256, W3, b256h, "writes, per shape:", [
        ("pbr_voxels_256/&lt;sid&gt;.vxz", "base_color, metallic, roughness, alpha"),
        ("emission_voxels_256/&lt;sid&gt;.vxz", "emissive"),
        ("&lt;sid&gt;.coords.npz", "coords · byproduct, for the validator"),
    ])
    parts.append(fcard)
    parts.append(_arrow(C2 + W2 + 6, y256 + b256h / 2, C3 - 6, y256 + b256h / 2, BLUE))

    # lane 512: geometry extraction (direct from GLB)
    y512a = LANE512[0] + LZ + 12
    h512a = 100
    b512geom = _box(C2, y512a, W2, h512a, ["mesh_to_flexible_", "dual_grid()"],
                     ["grid_size=512", "build_dataset_direct.py:154"],
                     fill=TILE, stroke=VIOLET)
    parts.append(b512geom)
    parts.append(_arrow(glb_cx + 20, glb_bottom + 4, C2 + W2 * 0.28, y512a - 6, VIOLET))

    b512geom_out = _box(C3, y512a, W3, h512a, ["voxel_indices,", "dual_vertices, intersected"],
                         ["dual_vertices rescaled to u8", "lines 164–165"],
                         fill=TILE, stroke=LINE, title_size=12.5)
    parts.append(b512geom_out)
    parts.append(_arrow(C2 + W2 + 6, y512a + h512a / 2, C3 - 6, y512a + h512a / 2, VIOLET))

    # lane 512: upsampler (fed from 256 files, downward)
    y512b = y512a + h512a + 24
    h512b = 100
    b512up = _box(C3, y512b, W3, h512b, ["Upsampler256to512"],
                   ["parent lookup (coord//2) +", "cKDTree fallback · line 170"],
                   fill=TILE, stroke=BLUE, title_size=13)
    parts.append(b512up)
    parts.append(_arrow(C3 + W3 / 2, y256 + b256h + 6, C3 + W3 / 2, y512b - 6, BLUE))

    # merge box: input.vxz / output.vxz
    hmerge = h512a + 24 + h512b
    bmerge = _box(C4, y512a, W4, hmerge, ["input.vxz", "output.vxz"],
                   ["512 pair, IDENTICAL coords", "emission binarized (>1/255)", "into output.vxz's base_color slot"],
                   fill=TILE, stroke=VIOLET, title_size=14.5)
    parts.append(bmerge)
    parts.append(_arrow(C3 + W3 + 6, y512a + h512a / 2, C4 - 6, y512a + hmerge * 0.32, VIOLET))
    parts.append(_arrow(C3 + W3 + 6, y512b + h512b / 2, C4 - 6, y512a + hmerge * 0.68, BLUE))

    # lane 32: encoder
    y32 = LANE32[0] + LZ + 12
    h32 = LANE32[1] - LZ - 24
    benc = _box(C4, y32, W4, h32, ["vxz_to_slat()"], ["encoder, UNCHANGED", "SegviGen code"],
                fill=TILE, stroke=ACCENT, title_size=14.5)
    parts.append(benc)
    parts.append(_arrow(C4 + W4 / 2, y512a + hmerge + 6, C4 + W4 / 2, y32 - 6, ACCENT))

    # lane 32: three .pth outputs
    pth_h = 36
    pth_gap = 12
    total_pth = pth_h * 3 + pth_gap * 2
    y32p0 = y32 + h32 / 2 - total_pth / 2
    pth_defs = [
        ("shape_slat.pth", "where the surface is", VIOLET),
        ("input_tex_slat.pth", "how it reflects light", BLUE),
        ("output_tex_slat.pth", "what it emits", ACCENT),
    ]
    for i, (fn, gloss, col) in enumerate(pth_defs):
        py = y32p0 + i * (pth_h + pth_gap)
        parts.append(_rect(C5, py, W5, pth_h, fill=TILE, stroke=col, sw=1.6))
        parts.append(_text(C5 + W5 / 2, py + 15, fn, 11.6, INK, "middle", "650",
                            family="ui-monospace,monospace"))
        parts.append(_text(C5 + W5 / 2, py + 29, "“" + gloss + "”", 10.6, INK3, "middle", "400"))
        parts.append(_arrow(C4 + W4 + 6, y32 + h32 / 2, C5 - 6, py + pth_h / 2, col))

    return "".join(parts), f"0 0 {W} {H}"


# ---------------------------------------------------------------------------
# Diagram 2: resolution ladder nesting arithmetic (2D cross-section)
def diagram_resolution_ladder():
    W, H = 1020, 700
    parts = []

    ox, oy = 60, 40
    cell = 30.0           # px per 512-cell (fine grid)
    n = 16                 # 16x16 fine cells shown = one latent token span
    grid_w = n * cell

    # fine grid (512)
    for i in range(n + 1):
        x = ox + i * cell
        parts.append(_line(x, oy, x, oy + grid_w, LINE, 0.9, opacity=0.85))
        y = oy + i * cell
        parts.append(_line(ox, y, ox + grid_w, y, LINE, 0.9, opacity=0.85))

    # 256-cell boundaries: every 2 fine cells (2x finer than 256)
    for i in range(0, n + 1, 2):
        x = ox + i * cell
        parts.append(_line(x, oy, x, oy + grid_w, BLUE, 2.0, opacity=0.85))
        y = oy + i * cell
        parts.append(_line(ox, y, ox + grid_w, y, BLUE, 2.0, opacity=0.85))

    # outer box: the whole 16x16 block = one 32-latent token
    parts.append(_rect(ox, oy, grid_w, grid_w, fill="none", stroke=ACCENT, sw=3.2, rx=0))

    # legend / callouts to the right
    lx = ox + grid_w + 50
    ly = oy + 10
    items = [
        (LINE, "1 cell = one 512³ grid cell",
         ["the pretrained encoder's", "native resolution"]),
        (BLUE, "2×2 cells = one 256³ cell",
         ["512 / 256 = 2× per axis;", "attribute values live here"]),
        (ACCENT, "16×16 cells = one latent token",
         ["512 / 32 = 16× per axis", "(the 32³ grid)"]),
    ]
    for i, (col, title, sub_lines) in enumerate(items):
        yy = ly + i * 92
        parts.append(_rect(lx, yy, 26, 26, fill="none" if col == LINE else col,
                            stroke=col, sw=2.2, opacity=1))
        parts.append(_mtext(lx + 38, yy + 12, [title], 13.5, INK, "start", 16, "650"))
        parts.append(_mtext(lx + 38, yy + 32, sub_lines, 11.5, INK3, "start", 15, "400"))

    # arithmetic strip at bottom
    ay = oy + grid_w + 44
    parts.append(_text(ox, ay, "1 latent token = 16³ of 512 cells (4,096 cells)", 15, INK, "start", "650"))
    parts.append(_text(ox, ay + 24, "= 8³ of 256 cells (512 distinct attribute values)", 15, INK, "start", "650"))
    parts.append(_text(ox, ay + 50,
                        "shown as a 2D cross-section; the 2× and 16× factors apply along all three axes",
                        11.8, INK3, "start", "400"))

    return "".join(parts), f"0 0 {W} {H}"
