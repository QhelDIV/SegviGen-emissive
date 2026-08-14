"""Computed-geometry inline SVG diagrams for the SegviGen emissive experiment
overview page (xgpage v2 editorial).

All diagrams are schematic box-and-arrow / grid layouts (xgpage D16a: schematic
SVG for schematics needs no render-and-look escalation). Every coordinate comes
from a small column/row/grid table in code, not a hand-placed pixel guess.
Colors reference theme2.css tokens so figures re-skin in light/dark. The
resolution-ladder diagram is adapted from data_pipeline/diagrams.py
(diagram_resolution_ladder), reused as the brief instructs ("a starting point,
improve on it"); the other three diagrams are new to this page.
"""

ACCENT = "var(--accent)"
INK = "var(--ink)"
INK2 = "var(--ink-2)"
INK3 = "var(--ink-3)"
LINE = "var(--line)"
BLUE = "var(--blue)"
VIOLET = "var(--violet)"
GOOD = "var(--good)"
BAD = "var(--bad)"
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
# low-level SVG emitters (copied from data_pipeline/diagrams.py so this page
# stays self-contained per SKILL.md rule 11)
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


# ---------------------------------------------------------------------------
# Diagram 1: the three latent channels, target highlighted
def diagram_three_channels():
    W, H = 1020, 380
    parts = []
    gap = 40
    bw = (W - 2 * gap - 3 * 30) / 3
    y0 = 70
    bh = 210
    defs = [
        ("shape_slat", "&ldquo;where the surface is&rdquo;", "geometry, from the 512&sup3; dual grid",
         VIOLET, False),
        ("input_tex_slat", "&ldquo;how it reflects light&rdquo;", "reflectance (PBR input channels)",
         BLUE, False),
        ("output_tex_slat", "&ldquo;what it emits&rdquo;", "binarized emission, the PREDICTION TARGET",
         ACCENT, True),
    ]
    x = 30
    for i, (name, gloss, sub, col, is_target) in enumerate(defs):
        bx = x
        sw = 2.6 if is_target else 1.4
        fill = TILE
        parts.append(_rect(bx, y0, bw, bh, fill=fill, stroke=col, sw=sw, rx=12))
        parts.append(_text(bx + bw / 2, y0 + 34, name, 15.5, INK, "middle", "700",
                            family="ui-monospace,monospace"))
        parts.append(_line(bx + 18, y0 + 48, bx + bw - 18, y0 + 48, LINE, 1, opacity=0.7))
        parts.append(_mtext(bx + bw / 2, y0 + 78, [gloss], 14, col, "middle", 18, "650"))
        # wrap sub into two lines around a natural break
        words = sub.split(" ")
        line1, line2 = [], []
        target_len = len(sub) / 2
        cur = 0
        for w_ in words:
            if cur < target_len:
                line1.append(w_)
                cur += len(w_) + 1
            else:
                line2.append(w_)
        parts.append(_mtext(bx + bw / 2, y0 + 118, [" ".join(line1), " ".join(line2)], 11.8,
                             INK3, "middle", 16, "400"))
        if is_target:
            parts.append(_rect(bx + bw / 2 - 58, y0 + bh - 40, 116, 26, fill=ACCENT, stroke="none", rx=13))
            parts.append(_text(bx + bw / 2, y0 + bh - 22, "PREDICTED", 11, "#FFFFFF", "middle", "750"))
        x += bw + 30
    # a light connecting rule under all three, with "one latent token, three channels" label
    parts.append(_text(W / 2, y0 - 26, "one latent token, three token-aligned channels", 13.5, INK2, "middle", "600"))
    parts.append(_text(W / 2, H - 14,
                        "same coordinates across all three: the model reads shape_slat + input_tex_slat at a voxel, predicts output_tex_slat there",
                        11.6, INK3, "middle", "400"))
    return "".join(parts), f"0 0 {W} {H}"


# ---------------------------------------------------------------------------
# Diagram 2: resolution ladder (adapted from data_pipeline/diagrams.py)
def diagram_resolution_ladder():
    W, H = 1020, 700
    parts = []

    ox, oy = 60, 40
    cell = 30.0           # px per 512-cell (fine grid)
    n = 16                 # 16x16 fine cells shown = one latent token span
    grid_w = n * cell

    for i in range(n + 1):
        x = ox + i * cell
        parts.append(_line(x, oy, x, oy + grid_w, LINE, 0.9, opacity=0.85))
        y = oy + i * cell
        parts.append(_line(ox, y, ox + grid_w, y, LINE, 0.9, opacity=0.85))

    for i in range(0, n + 1, 2):
        x = ox + i * cell
        parts.append(_line(x, oy, x, oy + grid_w, BLUE, 2.0, opacity=0.85))
        y = oy + i * cell
        parts.append(_line(ox, y, ox + grid_w, y, BLUE, 2.0, opacity=0.85))

    parts.append(_rect(ox, oy, grid_w, grid_w, fill="none", stroke=ACCENT, sw=3.2, rx=0))

    lx = ox + grid_w + 50
    ly = oy + 10
    items = [
        (LINE, "1 cell = one 512&sup3; grid cell",
         ["the pretrained encoder's", "native, locked resolution"]),
        (BLUE, "2&times;2 cells = one 256&sup3; cell",
         ["512 / 256 = 2&times; per axis;", "Dongchen's attribute bake lives here"]),
        (ACCENT, "16&times;16 cells = one latent token",
         ["512 / 32 = 16&times; per axis", "(the 32&sup3; grid the model trains on)"]),
    ]
    for i, (col, title, sub_lines) in enumerate(items):
        yy = ly + i * 92
        parts.append(_rect(lx, yy, 26, 26, fill="none" if col == LINE else col,
                            stroke=col, sw=2.2, opacity=1))
        parts.append(_mtext(lx + 38, yy + 12, [title], 13.5, INK, "start", 16, "650"))
        parts.append(_mtext(lx + 38, yy + 32, sub_lines, 11.5, INK3, "start", 15, "400"))

    ay = oy + grid_w + 44
    parts.append(_text(ox, ay, "1 latent token = 16&sup3; of 512 cells (4,096 cells)", 15, INK, "start", "650"))
    parts.append(_text(ox, ay + 24, "= 8&sup3; of 256 cells (512 distinct attribute values)", 15, INK, "start", "650"))
    parts.append(_text(ox, ay + 50,
                        "shown as a 2D cross-section; the 2&times; and 16&times; factors apply along all three axes",
                        11.8, INK3, "start", "400"))

    return "".join(parts), f"0 0 {W} {H}"


# ---------------------------------------------------------------------------
# Diagram 3: the channel hijack (emission binarized into the base_color slot)
def diagram_channel_hijack():
    W, H = 1020, 460
    parts = []
    slot_names = ["base_color", "metallic", "roughness", "alpha"]
    n = len(slot_names)
    slot_w = 150
    gap = 26
    total_slots_w = n * slot_w + (n - 1) * gap
    slots_x0 = (W - total_slots_w) / 2

    # header: pretrained encoder box
    enc_w, enc_h = total_slots_w, 56
    enc_y = 30
    parts.append(_rect(slots_x0, enc_y, enc_w, enc_h, fill=TILE, stroke=INK2, sw=1.6, rx=10))
    parts.append(_text(W / 2, enc_y + 24, "pretrained PBR encoder", 14.5, INK, "middle", "700"))
    parts.append(_text(W / 2, enc_y + 42, "unchanged, reused byte-identically: 4 input slots", 11.3, INK3, "middle", "400"))

    # the 4 slots (as the encoder expects them, INPUT convention)
    slot_y = enc_y + enc_h + 40
    slot_h = 64
    for i, name in enumerate(slot_names):
        sx = slots_x0 + i * (slot_w + gap)
        parts.append(_rect(sx, slot_y, slot_w, slot_h, fill=TILE, stroke=LINE, sw=1.3, rx=8))
        parts.append(_text(sx + slot_w / 2, slot_y + 28, name, 12.6, INK, "middle", "650",
                            family="ui-monospace,monospace"))
        parts.append(_text(sx + slot_w / 2, slot_y + 46, "u8 channel", 10.6, INK3, "middle", "400"))
        parts.append(_line(sx + slot_w / 2, enc_y + enc_h + 4, sx + slot_w / 2, slot_y - 4, INK3, 1.2, dash="3,3", opacity=0.6))

    # output.vxz content written into each slot
    out_y = slot_y + slot_h + 56
    out_h = 96
    out_defs = [
        ("emission, binarized<br>at &gt;1/255", "any nonzero emission counts", ACCENT, True),
        ("OUT_METALLIC_U8 = 0", "module constant, not per-voxel", INK3, False),
        ("OUT_ROUGHNESS_U8 = 255", "module constant, not per-voxel", INK3, False),
        ("OUT_ALPHA_U8 = 255", "module constant, not per-voxel", INK3, False),
    ]
    parts.append(_text(slots_x0, out_y - 18, "output.vxz writes, per slot:", 12.5, INK2, "start", "600"))
    for i, (label, sub, col, is_hijack) in enumerate(out_defs):
        sx = slots_x0 + i * (slot_w + gap)
        sw_ = 2.4 if is_hijack else 1.2
        fill = TILE
        parts.append(_rect(sx, out_y, slot_w, out_h, fill=fill, stroke=col, sw=sw_, rx=8))
        label_lines = label.split("<br>")
        parts.append(_mtext(sx + slot_w / 2, out_y + 26, label_lines, 12.2, col if is_hijack else INK,
                             "middle", 16, "700" if is_hijack else "500",
                             family="ui-monospace,monospace" if not is_hijack else None))
        parts.append(_mtext(sx + slot_w / 2, out_y + (52 if len(label_lines) > 1 else 46),
                             [sub], 10.4, INK3, "middle", 13, "400"))
        parts.append(_arrow(sx + slot_w / 2, out_y - 4, sx + slot_w / 2, slot_y + slot_h + 4, col, 1.8))
        if is_hijack:
            parts.append(_rect(sx - 6, out_y - 6, slot_w + 12, out_h + 12, fill="none", stroke=ACCENT,
                                sw=1, rx=11, opacity=0.35, dash="4,3"))

    parts.append(_text(W / 2, out_y + out_h + 34,
                        "no architecture change, no new encoder: the target rides in a slot built for reflectance",
                        13, INK2, "middle", "600"))

    return "".join(parts), f"0 0 {W} {H}"


# ---------------------------------------------------------------------------
# Diagram 4: dataset build funnel, missing decomposed into three causes
def diagram_data_funnel():
    W, H = 1020, 560
    parts = []

    top_w, top_h = 460, 74
    top_x = (W - top_w) / 2
    top_y = 20
    parts.append(_box(top_x, top_y, top_w, top_h,
                       ["74,503 shapes in the split"],
                       ["train 59,602 &middot; val 7,450 &middot; test 7,451"],
                       fill=TILE, stroke=INK2, title_size=15.5))

    branch_y = top_y + top_h + 60
    built_w, built_h = 400, 86
    built_x = W * 0.28 - built_w / 2
    parts.append(_box(built_x, branch_y, built_w, built_h,
                       ["72,546 built"], ["train 57,968 &middot; val 7,290 &middot; test 7,288"],
                       fill=TILE, stroke=GOOD, title_size=16))
    parts.append(_arrow(top_x + top_w * 0.32, top_y + top_h + 4, built_x + built_w / 2, branch_y - 4, GOOD, 2.2))

    miss_w, miss_h = 300, 74
    miss_x = W * 0.76 - miss_w / 2
    parts.append(_box(miss_x, branch_y, miss_w, miss_h,
                       ["1,957 missing"], ["1,036 + 584 + 337"],
                       fill=TILE, stroke=BAD, title_size=16))
    parts.append(_arrow(top_x + top_w * 0.72, top_y + top_h + 4, miss_x + miss_w / 2, branch_y - 4, BAD, 2.2))

    # three sub-causes under "missing"
    sub_y = branch_y + miss_h + 60
    sub_w = 300
    sub_gap = 26
    total_sub_w = 3 * sub_w + 2 * sub_gap
    sub_x0 = (W - total_sub_w) / 2
    sub_h = 132
    sub_defs = [
        ("1,036", "source never existed", "permanent, known since the build was planned", BAD),
        ("584", "Dongchen's rebake completed", "but produced no output for these shapes", INK2),
        ("337", "buildable right now", "source present; ours to rebuild (recommended skip, +0.6%)", BLUE),
    ]
    for i, (n, title, sub, col) in enumerate(sub_defs):
        sx = sub_x0 + i * (sub_w + sub_gap)
        parts.append(_rect(sx, sub_y, sub_w, sub_h, fill=TILE, stroke=col, sw=1.5, rx=10))
        parts.append(_text(sx + sub_w / 2, sub_y + 34, n, 22, col, "middle", "750",
                            family="ui-monospace,monospace"))
        parts.append(_mtext(sx + sub_w / 2, sub_y + 58, [title], 12.6, INK, "middle", 16, "650"))
        words = sub.split(" ")
        line1, line2 = [], []
        cur = 0
        target_len = len(sub) / 2
        for w_ in words:
            if cur < target_len:
                line1.append(w_)
                cur += len(w_) + 1
            else:
                line2.append(w_)
        parts.append(_mtext(sx + sub_w / 2, sub_y + 92, [" ".join(line1), " ".join(line2)], 10.6,
                             INK3, "middle", 14, "400"))
        parts.append(_arrow(miss_x + miss_w * (0.2 + 0.3 * i), branch_y + miss_h + 4,
                             sx + sub_w / 2, sub_y - 4, col, 1.8))

    return "".join(parts), f"0 0 {W} {H}"
