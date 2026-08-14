"""Computed inline-SVG diagrams for the LightGen paper-skeleton page.

Every diagram here is a schematic: boxes, arrows, grids and charts whose
coordinates come from a table or a loop in code, never a hand-placed guess
(design law D16a). Colors are theme2.css tokens, so each figure re-skins itself
in light and dark. Nothing here stands in for real data: the one chart in this
module (`diagram_coverage`) plots numbers taken verbatim from the content of
record, and its derived bin masses are differences of those numbers, stated as
such in the caption.

Every viewBox is 820 units wide so the page's `.diagram svg { min-width }`
override renders the labels near 1:1 at phone width instead of shrinking them
below the readable floor.
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
SURFACE = "var(--surface)"
PANEL_BG = "#F2F1EC"   # fixed: see diagram_mask_albedo

W = 820
MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

# The page's breakout grid line. On the v3 workspace shell the content column is
# the 820px measure (theme3.css remaps --breakout-max), so a figure carrying a
# wider inline max-width would run past the column instead of centering in it.
# Every viewBox is already 820 units wide, so at this cap the diagrams render
# 1:1 and their labels keep their declared size.
MAX_FIG_PX = 820


def svg_figure(inner_svg, viewbox, caption_html, width_px=MAX_FIG_PX, id=None):
    """Wrap raw inline SVG in a figure/figcaption using the .diagram class."""
    width_px = min(width_px, MAX_FIG_PX)
    vb = viewbox.split()
    vw, vh = float(vb[2]), float(vb[3])
    id_attr = f' id="{id}"' if id else ""
    cap = f"<figcaption>{caption_html}</figcaption>" if caption_html else ""
    return (
        f'<figure class="diagram" style="max-width:{width_px}px;margin-left:auto;'
        f'margin-right:auto"{id_attr}>'
        f'<svg viewBox="{viewbox}" style="aspect-ratio:{vw}/{vh}" role="img">{inner_svg}</svg>'
        f"{cap}</figure>"
    )


# --------------------------------------------------------------- primitives
def _rect(x, y, w, h, fill="none", stroke=INK, sw=1.5, rx=9, opacity=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{d}/>')


def _line(x1, y1, x2, y2, color=INK, width=1.5, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{d}/>')


def _text(x, y, s, size=14, color=INK, anchor="middle", weight="400", family=None,
          opacity=1, style=None):
    fam = f' font-family="{family}"' if family else ""
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" opacity="{opacity}"{fam}{st}>{s}</text>')


def _mtext(x, y, lines, size=12.5, color=INK2, anchor="middle", lh=16, weight="400",
           family=None):
    return "".join(_text(x, y + i * lh, s, size, color, anchor, weight, family)
                   for i, s in enumerate(lines))


def _arrow(x1, y1, x2, y2, color=ACCENT, width=2, head=9, dash=None):
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    hx1, hy1 = x2 - head * math.cos(ang - 0.4), y2 - head * math.sin(ang - 0.4)
    hx2, hy2 = x2 - head * math.cos(ang + 0.4), y2 - head * math.sin(ang + 0.4)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="0.9"{d}/>'
            f'<polygon points="{x2:.2f},{y2:.2f} {hx1:.2f},{hy1:.2f} {hx2:.2f},{hy2:.2f}" '
            f'fill="{color}" opacity="0.9"/>')


def _wrap(text, budget):
    """Greedy wrap into lines of at most `budget` characters."""
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) > budget and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _box(x, y, w, h, title, sub=None, stroke=INK3, sw=1.4, fill=TILE,
         title_color=None, title_size=14.5, sub_size=12, dash=None, mono_title=False):
    """A titled tile; `sub` may be a string (wrapped) or a list of lines."""
    parts = [_rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=10, dash=dash)]
    tlines = title if isinstance(title, list) else [title]
    budget = max(int(w * 0.92 / (sub_size * 0.56)), 8)
    slines = sub if isinstance(sub, list) else (_wrap(sub, budget) if sub else [])
    total = len(tlines) * 18 + (8 if slines else 0) + len(slines) * 15
    ty = y + h / 2 - total / 2 + 13
    parts.append(_mtext(x + w / 2, ty, tlines, title_size, title_color or INK, "middle", 18,
                        "650", MONO if mono_title else None))
    if slines:
        parts.append(_mtext(x + w / 2, ty + len(tlines) * 18 + 8, slines, sub_size, INK3,
                            "middle", 15, "400"))
    return "".join(parts)


# ==========================================================================
# 1. The generation pipeline: two stages exist, the third does not (or is ours)
def diagram_pipeline(with_ours=False):
    H = 330
    parts = []
    bw, bh, gap = 226, 118, 34
    x0 = (W - 3 * bw - 2 * gap) / 2
    y0 = 78
    stages = [
        ("Geometry", "sparse 3D structure, from an image", BLUE),
        ("PBR material", "albedo, metallic, roughness", VIOLET),
        ("Emission", "which surface emits, and what it emits", ACCENT),
    ]
    for i, (name, sub, col) in enumerate(stages):
        x = x0 + i * (bw + gap)
        third = i == 2
        if third and not with_ours:
            parts.append(_box(x, y0, bw, bh, ["Emission"], "no stage produces it",
                              stroke=INK3, sw=1.6, fill="none", dash="6,5",
                              title_color=INK3))
        else:
            parts.append(_box(x, y0, bw, bh, [name], sub, stroke=col,
                              sw=2.4 if third else 1.6))
        if i:
            parts.append(_arrow(x - gap + 5, y0 + bh / 2, x - 6, y0 + bh / 2,
                                INK3 if not (third and not with_ours) else INK3,
                                2, dash="5,4" if (third and not with_ours) else None))
        parts.append(_text(x + bw / 2, y0 - 22, f"stage {i + 1}", 12.5, INK3, "middle",
                           "600", MONO))

    # provenance rails under the stages
    rail_y = y0 + bh + 26
    parts.append(_line(x0, rail_y, x0 + 2 * bw + gap, rail_y, INK3, 2))
    parts.append(_text(x0 + (2 * bw + gap) / 2, rail_y + 22,
                       "TRELLIS.2 ships these two", 13.5, INK2, "middle", "600"))
    x3 = x0 + 2 * (bw + gap)
    parts.append(_line(x3, rail_y, x3 + bw, rail_y, ACCENT if with_ours else INK3, 2,
                       dash=None if with_ours else "6,5"))
    parts.append(_text(x3 + bw / 2, rail_y + 22,
                       "this paper" if with_ours else "nothing ships this",
                       13.5, ACCENT if with_ours else INK3, "middle", "700"))

    if with_ours:
        parts.append(_text(W / 2, H - 26,
                           "a third stage drops in without regenerating geometry or PBR",
                           13.5, INK2, "middle", "600"))
    else:
        parts.append(_text(W / 2, H - 26,
                           "a generated lamp reflects light correctly and emits nothing",
                           13.5, INK2, "middle", "600"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 2. The bake drops emission strength
def diagram_strength():
    H = 320
    parts = []
    bw, bh = 300, 150
    lx, rx_ = 40, W - 40 - bw
    y0 = 56

    parts.append(_rect(lx, y0, bw, bh, fill=TILE, stroke=INK2, sw=1.6, rx=10))
    parts.append(_text(lx + bw / 2, y0 + 30, "source GLB", 15, INK, "middle", "700"))
    parts.append(_line(lx + 16, y0 + 44, lx + bw - 16, y0 + 44, LINE, 1))
    parts.append(_text(lx + bw / 2, y0 + 70, "emissive texture", 13.5, INK2, "middle", "600"))
    parts.append(_text(lx + bw / 2, y0 + 88, "8-bit color", 12, INK3, "middle", "400"))
    parts.append(_text(lx + bw / 2, y0 + 116, "KHR_materials_emissive_strength", 12.2, ACCENT,
                       "middle", "700", MONO))
    parts.append(_text(lx + bw / 2, y0 + 133, "a scalar multiplier on that color", 12, INK3,
                       "middle", "400"))

    parts.append(_arrow(lx + bw + 10, y0 + bh / 2, rx_ - 10, y0 + bh / 2, INK3, 2.2))
    parts.append(_text((lx + bw + rx_) / 2, y0 + bh / 2 - 14, "bake", 13, INK2, "middle", "650"))

    parts.append(_rect(rx_, y0, bw, bh, fill=TILE, stroke=INK2, sw=1.6, rx=10))
    parts.append(_text(rx_ + bw / 2, y0 + 30, "our voxel target", 15, INK, "middle", "700"))
    parts.append(_line(rx_ + 16, y0 + 44, rx_ + bw - 16, y0 + 44, LINE, 1))
    parts.append(_text(rx_ + bw / 2, y0 + 70, "emission color", 13.5, INK2, "middle", "600"))
    parts.append(_text(rx_ + bw / 2, y0 + 88, "8-bit, bounded above by 255", 12, INK3,
                       "middle", "400"))
    parts.append(_text(rx_ + bw / 2, y0 + 116, "KHR_materials_emissive_strength", 12.2, BAD,
                       "middle", "700", MONO))
    parts.append(_line(rx_ + 24, y0 + 112, rx_ + bw - 24, y0 + 112, BAD, 1.6))
    parts.append(_text(rx_ + bw / 2, y0 + 133, "dropped: strength is not preserved", 12, BAD,
                       "middle", "600"))

    # the measured fact strip
    fy = y0 + bh + 44
    parts.append(_rect(lx, fy, W - 2 * lx, 58, fill=SURFACE, stroke=LINE, sw=1.2, rx=10))
    parts.append(_text(W / 2, fy + 25, "3 of 60 sampled source GLBs carry the extension",
                       14, INK, "middle", "650"))
    parts.append(_text(W / 2, fy + 45,
                       "restoring strength is a change to the data pipeline, not to the model",
                       12.2, INK3, "middle", "400"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 3. Two ways to score an emissive asset
def diagram_eval():
    H = 400
    parts = []
    pad = 40
    pw = (W - 2 * pad - 30) / 2
    y0 = 62

    for i, (title, verdict, col) in enumerate([
            ("compare the texture in isolation", "says nothing about the room", INK3),
            ("render it as the only light source", "scores what the asset does", ACCENT)]):
        x = pad + i * (pw + 30)
        parts.append(_rect(x, y0, pw, 268, fill="none", stroke=col, sw=2.0 if i else 1.4,
                           rx=12, dash=None if i else "6,5"))
        parts.append(_text(x + pw / 2, y0 - 20, ["A", "B"][i], 13, col, "middle", "700", MONO))
        parts.append(_text(x + pw / 2, y0 + 28, title, 14.5, INK, "middle", "650"))

        cy = y0 + 130
        if i == 0:
            # two maps side by side with a metric between them
            mw = 92
            for j, lab in enumerate(["prediction", "ground truth"]):
                mx = x + pw / 2 + (j * 2 - 1) * (mw / 2 + 34) - mw / 2
                parts.append(_rect(mx, cy - mw / 2, mw, mw, fill=SURFACE, stroke=LINE, sw=1.2, rx=6))
                for k in range(1, 4):
                    parts.append(_line(mx, cy - mw / 2 + k * mw / 4, mx + mw,
                                       cy - mw / 2 + k * mw / 4, LINE, 1))
                    parts.append(_line(mx + k * mw / 4, cy - mw / 2, mx + k * mw / 4,
                                       cy + mw / 2, LINE, 1))
                parts.append(_text(mx + mw / 2, cy + mw / 2 + 18, lab, 11.8, INK3, "middle", "400"))
            parts.append(_text(x + pw / 2, cy + 5, "IoU", 15, INK2, "middle", "700", MONO))
        else:
            # a room box, the asset inside it, rays out to the walls
            rw, rh = 210, 132
            rxx, ryy = x + pw / 2 - rw / 2, cy - rh / 2
            parts.append(_rect(rxx, ryy, rw, rh, fill=SURFACE, stroke=INK3, sw=1.4, rx=6))
            ox, oy = rxx + rw / 2, ryy + rh / 2 + 12
            import math
            for k in range(9):
                a = math.pi * (0.06 + 0.88 * k / 8)
                parts.append(_line(ox, oy, ox - 74 * math.cos(a), oy - 74 * math.sin(a),
                                   ACCENT, 1.4, opacity=0.55))
            parts.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="13" fill="{ACCENT}" '
                         f'opacity="0.9"/>')
            parts.append(_text(ox, ryy + rh + 18, "the asset is the only light",
                               11.8, INK3, "middle", "400"))

        parts.append(_text(x + pw / 2, y0 + 244, verdict, 12.6,
                           col if i else INK3, "middle", "650"))

    parts.append(_text(W / 2, H - 18,
                       "an emissive object cannot be judged alone: it has to be put somewhere",
                       13.5, INK2, "middle", "600"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 4. The two usable modes
def diagram_modes():
    H = 344
    parts = []
    bw, bh, gap = 214, 100, 74
    y = [52, 168]
    rows = [
        ("image", "a photograph or render", "3D shape with emission",
         "geometry, PBR and emission together"),
        ("existing 3D shape", "geometry and PBR already given", "emission for it",
         "no reference image of it glowing"),
    ]
    x0 = (W - 2 * bw - gap) / 2
    for i, (lt, ls, rt, rs) in enumerate(rows):
        parts.append(_text(x0 - 14, y[i] + bh / 2, f"mode {i + 1}", 12.5, INK3, "end", "600", MONO))
        parts.append(_box(x0, y[i], bw, bh, [lt], ls, stroke=INK3, sw=1.4))
        parts.append(_arrow(x0 + bw + 8, y[i] + bh / 2, x0 + bw + gap - 8, y[i] + bh / 2,
                            ACCENT, 2.2))
        parts.append(_box(x0 + bw + gap, y[i], bw, bh, [rt], rs, stroke=ACCENT, sw=2.0))
    parts.append(_text(W / 2, H - 46,
                       "mode 2 is the one a generation pipeline needs", 14, INK, "middle", "650"))
    parts.append(_text(W / 2, H - 24,
                       "inside a pipeline there is no photograph of the finished object to hand",
                       12.4, INK3, "middle", "400"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 5. EmissionGen's task setup against ours
def diagram_positioning():
    H = 372
    parts = []
    bw, bh, gap = 168, 92, 46
    x0 = 128
    rows = [
        ("EmissionGen", BLUE, [
            ("reference image", "already shows the glow"),
            ("multi-view generation", "Hunyuan3D-2.1 Paint"),
            ("fused to UV", "an emission texture"),
        ]),
        ("ours", ACCENT, [
            ("geometry and PBR", "no lit reference exists"),
            ("sparse 3D latent", "TRELLIS.2's own space"),
            ("per-voxel mask", "composes with the PBR stage"),
        ]),
    ]
    for r, (name, col, stages) in enumerate(rows):
        y = 66 + r * 148
        parts.append(_text(x0 - 18, y + bh / 2 - 6, name, 14.5, col, "end", "700"))
        parts.append(_text(x0 - 18, y + bh / 2 + 12, ["arXiv 2604.11006", "this paper"][r],
                           11.5, INK3, "end", "400"))
        for i, (t, s) in enumerate(stages):
            x = x0 + i * (bw + gap)
            parts.append(_box(x, y, bw, bh, [t], s, stroke=col, sw=1.9 if i == 0 else 1.4,
                              title_size=13.5, sub_size=11.5))
            if i:
                parts.append(_arrow(x - gap + 6, y + bh / 2, x - 7, y + bh / 2, col, 1.9))
        parts.append(_rect(x0 - 8, y - 8, bw + 16, bh + 16, fill="none", stroke=col, sw=1,
                           rx=13, opacity=0.4, dash="4,3"))
    parts.append(_text(W / 2, H - 24,
                       "the dashed frame marks the input: theirs contains the answer, ours does not",
                       13.5, INK2, "middle", "600"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 6. Dataset build funnel
def diagram_funnel():
    H = 470
    parts = []
    top_w, top_h = 380, 74
    top_x, top_y = (W - top_w) / 2, 18
    parts.append(_box(top_x, top_y, top_w, top_h, ["74,503 shapes in the split"],
                      ["train 59,602 &middot; val 7,450 &middot; test 7,451"],
                      stroke=INK2, sw=1.6, title_size=15.5))

    by = top_y + top_h + 52
    built_w, built_h = 344, 82
    built_x = W * 0.28 - built_w / 2
    parts.append(_box(built_x, by, built_w, built_h, ["72,546 built"],
                      ["train 57,968 &middot; val 7,290 &middot; test 7,288"],
                      stroke=GOOD, sw=1.8, title_size=16))
    parts.append(_arrow(top_x + top_w * 0.32, top_y + top_h + 4, built_x + built_w / 2, by - 5,
                        GOOD, 2.2))

    miss_w, miss_h = 232, 74
    miss_x = W * 0.775 - miss_w / 2
    parts.append(_box(miss_x, by, miss_w, miss_h, ["1,957 missing"], ["1,036 + 584 + 337"],
                      stroke=BAD, sw=1.8, title_size=16))
    parts.append(_arrow(top_x + top_w * 0.70, top_y + top_h + 4, miss_x + miss_w / 2, by - 5,
                        BAD, 2.2))

    sy = by + miss_h + 56
    sub_w, sub_gap, sub_h = 244, 22, 128
    sx0 = (W - 3 * sub_w - 2 * sub_gap) / 2
    subs = [
        ("1,036", "never had source data", BAD),
        ("584", "in a rebake that finished with no output", INK2),
        ("337", "buildable, and 0.5% of the dataset", BLUE),
    ]
    # an elbow bus, so no branch line crosses the "built" box on its way left
    bus_y = sy - 30
    parts.append(_line(miss_x + miss_w / 2, by + miss_h, miss_x + miss_w / 2, bus_y, BAD, 1.7))
    parts.append(_line(sx0 + sub_w / 2, bus_y, sx0 + 2 * (sub_w + sub_gap) + sub_w / 2,
                       bus_y, BAD, 1.7))
    for i, (n, title, col) in enumerate(subs):
        x = sx0 + i * (sub_w + sub_gap)
        parts.append(_rect(x, sy, sub_w, sub_h, fill=TILE, stroke=col, sw=1.5, rx=10))
        parts.append(_text(x + sub_w / 2, sy + 40, n, 24, col, "middle", "750", MONO))
        parts.append(_mtext(x + sub_w / 2, sy + 70, _wrap(title, 28), 12.6, INK2, "middle",
                            17, "500"))
        parts.append(_arrow(x + sub_w / 2, bus_y, x + sub_w / 2, sy - 5, col, 1.7))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 7. The coverage distribution: two populations, and the median-mean gap
SURVIVAL = [("&gt; 0", 90.8), ("&gt; 0.001", 79.7), ("&gt; 0.01", 59.3),
            ("&gt; 0.1", 36.9), ("&gt; 0.5", 22.9)]
MEDIAN, MEAN = 0.025, 0.244


def coverage_bins():
    """Per-bin share of shapes, by subtraction from the reported survival
    percentages. Returns (interval label, percent) newest-bin last."""
    vals = [100.0] + [v for _, v in SURVIVAL]
    labels = ["exactly 0", "0 to 0.001", "0.001 to 0.01", "0.01 to 0.1",
              "0.1 to 0.5", "0.5 to 1"]
    return [(labels[i], round(vals[i] - vals[i + 1], 1)) for i in range(5)] + \
           [(labels[5], SURVIVAL[-1][1])]


def diagram_coverage():
    H = 570
    parts = []
    bins = coverage_bins()
    pad_l, pad_r = 66, 40
    plot_w = W - pad_l - pad_r
    base_y, plot_h = 250, 178
    n = len(bins)
    slot = plot_w / n
    bar_w = slot * 0.62
    vmax = 25.0

    parts.append(_text(pad_l, 34, "share of shapes in each emissive-coverage band",
                       14.5, INK, "start", "650"))
    parts.append(_text(pad_l, 54, "n = 1,998 sampled training shapes", 12.2, INK3, "start", "400"))

    for g in range(0, 26, 5):
        y = base_y - plot_h * g / vmax
        parts.append(_line(pad_l, y, pad_l + plot_w, y, LINE, 1))
        parts.append(_text(pad_l - 10, y + 4, f"{g}%", 11.8, INK3, "end", "400"))

    for i, (lab, pct) in enumerate(bins):
        x = pad_l + i * slot + (slot - bar_w) / 2
        h = plot_h * pct / vmax
        end = i != 4          # the trough band, 0.1 to 0.5, is the muted one
        parts.append(_rect(x, base_y - h, bar_w, h,
                           fill=ACCENT if end else INK3,
                           stroke="none", sw=0, rx=3, opacity=1 if end else 0.45))
        parts.append(_text(x + bar_w / 2, base_y - h - 9, f"{pct}%", 13, INK, "middle", "700"))
        parts.append(_mtext(x + bar_w / 2, base_y + 20, _wrap(lab, 13), 11.8, INK2,
                            "middle", 14, "500"))
    parts.append(_line(pad_l, base_y, pad_l + plot_w, base_y, INK3, 1.4))

    # grouping brackets: the two populations and the trough between them
    gy = base_y + 62
    groups = [(0, 4, "63.1% at or below 0.1", ACCENT),
              (4, 5, "14.0%", INK3),
              (5, 6, "22.9% above 0.5", ACCENT)]
    for i0, i1, lab, col in groups:
        x1 = pad_l + i0 * slot + 4
        x2 = pad_l + i1 * slot - 4
        parts.append(_line(x1, gy, x2, gy, col, 2))
        parts.append(_line(x1, gy, x1, gy - 7, col, 2))
        parts.append(_line(x2, gy, x2, gy - 7, col, 2))
        parts.append(_text((x1 + x2) / 2, gy + 19, lab, 12.6, col, "middle", "700"))

    # the coverage number line, median against mean
    ny = 512
    nx0, nx1 = pad_l, pad_l + plot_w
    parts.append(_text(pad_l, ny - 96, "where the two summary statistics land",
                       14.5, INK, "start", "650"))
    parts.append(_line(nx0, ny, nx1, ny, INK3, 1.6))
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        x = nx0 + (nx1 - nx0) * t
        parts.append(_line(x, ny, x, ny + 7, INK3, 1.2))
        parts.append(_text(x, ny + 24, f"{t:g}", 11.8, INK3, "middle", "400"))
    xa = nx0 + (nx1 - nx0) * MEDIAN
    xb = nx0 + (nx1 - nx0) * MEAN
    for x, lab, anchor in [(xa, f"median {MEDIAN}", "start"), (xb, f"mean {MEAN}", "start")]:
        parts.append(_line(x, ny - 26, x, ny + 3, ACCENT, 2.4))
        parts.append(_text(x + 7, ny - 14, lab, 13, ACCENT, anchor, "700"))
    parts.append(_line(xa, ny - 44, xb, ny - 44, ACCENT, 1.2, dash="4,3"))
    parts.append(_text(xb + 16, ny - 40,
                       "the gap between them is the label noise", 12.4, INK2, "start", "600"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 8. emission = mask x albedo, on a voxel cross-section
def diagram_mask_albedo():
    """A 9x9 cross-section of a lamp: the shade emits, the stem and base do not.
    Cell classes and colors come from the grid table below, not from pixels."""
    H = 366
    parts = []
    # . empty, s shade (emits), t stem, b base
    grid = [
        ".........",
        "...sss...",
        "..sssss..",
        ".sssssss.",
        "....t....",
        "....t....",
        "....t....",
        "...bbb...",
        "..bbbbb..",
    ]
    fills = {"s": "#E8C88A", "t": "#9A9488", "b": "#6E6960"}
    cell = 21
    gw = 9 * cell
    y0 = 92
    panels = [
        ("albedo, given", "a"),
        ("mask, predicted", "m"),
        ("emission = mask &times; albedo", "p"),
    ]
    op_gap = 62
    total = 3 * gw + 2 * op_gap
    x0 = (W - total) / 2
    for pi, (label, kind) in enumerate(panels):
        px = x0 + pi * (gw + op_gap)
        # the panel background is a FIXED light tone, not the theme token: the cell
        # colors are fixed hexes, so a panel that flipped to near-black in dark
        # mode would swallow the "does not emit" cells (checked on the live page).
        parts.append(_rect(px - 6, y0 - 6, gw + 12, gw + 12, fill=PANEL_BG, stroke=LINE,
                           sw=1.2, rx=8))
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch == ".":
                    continue
                x, y = px + c * cell, y0 + r * cell
                if kind == "a":
                    f = fills[ch]
                elif kind == "m":
                    f = "#FFFFFF" if ch == "s" else "#2A2926"
                else:
                    f = fills[ch] if ch == "s" else "#111111"
                parts.append(_rect(x, y, cell - 1.5, cell - 1.5, fill=f, stroke="#C6C2B6",
                                   sw=0.6, rx=2))
        parts.append(_text(px + gw / 2, y0 - 22, label, 13.5, INK, "middle", "650"))
        if pi:
            ox = px - op_gap / 2
            parts.append(_text(ox, y0 + gw / 2 + 8, "&times;" if pi == 1 else "=", 26,
                               ACCENT, "middle", "600"))
    sub = [
        ("albedo, given", "already in the PBR input; the model never generates a color"),
        ("mask, predicted", "one bit per occupied voxel: does this surface emit"),
        ("emission = mask &times; albedo", "the input color, restricted to the mask"),
    ]
    for pi, (_, s) in enumerate(sub):
        px = x0 + pi * (gw + op_gap)
        parts.append(_mtext(px + gw / 2, y0 + gw + 32, _wrap(s, 30), 11.8, INK3, "middle",
                            15, "400"))
    parts.append(_text(W / 2, 40,
                       "white in the mask panel is an emitting voxel; dark is occupied and not emitting",
                       12.4, INK3, "middle", "400"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 9. The channel hijack
def diagram_hijack():
    H = 400
    parts = []
    slots = ["base_color", "metallic", "roughness", "alpha"]
    n = len(slots)
    sw_, gap = 168, 20
    total = n * sw_ + (n - 1) * gap
    sx0 = (W - total) / 2

    enc_y, enc_h = 24, 54
    parts.append(_rect(sx0, enc_y, total, enc_h, fill=TILE, stroke=INK2, sw=1.6, rx=10))
    parts.append(_text(W / 2, enc_y + 24, "pretrained PBR encoder", 15, INK, "middle", "700"))
    parts.append(_text(W / 2, enc_y + 42, "reused byte-identically: four input slots", 12, INK3,
                       "middle", "400"))

    slot_y, slot_h = enc_y + enc_h + 36, 56
    for i, name in enumerate(slots):
        x = sx0 + i * (sw_ + gap)
        parts.append(_rect(x, slot_y, sw_, slot_h, fill=TILE, stroke=LINE, sw=1.3, rx=8))
        parts.append(_text(x + sw_ / 2, slot_y + 25, name, 13, INK, "middle", "650", MONO))
        parts.append(_text(x + sw_ / 2, slot_y + 43, "u8 channel", 11.5, INK3, "middle", "400"))
        parts.append(_line(x + sw_ / 2, enc_y + enc_h + 4, x + sw_ / 2, slot_y - 4, INK3,
                           1.2, dash="3,3", opacity=0.6))

    out_y, out_h = slot_y + slot_h + 52, 92
    outs = [
        (["the binary target"], "emission, binarized at &gt; 1/255", ACCENT, True),
        (["metallic = 0"], "a module constant", INK3, False),
        (["roughness = 255"], "a module constant", INK3, False),
        (["alpha = 255"], "a module constant", INK3, False),
    ]
    for i, (title, sub, col, hijack) in enumerate(outs):
        x = sx0 + i * (sw_ + gap)
        parts.append(_rect(x, out_y, sw_, out_h, fill=TILE, stroke=col, sw=2.4 if hijack else 1.2,
                           rx=8))
        parts.append(_mtext(x + sw_ / 2, out_y + 32, title, 13.2, col if hijack else INK,
                            "middle", 16, "700" if hijack else "550",
                            None if hijack else MONO))
        parts.append(_mtext(x + sw_ / 2, out_y + 56, _wrap(sub, 24), 11.5, INK3, "middle",
                            14, "400"))
        parts.append(_arrow(x + sw_ / 2, out_y - 4, x + sw_ / 2, slot_y + slot_h + 4, col, 1.8))
    parts.append(_text(W / 2, H - 14,
                       "no new encoder, no new VAE, no architecture change",
                       13.5, INK2, "middle", "600"))
    return "".join(parts), f"0 0 {W} {H}"


# ==========================================================================
# 10. The conditioning asymmetry
def diagram_conditioning():
    H = 312
    parts = []
    bw, bh, gap = 190, 96, 66
    x0 = 168
    rows = [
        ("the baselines", BLUE, "thumbnail", "the emissive region blows out to white",
         "conditioned", "the region is visible in their input"),
        ("ours", ACCENT, "zeros", "the DINOv3 path is unbuilt", "unconditioned",
         "geometry and PBR only"),
    ]
    for r, (name, col, lt, ls, rt, rs) in enumerate(rows):
        y = 56 + r * 120
        parts.append(_text(x0 - 18, y + bh / 2, name, 14, col, "end", "700"))
        parts.append(_box(x0, y, bw, bh, [lt], ls, stroke=col, sw=1.8, title_size=13.5,
                          sub_size=11.5))
        parts.append(_arrow(x0 + bw + 8, y + bh / 2, x0 + bw + gap - 8, y + bh / 2, col, 2))
        parts.append(_box(x0 + bw + gap, y, bw, bh, [rt], rs, stroke=INK3, sw=1.3,
                          title_size=13.5, sub_size=11.5))
    parts.append(_text(W / 2, H - 24,
                       "an ablation that has not been built, not a licensing block",
                       13.5, INK2, "middle", "600"))
    return "".join(parts), f"0 0 {W} {H}"
