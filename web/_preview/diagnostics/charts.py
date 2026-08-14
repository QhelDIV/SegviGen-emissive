"""Hand-authored SVG charts for the N=300 diagnostics page.

The v2/v3 chart vocabulary ships one component, hbar_chart(), which draws a
single series scaled to its own maximum. Every figure this page needs is
something else: a grouped column chart on a fixed 0-1 IoU domain, two ranked
per-shape distributions, a per-shape dot strip, and a two-series line chart
across thresholds. Those forms are on the xgpage skill's "not yet
componentized, hand-author SVG" list, so they are built here.

Design rules followed (dataviz skill):
  form first, then color; categorical hues assigned in the documented fixed
  order and never cycled; bars capped at 22px with a 4px rounded cap and a 2px
  surface gap between neighbours; 2px lines; markers at r>=4 with a 2px surface
  ring; solid hairline gridlines (dashes are reserved for reference levels,
  where "threshold" is the reading intended); a legend whenever two or more
  series are plotted; values direct-labelled selectively, never on every mark;
  text in the ink tokens, never in a series colour.

The categorical slots are the dataviz reference palette's first four, in
order, validated with scripts/validate_palette.js on this page's own surfaces:
light #FAF9F5 and dark #23221F. Light passes every check with a contrast WARN
on aqua and yellow (2.67:1 and 2.06:1), which obliges the relief rule; the
stratified table under the grouped chart carries every plotted value, and the
model series is direct-labelled, so the relief is in place. Dark passes all
five checks outright.

One entity keeps one colour across every chart on the page: the model is slot
1, random slot 2, all-one slot 3, the albedo-brightness heuristic slot 4. The
round-trip ceiling is deliberately NOT a categorical slot: it is an upper
bound rather than a competitor, and it is drawn in the neutral ink token
wherever it appears.

Y domains are fixed by the measure, never by the data's own maximum: IoU
charts run 0 to 1 so that a bar's height means the same thing in every figure,
and the draw-variance chart runs 0 to 0.5 with its axis labelled, because a
standard deviation is not an IoU.
"""
from __future__ import annotations

VB_W = 820  # the v3 content column, so charts render 1:1 at desktop widths


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _fmt(x, n=3):
    return f"{x:.{n}f}"


# --------------------------------------------------------------- page-local CSS
# Series colours as CSS custom properties so the light/dark pair swaps in one
# place. The dark values are declared under BOTH the media query and the
# data-theme scope so the reader's theme toggle wins in either direction (the
# :not() guard lets an explicit light stamp beat an OS dark preference).
PALETTE_CSS = """
.dvz { --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
       --dv-surface: #FAF9F5; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .dvz {
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --dv-surface: #23221F; }
}
:root[data-theme="dark"] .dvz {
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --dv-surface: #23221F; }

.dvz text { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
  Helvetica, Arial, sans-serif; }
.dvz .ax { font-size: 12.5px; fill: var(--ink-2); }
.dvz .axsub { font-size: 11.5px; fill: var(--ink-3); }
.dvz .axtitle { font-size: 12.5px; fill: var(--ink-3); letter-spacing: .03em;
  text-transform: uppercase; }
.dvz .dlabel { font-size: 12.5px; fill: var(--ink); font-weight: 600; }
.dvz .anno { font-size: 12.5px; fill: var(--ink-2); }
.dvz .anno b { fill: var(--ink); font-weight: 600; }
.dvz .grid { stroke: var(--line); stroke-width: 1; }
.dvz .refline { stroke: var(--ink-3); stroke-width: 1.5; stroke-dasharray: 5 4;
  fill: none; }
.dvz .leader { stroke: var(--ink-3); stroke-width: 1; fill: none; }
.dvz .s1 { fill: var(--s1); } .dvz .s2 { fill: var(--s2); }
.dvz .s3 { fill: var(--s3); } .dvz .s4 { fill: var(--s4); }
.dvz .ln1 { stroke: var(--s1); stroke-width: 2; fill: none;
  stroke-linejoin: round; stroke-linecap: round; }
.dvz .lnk { stroke: var(--ink); stroke-width: 2; fill: none;
  stroke-linejoin: round; stroke-linecap: round; }
.dvz .ring { stroke: var(--dv-surface); stroke-width: 2; }
.dvz .bar { cursor: default; }
.dvz .bar:hover { opacity: .78; }
/* the 640px floor theme2.css sets would render this page's 820-wide charts at
   12.5px * 640/820 = 9.8px; lifted so caption-register type stays above 11px */
@media (max-width: 640px) { .xg2 .dvz svg { min-width: 720px; } }
"""

LEGEND_KEYS = {
    "model": ("s1", "the model"),
    "random": ("s2", "random at the true density"),
    "all_one": ("s3", "predict everything"),
    "pbr": ("s4", "albedo-brightness rule"),
}


def _legend_row(items, x, y, gap=22):
    """items: list of (css_class_or_None, label). A None class draws the
    dashed reference key instead of a swatch. Widths are advanced with a
    deliberately generous 7.2px per character at 12.5px type, so a row that
    fits on paper here fits on the page."""
    parts, cx = [], x
    for cls, label in items:
        if cls is None:
            parts.append(f'<line x1="{cx}" y1="{y}" x2="{cx + 16}" y2="{y}" '
                         f'class="refline"/>')
        else:
            parts.append(f'<rect x="{cx}" y="{y - 5}" width="11" height="11" rx="2" '
                         f'class="{cls}"/>')
        parts.append(f'<text x="{cx + (21 if cls is None else 16)}" y="{y + 4}" '
                     f'class="ax">{esc(label)}</text>')
        cx += (21 if cls is None else 16) + 7.2 * len(label) + gap
    return "".join(parts)


def _wrap(inner, height, aria):
    return (f'<div class="chart dvz"><svg viewBox="0 0 {VB_W} {height}" role="img" '
            f'aria-label="{esc(aria)}">{inner}</svg></div>')


def _yaxis(x0, x1, y0, y1, ticks, fmt=lambda v: f"{v:g}"):
    """Hairline horizontal gridlines with left-hand tick labels."""
    lo, hi = ticks[0], ticks[-1]
    parts = []
    for t in ticks:
        y = y1 - (t - lo) / (hi - lo) * (y1 - y0)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 9}" y="{y + 4:.1f}" text-anchor="end" '
                     f'class="ax">{fmt(t)}</text>')
    return "".join(parts)


# ----------------------------------------------------- 1. ceiling per-shape strip
def ceiling_strip(buckets, jitter_w=86):
    """buckets: [(label, n, mean_ceiling, [per-shape ceiling values]), ...].

    A jittered dot per shape plus the bucket mean as a tick, the six ticks
    joined. The joined ticks are the flat line the section argues; the dots are
    there so the flatness is not read as an average hiding a spread.
    """
    H, x0, x1, y0, y1 = 372, 66, 800, 62, 306
    colw = (x1 - x0) / len(buckets)
    p = [f'<text x="{x0 - 57}" y="30" class="axtitle">Round-trip ceiling IoU, one dot '
         f'per shape</text>']
    p.append(_yaxis(x0, x1, y0, y1, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    lambda v: f"{v:.1f}"))
    means = []
    for i, (label, n, mean_c, vals) in enumerate(buckets):
        cx = x0 + colw * (i + 0.5)
        if i:  # a hairline between columns, so the groups read as groups
            p.append(f'<line x1="{x0 + colw * i:.1f}" y1="{y0 - 12}" '
                     f'x2="{x0 + colw * i:.1f}" y2="{y1 + 6}" class="grid"/>')
        for j, v in enumerate(sorted(vals)):
            # deterministic jitter: a hash of the rank, so a rebuild never
            # reshuffles the cloud and the offsets carry no visible pattern
            # (a plain golden-ratio sequence over sorted values draws diagonal
            # streaks that read as structure in the data)
            frac = (((j * 2654435761) ^ 0x9E3779B9) % 10007) / 10007.0 - 0.5
            x = cx + frac * jitter_w
            y = y1 - v * (y1 - y0)
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="var(--ink-2)" '
                     f'opacity=".30"/>')
        ym = y1 - mean_c * (y1 - y0)
        means.append((cx, ym))
        p.append(f'<text x="{cx:.1f}" y="{y1 + 22}" text-anchor="middle" '
                 f'class="ax">{esc(label)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{y1 + 38}" text-anchor="middle" '
                 f'class="axsub">n = {n}</text>')
    p.append('<polyline class="lnk" points="'
             + " ".join(f"{cx:.1f},{ym:.1f}" for cx, ym in means) + '"/>')
    for i, (cx, ym) in enumerate(means):
        p.append(f'<circle cx="{cx:.1f}" cy="{ym:.1f}" r="4.5" fill="var(--ink)" '
                 f'class="ring"/>')
        p.append(f'<text x="{cx:.1f}" y="{ym - 13:.1f}" text-anchor="middle" '
                 f'class="dlabel" paint-order="stroke" stroke="var(--bg)" '
                 f'stroke-width="3.5">{_fmt(buckets[i][2])}</text>')
    p.append(f'<text x="{x0 - 57}" y="{y1 + 60}" class="axsub">ground-truth emissive '
             f'fraction of the shape</text>')
    return _wrap("".join(p), H,
                 "Round-trip ceiling IoU per shape, grouped by emissive fraction")


# --------------------------------------------- 2. ceiling and model vs threshold
def threshold_lines(thrs, ceiling, model):
    """Two flat series on one 0-1 IoU axis: neither moves with the threshold."""
    H, x0, x1, y0, y1 = 320, 66, 640, 62, 262
    p = [f'<text x="{x0 - 57}" y="30" class="axtitle">IoU against the true voxel '
         f'mask</text>']
    p.append(_yaxis(x0, x1, y0, y1, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    lambda v: f"{v:.1f}"))
    xs = [x0 + (x1 - x0) * i / (len(thrs) - 1) for i in range(len(thrs))]
    for x, t in zip(xs, thrs):
        p.append(f'<text x="{x:.1f}" y="{y1 + 22}" text-anchor="middle" '
                 f'class="ax">{t}</text>')
    for vals, line_cls, dot_fill in ((ceiling, "lnk", "var(--ink)"),
                                     (model, "ln1", "var(--s1)")):
        ys = [y1 - v * (y1 - y0) for v in vals]
        p.append(f'<polyline class="{line_cls}" points="'
                 + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + '"/>')
        for x, y, v, t in zip(xs, ys, vals, thrs):
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{dot_fill}" '
                     f'class="ring" data-tip="threshold {t}: IoU {v:.4f}"/>')
    cy = y1 - ceiling[-1] * (y1 - y0)
    my = y1 - model[-1] * (y1 - y0)
    p.append(f'<text x="{x1 + 14}" y="{cy - 8:.1f}" class="dlabel">round-trip '
             f'ceiling</text>')
    p.append(f'<text x="{x1 + 14}" y="{cy + 10:.1f}" class="anno">'
             f'{min(ceiling):.4f} to {max(ceiling):.4f}</text>')
    p.append(f'<text x="{x1 + 14}" y="{my - 8:.1f}" class="dlabel">the model</text>')
    p.append(f'<text x="{x1 + 14}" y="{my + 10:.1f}" class="anno">'
             f'{min(model):.4f} to {max(model):.4f}</text>')
    p.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 44}" text-anchor="middle" '
             f'class="axsub">probability threshold applied to the decoded '
             f'emission field</text>')
    return _wrap("".join(p), H, "Ceiling and model IoU at four thresholds")


# ------------------------------------------------- 3. ranked per-shape model IoU
def rank_curve(vals, mean_v, median_v, n_below, below_cut):
    """Every shape's IoU, sorted ascending. The floor on the left is the mass at
    zero; the median sits on that floor and the mean does not."""
    H, x0, x1, y0, y1 = 356, 66, 790, 58, 300
    n = len(vals)
    srt = sorted(vals)
    p = [f'<text x="{x0 - 57}" y="28" class="axtitle">IoU at threshold 0.5, one shape '
         f'per rank</text>']
    p.append(_yaxis(x0, x1, y0, y1, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    lambda v: f"{v:.1f}"))

    def X(i):
        return x0 + (x1 - x0) * i / (n - 1)

    def Y(v):
        return y1 - v * (y1 - y0)

    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(srt))
    p.append(f'<polygon points="{X(0):.1f},{y1} {pts} {X(n - 1):.1f},{y1}" '
             f'fill="var(--s1)" opacity=".12"/>')
    # the shapes below the cut, as a band over the x range they occupy: their
    # own area under the curve has no visible height, so the sub-population has
    # to be marked on the axis it lives on
    k = n_below
    # tinted in the page accent, not the series colour: it marks a slice of the
    # population on the x axis, and a blue block would read as part of the series
    p.insert(2, f'<rect x="{X(0):.1f}" y="{y0 - 6}" width="{X(k) - X(0):.1f}" '
                f'height="{y1 - y0 + 6:.1f}" fill="var(--accent)" opacity=".06"/>')
    pts_lo = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(srt[:k]))
    p.append(f'<polygon points="{X(0):.1f},{y1} {pts_lo} {X(k - 1):.1f},{y1}" '
             f'fill="var(--s1)" opacity=".28"/>')
    p.append(f'<polyline class="ln1" points="{pts}"/>')

    # reference level: the mean, labelled at the left where the curve is flat
    ym = Y(mean_v)
    p.append(f'<line x1="{x0}" y1="{ym:.1f}" x2="{x1}" y2="{ym:.1f}" class="refline"/>')
    p.append(f'<text x="{x0 + 6}" y="{ym - 9:.1f}" class="anno">'
             f'<tspan class="dlabel">mean {mean_v:.4f}</tspan>, lifted by the tail on '
             f'the right</text>')

    # the median, marked on the curve it sits on
    mi = n // 2
    p.append(f'<line x1="{X(mi):.1f}" y1="{Y(median_v) - 4:.1f}" x2="{X(mi) + 16:.1f}" '
             f'y2="{ym - 46:.1f}" class="leader"/>')
    p.append(f'<circle cx="{X(mi):.1f}" cy="{Y(median_v):.1f}" r="4.5" '
             f'fill="var(--s1)" class="ring"/>')
    p.append(f'<text x="{X(mi) + 21:.1f}" y="{ym - 44:.1f}" class="anno">'
             f'<tspan class="dlabel">median {median_v:.4f}</tspan> at rank {mi}</text>')

    # the mass at the floor
    p.append(f'<line x1="{X(k):.1f}" y1="{y1}" x2="{X(k):.1f}" y2="{y1 - 96}" '
             f'class="leader"/>')
    p.append(f'<text x="{X(k) - 12:.1f}" y="{y1 - 100}" text-anchor="end" '
             f'class="anno"><tspan class="dlabel">{k} of {n} shapes</tspan> score below '
             f'{below_cut}</text>')

    p.append(f'<text x="{x0}" y="{y1 + 22}" class="ax">rank 1</text>')
    p.append(f'<text x="{x1}" y="{y1 + 22}" text-anchor="end" class="ax">rank {n}</text>')
    p.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 22}" text-anchor="middle" '
             f'class="axsub">shapes ordered by their own IoU</text>')
    return _wrap("".join(p), H, "Per-shape model IoU, sorted ascending")


# ---------------------------------------------------- 4. draw-to-draw variability
def std_curve(stds, mean_iou, p90, n_over):
    """Per-shape standard deviation across three draws, sorted descending,
    against the model's own mean IoU."""
    H, x0, x1, y0, y1 = 330, 66, 790, 56, 272
    n = len(stds)
    srt = sorted(stds, reverse=True)
    top = 0.5
    p = [f'<text x="{x0 - 57}" y="28" class="axtitle">Standard deviation of IoU across '
         f'the three draws</text>']
    p.append(_yaxis(x0, x1, y0, y1, [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    lambda v: f"{v:.1f}"))

    def X(i):
        return x0 + (x1 - x0) * i / (n - 1)

    def Y(v):
        return y1 - v / top * (y1 - y0)

    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(srt))
    p.append(f'<polygon points="{X(0):.1f},{y1} {pts} {X(n - 1):.1f},{y1}" '
             f'fill="var(--s1)" opacity=".12"/>')
    pts_hi = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(srt[:n_over]))
    p.append(f'<polygon points="{X(0):.1f},{y1} {pts_hi} {X(n_over - 1):.1f},{y1}" '
             f'fill="var(--s1)" opacity=".28"/>')
    p.append(f'<polyline class="ln1" points="{pts}"/>')

    ym = Y(mean_iou)
    p.append(f'<line x1="{x0}" y1="{ym:.1f}" x2="{x1}" y2="{ym:.1f}" class="refline"/>')
    p.append(f'<text x="{x1}" y="{ym - 10:.1f}" text-anchor="end" class="anno">'
             f'<tspan class="dlabel">the model\u2019s own mean IoU, {mean_iou:.4f}</tspan>'
             f'</text>')
    # p90, marked where it falls on the curve
    pi = int(round(0.1 * n))
    p.append(f'<line x1="{X(pi):.1f}" y1="{Y(p90):.1f}" x2="{X(pi) + 40:.1f}" '
             f'y2="{Y(p90) - 26:.1f}" class="leader"/>')
    p.append(f'<circle cx="{X(pi):.1f}" cy="{Y(p90):.1f}" r="4.5" fill="var(--s1)" '
             f'class="ring"/>')
    p.append(f'<text x="{X(pi) + 46:.1f}" y="{Y(p90) - 24:.1f}" class="anno">'
             f'<tspan class="dlabel">p90 = {p90:.4f}</tspan>, three times the mean '
             f'IoU</text>')
    p.append(f'<text x="{X(n_over) + 12:.1f}" y="{y1 - 16}" class="anno">'
             f'<tspan class="dlabel">{n_over} shapes</tspan> move more between draws '
             f'than the model scores on average</text>')
    p.append(f'<line x1="{X(n_over):.1f}" y1="{y1}" x2="{X(n_over):.1f}" '
             f'y2="{y1 - 34}" class="leader"/>')
    p.append(f'<text x="{x0}" y="{y1 + 22}" class="ax">most variable</text>')
    p.append(f'<text x="{x1}" y="{y1 + 22}" text-anchor="end" class="ax">least '
             f'variable</text>')
    p.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 22}" text-anchor="middle" '
             f'class="axsub">{n} shapes, ordered by their own spread</text>')
    return _wrap("".join(p), H, "Per-shape standard deviation across three draws")


# ------------------------------------------------------- 5. the stratified chart
SERIES_ORDER = ["model", "random", "all_one", "pbr"]


def stratified(buckets, highlight=(1, 2), highlight_label=""):
    """buckets: [{"label", "n", "ceiling", "model", "random", "all_one", "pbr"}].

    Grouped columns on a fixed 0-1 IoU domain, with the round-trip ceiling drawn
    over each group as a dashed reference rather than a fifth column: it is a
    bound, not a competitor, and a column would invite reading it as one.
    """
    H, x0, x1, y0, y1 = 462, 62, 804, 106, 366
    p = [_legend_row([(LEGEND_KEYS[k][0], LEGEND_KEYS[k][1])
                      for k in ("model", "random", "all_one")], x0 - 53, 24)]
    p.append(_legend_row([(LEGEND_KEYS["pbr"][0], LEGEND_KEYS["pbr"][1]),
                          (None, "round-trip ceiling")], x0 - 53, 48))
    p.append(f'<text x="{x0 - 53}" y="82" class="axtitle">IoU at threshold 0.5, mean '
             f'over the shapes in each group</text>')
    p.append(_yaxis(x0, x1, y0, y1, [0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:.2f}"))
    colw = (x1 - x0) / len(buckets)
    bar_w, gap = 22, 2
    span = len(SERIES_ORDER) * bar_w + (len(SERIES_ORDER) - 1) * gap
    pad = (colw - span) / 2
    for i, b in enumerate(buckets):
        gx = x0 + colw * i
        if i in highlight:
            p.append(f'<rect x="{gx:.1f}" y="{y0}" width="{colw:.1f}" '
                     f'height="{y1 - y0}" fill="var(--accent)" opacity=".055"/>')
        for j, key in enumerate(SERIES_ORDER):
            v = b[key]
            bx = gx + pad + j * (bar_w + gap)
            h = max(v * (y1 - y0), 0.0)
            by = y1 - h
            r = min(4.0, h / 2) if h > 0 else 0
            if h < 1.2:  # a value near zero still gets a visible sliver
                p.append(f'<rect x="{bx:.1f}" y="{y1 - 1.2:.1f}" width="{bar_w}" '
                         f'height="1.2" class="bar {LEGEND_KEYS[key][0]}" '
                         f'data-tip="{esc(b["label"])} &middot; '
                         f'{esc(LEGEND_KEYS[key][1])}: IoU {v:.3f}"/>')
            else:
                p.append(f'<path class="bar {LEGEND_KEYS[key][0]}" '
                         f'data-tip="{esc(b["label"])} &middot; '
                         f'{esc(LEGEND_KEYS[key][1])}: IoU {v:.3f}" '
                         f'd="M{bx:.1f},{y1} V{by + r:.1f} '
                         f'q0,{-r:.1f} {r:.1f},{-r:.1f} '
                         f'h{bar_w - 2 * r:.1f} q{r:.1f},0 {r:.1f},{r:.1f} '
                         f'V{y1} Z"/>')
        # only the model is direct-laballed: it is the series the page is about
        mv = b["model"]
        mx = gx + pad + bar_w / 2
        p.append(f'<text x="{mx:.1f}" y="{y1 - mv * (y1 - y0) - 8:.1f}" '
                 f'text-anchor="middle" class="dlabel">{_fmt(mv)}</text>')
        # the ceiling, as a dashed rule over the group's bar span
        cy = y1 - b["ceiling"] * (y1 - y0)
        p.append(f'<line x1="{gx + pad - 3:.1f}" y1="{cy:.1f}" '
                 f'x2="{gx + pad + span + 3:.1f}" y2="{cy:.1f}" class="refline"/>')
        p.append(f'<text x="{gx + colw / 2:.1f}" y="{y1 + 22}" text-anchor="middle" '
                 f'class="ax">{esc(b["label"])}</text>')
        p.append(f'<text x="{gx + colw / 2:.1f}" y="{y1 + 38}" text-anchor="middle" '
                 f'class="axsub">n = {b["n"]}</text>')
    if highlight and highlight_label:
        hx0 = x0 + colw * min(highlight)
        hx1 = x0 + colw * (max(highlight) + 1)
        # placed in the empty middle of the band: above it sits the ceiling
        # rule, over the plot sits the axis title, and the annotation collided
        # with each in turn
        ay = y1 - 0.45 * (y1 - y0)
        p.append(f'<text x="{(hx0 + hx1) / 2:.1f}" y="{ay:.1f}" text-anchor="middle" '
                 f'class="dlabel" paint-order="stroke" stroke="var(--bg)" '
                 f'stroke-width="4">{esc(highlight_label)}</text>')
    p.append(f'<text x="{x0 - 53}" y="{y1 + 62}" class="axsub">ground-truth emissive '
             f'fraction of the shape, sparse on the left</text>')
    return _wrap("".join(p), H,
                 "Ceiling, baselines and model IoU by emissive-fraction group")
