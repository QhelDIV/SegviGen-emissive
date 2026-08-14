"""Distribution chart (SVG) + per-shape draw-strip HTML generators for the
fixedbake_galleries page extension. Imported by build.py."""
import html

# dataviz skill categorical palette, slots 1 (blue) and 2 (orange) -- the
# only two needed here, well inside the all-pairs-validated first-three cap.
VAL_LIGHT, VAL_DARK = "#2a78d6", "#3987e5"
TRAIN_LIGHT, TRAIN_DARK = "#eb6834", "#d95926"

# plain-word weight labels -- "rescue" is retired owner-facing, "resampled"
# everywhere, matching the rest of the page (both the SVG tooltips and the
# strip badges read from this one mapping).
WEIGHT_LABEL = {
    "raw": "raw", "ema": "ema",
    "rescue_raw": "resampled raw", "rescue_ema": "resampled ema",
    "paper_raw": "paper raw", "paper_ema": "paper ema",
}

CHART_CSS = """
.drawviz { color-scheme: light; }
.drawviz .row-label { fill: var(--dv-text-secondary); font: 12px/1 var(--sans, sans-serif); }
.drawviz .axis-label { fill: var(--dv-text-secondary); font: 11px/1 var(--mono, monospace); }
.drawviz .axis-line { stroke: var(--dv-grid); stroke-width: 1; }
.drawviz .gt-tick { stroke: var(--dv-text-primary); stroke-width: 2; opacity: 0.55; }
.drawviz .pt-val { fill: var(--dv-val); }
.drawviz .pt-train { fill: var(--dv-train); }
.drawviz .pt-picked { stroke: var(--dv-text-primary); stroke-width: 2; fill-opacity: 1; }
:root {
  --dv-text-secondary: #52514e; --dv-text-primary: #0b0b0b; --dv-grid: #d8d6d0;
  --dv-val: #2a78d6; --dv-train: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --dv-text-secondary: #c3c2b7; --dv-text-primary: #ffffff; --dv-grid: #3a3a38;
    --dv-val: #3987e5; --dv-train: #d95926;
  }
}
:root[data-theme="dark"] {
  --dv-text-secondary: #c3c2b7; --dv-text-primary: #ffffff; --dv-grid: #3a3a38;
  --dv-val: #3987e5; --dv-train: #d95926;
}
"""


def distribution_chart(draws_manifest, order):
    """One row per shape (in `order`), one dot per draw at its frac@0.5, a
    tick for GT, a ring on the displayed pick. Colored by split."""
    row_h = 26
    margin_left, margin_right, margin_top, margin_bottom = 210, 24, 28, 28
    chart_w = 620
    n = len(order)
    height = margin_top + n * row_h + margin_bottom
    width = margin_left + chart_w + margin_right

    def x_of(frac):
        return margin_left + max(0.0, min(1.0, frac)) * chart_w

    parts = [f'<svg class="drawviz" viewBox="0 0 {width} {height}" '
            f'width="100%" style="max-width:{width}px;height:auto" '
            f'xmlns="http://www.w3.org/2000/svg">']
    parts.append(f"<style>{CHART_CSS}</style>")

    # axis
    axis_y = margin_top - 8
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_of(t)
        parts.append(f'<line class="axis-line" x1="{x:.1f}" y1="{margin_top-4}" '
                     f'x2="{x:.1f}" y2="{height-margin_bottom}" opacity="0.4"/>')
        parts.append(f'<text class="axis-label" x="{x:.1f}" y="{axis_y}" '
                     f'text-anchor="middle">{t:.2f}</text>')
    parts.append(f'<text class="axis-label" x="{margin_left}" y="{height-8}">'
                 f'frac@0.5 (share of voxels predicted emissive)</text>')

    for i, sid in enumerate(order):
        d = draws_manifest[sid]
        y = margin_top + i * row_h + row_h / 2
        label = html.escape(d["caption"])
        parts.append(f'<text class="row-label" x="{margin_left-10}" y="{y+4:.1f}" '
                     f'text-anchor="end">{label}</text>')
        cls_pt = "pt-val" if d["split"] == "val" else "pt-train"
        # GT tick
        gx = x_of(d["gt_frac"])
        parts.append(f'<line class="gt-tick" x1="{gx:.1f}" y1="{y-8:.1f}" '
                     f'x2="{gx:.1f}" y2="{y+8:.1f}"><title>GT frac {d["gt_frac"]:.3f}'
                     f'</title></line>')
        for draw in d["draws"]:
            x = x_of(draw["frac"])
            is_picked = draw["tag"] == d.get("picked_tag")
            r = 5 if is_picked else 3.2
            extra = ' class="pt-picked"' if is_picked else ""
            wlabel = WEIGHT_LABEL.get(draw["weight"], draw["weight"])
            parts.append(
                f'<circle class="{cls_pt}"{extra} cx="{x:.1f}" cy="{y:.1f}" r="{r}" '
                f'fill-opacity="0.85"><title>{wlabel} {draw["idx"]} '
                f'frac={draw["frac"]:.3f}{" (displayed)" if is_picked else ""}'
                f'</title></circle>')

    # legend
    ly = height - margin_bottom + 20
    parts.append(f'<circle class="pt-val" cx="{margin_left}" cy="{ly}" r="4"/>')
    parts.append(f'<text class="axis-label" x="{margin_left+10}" y="{ly+4}">val (held out)</text>')
    parts.append(f'<circle class="pt-train" cx="{margin_left+140}" cy="{ly}" r="4"/>')
    parts.append(f'<text class="axis-label" x="{margin_left+150}" y="{ly+4}">train (seen-in-training)</text>')
    parts.append(f'<line class="gt-tick" x1="{margin_left+340}" y1="{ly-6}" x2="{margin_left+340}" y2="{ly+6}"/>')
    parts.append(f'<text class="axis-label" x="{margin_left+350}" y="{ly+4}">ground truth</text>')
    parts.append(f'<circle class="pt-val pt-picked" cx="{margin_left+460}" cy="{ly}" r="5"/>')
    parts.append(f'<text class="axis-label" x="{margin_left+470}" y="{ly+4}">displayed pick</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def draw_strip_html(shape_draws, src_fn):
    """One flex-wrap row of small thumbnails, badged, picked one outlined."""
    tiles = []
    for d in shape_draws["draws"]:
        if not d["thumb"]:
            continue
        is_picked = d["tag"] == shape_draws.get("picked_tag")
        label = f'{WEIGHT_LABEL.get(d["weight"], d["weight"])} {d["idx"]}'
        ring = "outline:2px solid var(--dv-text-primary,#0b0b0b);outline-offset:2px;" if is_picked else ""
        tiles.append(
            f'<figure style="margin:0;width:76px;flex:0 0 auto;text-align:center">'
            f'<img src="{src_fn(d["thumb"])}" loading="lazy" '
            f'style="width:76px;height:76px;object-fit:cover;border-radius:6px;'
            f'border:1px solid var(--dv-grid,#d8d6d0);{ring}" '
            f'alt="{html.escape(label)} frac {d["frac"]:.3f}">'
            f'<figcaption style="font:10px/1.3 var(--mono,monospace);'
            f'color:var(--dv-text-secondary,#52514e);margin-top:2px">'
            f'{html.escape(label)}<br>{d["frac"]:.3f}</figcaption></figure>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px">'
           + "".join(tiles) + "</div>")
