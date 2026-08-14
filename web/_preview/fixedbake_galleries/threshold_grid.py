"""Threshold-sweep grid: GT + 5 binarization thresholds + a continuous
(non-binarized) column, for five draws chosen to span confident/bimodal to
diffuse prediction character. Imported by build.py."""
import html

COL_LABELS = [("gt", "GT"), ("thr001", "0.01"), ("thr01", "0.1"),
              ("thr03", "0.3"), ("thr05", "0.5"), ("thr09", "0.9"),
              ("cont", "continuous")]


def threshold_grid_html(rows, src_fn):
    """rows: list of dicts with caption, max, mean, frac05, tag (row key used
    in the thrgrid_{tag}_{col}.png filenames)."""
    n_cols = len(COL_LABELS) + 1  # +1 for the row-label column
    parts = [f'<div style="overflow-x:auto;margin-top:10px">']
    parts.append(f'<div style="display:grid;grid-template-columns:180px repeat({len(COL_LABELS)},1fr);'
                 f'gap:4px;min-width:1080px;font-size:12px">')

    # header row
    parts.append('<div></div>')
    for _, label in COL_LABELS:
        weight = "font-weight:600" if label == "0.5" else ""
        parts.append(f'<div style="text-align:center;padding:4px 0;{weight};'
                     f'color:var(--dv-text-secondary,#52514e)">{html.escape(label)}</div>')

    for r in rows:
        label_html = (f'<div style="padding:6px 4px;font-size:12px;line-height:1.4">'
                      f'<div style="font-weight:600">{html.escape(r["caption"])}</div>'
                      f'<div style="font:11px/1.5 var(--mono,monospace);'
                      f'color:var(--dv-text-secondary,#52514e)">max {r["max"]:.3f}<br>'
                      f'mean {r["mean"]:.3f}<br>frac@0.5 {r["frac05"]:.3f}</div></div>')
        parts.append(label_html)
        for col_tag, _ in COL_LABELS:
            fname = f'thrgrid_{r["tag"]}_{col_tag}.png'
            highlight = 'border:2px solid var(--dv-text-primary,#0b0b0b);' if col_tag == "thr05" else 'border:1px solid var(--dv-grid,#d8d6d0);'
            parts.append(
                f'<div style="{highlight}border-radius:4px;overflow:hidden">'
                f'<img src="{src_fn(fname)}" loading="lazy" style="width:100%;display:block" '
                f'alt="{html.escape(r["caption"])} at {col_tag}"></div>')

    parts.append("</div></div>")
    return "\n".join(parts)
