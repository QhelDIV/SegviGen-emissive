"""Computed-geometry inline SVG diagrams for the O-Voxel / TRELLIS.2 explainer.

Every diagram below derives its coordinates from real, computed geometry (a
sampled parametric curve, a solved least-squares QEF, a bit-interleaved
Morton path, config numbers from the actual VAE yaml) rather than hand-placed
pixel coordinates, per the xgpage D16 design law. Colors reference the v2
theme2.css tokens (--accent, --ink, --ink-2, --ink-3, --line, --blue,
--violet, --good, --bad) so the diagrams re-skin correctly in light/dark.
"""
import math
import numpy as np


ACCENT = "var(--accent)"
INK = "var(--ink)"
INK2 = "var(--ink-2)"
INK3 = "var(--ink-3)"
LINE = "var(--line)"
BLUE = "var(--blue)"
VIOLET = "var(--violet)"
GOOD = "var(--good)"
BAD = "var(--bad)"


def svg_figure(inner_svg, viewbox, caption_html, width_px=680, aspect=None, id=None):
    """Wrap raw inline SVG in a figure/figcaption matching xgpage's fig() look,
    using the .diagram class (theme2.css) so it gets the self-contained-scroll
    treatment and centers per D11.

    theme2.css's `.xg2 .diagram svg` rule carries a `min-width: 640px` floor
    (same rationale as its `.chart` sibling: an SVG's text shrinks with the
    SVG, so small diagrams get unreadable labels on narrow columns). A figure
    max-width BELOW that floor doesn't shrink the SVG, it just leaves the
    figure's own box narrower than its child and the excess renders outside
    the visible box (silent clipping in a screenshot, a horizontal scrollbar
    in a live browser) -- clamp here so every caller is safe by construction."""
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
# Shared PIECEWISE-LINEAR input polygon, standing in for the cross-section of
# a triangle mesh (what mesh_to_flexible_dual_grid actually consumes -- a
# TRIANGLE MESH, not an implicit field; o-voxel/o_voxel/convert/
# flexible_dual_grid.py:30-40 takes `vertices`/`faces`). A smooth curve would
# misrepresent the algorithm's real input, so this shape is a coarse polygon,
# sampled sparsely off a star formula (16 vertices, deliberately not aligned
# to the grid) and connected with STRAIGHT segments -- every "crossing" below
# is a real 2D segment/segment intersection against this polygon's edges, not
# a linear interpolation of a smooth field.
def _blob_r(theta):
    return 1.0 + 0.22 * math.sin(3 * theta + 0.4) + 0.10 * math.cos(5 * theta)


def _input_polygon(n_facets=16):
    pts = []
    for i in range(n_facets):
        th = 2 * math.pi * i / n_facets
        r = _blob_r(th)
        pts.append((r * math.cos(th), r * math.sin(th)))
    return pts  # cyclic; edge i is (pts[i], pts[(i+1) % n])


def _point_in_polygon(x, y, poly):
    """Standard ray-casting point-in-polygon test."""
    inside = False
    n = len(poly)
    x1, y1 = poly[-1]
    for i in range(n):
        x2, y2 = poly[i]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def _seg_intersect(p1, p2, p3, p4):
    """Intersection of segment p1-p2 with segment p3-p4, or None. Both
    parameters must land in (0,1) (proper crossing, not touching an endpoint)."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _polygon_crossing(pa, pb, poly):
    """Where the grid edge pa-pb crosses the input polygon, plus the local
    outward normal (perpendicular of the polygon edge that was hit -- the
    real per-crossing normal a triangle mesh provides, not a numerical
    gradient of a field). poly is assumed counter-clockwise, so rotating an
    edge direction by -90 degrees points outward."""
    n = len(poly)
    for i in range(n):
        q1, q2 = poly[i], poly[(i + 1) % n]
        hit = _seg_intersect(pa, pb, q1, q2)
        if hit is not None:
            ex, ey = q2[0] - q1[0], q2[1] - q1[1]
            L = math.hypot(ex, ey) or 1.0
            normal = (ey / L, -ex / L)
            return hit, normal
    return None, None


def _grid_cells(n=13, half_extent=1.55):
    """n x n cells covering [-half_extent, half_extent]^2. Returns cell size h
    and a function cell(i,j) -> (x0,y0,x1,y1)."""
    h = 2 * half_extent / n
    def cell(i, j):
        x0 = -half_extent + i * h
        y0 = -half_extent + j * h
        return x0, y0, x0 + h, y0 + h
    return h, cell, n


def _edge_crossing(fa, fb, pa, pb):
    """Linear-interpolate the zero of f along segment pa->pb given corner
    values fa, fb (opposite sign). Returns the crossing point."""
    t = fa / (fa - fb)
    return (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))


def _grad(f, x, y, eps=1e-3):
    dx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
    dy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
    n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


def _to_px(x, y, half_extent, S, pad):
    """Map math coords (y-up) to SVG pixel coords (y-down)."""
    return (pad + (x + half_extent) / (2 * half_extent) * S,
            pad + (half_extent - y) / (2 * half_extent) * S)


def diagram_three_panel(n=13, half_extent=1.55, S=300, pad=14, gap=34):
    """Three panels, same computed PIECEWISE-LINEAR input polygon + grid:
    (a) occupancy, (b) marching-squares-style crossing segments (vertices ON
    the crossed edges), (c) dual contouring (one vertex INSIDE each active
    cell, fit from the same crossings, connected across shared edges).
    Every crossing below is a real segment/segment intersection against the
    input polygon's straight edges -- see _polygon_crossing. Returns
    (svg_inner, viewbox)."""
    h, cell, n = _grid_cells(n, half_extent)
    poly = _input_polygon()

    signs = {}
    for i in range(n + 1):
        for j in range(n + 1):
            x0 = -half_extent + i * h
            y0 = -half_extent + j * h
            signs[(i, j)] = _point_in_polygon(x0, y0, poly)

    active = []
    dual_vertex = {}
    crossings_of = {}
    for i in range(n):
        for j in range(n):
            x0, y0, x1, y1 = cell(i, j)
            c00, c10, c11, c01 = signs[(i, j)], signs[(i + 1, j)], signs[(i + 1, j + 1)], signs[(i, j + 1)]
            corners = [(x0, y0, c00), (x1, y0, c10), (x1, y1, c11), (x0, y1, c01)]
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            pts = []
            normals = []
            for a, b in edges:
                xa, ya, fa = corners[a]
                xb, yb, fb = corners[b]
                if fa != fb:
                    p, nrm = _polygon_crossing((xa, ya), (xb, yb), poly)
                    if p is not None:
                        pts.append(p)
                        normals.append(nrm)
            if len(pts) >= 2:
                active.append((i, j))
                crossings_of[(i, j)] = pts
                # QEF-lite dual vertex: least squares to the crossing tangent
                # planes, regularized toward the crossing centroid (same
                # structure as o-voxel's QEF: intersection term + reg term).
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                A = []
                b_ = []
                for (px, py), (nx, ny) in zip(pts, normals):
                    A.append([nx, ny])
                    b_.append(nx * px + ny * py)
                lam = 0.25
                A.append([lam, 0]); b_.append(lam * cx)
                A.append([0, lam]); b_.append(lam * cy)
                A = np.array(A); b_ = np.array(b_)
                sol, *_ = np.linalg.lstsq(A, b_, rcond=None)
                vx, vy = float(sol[0]), float(sol[1])
                vx = min(max(vx, x0), x1)
                vy = min(max(vy, y0), y1)
                dual_vertex[(i, j)] = (vx, vy)

    active_set = set(active)

    def draw_grid():
        s = []
        for i in range(n + 1):
            x0, y0, x1, y1 = cell(min(i, n - 1), 0)
            xa = x0 if i < n else x1
            p0 = _to_px(xa, -half_extent, half_extent, S, pad)
            p1 = _to_px(xa, half_extent, half_extent, S, pad)
            s.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{LINE}" stroke-width="1"/>')
        for j in range(n + 1):
            x0, y0, x1, y1 = cell(0, min(j, n - 1))
            ya = y0 if j < n else y1
            p0 = _to_px(-half_extent, ya, half_extent, S, pad)
            p1 = _to_px(half_extent, ya, half_extent, S, pad)
            s.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{LINE}" stroke-width="1"/>')
        return "".join(s)

    def draw_true_curve():
        d = "M " + " L ".join(f"{_to_px(x,y,half_extent,S,pad)[0]:.1f} {_to_px(x,y,half_extent,S,pad)[1]:.1f}" for x, y in poly) + " Z"
        return f'<path d="{d}" fill="none" stroke="{INK3}" stroke-width="1.1" stroke-dasharray="2,3"/>'

    def panel(kind, ox):
        s = [f'<g transform="translate({ox},0)">']
        s.append(f'<rect x="0" y="0" width="{S+2*pad}" height="{S+2*pad}" fill="none"/>')
        s.append(draw_grid())
        if kind == "occ":
            for (i, j) in active:
                x0, y0, x1, y1 = cell(i, j)
                p0 = _to_px(x0, y1, half_extent, S, pad)
                p1 = _to_px(x1, y0, half_extent, S, pad)
                s.append(f'<rect x="{p0[0]:.1f}" y="{p0[1]:.1f}" width="{p1[0]-p0[0]:.1f}" height="{p1[1]-p0[1]:.1f}" fill="{ACCENT}" fill-opacity="0.30" stroke="{ACCENT}" stroke-opacity="0.55" stroke-width="1"/>')
            s.append(draw_true_curve())
        elif kind == "ms":
            s.append(draw_true_curve())
            for (i, j), pts in crossings_of.items():
                if len(pts) == 2:
                    p0 = _to_px(*pts[0], half_extent, S, pad)
                    p1 = _to_px(*pts[1], half_extent, S, pad)
                    s.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{BLUE}" stroke-width="2.4"/>')
                for p in pts:
                    px, py = _to_px(*p, half_extent, S, pad)
                    s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.1" fill="{BLUE}"/>')
        elif kind == "dc":
            s.append(draw_true_curve())
            # connect dual vertices across shared active-active edges
            for (i, j) in active:
                for (di, dj) in ((1, 0), (0, 1)):
                    ni, nj = i + di, j + dj
                    if (ni, nj) in active_set:
                        p0 = _to_px(*dual_vertex[(i, j)], half_extent, S, pad)
                        p1 = _to_px(*dual_vertex[(ni, nj)], half_extent, S, pad)
                        s.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{ACCENT}" stroke-width="2.4"/>')
            for (i, j) in active:
                px, py = _to_px(*dual_vertex[(i, j)], half_extent, S, pad)
                s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{ACCENT}"/>')
            # Annotate the largest gap between the input polygon and the
            # reconstruction: for each active cell, the distance from its
            # dual vertex to the nearest of its own crossings (a cheap,
            # honest proxy for "how far the reconstruction drifted here").
            worst_d, worst_pair = -1.0, None
            for (i, j) in active:
                dv = dual_vertex[(i, j)]
                for p in crossings_of[(i, j)]:
                    d = math.hypot(dv[0] - p[0], dv[1] - p[1])
                    if d > worst_d:
                        worst_d, worst_pair = d, (dv, p)
            if worst_pair is not None and worst_d > 0.02:
                (dvx, dvy), (px_, py_) = worst_pair
                a = _to_px(dvx, dvy, half_extent, S, pad)
                b = _to_px(px_, py_, half_extent, S, pad)
                s.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{BAD}" stroke-width="1.3" stroke-dasharray="1.5,2"/>')
                mx = min(max((a[0] + b[0]) / 2, 70), (S + 2 * pad) - 70)
                my = min(a[1], b[1]) - 8
                s.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="10" fill="{BAD}">approximation error</text>')
        s.append('</g>')
        return "".join(s)

    W = S + 2 * pad
    total_w = 3 * W + 2 * gap
    parts = [
        panel("occ", 0),
        panel("ms", W + gap),
        panel("dc", 2 * (W + gap)),
    ]
    # panel titles
    titles = ["occupancy (blocky)", "crossings only, on-edge (marching squares)", "dual contouring, one interior vertex / voxel"]
    for k, t in enumerate(titles):
        x = k * (W + gap) + W / 2
        parts.append(f'<text x="{x:.1f}" y="{S+2*pad+20}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="12" fill="{INK2}">{t}</text>')
    inner = "".join(parts)
    viewbox = f"0 0 {total_w} {S+2*pad+34}"
    return inner, viewbox


def _lerp_hex(c0, c1, t):
    def h2rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    a = h2rgb(c0); b = h2rgb(c1)
    r = [round(a[i] + (b[i]-a[i])*t) for i in range(3)]
    return f'#{r[0]:02x}{r[1]:02x}{r[2]:02x}'


def diagram_framing_three_meanings(n=13, half_extent=1.55, S=230, pad=12, gap=30):
    """Section-1 framing diagram, same computed blob+grid as diagram_three_panel
    but showing the THREE THINGS "voxel" can mean: (a) occupancy, a bit per
    cell; (b) attribute on active voxels, a value per cell (illustrative
    color gradient stands in for a stored scalar, e.g. base_color); (c)
    sub-voxel geometry, a continuous dual-vertex position inside each cell,
    not snapped to the grid."""
    h, cell, n = _grid_cells(n, half_extent)
    poly = _input_polygon()
    signs = {}
    for i in range(n + 1):
        for j in range(n + 1):
            x0 = -half_extent + i * h
            y0 = -half_extent + j * h
            signs[(i, j)] = _point_in_polygon(x0, y0, poly)
    active = []
    dual_vertex = {}
    cell_color_t = {}
    for i in range(n):
        for j in range(n):
            x0, y0, x1, y1 = cell(i, j)
            c00, c10, c11, c01 = signs[(i, j)], signs[(i + 1, j)], signs[(i + 1, j + 1)], signs[(i, j + 1)]
            corners = [(x0, y0, c00), (x1, y0, c10), (x1, y1, c11), (x0, y1, c01)]
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            pts, normals = [], []
            for a, b in edges:
                xa, ya, fa = corners[a]; xb, yb, fb = corners[b]
                if fa != fb:
                    p, nrm = _polygon_crossing((xa, ya), (xb, yb), poly)
                    if p is not None:
                        pts.append(p); normals.append(nrm)
            if len(pts) >= 2:
                active.append((i, j))
                cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
                A = []; b_ = []
                for (px, py), (nx, ny) in zip(pts, normals):
                    A.append([nx, ny]); b_.append(nx*px+ny*py)
                lam = 0.25
                A.append([lam, 0]); b_.append(lam*cx)
                A.append([0, lam]); b_.append(lam*cy)
                sol, *_ = np.linalg.lstsq(np.array(A), np.array(b_), rcond=None)
                vx = min(max(float(sol[0]), x0), x1); vy = min(max(float(sol[1]), y0), y1)
                dual_vertex[(i, j)] = (vx, vy)
                cell_color_t[(i, j)] = (x0 + x1) / 2 / half_extent * 0.5 + 0.5

    def draw_grid():
        s = []
        for i in range(n + 1):
            x0 = -half_extent + i * h
            p0 = _to_px(x0, -half_extent, half_extent, S, pad); p1 = _to_px(x0, half_extent, half_extent, S, pad)
            s.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{LINE}" stroke-width="0.8"/>')
        for j in range(n + 1):
            y0 = -half_extent + j * h
            p0 = _to_px(-half_extent, y0, half_extent, S, pad); p1 = _to_px(half_extent, y0, half_extent, S, pad)
            s.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{LINE}" stroke-width="0.8"/>')
        return "".join(s)

    def panel(kind, ox):
        s = [f'<g transform="translate({ox},0)">', draw_grid()]
        if kind == "occ":
            for (i, j) in active:
                x0, y0, x1, y1 = cell(i, j)
                p0 = _to_px(x0, y1, half_extent, S, pad); p1 = _to_px(x1, y0, half_extent, S, pad)
                s.append(f'<rect x="{p0[0]:.1f}" y="{p0[1]:.1f}" width="{p1[0]-p0[0]:.1f}" height="{p1[1]-p0[1]:.1f}" fill="{INK}" fill-opacity="0.55"/>')
        elif kind == "attr":
            for (i, j) in active:
                x0, y0, x1, y1 = cell(i, j)
                p0 = _to_px(x0, y1, half_extent, S, pad); p1 = _to_px(x1, y0, half_extent, S, pad)
                col = _lerp_hex("#C96442", "#4E7FD0", cell_color_t[(i, j)])
                s.append(f'<rect x="{p0[0]:.1f}" y="{p0[1]:.1f}" width="{p1[0]-p0[0]:.1f}" height="{p1[1]-p0[1]:.1f}" fill="{col}"/>')
        elif kind == "geo":
            for (i, j) in active:
                px, py = _to_px(*dual_vertex[(i, j)], half_extent, S, pad)
                s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{ACCENT}"/>')
        s.append('</g>')
        return "".join(s)

    W = S + 2 * pad
    parts = [panel("occ", 0), panel("attr", W + gap), panel("geo", 2*(W+gap))]
    titles = ["(a) occupancy: 1 bit / voxel", "(b) attribute: 1 value / voxel", "(c) sub-voxel geometry: continuous point"]
    for k, t in enumerate(titles):
        x = k * (W + gap) + W / 2
        parts.append(f'<text x="{x:.1f}" y="{S+2*pad+20}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="11.5" fill="{INK2}">{t}</text>')
    total_w = 3 * W + 2 * gap
    inner = "".join(parts)
    viewbox = f"0 0 {total_w} {S+2*pad+32}"
    return inner, viewbox


def diagram_qef_corner():
    """Single-cell zoom: a sharp corner feature crosses two cell edges. Shows
    the true corner point, the naive crossing-average, and the QEF-solved
    dual vertex, all from a real (computed) line-intersection + least-squares
    solve: the classic Ju et al. 2002 dual-contouring corner-recovery demo,
    reproduced with o-voxel's QEF structure (intersection-plane term +
    regularization toward the crossing centroid)."""
    S = 340
    pad = 20
    half = 1.0  # cell = [0,1]^2 in math coords, plotted 0..1 -> px

    def to_px(x, y):
        return (pad + x * S, pad + (1 - y) * S)

    # True sharp corner at C, two rays hitting the left and bottom edges.
    C = (0.62, 0.68)
    d1 = (-1.0, -0.12)   # ray 1 direction, hits x=0
    d2 = (0.08, -1.0)    # ray 2 direction, hits y=0
    # intersect C + t*d1 with x=0 -> t = -C.x/d1.x
    t1 = -C[0] / d1[0]
    p1 = (C[0] + t1 * d1[0], C[1] + t1 * d1[1])
    t2 = -C[1] / d2[1]
    p2 = (C[0] + t2 * d2[0], C[1] + t2 * d2[1])
    # tangent-plane normals at each crossing (perpendicular to the ray, i.e.
    # the segment direction is the tangent; normal is rotate90)
    def normal_of(d):
        n = (-d[1], d[0])
        L = math.hypot(*n) or 1.0
        return (n[0] / L, n[1] / L)
    n1 = normal_of(d1)
    n2 = normal_of(d2)

    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2
    lam = 0.25
    A = np.array([[n1[0], n1[1]], [n2[0], n2[1]], [lam, 0], [0, lam]])
    b_ = np.array([n1[0]*p1[0]+n1[1]*p1[1], n2[0]*p2[0]+n2[1]*p2[1], lam*cx, lam*cy])
    sol, *_ = np.linalg.lstsq(A, b_, rcond=None)
    qx, qy = float(sol[0]), float(sol[1])

    s = []
    # cell box
    x0, y0 = to_px(0, 0)
    x1, y1 = to_px(1, 1)
    s.append(f'<rect x="{min(x0,x1):.1f}" y="{min(y0,y1):.1f}" width="{abs(x1-x0):.1f}" height="{abs(y1-y0):.1f}" fill="none" stroke="{LINE}" stroke-width="1.4"/>')
    # the two mesh edges (segments from C out past the cell boundary, clipped visually to just past crossing)
    ext = 0.18
    e1a = to_px(C[0] + (t1 - ext) * d1[0] / abs(t1) * abs(t1), C[1] + (t1 - ext) * d1[1])
    # simpler: draw from C to a bit beyond crossing p1/p2
    far1 = (p1[0] - 0.10 * (C[0]-p1[0])/(abs(C[0]-p1[0]) or 1), p1[1] - 0.10*(C[1]-p1[1])/(abs(C[1]-p1[1]) or 1))
    def seg(pa, pb, color, w=2.0, dash=""):
        a = to_px(*pa); b = to_px(*pb)
        dstr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}" stroke-width="{w}"{dstr}/>'
    s.append(seg(p1, C, INK, 2.2))
    s.append(seg(C, p2, INK, 2.2))
    # crossing points
    for p, lbl in ((p1, "x-edge crossing"), (p2, "y-edge crossing")):
        px, py = to_px(*p)
        s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{BLUE}"/>')
    # normals (short arrows) at crossings
    for p, nn in ((p1, n1), (p2, n2)):
        a = to_px(*p)
        b = to_px(p[0] + nn[0]*0.14, p[1] + nn[1]*0.14)
        s.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{BLUE}" stroke-width="1.3" marker-end="url(#arrowq)"/>')
    # true corner C: hollow ring
    pc = to_px(*C)
    s.append(f'<circle cx="{pc[0]:.1f}" cy="{pc[1]:.1f}" r="9" fill="none" stroke="{INK}" stroke-width="2.2"/>')
    # naive centroid
    pavg = to_px(cx, cy)
    s.append(f'<circle cx="{pavg[0]:.1f}" cy="{pavg[1]:.1f}" r="5.5" fill="{BAD}"/>')
    # QEF solution: small filled disc, nests inside C's ring since QEF ~= C
    pq = to_px(qx, qy)
    s.append(f'<circle cx="{pq[0]:.1f}" cy="{pq[1]:.1f}" r="4.2" fill="{GOOD}"/>')

    # Two label groups, spatially separated so they never collide: (1) the
    # C/QEF cluster (upper area, leader line up-right into open space), and
    # (2) the naive-average point (lower-left, leader line into open space).
    lead1_a = (pc[0] + 8, pc[1] - 8)
    lead1_b = (pc[0] + 46, pc[1] - 50)
    s.append(f'<line x1="{lead1_a[0]:.1f}" y1="{lead1_a[1]:.1f}" x2="{lead1_b[0]:.1f}" y2="{lead1_b[1]:.1f}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<text x="{lead1_b[0]:.1f}" y="{lead1_b[1]-8:.1f}" text-anchor="end" font-family="ui-monospace,monospace" font-size="12.5" fill="{INK}" font-weight="600">true corner C</text>')
    s.append(f'<text x="{lead1_b[0]:.1f}" y="{lead1_b[1]+8:.1f}" text-anchor="end" font-family="ui-monospace,monospace" font-size="12.5" fill="{GOOD}" font-weight="600">QEF vertex &#8776; C</text>')

    lead2_a = (pavg[0] - 8, pavg[1] + 6)
    lead2_b = (pavg[0] - 8, pavg[1] + 46)
    s.append(f'<line x1="{lead2_a[0]:.1f}" y1="{lead2_a[1]:.1f}" x2="{lead2_b[0]:.1f}" y2="{lead2_b[1]:.1f}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<text x="{lead2_b[0]-46:.1f}" y="{lead2_b[1]+16:.1f}" font-family="ui-monospace,monospace" font-size="12.5" fill="{BAD}" font-weight="600">naive average</text>')
    s.append(f'<text x="{lead2_b[0]-46:.1f}" y="{lead2_b[1]+32:.1f}" font-family="ui-monospace,monospace" font-size="12.5" fill="{BAD}" font-weight="600">(blurs the corner)</text>')

    # crossing-point micro-labels
    p1x, p1y = to_px(*p1)
    p2x, p2y = to_px(*p2)
    s.append(f'<text x="{p1x+8:.1f}" y="{p1y-6:.1f}" font-family="ui-monospace,monospace" font-size="11" fill="{BLUE}">x-edge crossing</text>')
    s.append(f'<text x="{p2x-118:.1f}" y="{p2y+13:.1f}" font-family="ui-monospace,monospace" font-size="11" fill="{BLUE}">y-edge crossing</text>')

    defs = (f'<defs><marker id="arrowq" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 Z" fill="{BLUE}"/></marker></defs>')
    inner = defs + "".join(s)
    viewbox = f"0 0 {S+2*pad} {S+2*pad}"
    return inner, viewbox


def diagram_morton(order=3):
    """A computed Morton (Z-order) traversal path over a 2^order x 2^order
    grid, via real bit-interleaving (not hand-drawn), illustrating how voxel
    coordinates are sequenced within one .vxz chunk before compression."""
    n = 2 ** order
    S = 300
    pad = 16
    cell = S / n

    def interleave(x, y, bits):
        code = 0
        for b in range(bits):
            code |= ((x >> b) & 1) << (2 * b)
            code |= ((y >> b) & 1) << (2 * b + 1)
        return code

    cells = []
    for x in range(n):
        for y in range(n):
            cells.append((interleave(x, y, order), x, y))
    cells.sort()

    s = []
    # grid
    for i in range(n + 1):
        s.append(f'<line x1="{pad+i*cell:.1f}" y1="{pad:.1f}" x2="{pad+i*cell:.1f}" y2="{pad+S:.1f}" stroke="{LINE}" stroke-width="1"/>')
        s.append(f'<line x1="{pad:.1f}" y1="{pad+i*cell:.1f}" x2="{pad+S:.1f}" y2="{pad+i*cell:.1f}" stroke="{LINE}" stroke-width="1"/>')
    pts = []
    for code, x, y in cells:
        cx = pad + (x + 0.5) * cell
        cy = pad + S - (y + 0.5) * cell
        pts.append((cx, cy))
    d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
    s.append(f'<path d="{d}" fill="none" stroke="{ACCENT}" stroke-width="2" marker-end="url(#arrowm)"/>')
    for k, (px, py) in enumerate(pts):
        s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{ACCENT}"/>')
    defs = (f'<defs><marker id="arrowm" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
            f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{ACCENT}"/></marker></defs>')
    inner = defs + "".join(s)
    viewbox = f"0 0 {S+2*pad} {S+2*pad}"
    return inner, viewbox


def diagram_resolution_ladder():
    """Spatial downsampling ladder for the f16c32 SC-VAE: 512^3 -> 32^3 over
    four SparseSpatial2Channel(2) stages, boxes sized proportionally to the
    real per-stage spatial resolution (512,256,128,64,32)."""
    stages = [512, 256, 128, 64, 32]
    base = 150
    pad = 20
    gap_label_h = 46
    boxes = []
    x = pad
    max_h = base
    ys = pad + gap_label_h
    xs = [x]
    sizes = []
    for i, res in enumerate(stages):
        size = max(base * (res / 512), 8)
        sizes.append(size)
        xs.append(xs[-1] + size + 46)
    total_w = xs[-1] - 46 + pad
    total_h = ys + max_h + 40
    s = []
    for i, res in enumerate(stages):
        size = sizes[i]
        bx = xs[i]
        by = ys + (max_h - size) / 2
        s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{size:.1f}" height="{size:.1f}" fill="{ACCENT}" fill-opacity="0.22" stroke="{ACCENT}" stroke-width="1.6" rx="3"/>')
        s.append(f'<text x="{bx+size/2:.1f}" y="{ys+max_h+22:.1f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="13" fill="{INK}" font-weight="600">{res}&#179;</text>')
        if i < len(stages) - 1:
            ax = bx + size + 6
            bxn = xs[i+1] - 6
            ay = ys + max_h/2
            s.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bxn:.1f}" y2="{ay:.1f}" stroke="{INK2}" stroke-width="1.6" marker-end="url(#arrowr)"/>')
            s.append(f'<text x="{(ax+bxn)/2:.1f}" y="{ay-8:.1f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="10.5" fill="{INK2}">S2C(2)</text>')
    s.append(f'<text x="{pad}" y="{pad+16}" font-family="ui-monospace,monospace" font-size="12" fill="{INK2}">spatial resolution per axis (4 stages, 2&#8308;&#215; total)</text>')
    defs = (f'<defs><marker id="arrowr" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
            f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{INK2}"/></marker></defs>')
    inner = defs + "".join(s)
    viewbox = f"0 0 {total_w:.1f} {total_h:.1f}"
    return inner, viewbox


if __name__ == "__main__":
    # smoke test
    i1, v1 = diagram_three_panel()
    i2, v2 = diagram_qef_corner()
    i3, v3 = diagram_morton()
    i4, v4 = diagram_resolution_ladder()
    print("ok", len(i1), len(i2), len(i3), len(i4))
