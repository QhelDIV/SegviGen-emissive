"""Computed-geometry inline SVG diagrams for the O-Voxel construction / dual
contouring / QEF deep-dive page.

Every diagram derives its coordinates from real computed geometry: a fixed,
verified simple polygon standing in for one 2D cross-section of a triangle
mesh (point-in-polygon corner signs, real segment/segment intersections for
edge crossings, real polygon-edge normals for Hermite data), a real 2x2
least-squares QEF solve (numpy) for the dual-vertex diagrams, and a real
isometric projection for the 3D rank-case schematics -- per xgpage D16, no
hand-placed pixel coordinates for anything the geometry itself determines.
Colors reference theme2.css tokens so figures re-skin in light/dark.
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
    """Wrap raw inline SVG in a figure/figcaption using the .diagram class
    (theme2.css), same idiom as the sibling ovoxel_explained page."""
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
def _line(x1, y1, x2, y2, color=INK, width=1.5, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{d}/>')


def _circle(x, y, r, fill=INK, stroke="none", sw=0, opacity=1):
    return (f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')


def _text(x, y, s, size=12, color=INK, anchor="middle", weight="400",
          family="ui-monospace, monospace", dy=0):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-family="{family}" '
            f'dy="{dy}">{s}</text>')


def _poly(points, fill="none", stroke=INK, width=1.5, opacity=1, dash=None):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}" opacity="{opacity}"{d}/>')


def _polyline(points, stroke=INK, width=1.5, dash=None, opacity=1, fill="none"):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}" opacity="{opacity}"{d}/>')


def _arrow(x1, y1, x2, y2, color=ACCENT, width=2, head=8):
    ang = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - head * math.cos(ang - 0.4)
    hy1 = y2 - head * math.sin(ang - 0.4)
    hx2 = x2 - head * math.cos(ang + 0.4)
    hy2 = y2 - head * math.sin(ang + 0.4)
    return (_line(x1, y1, x2, y2, color, width) +
            _poly([(x2, y2), (hx1, hy1), (hx2, hy2)], fill=color, stroke="none"))


# ---------------------------------------------------------------------------
# Shared piecewise-linear input polygon (one 2D cross-section of a triangle
# mesh). Verified simple (no self-intersections), CCW, with one sharp convex
# corner (vertex 3, interior angle 49.3 deg) that recurs through every figure
# on this page as the running example.
POLY = [(-2.3, -1.1), (-0.4, -2.2), (1.7, -0.6), (2.5, 1.8), (0.3, 0.9), (-1.9, 1.6)]
SHARP_VERTEX = POLY[3]  # (2.5, 1.8), interior angle 49.3 deg


def _point_in_polygon(x, y, poly):
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
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _edge_hit(pa, pb, poly):
    """Where grid edge pa-pb crosses the polygon, plus the outward normal of
    the polygon edge that was hit (poly is CCW; rotating the hit edge's
    direction by -90 deg gives the outward normal, same convention used to
    build TRELLIS.2's actual per-triangle plane quadric from a face normal)."""
    n = len(poly)
    for i in range(n):
        q1, q2 = poly[i], poly[(i + 1) % n]
        hit = _seg_intersect(pa, pb, q1, q2)
        if hit is not None:
            ex, ey = q2[0] - q1[0], q2[1] - q1[1]
            L = math.hypot(ex, ey) or 1.0
            normal = (ey / L, -ex / L)
            return hit, normal
    return None


def build_grid_data(poly, xr=(-3, 3), yr=(-3, 3)):
    """xr, yr: inclusive integer grid-line ranges. Returns corners (sign per
    grid corner), and per-cell crossing lists (point, normal) for the four
    edges of each cell, active iff >=1 crossing."""
    xs = list(range(xr[0], xr[1] + 1))
    ys = list(range(yr[0], yr[1] + 1))
    corners = {(gx, gy): (1 if _point_in_polygon(gx, gy, poly) else -1)
               for gx in xs for gy in ys}
    h_cross = {}  # (gx,gy) -> hit on edge (gx,gy)-(gx+1,gy)
    v_cross = {}  # (gx,gy) -> hit on edge (gx,gy)-(gx,gy+1)
    for gx in xs[:-1]:
        for gy in ys:
            if corners[(gx, gy)] != corners[(gx + 1, gy)]:
                hit = _edge_hit((gx, gy), (gx + 1, gy), poly)
                if hit:
                    h_cross[(gx, gy)] = hit
    for gx in xs:
        for gy in ys[:-1]:
            if corners[(gx, gy)] != corners[(gx, gy + 1)]:
                hit = _edge_hit((gx, gy), (gx, gy + 1), poly)
                if hit:
                    v_cross[(gx, gy)] = hit
    cell_crossings = {}
    for gx in xs[:-1]:
        for gy in ys[:-1]:
            xs_hits = []
            if (gx, gy) in h_cross: xs_hits.append(h_cross[(gx, gy)])       # bottom
            if (gx, gy + 1) in h_cross: xs_hits.append(h_cross[(gx, gy + 1)])  # top
            if (gx, gy) in v_cross: xs_hits.append(v_cross[(gx, gy)])       # left
            if (gx + 1, gy) in v_cross: xs_hits.append(v_cross[(gx + 1, gy)])  # right
            if xs_hits:
                cell_crossings[(gx, gy)] = xs_hits
    return dict(xs=xs, ys=ys, corners=corners, h_cross=h_cross, v_cross=v_cross,
                cell_crossings=cell_crossings)


def solve_qef_2d(crossings, lam=0.1, clamp=None):
    P = np.array([c[0] for c in crossings], dtype=float)
    Nn = np.array([c[1] for c in crossings], dtype=float)
    A = Nn
    b = np.einsum('ij,ij->i', Nn, P)
    AtA = A.T @ A
    Atb = A.T @ b
    qbar = P.mean(axis=0)
    x = np.linalg.solve(AtA + lam * np.eye(2), Atb + lam * qbar)
    if clamp is not None:
        x = np.clip(x, clamp[0], clamp[1])
    return x, qbar, AtA


# ---------------------------------------------------------------------------
# D1. Sign classification + edge crossings on the real grid
def diagram_sign_and_crossings():
    xr, yr = (-3, 3), (-3, 3)
    g = build_grid_data(POLY, xr, yr)
    S = 46.0
    OX, OY = 210.0, 210.0
    def T(x, y): return (OX + x * S, OY - y * S)

    parts = []
    # active cell shading
    for (cx, cy) in g["cell_crossings"]:
        x0, y0 = T(cx, cy); x1, y1 = T(cx + 1, cy + 1)
        parts.append(f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{x1-x0:.2f}" '
                      f'height="{y0-y1:.2f}" fill="{ACCENT}" opacity="0.13"/>')
    # grid lines
    for gx in g["xs"]:
        x0, y0 = T(gx, yr[0]); x1, y1 = T(gx, yr[1])
        parts.append(_line(x0, y0, x1, y1, LINE, 1))
    for gy in g["ys"]:
        x0, y0 = T(xr[0], gy); x1, y1 = T(xr[1], gy)
        parts.append(_line(x0, y0, x1, y1, LINE, 1))
    # polygon (the actual input surface)
    poly_svg = [T(x, y) for x, y in POLY]
    parts.append(_poly(poly_svg, fill="none", stroke=INK, width=2.4))
    # corner signs
    for (gx, gy), s in g["corners"].items():
        x, y = T(gx, gy)
        if s > 0:
            parts.append(_circle(x, y, 3.4, fill=GOOD))
        else:
            parts.append(_circle(x, y, 3.4, fill="none", stroke=INK3, sw=1.6))
    # crossings
    for hit, _n in list(g["h_cross"].values()) + list(g["v_cross"].values()):
        x, y = T(*hit)
        parts.append(_circle(x, y, 4.2, fill=ACCENT))
    # legend
    lx, ly = 26, 26
    parts.append(_circle(lx, ly, 3.4, fill=GOOD))
    parts.append(_text(lx + 12, ly + 4, "corner, sign +1 (inside)", 11.5, INK2, "start"))
    parts.append(_circle(lx, ly + 20, 3.4, fill="none", stroke=INK3, sw=1.6))
    parts.append(_text(lx + 12, ly + 24, "corner, sign &#8722;1 (outside)", 11.5, INK2, "start"))
    parts.append(_circle(lx, ly + 40, 4.2, fill=ACCENT))
    parts.append(_text(lx + 12, ly + 44, "crossing (sign change on this edge)", 11.5, INK2, "start"))
    parts.append(f'<rect x="{lx-4}" y="{ly+52}" width="14" height="14" fill="{ACCENT}" opacity="0.13"/>')
    parts.append(_text(lx + 12, ly + 64, "active cell (&#8805;1 crossing)", 11.5, INK2, "start"))
    return "".join(parts), "0 0 420 420"


# ---------------------------------------------------------------------------
# D2. Hermite-data zoom: one active cell, its crossings + surface normals
def diagram_hermite_zoom():
    g = build_grid_data(POLY, (-3, 3), (-3, 3))
    cell = (2, 1)
    crossings = g["cell_crossings"][cell]  # [(point,normal), ...], real, computed
    cx0, cy0 = cell
    S = 210.0
    pad = 0.35
    span = (1 + 2 * pad) * S  # local view height/width in px (here 357)
    OX, OY = 40.0, 90.0 + span  # 90px top margin for the "true corner" label
    def T(x, y):
        return (OX + (x - (cx0 - pad)) * S, OY - (y - (cy0 - pad)) * S)

    parts = []
    # cell square
    x0, y0 = T(cx0, cy0); x1, y1 = T(cx0 + 1, cy0 + 1)
    parts.append(f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{x1-x0:.2f}" height="{y0-y1:.2f}" '
                  f'fill="{ACCENT}" opacity="0.10" stroke="{INK3}" stroke-width="1.4"/>')
    # polygon edges through this region (clip loosely: draw the two edges
    # incident to the sharp vertex, they are the ones that actually cross
    # this cell)
    idx = POLY.index(SHARP_VERTEX)
    e_prev = (POLY[idx - 1], POLY[idx])
    e_next = (POLY[idx], POLY[(idx + 1) % len(POLY)])
    for a, b in (e_prev, e_next):
        parts.append(_line(*T(*a), *T(*b), INK, 2.2))
    vx, vy = T(*SHARP_VERTEX)
    parts.append(_circle(vx, vy, 4.0, fill=INK))
    parts.append(_text(vx + 10, vy - 8, "true corner (off-grid)", 11.5, INK2, "start"))

    # crossings + normal arrows + tangent line segments
    colors = [BLUE, VIOLET]
    for k, (pt, normal) in enumerate(crossings):
        px, py = T(*pt)
        parts.append(_circle(px, py, 4.6, fill=colors[k % 2]))
        nx, ny = T(pt[0] + normal[0] * 0.55, pt[1] + normal[1] * 0.55)
        parts.append(_arrow(px, py, nx, ny, colors[k % 2], 2.2, 7))
        # tangent line (perpendicular to normal) through the crossing, drawn
        # across the whole cell so the two tangent lines visibly intersect
        tx, ty = -normal[1], normal[0]
        a1 = T(pt[0] - tx * 1.3, pt[1] - ty * 1.3)
        a2 = T(pt[0] + tx * 1.3, pt[1] + ty * 1.3)
        parts.append(_line(*a1, *a2, colors[k % 2], 1.3, dash="4,3", opacity=0.75))
        parts.append(_text(px, py - 14, f"p{k+1}", 12, colors[k % 2], "middle", "600"))

    # solved dual vertex (QEF, lambda=0.1, real solve)
    xhat, qbar, AtA = solve_qef_2d(crossings, lam=0.1)
    dx, dy = T(*xhat)
    parts.append(_circle(dx, dy, 5.4, fill=GOOD, stroke=INK, sw=1.2))
    parts.append(_text(dx + 10, dy + 16, "solved dual vertex v&#770;", 11.5, INK2, "start"))
    qx, qy = T(*qbar)
    parts.append(_circle(qx, qy, 3.0, fill="none", stroke=INK3, sw=1.4))
    parts.append(_text(qx - 10, qy - 8, "q&#772; (mean of crossings)", 10.8, INK3, "end"))

    return "".join(parts), "0 0 620 540"


# ---------------------------------------------------------------------------
# D3. QEF minimizer = intersection of tangent lines, with error-field ellipses
#     (Example 1: clean hand-picked orthogonal-normal corner, exact arithmetic)
EX1_P1 = np.array([1.0, 0.35]); EX1_N1 = np.array([0.6, 0.8])
EX1_P2 = np.array([0.8625, 1.0]); EX1_N2 = np.array([0.8, -0.6])
EX1_C = np.array([0.6, 0.65])


def _error_ellipse_pts(center, AtA, k, n=80):
    eigvals, eigvecs = np.linalg.eigh(AtA)
    eigvals = np.maximum(eigvals, 1e-9)
    a = math.sqrt(k / eigvals[0])  # semi-axis along smaller eigval (long axis)
    b = math.sqrt(k / eigvals[1])
    v0, v1 = eigvecs[:, 0], eigvecs[:, 1]
    pts = []
    for i in range(n + 1):
        th = 2 * math.pi * i / n
        p = center + a * math.cos(th) * v0 + b * math.sin(th) * v1
        pts.append(tuple(p))
    return pts


def diagram_qef_corner():
    S = 300.0
    OX, OY = 40.0, 40.0 + S
    def T(x, y): return (OX + x * S, OY - y * S)

    A = np.stack([EX1_N1, EX1_N2])
    AtA = A.T @ A  # = Identity here (orthonormal normals)

    parts = []
    # unit cell box
    x0, y0 = T(0, 0); x1, y1 = T(1, 1)
    parts.append(f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{x1-x0:.2f}" height="{y0-y1:.2f}" '
                  f'fill="none" stroke="{INK3}" stroke-width="1.4"/>')
    # error-field level sets (concentric, here circles since AtA = I)
    for k in (0.02, 0.06, 0.12, 0.20):
        pts = [T(*p) for p in _error_ellipse_pts(EX1_C, AtA, k)]
        parts.append(_polyline(pts, stroke=INK3, width=1, opacity=0.45))
    # the two tangent lines (full length across the box)
    for pt, nrm, col in ((EX1_P1, EX1_N1, BLUE), (EX1_P2, EX1_N2, VIOLET)):
        tx, ty = -nrm[1], nrm[0]
        a1 = T(pt[0] - tx * 1.5, pt[1] - ty * 1.5)
        a2 = T(pt[0] + tx * 1.5, pt[1] + ty * 1.5)
        parts.append(_line(*a1, *a2, col, 2.0))
        px, py = T(*pt)
        parts.append(_circle(px, py, 4.6, fill=col))
        nxp, nyp = T(pt[0] + nrm[0] * 0.28, pt[1] + nrm[1] * 0.28)
        parts.append(_arrow(px, py, nxp, nyp, col, 2.0, 7))
    parts.append(_text(*[c + 10 for c in T(*EX1_P1)], "p&#8321;, n&#8321;", 12.5, BLUE, "start", "600"))
    parts.append(_text(T(*EX1_P2)[0] + 10, T(*EX1_P2)[1] - 8, "p&#8322;, n&#8322;", 12.5, VIOLET, "start", "600"))
    # the minimizer
    cx, cy = T(*EX1_C)
    parts.append(_circle(cx, cy, 5.6, fill=GOOD, stroke=INK, sw=1.2))
    parts.append(_text(cx + 12, cy + 4, "v&#770; = (0.60, 0.65) = C", 12.5, INK, "start", "600"))
    return "".join(parts), "0 0 640 560"


# ---------------------------------------------------------------------------
# D4. Ill-conditioned QEF: near-parallel normals send the raw solve far
#     outside the cell; regularization pulls it back to q-bar (Example 2)
EX2_P1 = np.array([0.0, 0.50]); EX2_N1 = np.array([0.0, 1.0])
_m = 0.001
EX2_N2 = np.array([_m, 1.0]) / math.hypot(_m, 1.0)
EX2_P2 = np.array([1.0, 0.52])


def diagram_qef_illconditioned():
    S = 320.0
    OX, OY = 60.0, 60.0 + S
    def T(x, y): return (OX + x * S, OY - y * S)

    A = np.stack([EX2_N1, EX2_N2])
    b = np.array([np.dot(EX2_N1, EX2_P1), np.dot(EX2_N2, EX2_P2)])
    AtA = A.T @ A
    Atb = A.T @ b
    x_uncon = np.linalg.solve(AtA, Atb)
    qbar = (EX2_P1 + EX2_P2) / 2
    lam = 0.1
    x_reg = np.linalg.solve(AtA + lam * np.eye(2), Atb + lam * qbar)

    parts = []
    x0, y0 = T(0, 0); x1, y1 = T(1, 1)
    parts.append(f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{x1-x0:.2f}" height="{y0-y1:.2f}" '
                  f'fill="none" stroke="{INK3}" stroke-width="1.4"/>')
    # the two nearly-parallel tangent lines, drawn across and slightly beyond the cell
    for pt, nrm, col in ((EX2_P1, EX2_N1, BLUE), (EX2_P2, EX2_N2, VIOLET)):
        tx, ty = -nrm[1], nrm[0]
        a1 = T(pt[0] - tx * 1.15, pt[1] - ty * 1.15)
        a2 = T(pt[0] + tx * 1.15, pt[1] + ty * 1.15)
        parts.append(_line(*a1, *a2, col, 2.0))
        px, py = T(*pt)
        parts.append(_circle(px, py, 4.6, fill=col))
    parts.append(_text(*[c for c in T(EX2_P1[0]+0.04, EX2_P1[1]+0.05)], "p&#8321;, n&#8321; (bottom edge)", 11.5, BLUE, "start", "600"))
    parts.append(_text(*[c for c in T(EX2_P2[0]-0.04, EX2_P2[1]+0.06)], "p&#8322;, n&#8322; (top edge, nearly parallel)", 11.5, VIOLET, "end", "600"))
    # q-bar
    qx, qy = T(*qbar)
    parts.append(_circle(qx, qy, 4.2, fill="none", stroke=INK3, sw=1.6))
    parts.append(_text(qx + 10, qy + 16, "q&#772; = (0.50, 0.51)", 11.5, INK3, "start"))
    # regularized solution
    rx, ry = T(*x_reg)
    parts.append(_circle(rx, ry, 5.4, fill=GOOD, stroke=INK, sw=1.2))
    parts.append(_text(rx + 10, ry - 10, "regularized v&#770; &#8776; (0.50, 0.51)", 12, INK, "start", "600"))
    # arrow off-canvas toward the unconstrained solution
    edge_x, edge_y = T(1.0, x_uncon[1])
    arrow_start = T(0.86, x_uncon[1])
    parts.append(_arrow(*arrow_start, edge_x + 14, edge_y, BAD, 2.2, 8))
    parts.append(_text(edge_x + 18, edge_y - 10, "unconstrained solve", 11.5, BAD, "start", "600"))
    parts.append(_text(edge_x + 18, edge_y + 6, f"&#8594; ({x_uncon[0]:.0f}, {x_uncon[1]:.1f}), 20+ cells away", 11.5, BAD, "start", "600"))
    return "".join(parts), "0 0 700 460"


# ---------------------------------------------------------------------------
# D5. Three rank cases: plane of solutions / line of solutions / unique point
def _iso(x, y, z):
    a = math.radians(30)
    sx = (x - z) * math.cos(a)
    sy = (x + z) * math.sin(a) - y
    return sx, sy


def _cube_wire(ox, oy, s):
    """Unit-cube wireframe (12 edges) in iso projection, offset (ox,oy) px,
    scale s px/unit. Returns svg string."""
    C = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    def P(x, y, z):
        px, py = _iso(x, y, z)
        return (ox + px * s, oy - py * s)
    edges = [((0,0,0),(1,0,0)),((0,0,0),(0,1,0)),((0,0,0),(0,0,1)),
             ((1,0,0),(1,1,0)),((1,0,0),(1,0,1)),((0,1,0),(1,1,0)),
             ((0,1,0),(0,1,1)),((0,0,1),(1,0,1)),((0,0,1),(0,1,1)),
             ((1,1,0),(1,1,1)),((1,0,1),(1,1,1)),((0,1,1),(1,1,1))]
    parts = []
    for a, b in edges:
        parts.append(_line(*P(*a), *P(*b), LINE, 1.3))
    return "".join(parts), P


def _rank_panel(ox, oy, s, kind):
    parts = []
    wire, P = _cube_wire(ox, oy, s)
    parts.append(wire)
    if kind == "flat":
        # single dominant normal (x-axis): solution set = the whole plane
        # x=0.5 slicing the cube; shade that plane, draw one normal arrow.
        quad = [P(0.5, 0, 0), P(0.5, 1, 0), P(0.5, 1, 1), P(0.5, 0, 1)]
        parts.append(_poly(quad, fill=ACCENT, stroke="none", opacity=0.22))
        parts.append(_poly(quad, fill="none", stroke=ACCENT, width=1.6, opacity=0.75))
        c = P(0.5, 0.5, 0.5)
        tip = P(0.85, 0.5, 0.5)
        parts.append(_arrow(*c, *tip, ACCENT, 2.2, 8))
        parts.append(_text(oy, 0, "", 1))  # no-op keep signature simple
        label = "rank 1: a flat patch"
        sub = "minimizer = a whole PLANE (free to slide on it)"
    elif kind == "crease":
        # two normals spanning a plane: solution set = a line (the crease)
        quadA = [P(0, 0, 0.5), P(1, 0, 0.5), P(1, 1, 0.5), P(0, 1, 0.5)]
        parts.append(_poly(quadA, fill=BLUE, stroke="none", opacity=0.16))
        quadB = [P(0.5, 0, 0), P(0.5, 1, 0), P(0.5, 1, 1), P(0.5, 0, 1)]
        parts.append(_poly(quadB, fill=VIOLET, stroke="none", opacity=0.16))
        line_a, line_b = P(0.5, 0, 0.5), P(0.5, 1, 0.5)
        parts.append(_line(*line_a, *line_b, GOOD, 3.0))
        c1 = P(0.75, 0.5, 0.5); t1 = P(0.75, 0.85, 0.5)
        parts.append(_arrow(*c1, *t1, BLUE, 2.0, 7))
        c2 = P(0.5, 0.5, 0.75); t2 = P(0.85, 0.5, 0.75)
        parts.append(_arrow(*c2, *t2, VIOLET, 2.0, 7))
        label = "rank 2: an edge / crease"
        sub = "minimizer = a LINE along the crease"
    else:  # corner
        quadA = [P(0.5, 0, 0), P(0.5, 1, 0), P(0.5, 1, 1), P(0.5, 0, 1)]
        parts.append(_poly(quadA, fill=BLUE, stroke="none", opacity=0.16))
        quadB = [P(0, 0.5, 0), P(1, 0.5, 0), P(1, 0.5, 1), P(0, 0.5, 1)]
        parts.append(_poly(quadB, fill=VIOLET, stroke="none", opacity=0.16))
        quadC = [P(0, 0, 0.5), P(1, 0, 0.5), P(1, 1, 0.5), P(0, 1, 0.5)]
        parts.append(_poly(quadC, fill=ACCENT, stroke="none", opacity=0.16))
        pt = P(0.5, 0.5, 0.5)
        parts.append(_circle(*pt, 6.0, fill=GOOD, stroke=INK, sw=1.2))
        label = "rank 3: a corner"
        sub = "minimizer = a UNIQUE POINT"
    tx, ty = ox, oy + s * 1.55
    parts.append(_text(tx, ty, label, 13.5, INK, "middle", "700"))
    parts.append(_text(tx, ty + 18, sub, 11.3, INK2, "middle"))
    return "".join(parts)


def diagram_rank_cases():
    s = 78
    parts = []
    parts.append(_rank_panel(140, 190, s, "flat"))
    parts.append(_rank_panel(420, 190, s, "crease"))
    parts.append(_rank_panel(700, 190, s, "corner"))
    return "".join(parts), "0 0 840 400"


# ---------------------------------------------------------------------------
# D6. Marching squares vs dual contouring, same input, same active cells
def diagram_mc_vs_dc():
    g = build_grid_data(POLY, (0, 4), (-1, 3))
    S = 66.0

    def panel(ox, oy, mode):
        def T(x, y): return (ox + x * S, oy - y * S)
        parts = []
        for (cx, cy) in g["cell_crossings"]:
            x0, y0 = T(cx, cy); x1, y1 = T(cx + 1, cy + 1)
            parts.append(f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{x1-x0:.2f}" '
                          f'height="{y0-y1:.2f}" fill="{ACCENT}" opacity="0.09"/>')
        for gx in g["xs"]:
            parts.append(_line(*T(gx, g["ys"][0]), *T(gx, g["ys"][-1]), LINE, 1))
        for gy in g["ys"]:
            parts.append(_line(*T(g["xs"][0], gy), *T(g["xs"][-1], gy), LINE, 1))
        poly_svg = [T(x, y) for x, y in POLY]
        parts.append(_poly(poly_svg, fill="none", stroke=INK3, width=1.6, dash="5,4", opacity=0.8))
        if mode == "mc":
            for (cx, cy), crossings in g["cell_crossings"].items():
                pts = [T(*hit) for hit, _n in crossings]
                for x, y in pts:
                    parts.append(_circle(x, y, 3.6, fill=BLUE))
                if len(pts) == 2:
                    parts.append(_line(*pts[0], *pts[1], BLUE, 2.4))
        else:
            dual = {}
            for (cx, cy), crossings in g["cell_crossings"].items():
                if len(crossings) >= 2:
                    xhat, _q, _A = solve_qef_2d(crossings, lam=0.08,
                                                 clamp=((cx, cy), (cx + 1, cy + 1)))
                    dual[(cx, cy)] = xhat
                elif len(crossings) == 1:
                    dual[(cx, cy)] = np.array(crossings[0][0])
            # connect dual vertices of cells sharing a crossing edge
            for (gx, gy), _ in g["h_cross"].items():
                a, b = (gx, gy - 1), (gx, gy)
                if a in dual and b in dual:
                    parts.append(_line(*T(*dual[a]), *T(*dual[b]), VIOLET, 2.4))
            for (gx, gy), _ in g["v_cross"].items():
                a, b = (gx - 1, gy), (gx, gy)
                if a in dual and b in dual:
                    parts.append(_line(*T(*dual[a]), *T(*dual[b]), VIOLET, 2.4))
            for x, y in dual.values():
                parts.append(_circle(*T(x, y), 4.0, fill=VIOLET, stroke=INK, sw=1))
        grid_bottom = oy - g["ys"][0] * S  # ys[0] is the smallest (most negative) y line
        cap_y = grid_bottom + 34
        title = "Marching squares" if mode == "mc" else "Dual contouring"
        parts.append(_text(ox + 2 * S, cap_y, title, 14, INK, "middle", "700"))
        return "".join(parts)

    left = panel(40, 280, "mc")
    right = panel(400, 280, "dc")
    return left + right, "0 0 740 420"
