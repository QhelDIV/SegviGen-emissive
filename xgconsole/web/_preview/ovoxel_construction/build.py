#!/usr/bin/env python3
"""Build the O-Voxel construction / dual contouring / QEF deep-dive page
(xgpage v2 editorial).

Every mechanism claim is grounded either in the TRELLIS.2 paper
(arXiv:2512.14692, "Native and Compact Structured Latents for 3D Generation"),
the original TRELLIS/SLAT paper (arXiv:2412.01506), or in code, cited
file:line against the lightgen_repo/TRELLIS2 submodule (o-voxel package).
Classical Dual Contouring background (Ju, Losasso, Schaefer, Warren,
SIGGRAPH 2002) is explicitly labeled as background, not a TRELLIS.2 claim.
Paper quotes below were extracted by fetching the paper's own text (arxiv.org
html rendering) and are reproduced verbatim; anything not directly quotable
from the paper is marked as such in-line.

Run: .venv_console/bin/python web/_preview/ovoxel_construction/build.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diagrams as D

import xgpage as lp
from xgpage.publish import publish_assets

WEB = "/local-scratch2/xya120/studio/misc/lightgen/web"
OUT_DIR = HERE


def code(text):
    return f'<code>{text}</code>'


def cite(path, lines=None):
    s = path if lines is None else f"{path}:{lines}"
    return f'<code class="cite">{s}</code>'


def paper(tag, quote):
    if not quote:
        return f'<span class="pqtag">{tag}</span>'
    return f'<span class="pq">&#8220;{quote}&#8221;</span> <span class="pqtag">{tag}</span>'


def build():
    assets_dir = os.path.join(WEB, "assets")

    # -------------------------------------------------------------- hero
    hero = lp.hero_header(
        "O-Voxel construction &middot; dual contouring &middot; QEF &middot; lightgen &middot; 2026-07-24",
        "One Solved Vertex Per Voxel",
        dek_html=(
            "TRELLIS.2 builds its O-Voxel geometry channel by solving one small "
            "least-squares problem per active voxel: a quadratic error function (QEF) "
            "fitted to the surface crossings inside that cell. This page derives that "
            "QEF from first principles, works two numerical examples by hand (a sharp "
            "corner recovered exactly, a near-flat patch that goes unstable without "
            "regularization), then reads the actual C++ that runs it and reconciles the "
            "paper's clean two-term energy with the code's three accumulation passes. "
            "Classical dual contouring background (Ju et al., SIGGRAPH 2002) is called "
            "out wherever it appears; everything else is TRELLIS.2's own adaptation of it."
        ),
        toc=[
            ("construction", "O-Voxel construction, end to end"),
            ("dc", "Dual contouring from first principles"),
            ("qef", "The QEF, derived"),
            ("worked", "Two worked examples, by hand"),
            ("code", "What the C++ actually solves"),
            ("flexible", "What &ldquo;flexible&rdquo; buys"),
            ("closing", "From dual vertex to O-Voxel"),
        ],
    )

    sections = []

    # ============================================================ 01 construction
    ladder = (
        lp.flow_stage("1", "mesh (.glb)", "arbitrary triangle mesh, incl. open/non-manifold") +
        lp.flow_arrow() +
        lp.flow_stage("2", "normalize", "scale/center into a unit cube AABB") +
        lp.flow_arrow() +
        lp.flow_stage("3", "determine active cells", "scan-convert every triangle against 3 grid directions") +
        lp.flow_arrow() +
        lp.flow_stage("4", "per-cell geometry solve", "one QEF &rarr; one dual vertex + face-existence flags") +
        lp.flow_arrow() +
        lp.flow_stage("5", "attribute sampling", "rasterize textures/factors into the same active cells")
    )
    sec1 = lp.section_v2(
        "construction", "01", "O-Voxel construction is two independent extractions over the same active cells",
        lp.prose(
            "<p>O-Voxel is a sparse container: only cells the surface actually touches "
            "carry data, and every active cell carries two kinds of data computed by two "
            "separate passes over the mesh. Formally "
            "(" + paper("TRELLIS.2, Eq. 1", "") + "):</p>"
        ) + lp.equation(
            r"\mathbf f = \{(\mathbf f_i^{\text{shape}},\, \mathbf f_i^{\text{mat}},\, \mathbf p_i)\}_{i=1}^{L}",
            comment="pᵢ is an active voxel's integer grid coordinate; only L active cells out of the full grid carry a tuple.",
        ) + lp.prose(
            "<p>The construction algorithm, concretely:</p>"
        ) + ladder + lp.prose(
            "<p><b>Step 1&ndash;2, normalization.</b> The example construction script "
            f"scales and centers the asset into a fixed unit-cube AABB before anything "
            f"else runs ({cite('o-voxel/examples/mesh2ovox.py', '9&ndash;14')}: "
            f"{code('scale = 0.99999 / (aabb[1]-aabb[0]).max()')}), then calls the two "
            "extractions over the SAME normalized mesh and the SAME "
            f"{code('grid_size')}/{code('aabb')} "
            f"({cite('o-voxel/examples/mesh2ovox.py', '21&ndash;44')}), which is what lets "
            "their outputs share one coordinate system.</p>"
            "<p><b>Step 3&ndash;4, geometry channel</b> "
            f"({code('mesh_to_flexible_dual_grid')}, "
            f"{cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '30&ndash;40')}): "
            "determines which cells are active by directly scan-converting every "
            "triangle against the grid (&sect;02 below), then solves one QEF per active "
            "cell for a continuous dual-vertex position, plus a 3-bit edge-intersection "
            "flag saying which of that cell's +x/+y/+z edges the surface actually "
            "crosses (governs which neighbor cells get connected into a face). Per "
            f"active voxel, the geometric feature stored is "
            "(" + paper("TRELLIS.2, &sect;3.1.1", "For each active voxel, our geometric "
            "feature f<sub>i</sub><sup>shape</sup> comprises: &bull; Dual vertex "
            "v<sub>i</sub>&isin;&#8477;<sup>[0,1]&sup3;</sup>&hellip; &bull; Edge "
            "intersection flags &delta;<sub>i</sub>&isin;{0,1}&sup3;&hellip; &bull; "
            "Splitting weights &gamma;<sub>i</sub>&isin;&#8477;<sub>&gt;0</sub>&hellip;") +
            "): dual vertex, intersection flags, and a learned splitting weight "
            "(&sect;05 covers the flags; the splitting weight is only produced by the "
            "trained VAE decoder, not by this raw construction step, so it is out of "
            "scope here). Code output: exactly these three arrays "
            f"({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '57&ndash;61')}).</p>"
            "<p><b>Step 5, material channel</b> "
            f"({code('textured_mesh_to_volumetric_attr')}, "
            f"{cite('o-voxel/o_voxel/convert/volumetic_attr.py', '127&ndash;161')}): an "
            "entirely separate rasterizer that finds, for every active voxel, which "
            "triangles pass through it and samples their UV textures there, mip-selected "
            "by voxel size. Per active voxel this stores "
            "(" + paper("TRELLIS.2, &sect;3.1.2", "our material feature f<sub>i</sub>"
            "<sup>mat</sup> for each active voxel consists of six channels: "
            "f<sub>i</sub><sup>mat</sup>=(c<sub>i</sub>,m<sub>i</sub>,r<sub>i</sub>,"
            "&alpha;<sub>i</sub>), where c<sub>i</sub> denotes the base color, "
            "m<sub>i</sub> the metallic ratio, r<sub>i</sub> the roughness, and "
            "&alpha;<sub>i</sub> the opacity") + "): base color (3ch) + metallic (1ch) "
            "+ roughness (1ch) + opacity (1ch) = 6 channels, confirmed by the code's own "
            f"output dict, which additionally computes a "
            f"{code('normal')} channel and (in this project's fork of the package) an "
            f"{code('emissive')} channel not in the paper's tuple "
            f"({cite('o-voxel/o_voxel/convert/volumetic_attr.py', '378&ndash;385')}). That "
            "extra channel is this project's own addition for emission generation, not "
            "part of TRELLIS.2's published format.</p>"
        ) + lp.callout(
            "The two extractions never talk to each other during construction. Each "
            "runs its own occupancy test (a cell can be active for the geometry pass "
            "and empty for the attribute pass, or vice versa, on a genuinely pathological "
            "mesh) and each returns its own coordinate list; a caller that wants both "
            "channels sorts both by a canonical voxel-index order (Morton code) and "
            "trusts that identical normalization on both sides makes the two coordinate "
            f"sets line up ({cite('o-voxel/examples/mesh2ovox.py', '30&ndash;35, 45&ndash;48')}).",
            title="Geometry and material are independent passes, reconciled only by shared normalization",
        ),
    )
    sections.append(sec1)

    # ============================================================ 02 dual contouring
    fig_signs = D.svg_figure(
        *D.diagram_sign_and_crossings(),
        "<b>Every grid corner gets a sign; an edge whose two corners disagree is a "
        "crossing.</b> The dashed&#8209;free black outline is the input polygon (a real, "
        "verified simple polygon standing in for one 2D slice of a triangle mesh, "
        "off&#8209;grid vertices, straight edges only). Green dot = corner classified "
        "inside (sign +1, by an actual point&#8209;in&#8209;polygon test at that grid "
        "corner); open dot = outside (&#8722;1). Every accent dot is a real computed "
        "segment/segment intersection between a grid edge and a polygon edge, i.e. a "
        "crossing. Any cell touching &#8805;1 crossing (tinted) is active.",
        width_px=760,
    )
    fig_hermite = D.svg_figure(
        *D.diagram_hermite_zoom(),
        "<b>Hermite data is a position AND a normal, not just a position.</b> Zoomed "
        "into the active cell containing this shape's sharpest corner (interior angle "
        "49.3&deg;). Two real crossings on this cell's boundary, p&#8321; (blue) and "
        "p&#8322; (violet), each carries the local outward normal of the polygon edge it "
        "hit (arrows); the dashed line through each crossing is that crossing's tangent "
        "line, perpendicular to its normal. A least&#8209;squares solve against BOTH "
        "tangent lines (&sect;03) places the dual vertex v&#770; near their intersection, "
        "much closer to the true off&#8209;grid corner than the plain average of the two "
        "crossings, q&#772;.",
        width_px=680,
    )
    fig_mcdc = D.svg_figure(
        *D.diagram_mc_vs_dc(),
        "<b>Same input, same active cells, two different reconstructions, both "
        "piecewise linear.</b> The dashed grey outline (both panels) is the true input; "
        "the grid and active-cell tinting are identical. Left, marching squares: a "
        "vertex sits ON each crossed edge (blue dots), and the two vertices belonging to "
        "one active cell are joined by a straight segment inside that cell, cutting the "
        "sharp corner off with a short diagonal. Right, dual contouring: one vertex sits "
        "INSIDE each active cell (violet, solved by the same QEF as the figure above), "
        "and dual vertices of cells sharing a crossing edge are connected instead, "
        "tracking the true corner much more closely. Neither reconstruction is curved: "
        "the difference is vertex placement and connectivity, not smoothness.",
        width_px=780,
    )
    sec2 = lp.section_v2(
        "dc", "02", "Dual contouring places one vertex INSIDE each active cell, not ON a crossed edge",
        lp.prose(
            "<p><span class=\"bg-tag\">Background, general graphics literature</span> "
            "Dual contouring (Ju, Losasso, Schaefer &amp; Warren, SIGGRAPH 2002) starts "
            "from an implicit scalar field (a signed distance function), evaluated at "
            "every grid corner. A corner's <b>sign</b> says which side of the surface it "
            "is on; an edge whose two corners disagree in sign must have the surface "
            "cross it somewhere, a <b>crossing</b>. A cell touching at least one crossing "
            "is <b>active</b>. None of this is TRELLIS.2-specific yet, and it does not "
            "depend on the field being smooth: the figure below classifies signs and "
            "finds crossings against a coarse, straight-edged polygon, exactly the shape "
            "of information a triangle mesh actually provides.</p>"
        ) + fig_signs + lp.prose(
            "<p>Classical DC gets its crossings from the field: walk along a corner-"
            "disagreeing edge until the field changes sign. TRELLIS.2 has no field at "
            "all, only the mesh, and gets the same crossings a different way "
            "(" + paper("TRELLIS.2, &sect;3.1.1", "Different from DC, we do not utilize "
            "any field representation. Our approach is straightforward: we directly use "
            "the asset's mesh surface to determine edge intersection flags (rather than "
            "detecting sign changes as in DC)") + "). Concretely, "
            f"{code('intersect_qef')} ({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '61&ndash;193')}) "
            "scan-converts every triangle against the grid along all 3 axis directions "
            "(a 2D scanline rasterizer run three times, once per projection plane) and "
            "records every axis-aligned grid edge the triangle's own straight facets "
            "cross, marking that edge's up-to-4 neighboring voxels active. This is the "
            "same active-cell criterion as classical DC (an edge with disagreeing "
            "endpoints) computed by mesh rasterization instead of field evaluation, and "
            "it is exactly why the paper calls O-Voxel &ldquo;field-free&rdquo; "
            "(" + paper("TRELLIS.2 abstract", "a new &lsquo;field-free&rsquo; sparse "
            "voxel structure termed O-Voxel") + "), and why it extends past watertight "
            "meshes: a scanline hit test has no trouble with an open, non-manifold, or "
            "self-intersecting surface, where a well-defined inside/outside sign may not "
            "even exist.</p>"
        ) + lp.prose(
            "<p>Each crossing carries not just a position but a <b>surface normal</b>, "
            "together the <b>Hermite data</b> for that crossing. This is the piece a "
            "plain occupancy grid, or a plain average-the-crossings approach, throws "
            "away, and it is what makes a sharp feature recoverable at all "
            "(&sect;03).</p>"
        ) + fig_hermite + lp.prose(
            "<p>Dual contouring's move: place exactly <b>one dual vertex per active "
            "cell</b>, fit from that cell's own Hermite data by a least-squares solve "
            "(&sect;03), then connect dual vertices with a straight segment across every "
            "shared crossing edge (a quad, in 3D; a segment, in the 2D cross-sections on "
            "this page). <b>Contrast with marching cubes/squares</b>, the older, more "
            "common isosurfacer: MC places a vertex ON each crossed edge (interpolated by "
            "the sign change) and triangulates within each cell from a lookup table; DC "
            "places its one vertex INSIDE the cell instead, fit to all of that cell's "
            "crossings jointly. <b>Both outputs are piecewise linear</b>, flat facets "
            "everywhere, no curves in either. A common mistake is describing DC as "
            "producing a &ldquo;smoother&rdquo; surface than MC; it does not. The real "
            "difference is only vertex placement and connectivity, visible directly "
            "below on the same input:</p>"
        ) + fig_mcdc + lp.callout(
            "MC needs no per-cell solve (its vertices are read directly off the "
            "field/crossing), but it cannot represent a corner as one point: a cell near "
            "a sharp feature gets its diagonal cut by a straight segment BETWEEN two "
            "edge points, never a vertex AT the corner. DC's one-point-per-cell format is "
            "compact (exactly what O-Voxel's one-feature-per-active-voxel storage needs) "
            "but that compactness only pays off if the per-cell solve actually lands "
            "near the true corner; that solve is the QEF, next.",
            title="Why O-Voxel needs a real per-cell solve, not just a crossing average",
        ),
    )
    sections.append(sec2)

    # ============================================================ 03 the QEF
    fig_corner = D.svg_figure(
        *D.diagram_qef_corner(),
        "<b>The QEF minimizer sits exactly at the intersection of the two tangent "
        "lines.</b> p&#8321;/n&#8321; (blue) and p&#8322;/n&#8322; (violet) are the "
        "example's two crossings; the dashed line through each is its tangent line. The "
        "faint grey rings are level sets of E(x), computed from the real 2&times;2 "
        "A&#7511;A matrix, circles here because the two normals happen to be orthogonal "
        "(A&#7511;A = I). The green point is the actual least-squares solve, landing "
        "exactly on C = (0.60, 0.65), the true corner, at E=0.",
        width_px=640,
    )
    sec3 = lp.section_v2(
        "qef", "03", "The QEF is a sum of squared point-to-plane distances, one per crossing",
        lp.prose(
            "<p>Every crossing i gives a point p<sub>i</sub> and a unit normal n<sub>i</sub>, "
            "which together define a tangent plane through p<sub>i</sub> perpendicular to "
            "n<sub>i</sub>: the local linear approximation of the surface right at that "
            "crossing. The quadratic error function scores a candidate vertex x by its "
            "summed squared distance to every one of those tangent planes:</p>"
        ) + lp.equation(
            r"E(\mathbf x) = \sum_i \big(\mathbf n_i \cdot (\mathbf x - \mathbf p_i)\big)^2",
            comment=(
                "n<sub>i</sub>&middot;(x&minus;p<sub>i</sub>) is the signed distance from x to "
                "the plane through p<sub>i</sub> with normal n<sub>i</sub> (unit length). "
                "Squaring and summing scores x by how far it sits off EVERY crossing's "
                "tangent plane at once: a vertex consistent with all of them scores near 0."
            ),
        ) + lp.prose(
            "<p>Writing A as the m&times;3 matrix whose rows are the crossings' "
            "n<sub>i</sub><sup>T</sup>, and b as the vector with "
            "b<sub>i</sub> = n<sub>i</sub>&middot;p<sub>i</sub>, this is exactly a linear "
            "least-squares problem:</p>"
        ) + lp.equation(
            r"E(\mathbf x) = \|A\mathbf x - \mathbf b\|^2, \qquad A^\top A\,\mathbf x = A^\top \mathbf b",
            comment="The normal equations. A&#7511;A is 3&times;3 (2&times;2 in this page's 2D examples), symmetric, positive semidefinite.",
        ) + lp.prose(
            "<p>The paper's own energy adds two further terms, boundary handling and "
            "regularization (" + paper("TRELLIS.2, Eq. 2", "") + "), covered in full in "
            "&sect;04; this section works the crossing term alone, the mechanism that "
            "makes sharp features recoverable in the first place.</p>"
        ) + lp.expandable(
            "Rank analysis of A&#7511;A: why one formula explains flat patches, creases, "
            "and corners alike",
            lp.prose(
                "<p>A&#7511;A is a sum of rank-1 outer products, "
                "A&#7511;A = &sum;<sub>i</sub> n<sub>i</sub>n<sub>i</sub><sup>T</sup>, so its "
                "rank is bounded by how many LINEARLY INDEPENDENT directions the "
                "crossings' normals actually span, not by how many crossings there are "
                "(ten crossings all facing the same way still give rank 1). In 3D there "
                "are exactly three cases:</p>"
            ) + D.svg_figure(
                *D.diagram_rank_cases(),
                "<b>What the minimizer SET looks like at each rank.</b> Left: every "
                "crossing's normal points the same way (one shaded facet, one arrow): "
                "A&#7511;A has rank 1, E(x)=0 for an entire PLANE of x (the flat patch's "
                "own plane), so the vertex is genuinely free to slide anywhere on it. "
                "Middle: two independent normal directions (two facets meeting at a "
                "crease): rank 2, the zero set collapses to a LINE, the crease itself. "
                "Right: three independent directions (three facets at a corner): rank 3, "
                "full rank in 3D, and the zero set is a single point, the corner.",
                width_px=760,
            ) + lp.prose(
                "<p>This is the payoff of writing the QEF as a sum of outer products "
                "rather than reasoning crossing-by-crossing: <b>the same formula "
                "recovers a flat patch, a crease, and a corner correctly</b>, with no "
                "case-specific logic, purely because rank(A&#7511;A) tracks how many "
                "genuinely different surface orientations the cell has seen. A corner "
                "needs 3 independent constraints to pin down in 3D; a crease only 2 (the "
                "third direction, along the crease, is unconstrained: sliding along it "
                "changes no tangent-plane distance); a flat patch only 1.</p>"
            ),
        ),
    )
    sections.append(sec3)

    # ============================================================ 04 worked examples
    fig_illcond = D.svg_figure(
        *D.diagram_qef_illconditioned(),
        "<b>Two nearly-parallel crossings send the raw solve almost 20 cells outside "
        "the voxel.</b> p&#8321;/n&#8321; (blue, on the bottom edge) and p&#8322;/n&#8322; "
        "(violet, on the top edge, tilted only 0.057&deg; from p&#8321;'s normal) are "
        "consistent with a real, exact intersection, just a very distant one: solving "
        "A&#7511;Ax=A&#7511;b with no regularization gives x&#8776;(21, 0.5), off the right "
        "edge of this figure. Adding the code's always-on regularization term (&lambda;=0.1, "
        "pulling toward q&#772;, hollow ring) collapses the solve back to (0.50, 0.51), "
        "essentially q&#772; itself, safely inside the cell.",
        width_px=680,
    )
    sec4 = lp.section_v2(
        "worked", "04", "Two examples, carried through by hand: an exact corner, then an unstable flat patch",
        lp.prose(
            "<p>Take one 2D cell (a slice through the reasoning above; the real solver "
            "runs in 3D, but the arithmetic is identical, just one row shorter). Two "
            "crossings, chosen with orthogonal unit normals so the arithmetic comes out "
            "exactly (this is a hand-picked idealization for legibility, distinct from "
            "the running polygon figure above, which uses the polygon's real, "
            "non-orthogonal crossings and lands close to but not exactly on its true "
            "corner):</p>"
        ) + lp.equation(
            r"\mathbf p_1=(1,\,0.35),\ \mathbf n_1=(0.6,\,0.8) \qquad \mathbf p_2=(0.8625,\,1),\ \mathbf n_2=(0.8,\,-0.6)",
            comment="Both lines pass through C=(0.60, 0.65) by construction; that is the true corner this cell should recover.",
        ) + lp.prose(
            "<p>Build A (rows = n<sub>i</sub><sup>T</sup>) and b (b<sub>i</sub> = "
            "n<sub>i</sub>&middot;p<sub>i</sub>):</p>"
        ) + lp.equation(
            r"A=\begin{pmatrix}0.6 & 0.8\\ 0.8 & -0.6\end{pmatrix}, \quad "
            r"\mathbf b=\begin{pmatrix}0.6(1)+0.8(0.35)\\ 0.8(0.8625)-0.6(1)\end{pmatrix}"
            r"=\begin{pmatrix}0.88\\ 0.09\end{pmatrix}",
        ) + lp.prose(
            "<p>Because n<sub>1</sub>&middot;n<sub>2</sub>=0.6(0.8)+0.8(&minus;0.6)=0, "
            "these two normals happen to be orthogonal unit vectors, which makes "
            "A&#7511;A come out to the identity matrix exactly:</p>"
        ) + lp.equation(
            r"A^\top A = \begin{pmatrix}1&0\\0&1\end{pmatrix}, \qquad "
            r"A^\top \mathbf b = \begin{pmatrix}0.6(0.88)+0.8(0.09)\\ 0.8(0.88)-0.6(0.09)\end{pmatrix}"
            r"=\begin{pmatrix}0.60\\ 0.65\end{pmatrix}",
        ) + lp.prose(
            "<p>So x&#770; = A&#7511;b = (0.60, 0.65) = C exactly: the QEF minimizer lands "
            "precisely on the intersection of the two tangent lines, the true corner, "
            "at E(x&#770;)=0. (Verified numerically, not just by hand: numpy's "
            f"{code('linalg.solve')} on the same A, b returns (0.6000000, 0.6500000) to "
            "float precision.)</p>"
        ) + fig_corner + lp.prose(
            "<p><b>Now the case that motivates regularization.</b> Two crossings "
            "representing an almost-flat surface, sampled on opposite edges of a cell: "
            "one on the bottom edge with a perfectly horizontal tangent, one on the top "
            "edge tilted by only 0.057&deg; (slope 0.001) from horizontal:</p>"
        ) + lp.equation(
            r"\mathbf p_1=(0,\,0.50),\ \mathbf n_1=(0,\,1) \qquad "
            r"\mathbf p_2=(1,\,0.52),\ \mathbf n_2\approx(0.000999,\,0.9999995)",
        ) + lp.prose(
            "<p>These two lines really do intersect at one specific point, since "
            "n<sub>1</sub> and n<sub>2</sub> are not EXACTLY parallel, so A is not "
            "exactly singular, and A&#7511;A is not exactly rank-deficient. But its "
            "eigenvalues are 1.9999995 and 0.0000005: a condition number over 4 million. "
            "Solving A&#7511;Ax=A&#7511;b unregularized (numpy, exact to the precision shown) "
            "gives:</p>"
        ) + lp.equation(
            r"\mathbf{\hat x}_{\text{unreg}} \approx (21.0,\ 0.50)",
            comment="Twenty-one cell-widths outside a unit cell, even though this solve is the EXACT global minimum of E (E&#8776;0 there): the two lines really do cross that far out, the QEF is doing exactly what it was asked, and what it was asked is unstable.",
        ) + lp.prose(
            "<p>The code's always-on regularization term (&sect;04's next section; "
            "here with &lambda;<sub>reg</sub>=0.1, matching the package's default) adds "
            "&lambda;&Vert;x&minus;q&#772;&Vert;&sup2; to E, where "
            "q&#772;=(p<sub>1</sub>+p<sub>2</sub>)/2=(0.50, 0.51) is the plain mean of the "
            "two crossings. Re-solving the regularized normal equations "
            "(A&#7511;A+&lambda;I)x = A&#7511;b+&lambda;q&#772; gives:</p>"
        ) + lp.equation(
            r"\mathbf{\hat x}_{\text{reg}} \approx (0.50,\ 0.51)",
            comment="Essentially q&#772; itself: with almost no information distinguishing directions near-parallel to both normals, the regularizer's pull toward the mean crossing dominates, and the vertex lands safely and sensibly inside the cell.",
        ) + fig_illcond,
    )
    sections.append(sec4)

    # ============================================================ 05 code's actual solve
    sec5 = lp.section_v2(
        "code", "05", "The paper's clean two-term energy runs as three accumulation passes in the C++",
        lp.prose(
            "<p>The paper states the full minimization the code is solving, boundary "
            "term and regularization included, in one line "
            "(" + paper("TRELLIS.2, Eq. 2", "") + "):</p>"
        ) + lp.equation(
            r"\min_{\mathbf v \in \text{voxel}}\; e(\mathbf v) = \sum_i d_{\Pi,i}^2 \;+\; "
            r"\lambda_{\text{bound}}\sum_j d_{L,j}^2 \;+\; \lambda_{\text{reg}}\, d_{\hat q}^2",
            comment=(
                "d&Pi;,i&sup2; is &sect;03's plane term (distance to crossing i's tangent "
                "plane). The paper states the regularization term explicitly: "
            ) + paper("TRELLIS.2, &sect;3.1.1", "a regularization term that encourages "
            "v to stay close to the average of the intersecting points, "
            "d<sub>q&#770;</sub>&sup2; = &Vert;v &minus; q&#772;&Vert;&sup2;") + (
                ", confirming exactly the q&#772; used numerically above (the mean "
                "of intersection points, not the cell's geometric center)."
            ),
        ) + lp.prose(
            "<p>The exposed Python signature carries the same idea as three named "
            f"weights, {code('face_weight')}, {code('boundary_weight')}, "
            f"{code('regularization_weight')} "
            f"({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '36&ndash;38')}), "
            f"solved in {cite('o-voxel/src/convert/flexible_dual_grid.cpp')}. Reading "
            "that file line by line, the paper's clean 2-term-plus-regularization "
            "energy is actually assembled by <b>three separate accumulation passes</b>, "
            "each adding a rank-1 quadric Q = plane&middot;plane<sup>T</sup> (a "
            "4&times;4 homogeneous packing of exactly the same A<sup>T</sup>A / "
            "A<sup>T</sup>b terms above; the top-left 3&times;3 block is "
            "n<sub>i</sub>n<sub>i</sub><sup>T</sup>, mathematically identical to the "
            "A<sup>T</sup>A this page derives) into a running per-voxel matrix:</p>"
            "<ul>"
            f"<li><b>{code('intersect_qef')}</b> "
            f"({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '61&ndash;193')}), "
            "always runs: for every triangle-vs-grid-edge crossing found by the "
            "scanline rasterizer (&sect;02), adds that triangle's plane quadric to "
            "EVERY ONE of the up-to-4 neighboring voxels sharing the crossed edge, and "
            "accumulates the crossing point itself into a running mean "
            f"({code('means[idx] += intersect; cnt[idx] += 1')}, lines "
            f"{cite('o-voxel/src/convert/flexible_dual_grid.cpp', '166&ndash;168')}) "
            "that later becomes q&#772;. This is the &Sigma;<sub>i</sub> d&Pi;,i&sup2; "
            "term.</li>"
            f"<li><b>{code('face_qef')}</b> "
            f"({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '196&ndash;309')}), "
            f"gated by {code('face_weight > 0.0f')}: a genuine triangle&ndash;voxel "
            "overlap test (separating-axis style, three 2D projection tests plus a "
            "plane-through-box test) that adds the SAME triangle's plane quadric to "
            "every voxel the triangle's full extent overlaps, not only the ones with an "
            "edge crossing. This extends the plane term's coverage beyond crossings to "
            "the triangle's whole footprint, still nominally the same &Sigma;<sub>i</sub> "
            "d&Pi;,i&sup2; term the paper writes, just with a broader index set i. "
            f"<b>Reading note:</b> despite the name, the C++ never multiplies this "
            f"contribution by {code('face_weight')} itself "
            f"({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '303')}, "
            f"{code('qefs[kv->second] += Q;')}, no scale factor): the argument "
            "gates whether the pass runs at all, it does not weight its result, which "
            "is consistent with the paper's equation assigning no explicit coefficient "
            "to this term either.</li>"
            f"<li><b>{code('boundry_qef')}</b> "
            f"({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '312&ndash;404')}), "
            f"gated by {code('boundary_weight > 0.0f')} and, unlike the previous pass, "
            f"actually scaled by it "
            f"({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '401')}, "
            f"{code('qefs[...] += boundary_weight * Q;')}): finds every mesh edge used "
            "by exactly one triangle (a naked/boundary edge of a non-watertight mesh), "
            "walks the voxels it passes through by 3D DDA traversal, and adds a "
            "LINE-distance quadric (project out the edge direction: A = I &minus; "
            "dd<sup>T</sup>) so the dual vertex is also pulled toward open mesh "
            "boundaries. This is exactly &lambda;<sub>bound</sub>&Sigma;<sub>j</sub> "
            "d<sub>L,j</sub>&sup2;.</li>"
            "</ul>"
        ) + lp.callout(
            "The paper does not spell out the correspondence between its two-term "
            "equation and these three C++ passes at file:line granularity; the mapping "
            "above (intersect_qef + face_qef both realizing &Sigma;<sub>i</sub> "
            "d&Pi;,i&sup2;, over two different index sets) is this page's own reading of "
            "the code against the paper's stated energy, not a claim the paper makes "
            "explicitly. What IS directly confirmed by the paper is which point "
            "regularization pulls toward (the mean of intersection points, q&#772;, not "
            "the cell's geometric center) and that the code's regularization strength "
            "additionally scales with each voxel's own crossing count "
            f"({code('regularization_weight * cnt[i] * Qreg')}, "
            f"{cite('o-voxel/src/convert/flexible_dual_grid.cpp', '610')}), a detail the "
            "paper's equation does not mention at all.",
            title="The 2-vs-3-term mapping above is this page's reading of the code, not a paper claim",
        ) + lp.prose(
            "<p><b>How the code actually solves and clamps</b>, since neither the paper "
            "nor a naive reading of &ldquo;QEF solve&rdquo; specifies this, and it is "
            "worth being precise: the paper's text does not state a numerical method "
            "(no SVD, no QR, no clamping scheme mentioned at all), so this is code-only, "
            "not paper-grounded. Textbook dual contouring (Ju et al. 2002, background) "
            "usually solves via SVD, truncating small singular values and shifting from "
            "the mass point; this code instead:</p>"
            "<ol>"
            "<li>Always folds in the regularization term FIRST if "
            f"{code('regularization_weight > 0')}, scaled by that voxel's crossing count "
            f"({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '599&ndash;611')}): "
            "a fixed additive Tikhonov-style prior applied unconditionally, not "
            "an adaptive small-singular-value truncation triggered only when the system "
            "is actually near-singular.</li>"
            f"<li>Solves the (now-regularized) 3&times;3 system with Eigen's "
            f"{code('colPivHouseholderQr')} ({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '613&ndash;616')}), "
            "a rank-revealing QR decomposition, not SVD.</li>"
            "<li>If that unconstrained solution falls outside the voxel's own "
            f"[min_corner, max_corner] box ({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '618&ndash;622')}), "
            "falls back to an EXHAUSTIVE enumeration of every box-boundary manifold: all "
            "6 faces (one axis fixed, solve the remaining 2&times;2 system), all 12 "
            f"edges (two axes fixed, solve the remaining 1D coordinate), and all 8 "
            f"corners (evaluate p<sup>T</sup>Qp directly), keeping whichever candidate "
            f"has the lowest error ({cite('o-voxel/src/convert/flexible_dual_grid.cpp', '623&ndash;782')}). "
            "This is an exact constrained-least-squares solve exploiting the box's KKT "
            "structure, not a naive per-axis clamp of the unconstrained answer.</li>"
            "</ol>"
        ),
    )
    sections.append(sec5)

    # ============================================================ 06 flexible
    sec6 = lp.section_v2(
        "flexible", "06", "Two independent things flex: where the vertex sits, and whether a face exists at all",
        lp.prose(
            "<p>Per the paper: " + paper("TRELLIS.2, &sect;3.1.1", "our algorithm "
            "flexibly adjusts the positions of dual vertices and the existence of dual "
            "grid faces to accurately represent arbitrary input surface data") + ".</p>"
            "<p><b>Flex #1, vertex position.</b> The dual vertex is a continuous point "
            "solved by the QEF (&sect;03&ndash;05), never pinned to the cell's center or "
            "any fixed offset; that is what lets a straight, off-grid input edge be "
            "reproduced at all (a fixed-center vertex could only ever reconstruct a "
            "blocky, grid-aligned surface).</p>"
            "<p><b>Flex #2, face existence.</b> Two active neighbor voxels are only "
            "connected into a face if BOTH agree the shared edge between them is a real "
            "crossing, tracked by the 3-bit "
            f"{code('intersected')} flags, decoded as {code('%2, //2%2, //4%2')} in "
            f"consuming code ({cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '17')}). "
            f"{code('flexible_dual_grid_to_mesh')} "
            f"({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '255&ndash;396')}) "
            "only emits a quad for an edge where the intersection flag is actually set, "
            "so a genuine non-manifold junction, or an open boundary, does not force a "
            "face to exist where the surface doesn't actually have one.</p>"
            "<p>The splitting weight &gamma;<sub>i</sub> mentioned in &sect;01's feature "
            "list is a related but separate flex, at the quad-to-triangle stage rather "
            "than the vertex-position or face-existence stage: it is a learned output of "
            "the trained VAE decoder (not the raw construction pipeline this page "
            f"covers), consumed as {code('split_weight')} to choose which of a quad's two "
            f"diagonals to cut ({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '370&ndash;378')}), "
            "governed by " + paper("TRELLIS.2, &sect;3.1.1", "splitting weights "
            "&gamma;<sub>i</sub>&isin;&#8477;<sub>&gt;0</sub>, controlling how "
            "quadrilateral faces are adaptively subdivided into triangles, following "
            "the flexible topology rule in [shen2023flexicubes]") + ", which defers to "
            "FlexiCubes (Shen et al., SIGGRAPH 2023) for the rule itself rather than "
            "defining it inline.</p>"
        ),
    )
    sections.append(sec6)

    # ============================================================ 07 closing
    sec7 = lp.section_v2(
        "closing", "07", "The per-cell solve becomes exactly two arrays: dual_vertices and intersected",
        lp.prose(
            "<p>Run the whole construction end to end and every active voxel's geometry "
            f"channel collapses to two small arrays, {code('dual_vertices')} (continuous "
            f"position, packed to a uint8 offset within the cell for storage, "
            f"{cite('o-voxel/examples/mesh2ovox.py', '52&ndash;53')}) and "
            f"{code('intersected')} (the 3-bit face-existence flags, packed into one "
            f"byte, {cite('o-voxel/examples/mesh2ovox.py', '54')}), sitting alongside "
            f"the material channel's base_color/metallic/roughness/alpha (&sect;01) at "
            f"the same sparse coordinates, written together into one "
            f"{code('.vxz')} file ({cite('o-voxel/examples/mesh2ovox.py', '55&ndash;57')}). "
            "Everything on this page, sign classification, Hermite data, the QEF, "
            "regularization, box clamping, is the machinery that fills in exactly those "
            "two arrays.</p>"
        ) + lp.prose(
            "<p>For how these arrays flow downstream into SC-VAE latents, the two "
            "dataset-preprocessing pipelines this project actually runs, and the "
            "where/how/what framing of shape vs. material vs. emission, see the sibling "
            "page, <a href=\"../ovoxel_explained/index.html\">O-Voxel &amp; TRELLIS.2, "
            "explained</a>; this page does not restate that ground.</p>"
        ),
    )
    sections.append(sec7)

    apx = lp.appendix("Sources", [
        "TRELLIS.2: Xiang, Chen, Xu, Wang, Lv, Deng, Zhu, Dong, Zhao, Yuan, Yang. "
        "&ldquo;Native and Compact Structured Latents for 3D Generation.&rdquo; Tech "
        "report, 2025. <a href=\"https://arxiv.org/abs/2512.14692\">arXiv:2512.14692</a>. "
        "Quotes on this page were fetched from the paper's own text (arxiv.org html "
        "rendering, section 3.1/3.1.1/Eq.1/Eq.2/abstract) and reproduced verbatim.",
        "TRELLIS (v1): Xiang, Lv, Xu, Deng, Wang, Sun, Yang, Peng, Yang. &ldquo;Structured "
        "3D Latents for Scalable and Versatile 3D Generation.&rdquo; CVPR 2025 "
        "(Spotlight). <a href=\"https://arxiv.org/abs/2412.01506\">arXiv:2412.01506</a>.",
        "Dual Contouring: Ju, Losasso, Schaefer, Warren. &ldquo;Dual Contouring of "
        "Hermite Data.&rdquo; SIGGRAPH 2002. Background reference for &sect;02/&sect;03; "
        "not a TRELLIS.2 citation. The classical solve method described in &sect;05 "
        "(SVD, mass-point shift) is this paper's own treatment, contrasted with what "
        "the actual C++ does.",
        "FlexiCubes: Shen, Munkberg, Hasselgren, Yin, Wang, Chen, Gao, Fidler. "
        "&ldquo;Flexible Isosurface Extraction for Gradient-Based Mesh Optimization.&rdquo; "
        "SIGGRAPH 2023. Cited by TRELLIS.2 for its &ldquo;flexible topology rule&rdquo; "
        "governing quad-to-triangle splitting; not independently re-derived on this page.",
        f"Code: lightgen_repo/TRELLIS2 submodule, {cite('o-voxel/o_voxel/convert/')}, "
        f"{cite('o-voxel/src/convert/flexible_dual_grid.cpp')}, "
        f"{cite('o-voxel/examples/mesh2ovox.py')} (branch lightgen).",
        "Every numerical example on this page (the corner solve, the ill-conditioned "
        "solve, the regularized solve) was independently verified with numpy "
        "(linalg.solve / eigh), not just carried through by hand; the diagrams compute "
        "their own geometry (point-in-polygon corner signs, real segment intersections, "
        "the same QEF solve) rather than placing points by eye.",
    ])

    page_html = lp.page(
        title="O-Voxel Construction, Dual Contouring, and the QEF (lightgen)",
        header_html=hero,
        body_sections=sections + [apx],
        assets_rel="../../assets",
        assets_dir=assets_dir,
        theme="v2",
        needs_katex=True,
        extra_head=EXTRA_CSS,
    )
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")


EXTRA_CSS = """<style>
.xg2 .pq { font-style: italic; color: var(--ink); }
.xg2 .pqtag { font-size: .82rem; color: var(--ink-3); font-family: ui-monospace, monospace; }
.xg2 code.cite { font-size: .82rem; background: var(--code-bg); border-radius: 4px; padding: 1px 5px; overflow-wrap: anywhere; }
.xg2 .bg-tag { display: inline-block; font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
  color: var(--ink-3); border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px; margin-right: 6px; }
.xg2 .eqcomment { font-size: .86rem; color: var(--ink-2); margin-top: 6px; }
.xg2 .flow-stage { background: var(--tile); }
.xg2 .flow-stage .flbl { color: var(--ink); }
.xg2 .flow-stage .fsub { color: var(--ink-2); }
.xg2 .flow-stage .fnum { color: var(--accent-ink); }
</style>
"""

if __name__ == "__main__":
    build()
