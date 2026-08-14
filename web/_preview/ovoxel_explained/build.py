#!/usr/bin/env python3
"""Build the O-Voxel / TRELLIS.2 deep explainer page (xgpage v2 editorial).

Every mechanism claim on this page is grounded either in the TRELLIS.2 paper
(arXiv:2512.14692, "Native and Compact Structured Latents for 3D Generation")
or the original TRELLIS paper (arXiv:2412.01506, SLAT), or in code, cited
file:line against the lightgen_repo/TRELLIS2 submodule (commit 2dabb82) and
lightgen_repo/SegviGen-emissive (commit f3443da). Classic graphics background
(Dual Contouring, Ju et al. 2002) is explicitly labeled as background, not as
a TRELLIS.2 claim.

Run: .venv_console/bin/python web/_preview/ovoxel_explained/build.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diagrams as D

import xgpage as lp
from xgpage.publish import publish_assets

REPO = "/local-scratch2/xya120/studio/misc/lightgen/lightgen_repo"
WEB = "/local-scratch2/xya120/studio/misc/lightgen/web"
OUT_DIR = HERE


def code(text):
    return f'<code>{text}</code>'


def cite(path, lines=None):
    s = path if lines is None else f"{path}:{lines}"
    return f'<code class="cite">{s}</code>'


def paper(tag, quote):
    return f'<span class="pq">&#8220;{quote}&#8221;</span> <span class="pqtag">{tag}</span>'


# ---------------------------------------------------------------------------
def build():
    assets_dir = os.path.join(WEB, "assets")

    # -------------------------------------------------------------- hero
    hero = lp.hero_header(
        "O-Voxel &amp; TRELLIS.2 &middot; deep explainer &middot; lightgen &middot; 2026-07-24",
        "O-Voxel is two channels, not one",
        dek_html=(
            "“It’s just sparse voxel features” is half right. That describes the "
            "<b>attribute channel</b>: a value stored on every active voxel that "
            "says how the surface reflects light (base color, metallic, roughness), plus "
            "opacity. It misses the <b>geometry channel</b>: a continuous dual-vertex "
            "position and a set of edge-crossing flags that TRELLIS.2 also stores on every "
            "active voxel, and that is how O-Voxel recovers a surface from a grid without "
            "an implicit field. This page builds both channels from first principles, then "
            "walks the actual pipeline this project runs, with file:line citations "
            "throughout."
        ),
        stats=[
            ("2", "channels per active voxel: geometry + material"),
            ("16&times;", "spatial compression, SC-VAE (4 stages)"),
            ("30-bit", "Morton code per voxel coordinate"),
            ("1.3B", "params, the texture flow-matching DiT"),
        ],
        toc=[
            ("framing", "Three things “voxel” can mean"),
            ("dc", "Dual contouring, from first principles"),
            ("slat", "SLAT and structured latents"),
            ("pipeline", "The pipeline, stage by stage"),
            ("pipelines", "Two preprocessing pipelines"),
            ("glossary", "Glossary"),
            ("scvae", "The SC-VAE and its two compressions"),
            ("vxz", ".vxz serialization"),
            ("situation", "Back to our situation"),
        ],
    )

    sections = []

    # ============================================================ 01 framing
    fr_inner, fr_vb = D.diagram_framing_three_meanings()
    fr_fig = D.svg_figure(
        fr_inner, fr_vb,
        "<b>Occupancy, attribute, and geometry are three different pieces of information, "
        "computed from the same piecewise-linear input polygon</b> (a coarse, faceted "
        "shape, standing in for one cross-section of a triangle mesh; dashed outline). "
        "(a) Occupancy is one bit per voxel: is the surface anywhere in this cell? "
        "(b) Attribute is one value per active voxel (here an illustrative color gradient "
        "standing in for base color); it says HOW that patch of surface reflects light, "
        "not WHERE inside the cell it sits. (c) Sub-voxel geometry is a continuous point "
        "per active voxel, computed from where the input polygon's own straight edges "
        "actually cross the cell's edges, not snapped to the grid.",
        width_px=900,
    )
    sec1 = lp.section_v2(
        "framing", "01", "“Voxel” means three different things, and O-Voxel keeps two of them",
        lp.prose(
            "<p>“Sparse voxel features” is a phrase that can mean any of three "
            "genuinely different things, and it matters which one a paper means:</p>"
            "<ol>"
            "<li><b>Occupancy</b>: a binary sparse grid. Each active cell says only "
            "“something is here,” nothing about what or where within the cell. "
            "This is the blocky, Minecraft-looking representation.</li>"
            "<li><b>Attributes on active voxels</b>: occupancy plus one value per active "
            "cell, describing how that surface reflects light (not what it is made of, "
            "which is not observable from voxels at all). This is what “it’s just sparse "
            "voxel features” usually pictures, and it is exactly what "
            f"{cite('o-voxel/o_voxel/convert/volumetic_attr.py', '378&ndash;385')} computes: "
            "base_color, metallic, roughness (the reflectance parameters), plus alpha "
            "(opacity) per active voxel.</li>"
            "<li><b>Sub-voxel geometry</b>: where, continuously, inside the cell the true "
            "surface actually sits. A cell is either occupied or not; the surface's exact "
            "position within an occupied cell is separate information, and it is what "
            f"{cite('o-voxel/o_voxel/convert/flexible_dual_grid.py')} computes.</li>"
            "</ol>"
        ) + fr_fig + lp.prose(
            "<p>TRELLIS.2's O-Voxel representation stores <b>both (2) and (3)</b> on the "
            "same sparse coordinates for every active voxel: a reflectance feature "
            "" + paper("(TRELLIS.2, &sect;3.1)", "f<sub>i</sub><sup>mat</sup>") + " and a shape "
            "feature " + paper("(TRELLIS.2, &sect;3.1)", "f<sub>i</sub><sup>shape</sup>") +
            ". The dataset this project trains on ran only the attribute extraction "
            "(2); the geometry channel (3) has to be recovered separately from the source "
            "mesh; that gap is the subject of "
            "<a href=\"#pipelines\">&sect;05</a> and "
            "<a href=\"#situation\">the closing section</a>.</p>"
        ) + lp.callout(
            "“It's just sparse voxel features” describes channel (2) exactly. It "
            "silently drops channel (3), which is the part that makes O-Voxel able to "
            "represent smooth, sharp-edged, and non-manifold surfaces without an implicit "
            "field: the rest of this page is mostly about (3).",
            title="What the half-right model gets right, and what it drops"
        ),
    )
    sections.append(sec1)

    # ============================================================ 02 dual contouring
    tp_inner, tp_vb = D.diagram_three_panel()
    tp_fig = D.svg_figure(
        tp_inner, tp_vb,
        "<b>Three reconstructions of the same piecewise-linear input, same grid, same "
        "computed edge crossings.</b> The dashed shape (all three panels) is the input: a "
        "coarse straight-edged polygon, standing in for a mesh cross-section, sampled at "
        "16 vertices deliberately off the grid. Left: occupancy alone is a staircase, "
        "it has thrown away everything about where the input sits inside each "
        "cell. Middle: marching squares places a vertex ON each crossed edge and connects "
        "the pair inside each cell with a straight segment, a real reconstruction, "
        "but there is no single point representing “the surface in this voxel,” each "
        "active cell contributes two boundary points, not one storable vertex. Right: dual "
        "contouring places exactly one vertex INSIDE each active voxel (fitted from the "
        "same crossings) and connects neighboring vertices with straight segments: "
        "this is the shape that fits O-Voxel's one-feature-per-active-voxel storage "
        "format. The dashed red segment marks the single largest gap between the "
        "reconstruction and the true input on this example: real, but small once the grid "
        "is fine enough.",
        width_px=940,
    )
    qef_inner, qef_vb = D.diagram_qef_corner()
    qef_fig = D.svg_figure(
        qef_inner, qef_vb,
        "<b>Averaging the crossings blurs a sharp corner; solving the QEF recovers it.</b> "
        "Two mesh edges meet at a true corner C inside this cell, crossing the cell's left "
        "and bottom edges (blue dots, with the local surface normal as a short arrow). The "
        "naive centroid of the two crossings (red) lands well off the corner. Minimizing "
        "the quadratic error function against each crossing's tangent plane, regularized "
        "toward that same centroid, lands almost exactly on C (green, nested inside the "
        "black ring). Both points are computed here by an actual least-squares solve, not "
        "placed by hand.",
        width_px=420,
    )
    sec2 = lp.section_v2(
        "dc", "02", "A grid corner and a straight-edged surface: how one vertex per voxel is fitted",
        lp.prose(
            "<p>Take a grid and a surface crossing it. Deliberately not a smooth curve: "
            f"{code('mesh_to_flexible_dual_grid(vertices, faces, ...)')} "
            f"({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '30&ndash;40')}) takes "
            "a triangle mesh, so its input is piecewise linear by construction, a set of "
            "flat facets, not an implicit function. (Classical Dual Contouring, the "
            "background literature this section draws on, is usually presented against a "
            "smooth implicit field, an SDF; that is a different input and worth keeping "
            "separate from what this project's pipeline actually consumes.) At each grid "
            "corner, evaluate an inside/outside sign against the input mesh. Where two "
            "adjacent corners disagree, the surface must cross that edge somewhere: "
            "here that crossing is a real 2D segment intersection against the "
            "input polygon's own straight edges, continuous, not snapped to either corner. "
            "A cell with at least one such crossing is <b>active</b>.</p>"
        ) + tp_fig + lp.prose(
            "<p>Dual contouring's move is to place exactly <b>one dual vertex per active "
            "cell</b>, positioned by fitting the cell's crossings (and their surface "
            "normals), then connect dual vertices with a STRAIGHT segment across every "
            "shared edge between two active cells (a planar quad, in 3D) to build a mesh. "
            "The output is piecewise linear too, there is no curve anywhere in it, only "
            "straight segments between grid-interior points. "
            "<span class=\"bg-tag\">Background, general graphics literature</span> this "
            "specific idea (one dual vertex per cell, fit from crossings and normals via a "
            "quadratic error function) is Dual Contouring, Ju, Losasso, Schaefer &amp; "
            "Warren, SIGGRAPH 2002, not a TRELLIS.2-specific invention. What follows "
            "is TRELLIS.2's own adaptation of it.</p>"
        ) + lp.callout(
            "TRELLIS.2 explicitly derives its dual grid from Dual Contouring but changes "
            "what feeds it: " + paper("TRELLIS.2, &sect;3.1.1",
            "inspired by Dual Contouring (DC) &hellip; Different from DC, we do not "
            "utilize any field representation &hellip; we directly use the asset's mesh "
            "surface to determine edge intersection flags (rather than detecting sign "
            "changes as in DC) and to assign Hermite data") + ". Classic DC needs a scalar "
            "field (an SDF) to get its corner signs and crossings; O-Voxel reads the "
            "crossings directly off the input triangle mesh, so it never needs to build an "
            "implicit field at all: that is what “field-free” means "
            "(" + paper("TRELLIS.2 abstract", "a new &lsquo;field-free&rsquo; sparse voxel "
            "structure termed O-Voxel") + ").",
        ) + lp.prose(
            "<p><b>What “flexible” in “flexible dual grid” refers to.</b> "
            "Per the paper: " + paper("TRELLIS.2, &sect;3.1.1",
            "our algorithm flexibly adjusts the positions of dual vertices and the "
            "existence of dual grid faces to accurately represent arbitrary input surface "
            "data") + ". Two independent things flex: the dual vertex's continuous position "
            "within its cell (never pinned to the cell center), and whether a quad face "
            "between two active neighbors is even instantiated at all (governed by the "
            f"edge-crossing flags, {code('&delta;')}, so a genuine non-manifold or open-surface "
            "junction does not force a face that shouldn't exist). Code: the per-voxel "
            "outputs are voxel indices, "
            f"{code('dual_vertices')}, and {code('intersected')} "
            f"({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '58&ndash;61')}); "
            f"{code('flexible_dual_grid_to_mesh')} "
            f"({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '255&ndash;396')}) "
            "only connects two active voxels across an edge that both of them flag as "
            "intersected.</p>"
        ) + qef_fig + lp.expandable(
            "The exact QEF TRELLIS.2 solves, and how it maps to the code's three weight arguments",
            lp.prose(
                "<p>The paper's minimization "
                "(" + paper("TRELLIS.2, &sect;3.1.1", "") + "):</p>"
            ) + lp.equation(
                r"\min_{\mathbf v}\; e(\mathbf v) = \sum_i d_{\Pi,i}^2 \;+\; \lambda_{\text{bound}}\sum_j d_{L,j}^2 \;+\; \lambda_{\text{reg}}\, d_{\hat q}^2",
                comment=(
                    "First term: squared distance from the candidate vertex to each "
                    "crossing's local tangent plane (uses the surface normal at that "
                    "crossing). Second term: distance to boundary edges (only engages for "
                    "voxels that touch the volume boundary). Third term: pull toward the "
                    "average intersection point, preventing an ill-conditioned or "
                    "under-constrained cell from sending the vertex to infinity."
                ),
            ) + lp.prose(
                "<p>The exposed Python signature carries the same three terms as literal "
                "keyword weights: "
                f"{code('face_weight')}, {code('boundary_weight')}, "
                f"{code('regularization_weight')} "
                f"({cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '36&ndash;38')}), "
                "solved in the C++ extension "
                f"({cite('o-voxel/src/convert/flexible_dual_grid.cpp')}). The two-panel "
                "figure above reproduces the same structure (intersection-tangent-plane "
                "term + centroid-regularization term) with a small NumPy least-squares "
                "solve, for illustration; TRELLIS.2's actual solver is the cited C++ "
                "code, not this page's Python.</p>"
            )
        ) + lp.prose(
            "<p><b>Why none of this is recoverable from occupancy + attributes alone.</b> "
            "Occupancy quantizes the surface to the cell: it says which cells the surface "
            "touches, and nothing about where inside them. An attribute value adds "
            "“how this patch reflects light,” still nothing about position within the cell. "
            "The dual vertex's continuous coordinate is information that was destroyed the "
            "moment a mesh got reduced to occupancy; no amount of post-hoc processing of an "
            "occupancy grid and its attributes can put it back, because there is nothing "
            "left in that data pointing to which of the infinitely many surfaces consistent "
            "with the same occupancy pattern was the real one.</p>"
        ) + lp.prose(
            "<p>This section stays at survey depth; for the full derivation of the QEF "
            "(normal equations, a rank analysis of A&#7511;A, two worked numerical examples, "
            "and how the paper's energy maps onto the actual C++), see "
            "<a href=\"../ovoxel_construction/index.html\">O-Voxel construction, dual "
            "contouring, and the QEF</a>.</p>"
        ),
    )
    sections.append(sec2)

    # ============================================================ 03 SLAT
    sec3 = lp.section_v2(
        "slat", "03", "A latent anchored to a coordinate is what “structured” means",
        lp.prose(
            "<p>The original TRELLIS paper (Xiang et al., "
            "<a href=\"https://arxiv.org/abs/2412.01506\">arXiv:2412.01506</a>, CVPR'25 "
            "Spotlight) defines a Structured LATent as a set of (feature, position) pairs "
            "on a sparse grid, not one vector for the whole object:</p>"
        ) + lp.equation(
            r"\mathbf z = \{(\mathbf z_i, \mathbf p_i)\}_{i=1}^{L}, \quad \mathbf z_i \in \mathbb R^{C}, \quad \mathbf p_i \in \{0,\dots,N-1\}^3",
            comment="TRELLIS, §3.1. pᵢ is an active voxel's position on an N³ grid; only L ≪ N³ positions are active.",
        ) + lp.prose(
            "<p>“Structured” is doing real work in that name: instead of one global "
            "latent vector standing for an entire shape, every latent is anchored to an "
            "explicit 3D coordinate. " + paper("TRELLIS, Introduction",
            "the active voxels pᵢ outline the coarse structure of the 3D asset, while "
            "the latents zᵢ capture finer details of appearance and shape") + ". This "
            "buys three things directly: compute is spent only where the object actually "
            "is (skip empty space entirely), edits stay local (touching one region's "
            "latents doesn't move the rest of the object), and the same sparse latent can "
            "decode into different output formats (meshes, Gaussians, radiance fields) "
            "because the geometry is carried by the coordinates, not baked into one "
            "opaque vector.</p>"
        ) + lp.callout(
            "TRELLIS v1's SLAT features came from rendering, not from the mesh directly: "
            "" + paper("TRELLIS, &sect;3.2", "each voxel is projected onto the multiview "
            "feature maps to retrieve features at corresponding locations, and their "
            "average is used as fᵢ") + ", using a pretrained DINOv2 encoder over "
            "rendered views. Mesh decoding used FlexiCubes, an isosurface method: "
            "" + paper("TRELLIS, &sect;3.2", "extract meshes from 0-level isosurfaces") +
            " after predicting a signed distance value at each of a voxel's 8 corners. "
            "That is a field-based method (an SDF), the same family O-Voxel was built to "
            "avoid.",
            title="TRELLIS v1's SLAT: multiview features in, FlexiCubes/SDF out"
        ) + lp.prose(
            "<p>TRELLIS.2's O-Voxel keeps the SLAT shape (feature anchored to sparse "
            "coordinate) but changes both ends. Features come directly from the 3D asset, "
            "not from rendering it: " + paper("TRELLIS.2, &sect;3.1",
            "a native structured latent space compared to that in [TRELLIS] which is "
            "built from multiview 2D information") + ". And decoding is field-free dual "
            "contouring (§02 above), not an implicit SDF field with FlexiCubes: "
            "" + paper("TRELLIS.2", "recent large 3D generation models predominantly "
            "leverage iso-surface fields (e.g., signed distance function, Flexicubes) to "
            "represent geometry, which have intrinsic limitations in handling open "
            "surfaces, non-manifold geometry, and enclosed interior structures") + ". "
            "O-Voxel's own formal definition mirrors SLAT's shape exactly, with an extra "
            "material slot:</p>"
        ) + lp.equation(
            r"\mathbf f = \{(\mathbf f_i^{\text{shape}},\, \mathbf f_i^{\text{mat}},\, \mathbf p_i)\}_{i=1}^{L}",
            comment="TRELLIS.2, §3.1. fᵢˢʰᵃᵖᵉ is the dual-vertex + intersection-flag geometry channel (§02); fᵢᵐᵃᵗ = (cᵢ, mᵢ, rᵢ, αᵢ), the reflectance parameters (base color, metallic, roughness), plus opacity α.",
        ) + lp.callout(
            "The paper does not frame f<sup>mat</sup> in rendering-equation terms; that "
            "framing below is this project's own, not TRELLIS.2's. The rendering equation "
            "(Kajiya 1986) splits outgoing radiance into an emitted "
            "term and a reflected term: " + paper("this project's method section, "
            "lightgen_overleaf/sec/4_method.tex:14&ndash;16", "L<sub>o</sub> = L<sub>e</sub> "
            "+ &int; f<sub>r</sub> L<sub>i</sub> (&omega;<sub>i</sub>&middot;n) d&omega;"
            "<sub>i</sub>") + ", where L<sub>e</sub> is emitted radiance and the integral "
            "is reflected radiance. Read against that split, the three latents this page "
            "keeps apart map onto it directly: "
            "<code>shape_slat</code> is <b>where</b> the surface is, the geometry the "
            "whole equation is evaluated on; <code>input_tex_slat</code> (f<sup>mat</sup>, "
            "i.e. base color/metallic/roughness, plus opacity) is <b>how</b> it reflects "
            "light, the reflected term's parameters; <code>output_tex_slat</code> "
            "(emission) is <b>what</b> it emits, the emitted term L<sub>e</sub> directly. "
            "&sect;05 uses this where / how / what triad throughout.",
            title="Where / how / what: this project's framing, not TRELLIS.2's"
        ),
    )
    sections.append(sec3)

    # ============================================================ 04 pipeline
    branch_a = lp.flow_branch([
        lp.flow_stage("A1", "mesh_to_flexible_dual_grid", "dual_vertices, intersected (§02)"),
    ])
    branch_b = lp.flow_branch([
        lp.flow_stage("B1", "textured_mesh_to_volumetric_attr", "base_color, metallic, roughness, alpha"),
    ])
    pipeline_html = lp.flow_wrap(
        lp.flow_stage("1", "mesh (.glb)", "trimesh Scene/Trimesh") +
        lp.flow_arrow() +
        lp.flow_stage("2", "load_and_merge / _tight_scene", "one canonical, normalized mesh") +
        lp.flow_arrow() +
        '<div class="flow-parallel">' + branch_a + branch_b + '</div>' +
        lp.flow_arrow() +
        lp.flow_stage("3", "shape_slat_encoder / tex_slat_encoder", "FlexiDualGridVaeEncoder / SparseUnetVaeEncoder") +
        lp.flow_arrow() +
        lp.flow_stage("4", "shape_slat, tex_slat", "32-ch latents, shared coords") +
        lp.flow_arrow() +
        lp.flow_stage("5", "slat_flow_imgshape2tex DiT", "DINOv3 image cond. + shape_slat concat_cond, flow matching") +
        lp.flow_arrow() +
        lp.flow_stage("6", "predicted tex_slat", "sampled, denormalized") +
        lp.flow_arrow() +
        lp.flow_stage("7", "tex_slat_decoder → pbr_voxel → bake", "textured output mesh")
    )
    sec4 = lp.section_v2(
        "pipeline", "04", "Two independent extractions feed two independent encoders",
        lp.prose(
            "<p>The production texturing pipeline "
            f"({cite('trellis2/pipelines/trellis2_texturing.py')}) runs the two channels "
            "from §01 side by side, off the same normalized mesh, then brings them back "
            "together only as conditioning for a flow-matching DiT:</p>"
        ) + pipeline_html + lp.prose(
            "<p>Concretely, for stage 2: this project's own data pipeline normalizes with "
            f"{code('load_and_merge')} ({cite('data_processing/uv_voxel_pipeline/loader.py', '57&ndash;111')}, "
            "one trimesh load, scene-graph transforms baked in, per-part materials "
            "preserved) and {tight}, which re-normalizes a COPY into the exact "
            "".format(tight=code('_tight_scene')) +
            f"[&minus;0.5, 0.5] voxel frame ({cite('data_processing/uv_voxel_pipeline/voxelize.py', '181&ndash;207')}). "
            "Stage 3, shape branch: encoding a mesh calls "
            f"{code('o_voxel.convert.mesh_to_flexible_dual_grid')} directly inside "
            f"{code('encode_shape_slat')} "
            f"({cite('trellis2/pipelines/trellis2_texturing.py', '198&ndash;222')}), then feeds "
            f"the result through {code('FlexiDualGridVaeEncoder')} "
            f"({cite('trellis2/models/sc_vaes/fdg_vae.py', '23&ndash;50')}). Stage 5, the DiT "
            f"itself, has the checkpoint name "
            f"{code('slat_flow_imgshape2tex_dit_1_3B_512_bf16')} "
            f"({cite('configs/gen/emission_dit_74k.yaml', '106')}): image + shape in, texture "
            "latent out. In this project's 74k emission config it is retargeted (channel-"
            "hijacked) to predict emission instead of PBR, but the architecture is byte-"
            "identical: 1536-wide, 30 blocks, 12 heads, DINOv3-L image conditioning "
            f"({cite('configs/gen/emission_dit_74k.yaml', '16&ndash;33, 128&ndash;132')}), "
            "matching the paper's own reported DiT size almost exactly "
            "(" + paper("TRELLIS.2, Appendix A.2", "approximately 1.3B parameters in "
            "total") + ", width 1536, 30 blocks, 12 heads, DINOv3-L conditioning).</p>"
        ) + lp.callout(
            "Two different alignment mechanisms operate at two different stages here, and "
            "conflating them is an easy mistake (an earlier draft of this page made it). "
            "WITHIN one mesh, the dual-grid and attribute extractions are merged with no "
            "reconciliation step at all: "
            f"{code('glb_to_vxz')} writes both into one <code>.vxz</code> "
            f"({cite('SegviGen-emissive/data_toolkit/glb_to_vxz.py', '76&ndash;82')}), trusting "
            "that both, Morton-sorted independently, land in the same coordinate order "
            "because both ran over the identical normalized mesh. "
            f"{code('get_common_coords()')} is a separate, later, MANDATORY stage that has "
            "nothing to do with that merge: it runs AFTER encoding, on latent SLATs, and it "
            "reconciles FOUR latents drawn from TWO different <code>.vxz</code> files, an "
            "input mesh and an output (edited or target) mesh, not the two extractions of "
            "one mesh. It is called in both the interactive and non-interactive branches of "
            f"{code('vxz_to_slat')} "
            f"({cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '96, 99')}), and its "
            "result is what all three saved latents get re-indexed to "
            f"({cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '102&ndash;107')}). "
            f"{code('build_dataset.py')}, the actual dataset builder for the SegviGen "
            f"fine-tune, calls exactly that non-interactive path, "
            f"{code('vxz_to_slat(..., interactive=False)')}, as step 4 of its own pipeline "
            f"({cite('SegviGen-emissive/build_dataset.py', '11, 185')}), so this is a "
            "required stage of the real builder, not an edge case. Training itself then "
            f"does no coordinate checking: {code('train_emissive.py')} loads the three "
            "<code>.pth</code> files independently and pairs them positionally, trusting "
            "the reconciliation that already happened at build time (its only assertion is "
            + code("cond_mode in (\"real\", \"zero\")") + ", unrelated, "
            f"{cite('SegviGen-emissive/train_emissive.py', '66')}). The separate TRELLIS2-"
            f"native emission-generation branch ({code('lightgen_slat.py')}, &sect;06's "
            "256-resolution config) uses neither of these mechanisms: it hard-asserts the "
            "two extractions' coordinates already match "
            f"({cite('trellis2/datasets/lightgen_slat.py', '513&ndash;515, 522&ndash;525')}), "
            "which holds because both are run over the identical normalized vertex set and "
            f"aabb/resolution ({cite('data_toolkit/prep_74k_dual_grid.py', '2&ndash;4, 61&ndash;67')}; "
            f"{cite('data_toolkit/voxelize_pbr.py', '88')}).",
            title="get_common_coords reconciles two meshes' latents, after encoding; it is not what aligns one mesh's two extractions",
        ),
    )
    sections.append(sec4)

    # ============================================================ 05 two pipelines
    pipeline_dongchen = lp.flow_wrap(
        lp.flow_stage("1", "mesh (.glb)", "") +
        lp.flow_arrow() +
        lp.flow_stage("2", "load_and_merge / _tight_scene", "") +
        lp.flow_arrow() +
        lp.flow_stage("3", "textured_mesh_to_volumetric_attr", "attribute channel ONLY") +
        lp.flow_arrow() +
        lp.flow_stage("4", "pbr_voxels_*.vxz, emission_voxels_*.vxz, UV atlas", "on disk", highlight=False)
    )
    pipeline_segvigen = lp.flow_wrap(
        lp.flow_stage("1", "mesh (.glb)", "") +
        lp.flow_arrow() +
        lp.flow_stage("2", "glb_to_vxz.py", "BOTH extractions, merged into one .vxz") +
        lp.flow_arrow() +
        lp.flow_stage("3", "vxz_to_slat.py", "shape_encoder + tex_encoder") +
        lp.flow_arrow() +
        lp.flow_stage("4", "shape_slat.pth, input_tex_slat.pth, output_tex_slat.pth", "on disk", highlight=True)
    )
    sec5 = lp.section_v2(
        "pipelines", "05", "Two preprocessing pipelines, built for two different consumers",
        lp.prose(
            "<p>A natural question once §04 is in view: this project already runs a "
            "voxel-preprocessing step over its dataset, so shouldn't that step already "
            "contain everything a fine-tune needs? In theory, preprocessing that includes "
            "everything downstream needs is exactly the right expectation, and it does "
            "sometimes hold. Here it does not, and the reason is worth stating precisely: "
            "there are two different preprocessing chains in this project, written for two "
            "different consumers, and both names have loosely been called “the "
            "preprocessing.”</p>"
        ) + lp.prose("<p><b>Pipeline A</b>, this project's dataset generator "
            f"({cite('data_processing/uv_voxel_pipeline/voxelize.py')}):</p>") + pipeline_dongchen
        + lp.prose(
            "<p><b>Pipeline B</b>, SegviGen-emissive's own conversion chain "
            f"({cite('SegviGen-emissive/data_toolkit/glb_to_vxz.py')}, "
            f"{cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py')}), the exact input "
            f"format {code('train_emissive.py')} requires "
            f"({cite('SegviGen-emissive/train_emissive.py', '72')} lists "
            + code('["shape_slat.pth", "input_tex_slat.pth", "output_tex_slat.pth"]') +
            " as the three required files per sample):</p>"
        ) + pipeline_segvigen + lp.prose(
            "<p>Concretely, "
            f"{code('glb_to_vxz')} "
            f"({cite('SegviGen-emissive/data_toolkit/glb_to_vxz.py', '47&ndash;82')}) calls "
            f"{code('o_voxel.convert.mesh_to_flexible_dual_grid')} "
            f"({cite('SegviGen-emissive/data_toolkit/glb_to_vxz.py', '59&ndash;62')}) AND "
            f"{code('o_voxel.convert.textured_mesh_to_volumetric_attr')} "
            f"({cite('SegviGen-emissive/data_toolkit/glb_to_vxz.py', '69&ndash;71')}) over "
            "the SAME normalized mesh, Morton-sorts each result independently, then writes "
            "one combined <code>.vxz</code> holding "
            f"{code('dual_vertices')}, {code('intersected')}, {code('base_color')}, "
            f"{code('metallic')}, {code('roughness')}, {code('alpha')} together under one "
            f"coordinate array ({cite('SegviGen-emissive/data_toolkit/glb_to_vxz.py', '76&ndash;82')}). "
            f"{code('vxz_to_slat')} then reads that file twice, once for an “input” mesh "
            "and once for an “output” (edited or target) mesh, and calls the shape encoder "
            "and the texture encoder on each "
            f"({cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '13&ndash;48')}). "
            f"The input mesh and the output mesh are two independent <code>.vxz</code> "
            f"files, so {code('vxz_to_slat')} then explicitly reconciles the four resulting "
            f"latents (input shape, input texture, output shape, output texture) onto one "
            f"shared coordinate set with {code('get_common_coords()')}, called for every "
            f"sample, interactive or not "
            f"({cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '50&ndash;64, 96, 99')}), "
            "and re-indexes each saved latent to it "
            f"({cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '102&ndash;107')}). "
            "This is a separate reconciliation from the shape/attribute merge inside one "
            "<code>.vxz</code> above, which has no such step: that merge just trusts both "
            "independently-sorted arrays already agree, while this one actively re-indexes "
            "across two files, producing exactly the where / how / what trio from &sect;03. Downstream "
            f"training ({code('train_emissive.py')}) does not re-check this: it trusts the "
            "build-time reconciliation and loads the three files positionally.</p>"
        ) + lp.callout(
            "Pipeline A is not missing a step by oversight. It was built to feed the UV-"
            "atlas work and the 256-resolution emission-generation branch, neither of "
            "which ever needed <code>shape_slat</code>, so it never runs "
            f"{code('mesh_to_flexible_dual_grid')} at all. It is complete for what it was "
            "designed to serve; the gap only appears against Pipeline B's requirements, "
            "which this project did not adopt wholesale for its dataset generator. This is "
            "a scope mismatch between two pipelines built for different jobs, not a bug in "
            "either one.",
            title="Pipeline A is complete for its own consumers, just not for train_emissive.py"
        ) + lp.prose(
            "<p><b>Why <code>shape_slat</code> specifically is required, not optional, for "
            f"fine-tuning with {code('train_emissive.py')}.</b> Two separate reasons:</p>"
            "<ul>"
            "<li><b>Architectural.</b> The flow model is loaded from the pretrained "
            f"checkpoint {code('slat_flow_imgshape2tex_dit_1_3B_512_bf16')} "
            f"({cite('SegviGen-emissive/train_emissive.py', '157')}), then further warm-"
            f"started from a SegviGen checkpoint, {code('full_seg.ckpt')} by default "
            f"({cite('SegviGen-emissive/train_emissive.py', '122&ndash;125, 154&ndash;157')}). "
            "Inside the network, the shape latent is not attention-based conditioning: it "
            f"is concatenated onto the noised texture latent along the feature axis, "
            f"{code('x = sp.sparse_cat([x, concat_cond], dim=-1)')} "
            f"({cite('trellis2/models/structured_latent_flow.py', '177&ndash;178')}), before "
            "the transformer ever runs. A checkpoint trained with that channel slot filled "
            "by real shape geometry has learned weights that expect it; leaving the slot "
            "empty or zeroed is a distribution shift the pretrained weights were never "
            "exposed to. Image conditioning (DINOv3) is the separate cross-attention "
            "pathway; shape conditioning is concatenation, a different mechanism.</li>"
            "<li><b>Semantic, motivation only, not a claim from either paper:</b> geometry "
            "is a plausible cue for where emission belongs, a bulb, a screen, a thin strip "
            "along an edge, versus a flat uniform panel. Neither the TRELLIS.2 paper nor "
            "this project's own draft states this explicitly; it is offered here as "
            "motivation for why shape conditioning is a reasonable design, not as a "
            "grounded result.</li>"
            "</ul>"
        ),
    )
    sections.append(sec5)

    # ============================================================ 06 glossary
    def row(term, meaning, produced, shape, feeds):
        return (f'<tr><td class="gterm">{term}</td><td>{meaning}</td>'
                f'<td class="gcite">{produced}</td><td class="gshape">{shape}</td>'
                f'<td>{feeds}</td></tr>')

    glossary_rows = "".join([
        row(code("coords"), "Integer voxel index (plus a batch column): the sparse "
            "tensor's “where.”",
            f"{cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '58&ndash;59')}; batched "
            f"as SparseTensor in {cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '15')}",
            "(N,4) int32<br>[batch,x,y,z]", "both channels; the set that get_common_coords reconciles"),
        row(code("feats"), "The per-voxel feature vector attached to each coords row "
            "(SparseTensor's other half).",
            f"whichever stage populated it; convention: {cite('TRELLIS2/CLAUDE.md', '“Sparse Tensor Convention”')}",
            "(N,C) float<br>C: 6 raw / 32 latent", "both channels"),
        row(code("dual_vertices"), "Continuous 3D position of the dual vertex inside each "
            "active cell (where the surface sits, §02).",
            f"{cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '130&ndash;141')} (Python), "
            f"{cite('o-voxel/src/convert/flexible_dual_grid.cpp')} (QEF solve); stored on disk "
            f"as key <code>vertices</code>, {cite('trellis2/datasets/flexi_dual_grid.py', '130&ndash;131')}",
            "(N,3) uint8 on disk<br>&rarr; float [0,1]", "geometry channel"),
        row(code("intersected"), "Packed 3-bit edge-crossing flags per voxel (does the "
            "surface cross the +x/+y/+z edge from here).",
            f"same call as dual_vertices, paper's &delta;ᵢ (&sect;3.1.1); decoded via "
            f"<code>%2, //2%2, //4%2</code>, {cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '17')}",
            "(N,1) uint8 packed<br>&rarr; (N,3) bool", "geometry channel; gates which quads exist"),
        row(code("shape_slat"), "<b>Where</b> the surface is: 32-channel latent encoding "
            "of the geometry channel (f<sup>shape</sup>) after the SC-VAE. The geometry "
            "the rest of the rendering equation is evaluated on.",
            f"{cite('trellis2/models/sc_vaes/fdg_vae.py', '45&ndash;50')}, called at "
            f"{cite('trellis2/pipelines/trellis2_texturing.py', '219')} or "
            f"{cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '13&ndash;22')}",
            "SparseTensor (M,32)<br>float, M @ res/16 grid", "geometry channel latent; DiT conditioning via concatenation (&sect;05)"),
        row(code("input_tex_slat"),
            "<b>How</b> the surface reflects light: 32-channel latent of the reflectance "
            "channel (base color, metallic, roughness, plus opacity) for the INPUT mesh. "
            "One of the three files train_emissive.py requires.",
            f"{cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '104&ndash;105')}; required "
            f"by {cite('SegviGen-emissive/train_emissive.py', '72')}",
            "SparseTensor (M,32) float", "reflectance channel latent (&sect;05)"),
        row(code("output_tex_slat"),
            "<b>What</b> the surface emits: the same latent slot, but populated from the "
            "OUTPUT/target mesh's channel (PBR for interactive editing; retargeted to "
            "emission for this project's fine-tune, &sect;05).",
            f"{cite('SegviGen-emissive/data_toolkit/vxz_to_slat.py', '106&ndash;107')}",
            "SparseTensor (M,32) float", "emitted-term latent, the fine-tune target (&sect;05)"),
        row(code("base_color, metallic,<br>roughness, alpha"),
            "The reflectance parameters stored per active voxel, plus opacity "
            "(f<sup>mat</sup> = (c,m,r,&alpha;), TRELLIS.2 &sect;3.1): HOW the surface "
            "reflects light, not what it is made of.",
            f"{cite('o-voxel/o_voxel/convert/volumetic_attr.py', '378&ndash;385')}, a C++ voxel "
            f"rasterizer sampling the mesh's UV textures/factors",
            "each (N,3) or (N,1)<br>uint8 [0,255]", "attribute channel; concatenated to 6ch as the tex-VAE input"),
        row(code("emissive"), "A fifth attribute the same rasterizer can also produce, but "
            "it is <b>not</b> part of TRELLIS.2's own f<sup>mat</sup> tuple (paper lists "
            "only c,m,r,&alpha;): this is the lightgen-specific channel bolted on. "
            "WHAT the surface emits.",
            f"{cite('o-voxel/o_voxel/convert/volumetic_attr.py', '382')}; dropped by the stock "
            f"PBR-VAE, see {cite('TRELLIS2/CLAUDE.md', '“Emission (LightGen) Specifics”')}",
            "(N,3) uint8", "lightgen-only emission channel"),
        row("SLAT", "“Structured LATent”: a set of (feature, position) pairs "
            "anchored to a sparse grid, the original TRELLIS formalism (§03).",
            "TRELLIS, arXiv:2412.01506, &sect;3.1", "z = {(zᵢ, pᵢ)}", "the general format shape_slat / tex_slat / emission latents all instantiate"),
        row("O-Voxel", "TRELLIS.2's “omni-voxel” representation: a SLAT-shaped "
            "container holding BOTH the geometry channel and the material channel at the "
            "same coordinates.",
            "TRELLIS.2, arXiv:2512.14692, &sect;3.1", "f = {(fᵢˢʰᵃᵖᵉ, fᵢᵐᵃᵗ, pᵢ)}", "umbrella representation for §01's channels (2)+(3)"),
        row(".vxz", "On-disk container for a sparse voxel grid + attributes: octree-"
            "chunked, Morton/Hilbert-coded within a chunk, entropy-compressed.",
            f"format spec {cite('o-voxel/o_voxel/io/vxz.py', '22&ndash;54')}; writer "
            f"{cite('o-voxel/o_voxel/io/vxz.py', '223&ndash;365')}",
            "chunk_size=256 default", "on-disk serialization of either channel"),
        row("f16/c32 naming", "TRELLIS.2's own checkpoint-naming convention: f16 = 16&times; "
            "spatial downsample (4 &times; SparseSpatial2Channel(2) stages), c32 = 32 latent "
            "channels per surviving voxel.",
            f"{cite('TRELLIS2/CLAUDE.md')}; confirmed at "
            f"{cite('configs/scvae/tex_vae_next_dc_f16c32_fp16_ft_512.yaml', '12&ndash;17')}",
            "n/a", "names every SC-VAE checkpoint (shape/tex/emission)"),
    ])
    glossary_table = f'''<div class="table-scroll">
    <table class="results glossary">
      <colgroup><col class="c-term"><col class="c-mean"><col class="c-cite"><col class="c-shape"><col class="c-feed"></colgroup>
      <thead><tr><th style="text-align:left">term</th><th style="text-align:left">plain-English meaning</th>
      <th style="text-align:left">produced by</th><th style="text-align:left">shape / dtype</th>
      <th style="text-align:left">feeds</th></tr></thead>
      <tbody>{glossary_rows}</tbody>
    </table></div>'''
    sec6 = lp.section_v2(
        "glossary", "06", "The variable names, in one table",
        lp.prose(
            "<p>Every row below is grounded to a specific file:line in the TRELLIS2 "
            "submodule (commit <code>2dabb82</code>) or SegviGen-emissive (commit "
            "<code>f3443da</code>), or to the paper section that defines it.</p>"
        ) + glossary_table,
    )
    sections.append(sec6)

    # ============================================================ 06 SC-VAE
    ladder_inner, ladder_vb = D.diagram_resolution_ladder()
    ladder_fig = D.svg_figure(
        ladder_inner, ladder_vb,
        "<b>Spatial resolution shrinks 16&times; per axis over four stages.</b> Box side "
        "length is drawn proportional to the real per-stage resolution (512, 256, 128, 64, "
        "32); each arrow is one <code>SparseSpatial2Channel(2)</code> downsample "
        f"({cite('trellis2/models/sc_vaes/sparse_unet_vae.py', '51, 196')}). This is a "
        "reduction in the NUMBER OF ACTIVE VOXEL POSITIONS, not in the feature dimension "
        "carried at each surviving position.",
        width_px=680,
    )
    channel_chart = lp.hbar_chart(
        [
            {"label": "input (raw attrs)", "value": 6, "display": "6 ch"},
            {"label": "stage 1 model_channels", "value": 64, "display": "64 ch"},
            {"label": "stage 2 model_channels", "value": 128, "display": "128 ch"},
            {"label": "stage 3 model_channels", "value": 256, "display": "256 ch"},
            {"label": "stage 4 model_channels", "value": 512, "display": "512 ch"},
            {"label": "stage 5 model_channels", "value": 1024, "display": "1024 ch"},
            {"label": "stored latent", "value": 32, "display": "32 ch"},
        ],
        title="feature-vector width at each stage (internal model_channels, not spatial resolution)",
        note=("<b>The feature dimension GROWS through the network, then drops to 32 for "
              "storage.</b> 32 stored latent channels is MORE than the 6 raw input "
              "channels, not fewer: the representation's compactness comes almost "
              "entirely from having far fewer active voxel positions after 16&times; "
              "spatial downsampling, not from a smaller per-voxel vector."),
        label_w=230,
    )
    sec7 = lp.section_v2(
        "scvae", "07", "Two different compressions, on two different axes",
        lp.prose(
            "<p>The Sparse Compression VAE (SC-VAE) that turns raw O-Voxel attributes into "
            "a latent runs a fully sparse-convolutional U-Net "
            "(" + paper("TRELLIS.2, &sect;3.2.1", "our SC-VAE employs a fully sparse-"
            "convolutional network that is both computationally efficient at high "
            "resolutions and generalizes well across scales") + "), and TRELLIS.2 reports "
            "" + paper("TRELLIS.2", "16× spatial downsampling, a high ratio not seen "
            "in prior voxel-based methods") + ". The production PBR-VAE config confirms this "
            f"exactly: four downsample stages ({cite('configs/scvae/tex_vae_next_dc_f16c32_fp16_ft_512.yaml', '13&ndash;17')}), "
            f"each a {code('SparseResBlockS2C3d')} block wrapping "
            f"{code('sp.SparseSpatial2Channel(2)')} "
            f"({cite('trellis2/models/sc_vaes/sparse_unet_vae.py', '51')}): four halvings "
            "of the spatial grid, 2⁴ = 16× total, matching the checkpoint name "
            f"{code('tex_enc_next_dc_f16c32_fp16')}.</p>"
        ) + ladder_fig + lp.prose(
            "<p>This is a POSITION-COUNT reduction: fewer active voxel coordinates survive "
            "at each stage. It is a completely different axis from the feature/channel "
            "width carried at each surviving position, which is what the checkpoint's "
            f"{code('model_channels')} list actually controls "
            f"({cite('configs/scvae/tex_vae_next_dc_f16c32_fp16_ft_512.yaml', '6&ndash;12')}). "
            "Conflating the two has confused this project before, so:</p>"
        ) + channel_chart + lp.prose(
            "<p>The shape (dual-grid) VAE uses the identical architecture pattern "
            f"({code('FlexiDualGridVaeEncoder')} subclasses the same "
            f"{code('SparseUnetVaeEncoder')}, {cite('trellis2/models/sc_vaes/fdg_vae.py', '23&ndash;43')}) "
            f"at the same f16c32 ratio ({cite('configs/scvae/shape_vae_next_dc_f16c32_fp16_ft_512.yaml', '5&ndash;17')}), "
            "with 6 input channels (3 dual-vertex offset + 3 intersection flags, both "
            f"shifted by &minus;0.5, {cite('trellis2/models/sc_vaes/fdg_vae.py', '45&ndash;50')}) "
            "and 7 output channels (3 vertex + 3 intersection logits + 1 quad-split weight "
            f"&gamma;, {cite('trellis2/models/sc_vaes/fdg_vae.py', '83&ndash;110')}); that "
            "&gamma; channel is exactly the paper's learned splitting weight "
            "" + paper("TRELLIS.2, &sect;3.1.1",
            "splitting weights &gamma;ᵢ &isin; &Ropf;&gt;0, controlling how "
            "quadrilateral faces are adaptively subdivided into triangles") + ", consumed "
            f"as {code('split_weight')} by "
            f"{cite('o-voxel/o_voxel/convert/flexible_dual_grid.py', '370&ndash;378')}.</p>"
        ) + lp.callout(
            "This project's own 74k emission training does not use the 512&rarr;32 branch "
            f"above: its dataset config points at {code('tex_enc_next_dc_f16c32_fp16_256')} "
            f"and {code('shape_enc_next_dc_f16c32_fp16_256')} "
            f"({cite('configs/gen/emission_dit_74k.yaml', '39&ndash;41')}): the SAME "
            "f16c32 architecture, trained/applied at voxel resolution 256 instead of 512 "
            "(giving a 16³ latent grid, not 32³). The 256 and 512 configs are "
            "separate checkpoints of the same architecture, not different architectures.",
            title="Our project uses the 256-resolution branch, not the 512 one above",
        ),
    )
    sections.append(sec7)

    # ============================================================ 07 vxz
    morton_inner, morton_vb = D.diagram_morton(order=3)
    morton_fig = D.svg_figure(
        morton_inner, morton_vb,
        "<b>Voxel coordinates are Morton (Z-order) sorted before compression, not raster-"
        "scanned.</b> This 8&times;8 path is a real bit-interleave of (x,y): each step "
        "recurses into same-size quadrants, keeping spatially nearby voxels nearby in the "
        "sorted sequence (unlike a row-major scan, which jumps across the whole width "
        "every row). Better locality compresses better.",
        width_px=340,
    )
    sec8 = lp.section_v2(
        "vxz", "08", "One file, chunked, Morton-sorted, then compressed",
        lp.prose(
            "<p>A voxel coordinate is packed into a <b>30-bit code</b>, 10 bits per axis, "
            f"headroom to 1024³ ({cite('o-voxel/o_voxel/serialize.py', '7&ndash;9, 39&ndash;44')}), "
            "defaulting to Z-order (Morton) with Hilbert available as an alternative mode. "
            "Within one 256³ chunk "
            f"(default {code('chunk_size=256')}, {cite('o-voxel/o_voxel/io/vxz.py', '227')}), "
            "coordinates are sorted by this code before the octree and attributes are "
            "compressed (LZMA by default). A 512³ grid therefore shards into up to "
            "2×2×2 = 8 chunks, each independently compressed and byte-addressed "
            f"in the file header ({cite('o-voxel/o_voxel/io/vxz.py', '22&ndash;54')} is the "
            "full format spec as a comment).</p>"
        ) + morton_fig,
    )
    sections.append(sec8)

    # ============================================================ 09 situation
    sec9 = lp.section_v2(
        "situation", "09", "This project's dataset has the attribute channel, not the geometry channel",
        lp.prose(
            "<p>&sect;05's Pipeline A is what actually built this project's dataset. It "
            f"runs only {code('textured_mesh_to_volumetric_attr')} "
            f"({cite('data_processing/uv_voxel_pipeline/voxelize.py', '210&ndash;226')}, "
            f"confirmed: no call to {code('mesh_to_flexible_dual_grid')} anywhere in that "
            "file): the attribute channel from &sect;01 (how the surface reflects "
            f"light), and only that channel. {code('shape_slat')} has no source in it. "
            "Recovering the geometry channel means reading the original mesh again and "
            f"running the dual-grid extraction separately, which is exactly what "
            f"{code('prep_74k_dual_grid.py')} does, over the same normalized "
            f"pickle dump ({cite('data_toolkit/prep_74k_dual_grid.py', '2&ndash;4')}) that "
            "built the attribute channel, for the subset of the dataset that needed shape "
            "conditioning for the emission DiT "
            f"({cite('configs/gen/emission_dit_74k.yaml', '41')}), a third, narrower "
            "pass, distinct from both pipelines in &sect;05: it reuses TRELLIS.2's own "
            "extraction function but is run as a separate backfill step, not merged into "
            "one combined <code>.vxz</code> the way Pipeline B does.</p>"
        ) + lp.prose(
            "<p>See <a href=\"../training_inputs/index.html\">the training-inputs "
            "explainer</a> for the rest of that argument (what SegviGen training needs, "
            "and what this dataset actually has); this page does not restate it.</p>"
        ),
    )
    sections.append(sec9)

    apx = lp.appendix("Sources", [
        "TRELLIS.2: Xiang, Chen, Xu, Wang, Lv, Deng, Zhu, Dong, Zhao, Yuan, Yang. "
        "“Native and Compact Structured Latents for 3D Generation.” Tech report, "
        "2025. <a href=\"https://arxiv.org/abs/2512.14692\">arXiv:2512.14692</a> "
        "(bib key <code>xiang2025trellis2</code>, "
        f"{cite('lightgen_overleaf/bibliography.bib', '321')}).",
        "TRELLIS (v1): Xiang, Lv, Xu, Deng, Wang, Sun, Yang, Peng, Yang. “Structured "
        "3D Latents for Scalable and Versatile 3D Generation.” CVPR 2025 (Spotlight). "
        "<a href=\"https://arxiv.org/abs/2412.01506\">arXiv:2412.01506</a>.",
        "Dual Contouring: Ju, Losasso, Schaefer, Warren. “Dual Contouring of Hermite "
        "Data.” SIGGRAPH 2002. Background reference for &sect;02; not a TRELLIS.2 "
        "citation.",
        f"Code: lightgen_repo/TRELLIS2 submodule, commit <code>2dabb82</code> "
        f"(3dlg-hcvc/TRELLIS.2-lightning, branch lightgen); "
        f"lightgen_repo/SegviGen-emissive, commit <code>f3443da</code>.",
    ])

    page_html = lp.page(
        title="O-Voxel & TRELLIS.2, explained (lightgen)",
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
.xg2 table.glossary code.cite { white-space: normal; }
.xg2 .bg-tag { display: inline-block; font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
  color: var(--ink-3); border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px; margin-right: 6px; }
.xg2 table.glossary { table-layout: fixed; }
.xg2 table.glossary td, .xg2 table.glossary th { vertical-align: top; padding: 8px 10px; font-size: .87rem; overflow-wrap: anywhere; }
.xg2 table.glossary col.c-term { width: 15%; } .xg2 table.glossary col.c-mean { width: 30%; }
.xg2 table.glossary col.c-cite { width: 33%; } .xg2 table.glossary col.c-shape { width: 12%; }
.xg2 table.glossary col.c-feed { width: 10%; }
.xg2 table.glossary .gterm { font-family: ui-monospace, monospace; font-weight: 600; }
.xg2 table.glossary .gcite { font-size: .78rem; color: var(--ink-2); }
.xg2 table.glossary .gshape { font-family: ui-monospace, monospace; font-size: .8rem; }
.xg2 .flow-parallel { display: flex; gap: 18px; justify-content: center; flex-wrap: wrap; }
/* flow_stage/flow_wrap are v1 components with a hardcoded #0d1014 chip
   background (theme.css); v2's --text/--muted tokens are dark ink meant for
   a light chip, so left unfixed this renders illegible dark-on-black. */
.xg2 .flow-stage { background: var(--tile); }
.xg2 .flow-stage .flbl { color: var(--ink); }
.xg2 .flow-stage .fsub { color: var(--ink-2); }
.xg2 .flow-stage .fnum { color: var(--accent-ink); }
.xg2 .eqcomment { font-size: .86rem; color: var(--ink-2); margin-top: 6px; }
</style>
"""

if __name__ == "__main__":
    build()
