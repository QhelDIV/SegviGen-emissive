#!/usr/bin/env python3
"""Build the lightgen emissive data pipeline explainer (xgpage v2 editorial):
GLB -> ovoxels -> latent, with the three resolutions (256/512/32) and the
file/variable flow made unambiguous.

Every fact on this page is grounded in a specific file:line, cited inline and
collected again in the provenance section at the end. Nothing beyond the
verified facts in the build brief is asserted; where a detail (e.g. the exact
contents of atlas.npz) was not in the cited source excerpt, the page says so
rather than guessing.

Run: .venv_console/bin/python web/_preview/data_pipeline/build.py
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
SITE_ASSETS = "/projects/omages/yanxg/lightgen/assets"


def code(text):
    return f'<code>{text}</code>'


def cite(path, lines=None):
    s = path if lines is None else f"{path}:{lines}"
    return f'<code class="cite">{s}</code>'


def row(*cells):
    return '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'


def build():
    assets_dir = os.path.join(WEB, "assets")

    # -------------------------------------------------------------- hero
    fig_flow = D.svg_figure(
        *D.diagram_pipeline_flow(),
        caption_html=(
            "<b>GLB feeds two independent o_voxel extractions that run at different "
            "resolutions and different times.</b> Dongchen's appearance bake "
            f"({code('textured_mesh_to_volumetric_attr')}) runs first, at 256&sup3;, and "
            f"writes the pbr/emission voxel files. Our builder then re-extracts geometry "
            f"directly from the GLB at 512&sup3; ({code('mesh_to_flexible_dual_grid')}) and "
            "upsamples the 256&sup3; attribute values onto that same 512&sup3; grid, "
            "merging both into the <code>input.vxz</code> / <code>output.vxz</code> pair. "
            "The unchanged SegviGen encoder then compresses that pair to the 32&sup3; latent "
            "the model trains on, three token-aligned <code>.pth</code> files."
        ),
        width_px=960,
        id="hero-flow",
    )

    hero = lp.hero_header(
        "lightgen emissive data pipeline &middot; GLB to latent &middot; 2026-07-29",
        "Three Resolutions, One Pipeline",
        dek_html=(
            "The emissive dataset pipeline crosses three grid resolutions between the "
            "source GLB and the latent the model trains on: 256&sup3; where attribute "
            "values are baked, 512&sup3; where geometry is extracted and the pretrained "
            "encoder expects its input, and 32&sup3; where the model actually sees the "
            "data. This page makes that crossing, and the files and variables that carry "
            "it, unambiguous: one end-to-end flow diagram, the exact nesting arithmetic "
            "between the three resolutions, and a table of every file and variable at "
            "each stage, all grounded in file:line citations against the running code."
        ),
        toc=[
            ("flow", "The end-to-end flow"),
            ("ladder", "The resolution ladder"),
            ("stage1", "Stage 1: GLB &rarr; ovoxels (256&sup3;)"),
            ("stage2", "Stage 2: GLB + ovoxels &rarr; the 512&sup3; pair"),
            ("stage3", "Stage 3: vxz &rarr; latent (32&sup3;)"),
            ("functions", "The two o_voxel functions"),
            ("upsample", "Why upsample, not re-bake"),
        ],
    )

    sections = []

    # ============================================================ 01 flow
    sec1 = lp.section_v2(
        "flow", "01", "The pipeline crosses three resolutions and two independent extractions",
        lp.prose(
            "<p>Two things happen to the same GLB, at different resolutions and different "
            "times, and everything downstream is the bookkeeping that keeps them lined up. "
            "Dongchen's pipeline bakes appearance at 256&sup3; (&sect;03 below); this "
            "project's dataset builder separately extracts geometry at 512&sup3;, the "
            "pretrained encoder's native grid, and upsamples the 256&sup3; attribute values "
            "onto it (&sect;04); the unchanged SegviGen encoder then compresses the result "
            "to the 32&sup3; latent the model trains on (&sect;05). The diagram below is the "
            "map for the rest of this page: the colored band behind each box names the "
            "resolution it operates at.</p>"
        ) + fig_flow + lp.prose(
            "<p>The one non-obvious point worth stating up front: <b>stage 1 produces no "
            "geometry channel.</b> It is an appearance bake, full stop. Occupancy (which "
            "cells are active) is incidental scaffolding to hang attribute values on, not a "
            "geometry deliverable in its own right; geometry only enters the pipeline in "
            "stage 2, from a fresh, independent extraction at 512&sup3;. This is the single "
            "fact that resolves most of the confusion between the two pipelines (&sect;06 "
            "makes it precise).</p>"
        ),
    )
    sections.append(sec1)

    # ============================================================ 02 ladder
    fig_ladder = D.svg_figure(
        *D.diagram_resolution_ladder(),
        caption_html=(
            "<b>The three resolutions nest exactly: 512 / 256 = 2&times; per axis, "
            "512 / 32 = 16&times; per axis.</b> One latent token (the accent outer square) "
            "spans a 16&sup3; block of 512&sup3; cells, 4,096 cells, which is the same "
            "footprint as an 8&sup3; block of 256&sup3; cells, 512 distinct attribute "
            "values. A 2D cross-section is shown for legibility; the same factors apply "
            "along all three axes."
        ),
        width_px=800,
        id="ladder-fig",
    )
    sec2 = lp.section_v2(
        "ladder", "02", "One latent token covers 4,096 cells at 512&sup3;, 512 attribute values at 256&sup3;",
        lp.prose(
            "<p>The nesting arithmetic that makes the resolution ladder tractable:</p>"
        ) + fig_ladder + lp.prose(
            "<p>Concretely: <b>512 / 256 = 2&times; per axis</b>, so every 256&sup3; cell "
            "corresponds to a 2&times;2&times;2 = 8 block of 512&sup3; cells, all carrying "
            "the SAME attribute value (attributes are piecewise-constant per 256 cell by "
            "construction, &sect;06). And <b>512 / 32 = 16&times; per axis</b>, so every "
            "latent token corresponds to a 16&times;16&times;16 = 4,096 block of 512&sup3; "
            "cells, equivalently an 8&times;8&times;8 = 512 block of 256&sup3; cells. A "
            "single latent token can therefore span up to 512 distinct baked attribute "
            "values, compressed through the encoder into one 32-channel feature.</p>"
        ),
    )
    sections.append(sec2)

    # ============================================================ 03 stage 1
    stage1_files_rows = "".join([
        row(code("pbr_voxels_256/&lt;sid&gt;.vxz"),
            "base_color, metallic, roughness, alpha",
            "base_color u8 (N,3)<br>metallic u8 (N,1)<br>roughness u8 (N,1)<br>alpha u8 (N,1)"),
        row(code("emission_voxels_256/&lt;sid&gt;.vxz"), "emissive",
            "u8 (N,3)"),
        row(code("&lt;sid&gt;.coords.npz"),
            "one array <code>coords</code>; byproduct, docstring purpose is literally "
            "&ldquo;for the validator&rdquo; (voxelize.py:6), not a geometry deliverable",
            "int32 (N,3), range 0&ndash;255"),
        row(code("atlas.npz"),
            "present in the output directory; contents not detailed in the cited source "
            "excerpt for this page, not verified here",
            "not verified"),
    ])
    stage1_table = lp.results_table(
        ["file", "what's inside", "dtype / shape"], stage1_files_rows,
    )
    sec3 = lp.section_v2(
        "stage1", "03", "Stage 1 bakes appearance at 256&sup3;; nothing here is a geometry deliverable",
        lp.prose(
            "<p>Dongchen's pipeline (<code>uv_voxel_pipeline</code>) calls one o_voxel "
            f"function per shape ({cite('uv_voxel_pipeline/voxelize.py', '218')}):</p>"
        ) + lp.code_block(
            "o_voxel.convert.textured_mesh_to_volumetric_attr(\n"
            "    scene, grid_size=256,\n"
            "    aabb=[[-VOXEL_HALF]*3, [VOXEL_HALF]*3],\n"
            "    mip_level_offset=_FINEST_MIP_OFFSET,\n"
            ")"
        ) + lp.prose(
            "<p>which returns <code>(coord, attr)</code>. <code>attr</code> carries "
            "<code>base_color</code>, <code>metallic</code>, <code>roughness</code>, "
            f"<code>emissive</code>, <code>alpha</code>, and <code>normal</code>; "
            f"<code>normal</code> is popped and discarded "
            f"({cite('uv_voxel_pipeline/voxelize.py', '227')}). Written per shape into "
            f"{cite('/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k/&lt;sid&gt;/')}:</p>"
        ) + stage1_table + lp.callout(
            "Both .vxz files share the SAME coords, in the SAME order (verified). This is "
            "why the two files can be read as one attribute set even though they are "
            "written to separate directories: base_color/metallic/roughness/alpha and "
            "emissive index the same voxels.",
            title="pbr_voxels_256 and emission_voxels_256 are coordinate-aligned"
        ) + lp.prose(
            "<p>The <code>.vxz</code> container format: an 8-byte magic "
            "(<code>&ldquo;VXZ&rdquo;</code> + version), then a JSON header, then the "
            "compressed payload. A real header, read from sid "
            "<code>1142867141f74ee6955593ff6f59e51a</code>:</p>"
        ) + lp.code_block(
            '{"num_voxel": 796737, "chunk_size": 256, "filter": "none", '
            '"compression": "lzma", "compression_level": 9, "raw_size": 14341266, '
            '"compressed_size": 1171425, "compress_ratio": 12.24}'
        ) + lp.prose(
            "<p>Note what is absent from that returned attribute dict and this stage's "
            "output: no dual vertex, no edge-crossing flag, nothing that says WHERE inside "
            "a voxel the surface actually sits. Occupancy here is incidental scaffolding "
            "to hang appearance values on, and that is a complete, correct job for what "
            "this pipeline was built to do, not a gap or a bug (&sect;06).</p>"
        ),
    )
    sections.append(sec3)

    # ============================================================ 04 stage 2
    stage2_geom_rows = "".join([
        row(code("voxel_indices"), "which 512&sup3; cells are active", "int, (N,3)"),
        row(code("dual_vertices"), "continuous dual-vertex position inside each active cell; "
            "rescaled: <code>clamp(dual_vertices*512 - voxel_indices, 0, 1)*255</code>, cast "
            "to uint8 (lines 164&ndash;165)", "u8 (N,3)"),
        row(code("intersected"), "packed edge-crossing flags", "u8 (N,1) packed"),
    ])
    stage2_input_rows = "".join([
        row(code("base_color, metallic,<br>roughness, alpha"),
            "upsampled from the 256&sup3; bake (Upsampler256to512)", "u8, (N,3)/(N,1)&times;3"),
        row(code("dual_vertices, intersected"), "from the fresh 512&sup3; geometry extraction",
            "u8 (N,3) / u8 (N,1) packed"),
    ])
    stage2_output_rows = "".join([
        row(code("base_color") + " slot",
            "emission, binarized at &gt;1/255 (any nonzero emission), written into the "
            "base_color channel", "u8 (N,3)"),
        row(code("metallic, roughness, alpha"),
            "module CONSTANTS, not derived per-voxel: "
            "<code>OUT_METALLIC_U8=0</code>, <code>OUT_ROUGHNESS_U8=255</code>, "
            "<code>OUT_ALPHA_U8=255</code> (lines 97&ndash;99)",
            "u8, fixed"),
        row(code("dual_vertices, intersected"), "SAME values as input.vxz (identical coords)",
            "u8 (N,3) / u8 (N,1) packed"),
    ])
    sec4 = lp.section_v2(
        "stage2", "04", "Stage 2 re-extracts geometry at 512&sup3; and upsamples the 256&sup3; attributes onto it",
        lp.prose(
            "<p>Two independent paths, both feeding the same 512&sup3; coordinate grid "
            f"({cite('segvigen_emissive/code/build_dataset_direct.py')}, "
            f"{code('GRID = 512')} at line 94):</p>"
            "<p><b>Geometry path.</b> "
            f"{code('merged_mesh_512_frame(glb)')} &rarr; {code('dual_grid_512()')} &rarr;</p>"
        ) + lp.code_block(
            "o_voxel.convert.mesh_to_flexible_dual_grid(\n"
            "    vertices, faces, grid_size=512,\n"
            "    aabb=[[-0.5]*3, [0.5]*3],\n"
            "    face_weight=1.0, boundary_weight=0.2,\n"
            "    regularization_weight=1e-2,\n"
            ")"
        ) + lp.prose(
            f"<p>({cite('build_dataset_direct.py', '154')}), returning:</p>"
        ) + lp.results_table(["variable", "what it is", "dtype / shape"], stage2_geom_rows) +
        lp.prose(
            "<p><b>Attribute path.</b> <code>class Upsampler256to512</code> "
            f"({cite('build_dataset_direct.py', '170')}): for each 512&sup3; cell, look up "
            "<code>parent = coord512 // 2</code> in the 256&sup3; attribute dict by exact "
            "match; where a 512 cell's parent has no attribute (the two occupancy sets do "
            "not nest exactly), fall back to a scipy <code>cKDTree</code> nearest-occupied-"
            "256-voxel lookup. Writes, per shape:</p>"
            "<p><code>input.vxz</code>:</p>"
        ) + lp.results_table(["field", "source", "dtype / shape"], stage2_input_rows) +
        lp.prose("<p><code>output.vxz</code>:</p>") +
        lp.results_table(["field", "source", "dtype / shape"], stage2_output_rows) +
        lp.callout(
            "Emission is deliberately written into the base_color slot, not a new channel. "
            "That lets the pretrained PBR encoder be reused byte-identically, with zero "
            "architecture change, to encode the emission TARGET the same way it encodes "
            "reflectance INPUTS.",
            title="The channel hijack: emission rides in base_color's slot on purpose"
        ) + lp.prose(
            "<p>Both files share IDENTICAL coords, the freshly extracted 512&sup3; dual "
            "grid, so <code>input.vxz</code> and <code>output.vxz</code> are the same "
            "voxel set with two different attribute payloads.</p>"
        ),
    )
    sections.append(sec4)

    # ============================================================ 05 stage 3
    pth_rows = "".join([
        row(code("shape_slat.pth"), "&ldquo;where the surface is&rdquo;",
            "geometry, from the 512&sup3; dual grid", "feats (1041,32) f32<br>coords (1041,4) i32"),
        row(code("input_tex_slat.pth"), "&ldquo;how it reflects light&rdquo;",
            "input.vxz (reflectance)", "feats (1041,32) f32<br>coords (1041,4) i32"),
        row(code("output_tex_slat.pth"), "&ldquo;what it emits&rdquo;",
            "output.vxz (binarized emission target)", "feats (1041,32) f32<br>coords (1041,4) i32"),
    ])
    sec5 = lp.section_v2(
        "stage3", "05", "n_common_coords is the latent token count: 1,041 tokens for the measured shape",
        lp.prose(
            "<p>The unchanged SegviGen <code>vxz_to_slat</code> code encodes the 512&sup3; "
            "pair into three token-aligned latents, written per shape into "
            f"{cite('/cs/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/train_72k/&lt;sid&gt;/')}: "
            "<code>shape_slat.pth</code>, <code>input_tex_slat.pth</code>, "
            "<code>output_tex_slat.pth</code>, plus the source <code>input.vxz</code>, "
            "<code>output.vxz</code>, and <code>meta.json</code>. Each <code>.pth</code> is "
            "a dict of two tensors; measured on real shape "
            "<code>000054a8d92d4c80828161fbb235d141</code>:</p>"
        ) + lp.results_table(["file", "channel gloss", "encodes", "tensors"], pth_rows) +
        lp.callout(
            "All THREE .pth files have identical coords, which is what makes input and "
            "target line up token for token: the model reads one geometry token "
            "(shape_slat), one appearance token (input_tex_slat) at a coordinate, and is "
            "trained to predict the corresponding output_tex_slat token at that SAME "
            "coordinate.",
            title="Identical coords across all three latents is the alignment mechanism"
        ) + lp.prose(
            "<p><code>meta.json</code> for the same shape, verbatim:</p>"
        ) + lp.code_block(
            '{"sid": "000054a8d92d4c80828161fbb235d141", "emissive_frac": 0.9446967244148254, '
            '"gap_frac": 8.317989163928661e-05, "n_voxels_512": 264487, "n_voxels_256": 66147, '
            '"n_common_coords": 1041, "timing_s": {"load_ovox": 0.084, "load_mesh": 0.134, '
            '"dual_grid": 0.358, "upsample": 0.522, "write_vxz": 2.147, "encode_slat": 0.548, '
            '"emis_mask": 0.072, "total": 3.866}}'
        ) + lp.prose(
            "<p>This shape's <code>dual_grid</code> timing (0.358s) is ONE example, not the "
            "average: measured across 41,671 shapes in the production build, dual_grid "
            "costs 1.17s of a 9.58s shape on average (12.2% of build time, &sect;07).</p>"
        ) + lp.prose(
            "<p><b><code>n_common_coords</code> (1,041) IS the latent token count</b>: it is "
            "the number of rows in every <code>coords</code> tensor above, and by "
            "&sect;02's arithmetic, up to 512 baked 256&sup3; attribute values can sit "
            "behind any one of those 1,041 tokens.</p>"
        ),
    )
    sections.append(sec5)

    # ============================================================ 06 functions
    func_rows = "".join([
        row(code("mesh_to_flexible_dual_grid()"),
            "vertices, faces (geometry ONLY, no materials)",
            "&ldquo;where exactly is the surface inside this cell&rdquo;",
            "voxel_indices, dual_vertices, intersected"),
        row(code("textured_mesh_to_volumetric_attr()"),
            "the whole textured Scene",
            "&ldquo;what does the surface look like here&rdquo;",
            "coords + attribute dict"),
    ])
    sec6 = lp.section_v2(
        "functions", "06", "o_voxel has exactly two extraction functions, and TRELLIS.2 calls both",
        lp.prose(
            "<p><code>o_voxel.convert</code> contains exactly two modules: "
            "<code>flexible_dual_grid.py</code> and <code>volumetic_attr.py</code>.</p>"
        ) + lp.results_table(["function", "input", "answers", "returns"], func_rows) +
        lp.prose(
            "<p>TRELLIS.2's own preprocessing calls BOTH, ten lines apart, both at "
            "<code>grid_size=512</code>, the same aabb: "
            f"{cite('segvigen_emissive/code/SegviGen/data_toolkit/glb_to_vxz.py', '59')} and "
            f"{cite('segvigen_emissive/code/SegviGen/data_toolkit/glb_to_vxz.py', '69')}.</p>"
        ) + lp.callout(
            "Dongchen calls only the appearance function, at 256&sup3;, because appearance "
            "is all his pipeline needed. Nothing is missing or broken in his code; it is a "
            "scope mismatch between two pipelines built for different jobs, not a defect "
            "in either one.",
            title="Stage 1's missing geometry channel is scope, not a defect"
        ) + lp.prose(
            "<p>One more consequence of the split worth stating precisely: the attribute "
            "pass is NEVER passed <code>dual_vertices</code>, so it cannot be sampling at "
            "the dual-contouring vertex. <b>Attributes are piecewise-constant per cell.</b> "
            "That is exactly the property &sect;02's arithmetic and &sect;07's upsampling "
            "argument both depend on.</p>"
        ),
    )
    sections.append(sec6)

    # ============================================================ 07 upsample
    sec7 = lp.section_v2(
        "upsample", "07", "Upsampling attribute values is not an approximation; the mip bug that motivated it is orthogonal to grid resolution",
        lp.prose(
            "<p>Because attributes are piecewise-constant per 256&sup3; cell by "
            "construction (&sect;06), copying a cell's value to its 8 child 512&sup3; "
            "cells (&sect;02's 2&times; nesting) is the natural operation, not an "
            "approximation of some finer ground truth that a 512&sup3; re-bake would "
            "recover.</p>"
        ) + lp.callout(
            "The emissive mip bug is real but orthogonal to grid resolution: "
            f"{cite('uv_voxel_pipeline/voxelize.py', '49&ndash;58')} documents that o_voxel's "
            "LINEAR mip sampling of emissive textures reads garbage from coarse mip levels "
            "for textures of 512&times;512 pixels or larger (and smaller ones on "
            "fine-triangle geometry). <b>This 512 refers to texture pixels, not the voxel "
            "grid. The bug is independent of grid resolution.</b> Because o_voxel's "
            "emissive default is white, a black or sparse emissive texture bakes "
            "to near-white, a false positive; only emissive is affected, base_color, "
            "metallic and roughness are not. The fix is "
            f"<code>mip_level_offset=_FINEST_MIP_OFFSET</code> "
            f"({cite('uv_voxel_pipeline/voxelize.py', '223')}, "
            f"{cite('uv_voxel_pipeline/voxelize.py', '59')}), forcing the finest mip, which "
            "point-samples the full-resolution texture at each voxel center "
            f"({cite('uv_voxel_pipeline/voxelize.py', '55&ndash;57')}). This matches the "
            "nvdiffrast atlas bake, which also point-samples. A 512&sup3; re-bake would "
            "pass the same parameter and be equally clean: the bug is not a reason to "
            "prefer upsampling.",
            warn=True,
            title="The mip bug is a texture-size bug, not a grid-size argument for upsampling"
        ) + lp.prose(
            "<p>The real reasons this pipeline upsamples the 256&sup3; attributes onto "
            "the 512&sup3; grid rather than re-baking appearance natively at 512&sup3; are "
            "not correctness arguments:</p>"
        ) + lp.results_table(["reason", "detail"], "".join([
            row("Cost", "Geometry extraction (the dual grid) costs 1.17s of a 9.58s shape "
                "on average, 12.2% of build time, measured over 41,671 shapes in the "
                "production build "
                f"({cite('segvigen_emissive/logs/build_72k_237094_*.log')}, "
                "<code>dualgrid=</code> per-stage timings; a build-log aggregate, not a "
                "code citation). Appearance baking (stage 1) is the more expensive step by "
                "comparison: GENERATION.md notes its wall-clock is CPU-bound on xatlas UV "
                "unwrapping, not GPU."),
            row("Fixed ground truth", "The 256&sup3; bake is the agreed ground truth for "
                "this project; re-baking appearance at 512&sup3; mid-build would change "
                "what the model is trained against."),
            row("Information ceiling", "Per &sect;02's arithmetic, one latent token "
                "already averages up to 512 distinct 256&sup3; attribute values; native "
                "512&sup3; attribute detail would be averaged away by the encoder before "
                "the model sees it."),
        ])) + lp.prose(
            "<p>A 512&sup3; re-bake would be correct, not just cheaper to avoid: none of "
            "these three reasons argues that upsampling is more correct than re-baking, "
            "only that it is less disruptive.</p>"
        ),
    )
    sections.append(sec7)

    # ============================================================ appendix
    apx = lp.appendix("Provenance", [
        f"Stage 1 code: {cite('uv_voxel_pipeline/code_snapshot/data_processing/uv_voxel_pipeline/voxelize.py')}: "
        "call and grid_size at line 218, normal popped at line 227, coords.npz's "
        "&ldquo;for the validator&rdquo; docstring at line 6, the broken linear-mip-on-"
        "emissive documentation at lines 49&ndash;58.",
        f"Stage 1 output directory: {cite('/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k/&lt;sid&gt;/')}. "
        ".vxz header example above is from sid <code>1142867141f74ee6955593ff6f59e51a</code>.",
        f"Stage 2 code: {cite('segvigen_emissive/code/build_dataset_direct.py')}: "
        "GRID=512 at line 94, mesh_to_flexible_dual_grid call at line 154, dual_vertices "
        "rescale at lines 164&ndash;165, Upsampler256to512 at line 170, "
        "OUT_METALLIC_U8/OUT_ROUGHNESS_U8/OUT_ALPHA_U8 constants at lines 97&ndash;99.",
        f"Stage 3 output directory: {cite('/cs/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/train_72k/&lt;sid&gt;/')}. "
        "The .pth tensor shapes and meta.json above are from sid "
        "<code>000054a8d92d4c80828161fbb235d141</code>; this is a DIFFERENT shape from the "
        ".vxz header example in &sect;03, each example is labeled with its own sid rather "
        "than mixed.",
        f"TRELLIS.2's own combined-call preprocessing: {cite('segvigen_emissive/code/SegviGen/data_toolkit/glb_to_vxz.py', '59, 69')}.",
        f"&sect;07's dual_grid timing (1.17s mean, 12.2% of build time): "
        f"{cite('segvigen_emissive/logs/build_72k_237094_*.log')}, aggregated over 41,671 "
        "&lsquo;[ok]&rsquo; lines' <code>dualgrid=</code> fields. A build-log aggregate from "
        "this project's own build run, not a citation into Dongchen's pipeline code.",
        "Every figure on this page is a computed schematic (box/arrow/grid layout drawn "
        "from a coordinate table in code), not a hand-placed illustration; no rendered "
        "3D data is claimed by either diagram.",
        "Not independently verified for this page (flagged, not silently assumed): the "
        "exact contents of <code>atlas.npz</code> (&sect;03) were not in the cited source "
        "excerpt.",
    ])

    page_html = lp.page(
        title="GLB to Latent: Three Resolutions, One Pipeline (lightgen)",
        header_html=hero,
        body_sections=sections + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v2",
        needs_katex=False,
        extra_head=EXTRA_CSS,
    )
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")


EXTRA_CSS = """<style>
.xg2 code.cite { font-size: .82rem; background: var(--code-bg); border-radius: 4px; padding: 1px 5px; overflow-wrap: anywhere; }
</style>
"""

if __name__ == "__main__":
    build()
