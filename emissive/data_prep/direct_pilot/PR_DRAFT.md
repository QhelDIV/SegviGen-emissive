# Upstream PR draft (OWNER-GATED — do not open without owner approval)

Target: microsoft/TRELLIS.2 (base commit 75fbf01)
Branch: fix/emissive-mipmap-sampler  (local sha f0e1272)
Change: one line, o-voxel/src/convert/volumetic_attr.cpp:559
        roughnessMipmaps[mid] -> emissiveMipmaps[mid]

## PR title
Fix emissive texture sampled through the roughness mipmap pyramid

## PR body
`textured_mesh_to_volumetric_attr`'s per-voxel emissive sampling
(`o-voxel/src/convert/volumetic_attr.cpp:559`) passes `roughnessMipmaps[mid]` to the
emissive texture sampler instead of `emissiveMipmaps[mid]`. The emissive mipmaps are
built at L364-368 but never used — a copy-paste from the roughness block above — so the
emissive texture is sampled through the roughness mip pyramid and the per-voxel
`emissive` attribute is decoupled from the material's real emission. (base_color /
metallic / roughness sampling is unaffected.)

Evidence, 54-shape TexVerse batch vs the emission-only render as ground truth:
correlation of the per-voxel emissive fraction with true emission was 0.25 before and
0.834 after; shapes that emit nothing dropped from 0.33 mean fabricated coverage
(up to 96%) to 0.010. Consistent with `data_toolkit/voxelize_pbr.py` deleting the
attribute (`del attr['emissive']`).

## How to apply (owner)
On a fork of microsoft/TRELLIS.2 at (or rebased onto) 75fbf01:
  git am 0001-emissive-mipmap-fix.patch
  # or, single line, by hand: volumetic_attr.cpp:559 roughnessMipmaps -> emissiveMipmaps
Rebuild the o-voxel extension. Validation harness + panels: direct_pilot/.
