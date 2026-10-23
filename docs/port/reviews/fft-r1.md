# Review fft-r1: GPU FFT (TSL compute)

Reviewer: fresh-context critic, 2026-10-23. Scope: `world-ocean/three/fft.ts`, `demo/fft-debug.*`, `tools/fft-check.mjs`,
and the FFT displacement and motion vectors in `three/index.ts`. The material and foam are out of scope. Other agents changed
`material.ts`, `index.ts` and `cpu-fft.ts` while this review ran, and that is noted where it matters.

## Verdict: **FAIL**

The FFT is correct: it matches the CPU to about 1e-6 m and matches the reference wave for wave. The **tier design** fails.
"low" costs about as much as medium and uses more memory than the whole reference FFT. "high" now looks identical to
"medium". "ultra" adds no waves for 2x the memory and 2.5-10x the GPU time. A tier switch stalls the main thread for
55-76 ms. Displacement sampling is bilinear where the reference uses a bicubic B-spline.

## Single biggest gap (instruction for the builder)

**Make the tiers scale cost, as spec §4 requires ("tiers scale the grid 64 → 512 and the number of cascades").** Today
c0/c1 are fixed at 256² on every tier, so "low" (256/256/128) saves only about 25 % of the bandwidth. Builder's own
numbers: low 0.24-0.28 ms vs medium 0.24-0.32 ms. Low is also 22 MB, more than the reference's ~20 MB FFT.

What to do:
- Make the cascade band split tier-aware: `upper = πN/L/1.5` for the tier's N, with domains re-chosen so the bands stay contiguous.
- Configure the core query at the same per-cascade sizes. `createOcean({ query: [...] })` already takes sizes and a
  `'cubic'` filter, so CPU parity does not force 256.
- Feed what the low grid drops into `unresolvedSlopeVariance`. That mechanism already exists.
- Target low ≤ ¼ of medium in GPU time and memory, for example 3 × 128² or 128/128/64.
- Delete the zero-padded 512 "ultra". Put B-spline bicubic for LOD < 0.5 in `createFftSampling` instead; it helps every tier.
- Either delete c3 or make it a real opt-in gravity-capillary band. Today it is computed on high/ultra and then not shaded
  (see defect 2).
- Re-measure and write a per-tier cost table, including an estimate for an integrated GPU.

## Evidence

### Tests / spec greps
- `npm test`: 42/43 pass. The failure is `lod.test.ts:190` (heap grew 3.3 MB). That is the LOD piece, not the FFT.
- `three/fft.ts` is TSL only: `Fn().compute`, `workgroupArray`, `StorageArrayTexture`, `textureStore`. It has no
  `wgslFn`/GLSL/ShaderMaterial, and every dispatch goes through `renderer.compute(list)` inside the pass. Async compile uses
  `renderer.compileComputeAsync` (present in three 0.186.1). Explicit time comes from `ctx.time`.
- `performance.now` is used only to time the compile.
- Wrapped tile coordinates go through `lod.nodes.tileCoord(P)` (`origin.wrapped`).
- Pass desc: stage `compute`, writes `ocean.displacement/normal/derivatives`.
- Rule 7 (no per-frame allocation), read line by line. `execute()` allocates nothing in steady state. `subarray()` and
  `addUpdateRange` allocate only when a spectrum slot version changes.
- The c3 spectrum is regenerated on the render thread, 32 rows per frame, whenever the core regenerates a slot.
- I checked that three does not overwrite the compute mip chain: `scratch/fft-r1/mip-probe.mjs` counted 0
  `generateMipmaps` calls over 61 frames.

### Numerics (verified myself)
- `tools/fft-check.mjs`, breeze preset, all 4 tiers. Results: `docs/port/reviews/scratch/fft-r1/fft-check-breeze.json`.
  - GPU f32 field vs core `CascadeSim`: max error ≤ 2.4e-6 m (≤ 6e-7 Hs) at t = 12.5, 8191.7 (loop wrap) and 100000.37 s,
    and at the middle of a breeze→storm crossfade.
  - Chained rgba16f texture lookup vs the core's `sampleRest`: ≤ 2.25e-3 Hs over 4096 points.
- Independent check (`scratch/fft-r1/dft-check.mjs`): a brute-force f64 direct DFT from the core's h0 slots, with dispersion
  recomputed independently (not the core's ω table).
  - storm, t = 4321.123 / 98765.4321 / 7777.77, tiers high / low / ultra: c0-c2 max |err| ≤ 1.8e-6 m. Amplitudes are 0.02-3.9 m.
  - c3 max |err| ≤ 2e-5 m. This is the rounding difference between my ω index and the f32-rounded k (16 bins differ).
  - Phase precision at large t is fine: (m·T_int mod 8192 in u32) + m·T_frac.
- Conventions (sign, checkerboard, chop, negated height, east/north) agree with my independent derivation.

### Visual (shots in `docs/port/shots/fft-r1/`)
- **Wave shape, scale and direction match the reference wave for wave.**
  - C1-storm and C5-storm at t = 30, and C5 at t = 33.7 (`motion/C5-storm-{demo,ref}-t33.7.png`): same crests, same bumps,
    same glint positions. So dispersion, speed and direction are right.
  - C5 crest silhouettes are slightly kinked/polygonal vs the reference's smooth curve (`motion/crop-C5-{demo,ref}-crest.png`).
- **No bicubic: blocky geometry.**
  - `?debug=facets` shows bilinear texel blocks and diagonal stair-steps in the displaced geometry: c0 4 m texels at 150 m
    altitude (`motion/facets-alt150.png`) and c2/c1 blocks at C6/C5 (`motion/facets-C6-medium.png`, `facets-C5-medium.png`).
  - The reference avoids this with the 4-tap B-spline at LOD < 0.5 (REFERENCE.md §1.3).
- **c3 (0.84 m tile) made the surface worse.**
  - With c3 normals shaded, high C6-azure (`high/C6-azure.png`) turns the reference's broad glints into a crinkled-foil speckle.
  - medium (`medium/C6-azure.png`) matches the reference's glints almost exactly (`reference-shots/C6-azure.png`).
  - Diff medium vs high: mean 12.3, 26 % of pixels differ by > 8 levels. C1-storm high is also grainier than medium
    (mean 4.3, 16 %).
  - The material agent has since disabled c3 normals by default (`material.ts:138-140`). Now high vs medium differs by
    mean 0.68 (1.7 %) (`high/C6-azure-after-material-change.png`). c3 is pure cost.
- **Ultra 512 vs medium**: mean 3.9 (10.7 % of pixels), a slightly crisper highlight. This is the same band zero-padded, so
  it adds no waves.
- **Low vs medium (C1)**: mean 3.9 (14 %). There is no visible band hole, but low is not cheaper either.
- **Tiling and seams**: no visible 1024/96/9 m repetition at C4, 03 or 150 m altitude. Haze hides 300 m altitude in both
  renderers. No seams between cascades (chained lookup, smoothstep bands).
- **Distance shimmer** at 100-300 m (single-pixel sparkles, `motion/crop-demo-band.png`) is mostly a material matter. The FFT
  side (footprint LOD, unnormalised mips) follows the reference.
- **Motion vectors (`?debug=velocity`, running)**: smooth fields that follow the waves, no noise (`motion/velocity-run-C1.png`,
  `velocity-run-C4.png`).
  - The ×200 scale saturates everywhere near the camera, so magnitudes cannot be read.
  - In `?shot` (paused) the vectors are exactly 0 (`velocity-C1.png`), which is correct for a paused simulation.
  - The previous displacement is last frame's actual texture (history layers), so it is consistent by construction.

### Cost
GPU measurements were heavily contended: 3+ other agents' headless browsers and dev servers were using the GPU.

- fft-check run 1, FFT pass mean / p50 (ms):

  | tier | mean | p50 |
  |---|---|---|
  | low | 0.89 | 0.74 |
  | medium | 0.45 | 0.31 |
  | high | 1.13 | 0.66 |
  | ultra | 3.72 | 3.26 |

- Demo `__perf` compute: low 0.19, medium 0.25-0.27 ms. In the other runs compute timestamps came back null.
- Builder's numbers: 0.24-0.88 ms.
- I could not re-run fft-check: the core's `cpu-fft.ts` changed at 20:10 and dropped `CALL`, `fft-debug.ts` now throws
  `SyntaxError ... does not provide an export named 'CALL'`, and fft-check times out.
- Memory (`debug.memoryBytes`): low 22.3, medium 35.7, high 35.7, ultra 71.2 MB. The reference FFT is ~20 MB (REFERENCE.md §4).
  - Base group ≈ 18.7 MB: h0 3 slots × 2 layers × f32 vec4 = 6.3 MB, work + field 4.2, f32 mip pyramids 2.1,
    textures 5.6 (incl. 2 displacement history layers and an unused derivatives texture), ω 0.5.
- Bandwidth estimate: ≈ 210 B per texel-layer per frame (evolve 52, 2 line FFTs 64, derive ~56, mips ~40). That gives
  low 31 MB, medium 41 MB, high 55 MB, ultra 138 MB of traffic per frame.
- Integrated GPU (UHD 620 class, ~20-25 GB/s usable, ~25 dispatches): low ≈ 1.4-1.9 ms, medium ≈ 1.8-2.3 ms,
  high ≈ 2.5-3 ms, ultra ≈ 6-7 ms. **"low" is ~10 % of a 60 Hz frame on the GPUs that need a low tier.**
- Tier switch (`scratch/fft-r1/tier-switch-t.mjs`):
  - The old tier stays on screen. Async compile took 2.2-17.6 s under load (builder: 1.0-1.3 s).
  - **`setTier('ultra')` blocks the main thread for 55-76 ms.** The long frame lands 69 ms after the call: `makeDetail` runs
    synchronously (512² buffers, ω, TSL graph). `setTier('low')` has no large hitch.
- Full demo vs reference (`perf.mjs`, 1080p): demo 460-760 fps vs reference 160 fps. This is outside the FFT scope; it is
  shown for context only.

### Is "256 for c0/c1 on all tiers" defensible?
Only partly. The builder's band-hole and parity arguments are real if the band split stays fixed at the 256 master. But
both constraints come from the builder's own choices, and the core already supports per-cascade query sizes and a cubic filter.

The spec says the grid scales from 64 to 512. In practice 64 is never used, and 512 is only interpolation. The result does not
meet the purpose of tiers: low is not low-cost, and 3 of the 4 tiers now look the same.

## Ranked defects
1. **Tier ladder does not scale cost or quality.** low ≈ medium cost and > reference memory; high = medium visually; ultra = zero-padded interpolation at 2× memory and 2.5-10× GPU. Spec §4/§5 miss. *(biggest gap)*
2. **c3 cascade is wrong and now useless.**
   - It uses pure gravity dispersion for k = 60-635 rad/m. In that range the capillary term σk³/ρ dominates (about 3× gk at
     k = 635), so these ripples move about 1.7× too slow.
   - It made the glints speckled vs the reference (high C6: 26 % of pixels differ).
   - It is now unshaded, but it is still FFT'd every frame on high/ultra.
   - Its h0 is regenerated on the render thread, 8 frames late vs the core's crossfade slot. During that window c3 blends
     from/to a stale slot.
3. **Displacement sampling is bilinear at LOD 0, with no B-spline bicubic for LOD < 0.5.**
   - Texel blocks and stair-steps show in the geometry (facets views), and crest silhouettes are kinked.
   - "CPU parity" does not justify it: the core has a `'cubic'` query filter.
   - The header claims ultra's 512 "replaces bicubic on c2/c3". It does not cover c0/c1, where the kinks are largest.
4. **Tier switch stalls 55-76 ms** (synchronous `buildGroup` in `setTier`). The spec says "changing tier must not stall".
   Build the group (buffers, ω, node graphs) incrementally or in a worker before the async compile.
5. **Wasted memory and bandwidth.**
   - The detail group always allocates 2 layers (c3) even when 1 is active: about 9 MB dead on medium.
   - The `derivatives` texture and its mips are written every frame but read by nothing (material and foam don't sample it):
     about 1/3 of the derived-store and mip traffic.
   - h0 keeps 3 f32 slots.
   - Result: low 22 MB ≥ the reference's 20 MB.
6. **Verification tooling is broken and fragile.**
   - `demo/fft-debug.ts` imports the core's internal `CALL` scratch array, so the core change at 20:10 broke the page and
     `tools/fft-check.mjs` (timeout).
   - GPU timing is irreproducible: claimed 0.24-0.88 ms vs measured p50 0.31-3.26 ms (contended), and demo compute
     timestamps are often null.
   - No cost table with an integrated-GPU estimate.
7. Motion-vector debug scale (×200) saturates near the camera, so magnitudes can't be checked. There is no running calm-sea
   check (spec §7.9).
8. Minor issues:
   - `startBuild` leaks: request t → u → t while the first t is in flight, and two TierStates can land in `pending` without
     the first being disposed.
   - The foam mask uses the current target wind direction rather than the blended spectrum's.
   - `computeLostSlope` runs a 65k-element loop with `**` on the render thread at every c2 upload.

## Positives (verified)
- Correct, deterministic, precise FFT. GPU = CPU to 1e-6 m, including at t = 1e5 s and during a crossfade.
- Two dispatches per line FFT in shared memory, with f64-built twiddles.
- Explicit-LOD footprint sampling and unnormalised normal mips, like the reference.
- Exact last-frame history for motion vectors.
- Old tier stays drawn during async compile.
- Wave field matches the reference wave for wave in shape, scale, speed and direction.

## Screenshots
- `docs/port/shots/fft-r1/high/C6-azure.png` vs `docs/port/shots/fft-r1/medium/C6-azure.png` vs
  `docs/port/reference-shots/C6-azure.png`: c3 speckle vs the reference glints.
- `docs/port/shots/fft-r1/motion/facets-alt150.png`: bilinear c0 texel blocks and stair-steps in the geometry (no bicubic).
- `docs/port/shots/fft-r1/motion/C5-storm-demo-t33.7.png` vs `C5-storm-ref-t33.7.png`: wave-for-wave match in motion.
