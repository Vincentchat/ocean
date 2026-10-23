# Foam review r1 (three/foam.ts + material/refraction consumption)

Reviewer: fresh context, harsh. Scope per the coordinator's direction change: FUNCTIONALITY + SPEC only. Look differences
from the WebGL reference are notes, not fail reasons.

## Verdict: FAIL (narrow: the core works; a reset bug and spec/integration problems remain)

The compute core is solid. It is deterministic, frame-rate independent, exact under re-projection, origin invariant
at 1e7 m, continuous across tier switches, and cheap. It fails on: (1) reset handling that wipes all whitecaps
(and leaves them at exactly 0 while paused); (2) a pass graph that declares `ocean.foam` with no reader;
(3) the asset hook not reachable through `createOceanGpu`; (4) a dead public API field (`fineInSource`) with a
false doc.

## Single biggest gap

A history reset (first frame, an anchor jump > 200 m, dt < 0 or dt > 2 s) cold-starts the history at 0
(`FOAM_PARAMS.coldStart = true`, frozen and not an option). When this happens while paused, dt = 0 keeps
retention at 1, so the history stays at **exactly 0 for as long as the pause lasts**. Measured: paused, free
camera moved 1000 m, history mean 0.3268 -> **0.0000**. That breaks photo mode and editor cameras. While running,
every camera cut, teleport, rewind or > 2 s hitch makes all whitecaps vanish and fade back over ~1.5 s (only 65 % of
equilibrium after 1.5 s: mean 0.2116 vs 0.3263). Fix: on reset, warm-start at the instantaneous source (`coldStart = false`
path: `uRetention = 0`) by default, or make it a runtime option. Clamp dt > 2 s to 2 s instead of resetting. Keep
cold start only behind an explicit "reference parity" flag.

## Ranked defects

1. **Reset wipes foam / zero while paused** (above). Functional bug for engine use (cuts, photo mode, load hitches).
2. **Pass graph: `ocean.foam` has no declared reader.** The foam pass writes `['ocean.foam']`. The water pass
   (refraction.ts:297) reads only `scene.color, scene.depth, ocean.displacement`, not `ocean.foam` (nor
   `ocean.normal`). pass-types.ts says a pass whose writes nobody reads may be culled by the engine graph. The
   result would be frozen or empty foam in the engine. The water pass desc must list `ocean.foam` and `ocean.normal`.
   Without refraction the material has no pass at all, so the scene-mesh path needs a documented reader too.
3. **Asset hook not plumbed.** `createOceanFoam` has `assets` (resolve / preloaded texture), but `createOceanGpu`
   (index.ts:179) does not forward any `assets` option. Through the public factory the pattern always loads from
   the hard default `resolveAsset`. Spec: assets go through a loader hook.
4. **Dead API: `foam.uniforms.fineInSource`.** It is declared, documented as "1 = sum the 4th (0.84 m) cascade into
   the source", and set by the demo's `?debug=foamfine`, but the kernel never reads it (the loop runs over
   `fftSampling.cascades` only). Remove it from the interface and the demo, or wire it.
5. **Frozen tuning constants.** `FOAM_PARAMS` is `Object.freeze`d, and `coldStart` / `storageLoss` / `tau` are not
   options. The engine cannot tune or disable the reference-emulation hacks without editing source.
6. **Stale integration docs in the foam.ts header.** "MATERIAL INTEGRATION" still says foam is "GREY, unlit by sun /
   sky colour". The material (round 2, in flight) now lights foam with SKY_LIGHT (sky irradiance x up-cosine + sun N.L,
   albedo 0.8, material.ts:289-294). Lighting state: fixed in the material, doc not updated.
7. **Demo foam debug views are broken right now.** `?debug=foam|foamraw|foamcov` give a flat grey frame plus
   `THREE.TSL: TypeError: Cannot read properties of undefined (reading 'build')`. The debug material copies
   `gpu.mesh.material.positionNode`, which is now undefined after the concurrent index/lod changes. This is
   demo-only, but it is a fragile coupling, and it breaks the builder's own tools/foam-check.mjs (its stats still
   run because they read the buffer).
8. **storageLoss = 30 ulp16/s** is a calibration to a hardware quirk: the reference's RGBA16F writes round toward
   zero on the builder's NVIDIA GPU, and other GPUs round to nearest. As implemented it behaves correctly: it is
   linear in dt (frame-rate independent, verified), deterministic, and a step function of h (the rate doubles at
   each power of two, which does no harm). Treat it as an artistic -3 % bias. Rename it (for example `historyBias`)
   and document it as such, not as "physics". Note: the reference's own loss is per frame, so at 144 Hz the
   reference itself would lose 2.4x more. The port pins the 60 Hz behaviour, which is the right choice.
9. **detailGain on the low tier = 2.51x on the c2 slope and compression.** This is an empirical exponent (0.86)
   fitted on one storm camera. The statistics hold (low 0.3173 vs high 0.3203, -0.9 %), but it amplifies the
   64-grid c2 features 2.5x, which affects the spatial character of low-tier whitecaps. Acceptable for now; it needs
   a calmer-preset check.
10. **The CPU `detailGainFor` runs a 65 k-iteration sin loop synchronously inside `pass.execute`** on every slot
    regeneration (preset change), which can cause a main-thread hitch on that frame. Move it to the existing FFT
    worker job (fft.ts already posts a 'dropped' job for the same slot).
11. LOD internals: `lod.state.footU/footV` was replaced by `lod.getFoot(foot)` in the working tree during this
    review (OK if `getFoot` is allocation-free, which it appears to be).

Notes (look, not fail reasons): whitecaps are structurally right (same patches, same streak orientation, 1.9x
stretch along the wind, lace scale matches at 05). They are much darker and greyer than the reference
(C4-storm / 05 vs the reference shots) because of the new SKY_LIGHT foam lighting in the material. The far field
beyond the 200 m window has only surface lace, same as the reference, and haze hides the boundary at C4. No drift,
matching the reference (`uFoamOffset` is unused there).

## Evidence (verified myself)

- `npm test`: 66/66 pass.
- **Determinism**: two page loads, `?shot` C4 storm, history buffer hash identical (849344588), mean 0.32029.
- **Frame-rate independence** (static source, cold start, stepped manually; scratch/foam-r1/fps-check.mjs):
  mean at 1.5 s: 20 Hz 0.21149 / 30 Hz 0.21153 / 60 Hz 0.21156 / 144 Hz 0.21159 / 480 Hz 0.21160. Equilibrium at 12 s:
  0.32618 / 0.32624 / 0.32630 / 0.32633 / 0.32635. Per-texel mean |diff| vs 480 Hz <= 1.7e-4.
- **Re-projection**: paused, camera +3.3 / -2.1 m, anchor shift (8, -6) texels, max |diff| 6e-8 over 1.03 M texels (exact).
- **Snap turn** at C4 (60 m altitude): anchor moved 128 m, no reset, mean continuous (0.3209 -> 0.3269).
- **Tier switch** while paused, high -> low -> high: mean 0.32686 -> 0.32677 -> 0.32677 (resampled, continuous).
- **Tier stats** (foam-check, storm C4 t = 30): low 512^2 mean 0.3173, high 1024^2 mean 0.3203. detailGain 2.509 / 1.
- **Origin**: camera at world (1e7, 60, 1e7+8); origin (1e7, 0, 1e7) vs (1e7+1000, 0, 1e7-700): full frame
  **bit-identical** (meanAbsDiff 0). At 1e7 m the lace looks the same as at the origin (W7-close.png). Origin 1e5 with
  the camera at 0: 0.26 % of pixels > 8 levels (camera-relative raster only).
- **Paused teleport** 1000 m: history mean -> 0.0000 (defect 1).
- **GPU**: foam dispatch 0.080 ms (high, 1024^2), 0.027 ms (low, 512^2), RTX 2070. Memory 21.3 MB high (16 MB history +
  5.3 MB pattern), 9.3 MB low.
- **Spec grep (foam.ts)**: TSL only, no GLSL / ShaderMaterial / WebGLRenderer. `performance.now` only times the
  async compile (not sim). No `new` or literals in `execute` / `updateWind` / `updatePhases` / `updateDetailGain`
  (`ocean.getState()` returns the cached state, `blendAt` is allocation-free). No world coordinates on the GPU
  (anchor - foot, fract64 phases, anchor mod P). Compute pass desc: stage compute, reads ocean.normal, writes ocean.foam.
- Wind direction ?wind=15,90: lace and streaks re-orient (C4-wind90.png), no artefacts.

## Screenshots (docs/port/reviews/scratch/foam-r1/)
- C4-storm.png vs docs/port/reference-shots/C4-storm.png: structure matches, foam much dimmer (material lighting).
- W7-o0.png / W7-o1.png: 1e7 m, two origins, bit-identical.
- W7-close.png: close-up lace at 1e7 m, clean.
- W7-raw-o0.png: the broken foam debug view (flat grey, TSL build error).
