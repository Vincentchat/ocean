# Review: LOD / mesh + floating origin, round 2 (lod-r2)

Reviewer: fresh context, harsh. Scope: `world-ocean/three/lod.ts`, `three/lod-math.ts`, `tests/lod.test.ts`, `?origin` handling in `demo/demo.ts`.

Setup:
- Demo on port 5195, using the builder's `serve-nohmr.mjs` (no HMR, because other agents were editing files). Reference on port 5196.
- All my evidence is under `docs/port/reviews/scratch/lod-r2/`. I did not use any image or number from `docs/port/shots/lod-r2/`. I reused only its `serve-nohmr.mjs`, and `imgtool.mjs` (crop only, after reading it). `sweep.mjs` is my own extension of `farsweep.mjs`.
- The GPU and CPU were shared with other agents throughout. Frame intervals in uncapped mode were about 55 ms in every demo variant, including `nomesh`. All perf statements below are therefore relative comparisons within one session.

## Verdict: **FAIL** (much closer than round 1)

Round-1 demands, as verified:

| Demand | Status |
|---|---|
| Skirt independent of `camera.far`, incl. ∞ | **Mostly done.** 0 gap pixels in 128 + 70 sweep shots. But the far-plane-mapped closure now **hides terrain at ≥ 0.99 far** in the demo's default pipeline (see the biggest gap). |
| Non-stalling tier change | **Improved, not done.** The build is sliced at 2 ms per frame, but the swap frame still costs 5–14.5 ms of main thread on high/ultra. |
| `?origin` keeps the camera's world position | **Done** in `demo.ts`, and auto-rebase is off with an explicit `?origin`. This now exposes the fact that the mesh is **not drawn camera-relative**: the geometry shifts 3.75 % px at 1e5 and 74 % px at 1e7. |
| Motion vectors for the morph band and skirt | **Done** in `previousLocalPos` (smooth fields, no ring seams, horizon at zero). But nothing feeds it to three's velocity path. |
| Geomorph sliding | **Much reduced** (0.1 m move: 0.005 % px against 0.21 % in round 1), but not zero (F21). |

## The single biggest gap

**The horizon closure z-ties with distant terrain and hides it.**

The skirt sits at view depth 0.995–0.999 far. With the demo's standard (non-reversed) float depth (near 0.1, far 20 000), everything from about 0.985 far up to the far plane quantizes to the **same depth value** as the closure. The difference between 0.99 F and 0.995 F is 2.5e-8, below one float32 ulp at 1.0 (6e-8).

three's default `LessEqual` depth test then lets the later draw win the tie. With refraction on (the default), the water draws later, in the transparent stage, so it covers the terrain. Measured with an opaque red "island" at 0.99 F and 0.994 F (`occl.mjs`):
- **100 % of the island's pixels below eye level are replaced by water**: 888/888 px at h = 400 m, 6512/6512 px at h = 2000 m.
- At 0.985 F, at far = 5000, and with `?refraction=0`: 0 px hidden.

So the claim "hides nothing nearer than 0.995 far" is false in the shipped configuration. The band widens with far/near: at far = 1e5 the ties reach about 0.94 F.

Fix: the closure must lose every depth tie and must not depend on depth precision at all. Any one of these works:
- Draw the skirt as its own draw, first or as background, with a depth that is exactly the far value (clip z = w, or z = 0 for reversed-Z, through `vertexNode` / `depthNode`).
- Or give the water material `depthFunc = LessDepth` and keep the closure strictly behind the scene.

Add a regression check: an island at {0.97, 0.99, 0.994} F × far ∈ {5000, 20000, 1e5} × refraction on/off, with zero hidden pixels.

## Ranked defects

1. **[FAIL, robustness] Closure occludes terrain near far** (above). Evidence:
   - `occl-r1/occl-far20000-h2000-f0.99.png` and `occl-r1/crop-f099.png`: the island is cut flat at the horizon line.
   - `occl-r1/rows.json` holds the numbers. `occl-r0` (refraction off) is clean.

2. **[FAIL, spec §5 "must not stall"] The tier swap frame hitches.** `setTier` returns in ≤ 0.1 ms, and the build runs at 2 ms per update for 2–13 frames. But the atomic swap creates a new `BufferGeometry`, which the renderer uploads in that one frame: about 14 MB on high and 25 MB on ultra. Main-thread time inside `renderer.render` on the swap frame (`tier.mjs`, two runs each):

   | Swap to | Swap-frame render ms |
   |---|---|
   | ultra | 14.5, 9.5 |
   | high | 5.1–7.5 |
   | medium | 2.8–3.4 |
   | low | 2.0–3.3 |
   | baseline frame | 0.6–0.8 (max 2.3) |

   A 9–15 ms spike drops a frame at 60 Hz. Fix: allocate the buffers for the largest tier once and switch tiers through `drawRange` and uniforms only. Otherwise, pre-upload the new attributes in slices (`addUpdateRange`) before the swap.

3. **Not drawn camera-relative.** The mesh is placed at the foot in render space and drawn through three's default `mediumpModelViewMatrix`, which is `cameraViewMatrix × modelWorldMatrix` multiplied in float32 on the GPU.
   - With a correctly following floating origin, the geometry is **bit-identical at 9 999 360 m** (0 px differ, `geo-o0` against `geo-w1e7`).
   - With the demo's `?origin` as now shipped (camera kept, render coordinates −1e5 / −1e7), a geometry-only colour view differs by **3.75 % px at 1e5** (sub-pixel edge shifts) and **74 % px at 1e7** (the whole sea offset by about 0.5 m; `geo/geo-o1e7.png`, `geo/diffs/`).
   - Forcing `highpModelViewMatrix` (an in-page experiment) brings 1e7 down to **0.098 %**.
   - So acceptance check 7, as the demo now runs it, is not "identical". Live shifts within 2 km: 0.04 % px (fine).
   - Fix: use `highpModelViewMatrix` (`vertexNode`, or the material context) for the ocean, or document that the render-space camera must stay within a few km.

4. **Motion vectors are not reachable by three's velocity pipeline.**
   - `previousLocalPos` itself is right. Running water with a still camera gives a smooth field with no level seams and zero at the horizon (`vel/run-calm-still.png`, `vel/run-windy-still.png`). Strafing gives a uniform translation field with no seams (`vel/run-calm-strafe.png`).
   - But `index.ts` only exports `previousPositionNode`. three's `VelocityNode` uses `positionPrevious`, which is `positionGeometry`, and here that is the **packed grid data** `(gx, gy, level*4+kind)`. A TRAA or motion-blur MRT would get garbage on the ocean.
   - The velocity debug view is also mislabelled. It pairs `previousLocalPos` (last frame's local frame) with the *current* `modelWorldMatrix`, so it shows camera-translation motion, not "object motion only". The contract only holds with `previousModelWorldMatrix`.
   - This is integration-owned, but it blocks rule 10.

5. **No culling.** One merged draw with `frustumCulled = false`. Only **27 %** of the drawn grid triangles intersect the frustum: 164 k of 602 k at C1 high, 26 % at C5, 22–24 % at 60 m altitude (`cullfrac.ts`). This is wasted vertex work, and every vertex runs the multi-cascade FFT fetch. GPU render time (contended, same session): high 2.6 / 4.6 ms against `nomesh` 0.4 / 0.85 ms.

6. **Geomorph sliding (F21: "no vertex sliding").**

   | Foot move, view fixed | Water: px > 8 levels | Facets: px > 8 levels |
   |---|---|---|
   | 0.1 m | 0.005 % (round 1: 0.21 %) | 0.12 % |
   | 1 m | 0.58 % (round 1: 0.87 %) | 2.65 % |

   The changes sit in the morph bands in the near-field corners (`swim/diffs/diff_swim-wat-0__swim-wat-1.png`). The `nosnap` control gives 3.35 %. The static reference mesh gives 0 by construction. Much better than round 1, but still worse than the reference in motion.

7. **Closure geometry assumes the rendering camera is the LOD camera.** The depth solve uses `uCamHeight` and treats the camera as sitting at local (0, h, 0). A second camera rendering the same mesh gets a wrong closure depth: a reflection or cube probe, a split view, or a shadow camera (orthographic, with its own far). Clipmap rings share this limit, but the header claims "per rendering camera" and "other cameras/passes stay correct", which is only true for co-located cameras.

8. **Isolation leaks for a future lod-tree.** `foam.ts` reads `lod.state.footU`, `lod.state.footV` and `lod.state.hasPrev`, the clipmap's internal lattice state, not `lod.nodes`. `update()` still reads `camera.far` implicitly.

9. **Tests.**
   - `lod.test.ts` passes 10/10.
   - But `emulateVertex`'s skirt branch (radial `hypot`) no longer mirrors the shader's depth-solved closure.
   - There is no test of the closure against depth precision, of `previousLocalPos`, of render-space magnitude (mediump), or of the swap-frame upload cost. The allocation test covers only `updateLodState`, not `lod.update`.
   - Full `npm test`: 57/59. The two failures are core heap/perf tests (`zero allocation: tick…`, `heightAt >= 1 M calls/s`), not the LOD, and were under CPU contention.

10. **Near-field density.** The tangential density on high is still below the reference: C5 near-crest silhouettes show slight extra kinks (`canon/crop-C5-*-near.png`). This is minor and not a visible defect at normal viewing. Separately, the far swell crest at C5 storm is about 18 px taller than in the reference (`canon/crop-C5-*-crest.png`). Its cause is unclear: the FFT, or the displacement footprint (`cellSize` against the reference's `(r+0.25)·radialStep`). The FFT and LOD owners should check it.

11. **Nits.**
    - `stats()` still allocates without `out`.
    - The tier build allocates iterator results each slice (only during builds).
    - `cosView` is clamped at 0.2, so the depth solve is wrong for directions more than 78° off-axis. That is harmless today; it would matter at a horizontal FOV above 156°.

## What is good (verified)

- **Zero-gap claim holds.**
  - Medium: 128 shots (far 3000 / 5000 / 20 000 / ∞ × h 1.2 / 4.8 / 60 / 400 × pitch 0 / −2 / −10 / −25 × 2 yaws): **0** magenta pixels below the horizon (`sweep-medium/`).
  - High, extended to h 2000 / 10 000 m and FOV 100°: 72 shots, 0 real gaps. The two flagged shots are a 10 km camera with far = 5000 looking 45° down; there the whole sea is physically beyond the far plane.
  - Infinite far (hand-built infinite projection) works, and the closure clamps to 1e6 m.
- **Floating origin with a following origin is perfect:**
  - geometry bit-identical at 9 999 360 m;
  - all origin-dependent maths in float64 on the CPU;
  - `onShift` re-places the mesh in the same frame.
- **Water-view differences at 1e7 are not the LOD's.** With a proper origin, the water view still differs 47 % px from the view at 0 while the geometry is identical. So those differences come from the material or foam: world-periodic patterns that are not commensurate with 9216 m, or world-space terms. That is not the LOD.
- **Isolation.** The material and displacement talk to `lod.nodes` only (except foam, defect 8). Normal frames make no per-frame allocation in `update()`.

## Key numbers

| Item | Value |
|---|---|
| Tests | `lod.test.ts` 10/10; full suite 57/59 (2 core perf/heap failures under contention) |
| Gap sweep | medium 0/128 shots, high-extended 0 real gaps / 72 |
| Island occlusion, refraction on | far 20 000: 0 % hidden at ≤ 0.985 F, **100 %** of the below-eye band hidden at 0.99 / 0.994 F; far 5000: 0 |
| Swap-frame render ms | 14.5 / 9.5 (ultra), 5.1–7.5 (high), 2.8–3.4 (medium), 2.0–3.3 (low); baseline 0.6–0.8 |
| Build | 2 ms per frame for 2–13 frames |
| Geometry vs origin | following origin 1e7: 0 px; camera 1e5 from origin: 3.75 %; 1e7: 74 %; 1e7 with highp: 0.098 %; live shifts ≤ 2 km: 0.04 % |
| Swim (water px > 8 levels) | 0.1 m: 0.005 %; 1 m: 0.58 %; nosnap control 3.35 % |
| In-frustum grid triangles | 27 % at C1 (164 k / 602 k on high) |
| Geometry at h = 4.8 | high: 399 k verts, 781 k tris, 606 k drawn, 10 levels, ring reach 3136 m |

## Telling screenshots

- `docs/port/reviews/scratch/lod-r2/occl-r1/crop-f099.png`: an island at 0.99 far (far 20 km, camera 2 km up) is cut flat at the horizon. Everything below eye level is hidden by the closure.
- `docs/port/reviews/scratch/lod-r2/geo/diffs/diff_geo-o0__geo-o1e7.png`: geometry-only view with the camera 1e7 from the origin (demo `?origin`). The whole sea is shifted by the mediump model-view matrix.
- `docs/port/reviews/scratch/lod-r2/swim/diffs/diff_swim-wat-0__swim-wat-1.png`: the remaining geomorph sliding after a 1 m foot move (near-field corners).

Scripts: `sweep.mjs`, `occl.mjs`, `tier.mjs`, `shots.mjs` with `list-*.json` and `mklist-geo.mjs` (geometry-only material), `mdiff.mjs`, `velrun.mjs`, `cullfrac.ts`. Perf JSON is in `perf/`.
