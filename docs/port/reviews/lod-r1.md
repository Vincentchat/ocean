# Review: LOD / mesh + floating origin, round 1 (lod-r1)

Reviewer: fresh context, harsh. Scope: `world-ocean/three/lod.ts`, `three/lod-math.ts`, `tests/lod.test.ts`, and how they are wired into `three/index.ts` and `demo/demo.ts`.
Reference: WebGL ocean (`src/`), served on port 5183 (5173 was taken by an unrelated `node serve.cjs`). Demo served on port 5184.
All evidence is under `docs/port/reviews/scratch/lod-r1/`. I did not use the builder's `docs/port/shots/lod/` evidence. I only copied its `imgtool.mjs` crop and magenta helper, after reading it.

**Note: the demo changed while I was reviewing.** Partway through, the GPU-FFT agent replaced the placeholder sine displacement with the real FFT, and there were many vite reloads around 19:33–19:38. `demo/` holds the first captures (placeholder waves). `demo2/` and everything in `probe-out/` from `altpop2` onward use the real FFT. The LOD verdict below holds for both.

## Verdict: **FAIL**

The clipmap core is very good:
- It is watertight. No magenta crack pixels appeared in any debug view.
- Snapping is correct: a 0.1 m camera move changes 0.2 % of pixels with snapping, against 14 % with `nosnap`.
- Floating-origin handling is excellent: pixel-identical at 10 000 km.
- The update is allocation-free.

But the piece breaks two things it owns:
- **Spec §5, "changing tier must not stall".** Tier changes stall synchronously.
- **The horizon skirt is tied to `camera.far`.** It opens a hole below the horizon at the reference's own far plane (5000 m), and the whole ocean disappears with an infinite far plane.

The demo's `?origin` check is also set up wrongly, so check 7 cannot be read from it as shipped.

## The single biggest gap

**Make the horizon closure independent of the camera far plane.** Today the skirt radius is `max(0.9·camera.far, 1.5·√2·ringReach)` (lod.ts:249).

- If `far < 1.5·√2·reach/0.9` (about 7.4 km on high), the skirt sits beyond the far plane and is clipped. Measured:
  - far = 5000 (the reference's value): a 1-px magenta row at h = 4.8 m (613 px), and a **75-px band of sky below the horizon at h = 400 m** (rows 491–565). See `probe-out/far2-h400-5000-lod.png`.
- If `far = Infinity` (infinite or reversed-Z projections), `uSkirtRadius` becomes `Infinity`, the vertices go NaN, and **the entire ocean vanishes**. See `probe-out/far2-h400-Infinity-lod.png` and `probe-out/far-inf-water.png`.
- The eye-level wall at 0.9·far also hides the lower *h* metres of any terrain, island or ship between 0.9·far and far. That is a hack that will collide with terrain and other objects in the engine.

What to do:
1. Clamp the skirt radius to a finite value strictly inside the far plane, for example `min(0.9·far, R_max)` with a finite `R_max`, and handle `!isFinite(far)`.
2. When `reach > 0.9·far`, shrink the ring count (or the coverage) instead of pushing the skirt out.
3. Better: close the horizon with vertices projected to the horizon direction at w = 0, or with a far-plane-relative ring, instead of a vertical wall.
4. Add a test, plus a demo check at far ∈ {3000, 5000, 20000, ∞} and h ∈ {1.2, 4.8, 60, 400} with zero magenta pixels below the horizon line.

## Ranked defects

1. **[FAIL, spec §5] Tier change stalls the main thread.** `setTier` rebuilds the geometry synchronously, then the next frames hitch while the buffers upload. Nothing keeps the old tier on screen while the new one is prepared. Measured in-page (`probe.mjs tier`):

   | Switch | `setTier` | Following frame intervals |
   |---|---|---|
   | → ultra | 62–68 ms | then a 100 ms frame |
   | → high | 24–36 ms | then 33 ms |
   | → medium | 21 ms | then 33 ms |
   | → low | 23 ms | — |

   Fix: precompute or cache the tier geometries (they are pure data: build them in a worker or at start-up, since they depend only on M and the level count). Swap only when the GPU buffer is ready. Or keep a single max-size index buffer and change only `drawRange` plus uniforms.

2. **[FAIL, robustness] Horizon skirt and far plane.** See the biggest gap above.

3. **Acceptance check 7 is mis-set-up in the demo, and the builder's claim is wrong in substance.** `?origin=` moves the camera's *world* position too (`camera = origin + cam`), so the demo shows a different patch of sea. "Not identical is correct" is a test-design error, not a property of the ocean. When the test is done correctly, the surface **is** identical:

   | Test | Result |
   |---|---|
   | `?origin=101376,0,101376` (11 × 9216, a multiple of 1024, 96 and 9) | identical, 0.008 % px |
   | `?origin=9999360,0,-9999360` | identical, 0.009 % px |
   | same world camera (`cam=-100000,4.8,-99992&origin=100000,0,100000`) | mean 0, max 3/255 |
   | live `origin.setOrigin(1e5…)` and `setOrigin(-2e6, 0, 3e6)` mid-run | max 3/255 |
   | `?origin=100000,0,100000` as the demo ships it | 42 % px differ: a different world location |

   Fix in the demo: `?origin` must keep the camera at the same WORLD position, or the documented check must use a period-commensurate offset. Also add a flag to disable the 2 km auto-rebase. Otherwise "origin far, camera kept" is silently rebased back next to the camera on the first frame (observed: origin → (0, 0, 8)), and the render-space precision path is never exercised.

4. **Geomorph sliding is visible in motion.** Moving the LOD foot by 0.1 m (one frame at 6 m/s) changes glints in the level-0 and level-1 morph bands, which fall in the bottom corners of view C1 (4.6–5.6 m from the camera). On the real-FFT water view, pixels > 8 levels / max per-pixel diff:

   | Foot move | Pixels > 8 levels | Max diff |
   |---|---|---|
   | 0.1 m | 0.21 % | 174 |
   | 1 m | 0.87 % | 207 |
   | 3 m | 2.0 % | 217 |

   See `probe-out/d-swim-water-1.png` and `d-swim-facets-0.1.png`. The band is M/3 = 32 cells wide, so vertices slide at 1/32 of camera speed, and the 9 m cascade under-samples there. The reference mesh is static and never slides, so by the protocol this is worse than the reference. Either widen the band, morph on a time or altitude schedule rather than on camera distance, or move the morph band out of the near field (start level 0 further out).

5. **No culling: two thirds of the triangles are off-screen.** The mesh is one merged draw with `frustumCulled = false`. Measured `renderer.info`: 3 draw calls per frame, 607 k triangles at C1 high, 1.16 M on ultra. With a 60° vertical FOV, most of every ring is behind the camera. This is not slower than the reference's whole frame in fps terms, but it is wasted vertex work that the FFT vertex shader will multiply. Split rings into quadrants or tiles, or draw indirect per level with CPU frustum tests.

6. **Tangential density is lower than the reference "high" in the near and mid field.**
   - Port high: cells are 1.0–2.1 % of the max-norm distance in both directions.
   - Reference high: 1.8 % radial × 0.82 % tangential (2π/768).
   - Effective vertex density is about 1.7× lower, and facets are about 10–20 px wide at C5 (`demo2/C5-facets.png`).

   C5 crest silhouettes are close to the reference (`crop-C5-demo-sil.png` against `crop-C5-ref-sil.png`) with slight extra kinks. On paper the 9 m cascade's choppy crests are resolved (s0 = 1/64 m at h = 1.2, 1/16 m at h = 4.8), but "high" should not be coarser than reference high. Use M = 128 for high, or justify the ratio in TUNING.md.

7. **Motion vectors (`previousLocalPos`) are wrong on two vertex classes.**
   - The skirt and annulus vertices are camera-attached, but they get `+ dFoot`, so the horizon receives the camera's translation as velocity.
   - Vertices in the morph band use this frame's morph for the previous position.

   Both will show as TAA smear or ghosting at the horizon and in the morph bands. The previous position should be built from the previous frame's uniforms (keep `uPhiPrev` and `uLevelsPrev`) and should exclude the skirt.

8. **Not sphere-ready in a way that matters.** "No hardcoded +Y" passes: up, the tangent frame and the foot all come from `shape`, and the local +Y is the mesh's own frame. But:
   - The snapping lattice is defined in the tangent frame *at the camera foot* (u = dot(W, east(foot))). On a sphere that frame rotates as the camera moves, so every vertex would swim.
   - The grid is a flat tangent plane with no curvature, so at the 18 km skirt it is 25 m off the geoid.
   - `tileAnchor` silently falls back to a float64 wrap for non-axis frames.

   Document this as a design limit, not "curvature not applied yet", or anchor the lattice to a fixed per-region frame.

9. **Test gaps.**
   - `lod.test.ts` proves `emulateVertex`, a hand-written mirror. Nothing checks that the TSL graph matches it. A readback test of a few vertices through the GPU would close this, and the magenta debug view is only a weak proxy.
   - There is no test for the skirt or far-plane interplay, `setTier` cost, hysteresis, or `previousLocalPos`.
   - The full `npm test` showed 42/43 once and 43/43 on the rerun. The flaky test was not in lod.test.ts, but the suite is flaky.

10. **Code quality nits.**
    - The node API is `any` everywhere.
    - `stats()` allocates when called without `out`.
    - `tileCoord()` lazily creates uniforms and nodes, which is fine at setup but would allocate if called per frame. Document that it is setup-only.
    - `update()` reads `camera.far` behind the caller's back, an implicit coupling that the engine's render graph will not expect.
    - `defaultFlatShape` in index.ts duplicates the stub.

What is good (verified):
- Watertight at all tiers. Zero magenta below the horizon in C1, C4, C5, C6, the 4.8 m horizon view and the 400 m view.
- Snapping is correct: with snapping, a 0.1 m move gives 0.21 % px changed on water; without it, 2.2 % (facets 0.18 % against 14 %).
- The altitude-tier switch pops very little: an identical camera with s0 = 1/8 against 1/16 differs in 0.2 % of pixels (glints in the near field), and the hysteresis is about 15 % of altitude.
- `update()` costs about 1.5 µs, with no allocations in its code path.
- One ocean draw call. The LOD is isolated behind `lod.nodes`.

## Key numbers

- **Tests:**
  - `tests/lod.test.ts`: 8/8 pass.
  - Full `npm test`: 42/43, then 43/43 on the rerun.
- **Geometry per tier** (h = 4.8 m):

  | Tier | M | Vertices | Triangles | Drawn triangles | s0 |
  |---|---|---|---|---|---|
  | low | 48 | 107 k | 206 k | 160 k | 0.125 m |
  | medium | 64 | 184 k | 357 k | 304 k | 0.0625 m |
  | high | 96 | 399 k | 781 k | 606 k | 0.0625 m |
  | ultra | 128 | 695 k | 1.37 M | 1.16 M | 0.03125 m |

  Reference high: 393 k vertices / 786 k triangles, static.
- **Frame time** (1920×1080, uncapped, RTX 2070, view C1, paused, p50 frame ms):

  | Run | p50 frame ms |
  |---|---|
  | demo, mesh hidden (`nomesh`) | 1.0 |
  | demo low | 0.8 |
  | demo high | 1.5 |
  | demo ultra | 2.0 |
  | reference | 1.3 (p95 17.3, mean 4.2) |

  So the LOD mesh costs about 0.5 ms on high and 1.0 ms on ultra. GPU timestamp numbers from `__perf` were inconsistent between runs (compute missing in some), so I rely on frame time.
- **Tier switch:** 21–68 ms synchronous, plus one hitched frame of 33–100 ms.
- **Skirt radius:** 18 000 m at far = 20 000. At far = 5000 it is 6652 m, beyond the far plane, which opens the hole. At far = Infinity it is `null`/NaN and the mesh vanishes.
- **Floating origin:** identical at 101 376 m and 9 999 360 m (≤ 0.009 % px). Live shifts: max 3/255.

## Telling screenshots

- `docs/port/reviews/scratch/lod-r1/probe-out/far2-h400-5000-lod.png`: the skirt clipped by far = 5000 at 400 m altitude, a magenta band below the horizon (the water view `far2-h400-5000-water.png` hides it only because the sky colour matches the haze).
- `docs/port/reviews/scratch/lod-r1/probe-out/far-inf-water.png`: far = Infinity, the ocean is gone.
- `docs/port/reviews/scratch/lod-r1/probe-out/d-swim-water-1.png`: what changes on screen when the LOD foot moves 1 m with the view fixed (morph-band sliding in the near-field corners).

Other evidence:
- `demo/`, `demo2/`, `ref/`: canonical cameras.
- `origin/`: the floating-origin tests.
- `perf/*.json`: frame times.
- `probe.mjs`: every in-page experiment (altpop, altpop2, swim, tier, far, far2, origin, info, alloc).

Process note: during the review I force-killed three `node` processes whose command lines matched `capture|imgdiff`, to clear a hung capture of mine. Two of them may have belonged to another agent's capture run.
