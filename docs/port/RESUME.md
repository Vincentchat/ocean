# Resume here: WebGPU ocean port, state at end of session 1 (2026-10-23)

Read this first, then `docs/port/BRIEF.md` (rules for every agent) and `docs/port/REVIEW-PROTOCOL.md`.
The spec is `docs/Ocean-WebGPU-Port-Rules.md`. Live progress page: https://claude.ai/artifact/V5B8C16c5bT4Sfn4yPoKoJ
(rebuild with `node docs/port/progress/build.mjs` from `docs/port/STATUS.json`, then republish the artifact).

## Ground rules agreed with the user
- **Nobody commits, stages or resets git.** The user commits. Changes stay in the working tree.
- **Goal now: working + spec-compliant, not pixel-perfect.** Look-dev will be redone in the engine (its own HDR,
  TAA, exposure, sky) against new references. Reviewers fail pieces only for spec violations, broken
  functionality, gross artefacts or perf regressions; visual differences vs the WebGL reference are notes.
- Process per piece: builder → fresh harsh reviewer (actual renders + code) → if FAIL, the single biggest gap goes
  back to the builder → repeat. Between waves one fresh integration agent tests the whole module.
- Agents: Opus 5.5, medium effort. Each uses its own dev-server port; Vite HMR from concurrent edits disrupts
  captures (use `docs/port/shots/lod-r2/serve-nohmr.mjs`). The GPU is shared, so perf numbers are noisy:
  take final perf numbers in a quiet run.

## Plan for the next session
1. **Finish these passes** (below, in order of priority), each with a fresh review:
   1. Core round 3 (blockers).
   2. Foam round 3.
   3. Parity decisions + CPU inversion fix.
   4. Re-review the FFT (round 3 not reviewed), the LOD (round 3 not reviewed), the material (round 2 not
      reviewed), refraction (never reviewed), and parity.
2. **Integration wave:** one fresh agent runs every §7 acceptance check end to end and fills `docs/port/STATUS.json`
   checks.
3. **Docs** (spec §8, §9): `world-ocean/README.md` (agent quick-start first), `world-ocean/TUNING.md` (every
   param, presets table, per-tier cost table at 1080p measured on a quiet GPU), `world-ocean/HANDOFF.md`, asset list
   with licences. None of these exist yet.
4. Then the user ports the module to the engine.

## Piece status

| Piece | Files (world-ocean/) | State | Latest review |
|---|---|---|---|
| Infra / harness | package.json, module.json, stubs/, demo/, tools/ | done | — |
| Core (spectrum, CPU FFT, WATER_QUERY, presets) | index.ts, types.ts, spectrum.ts, cpu-fft.ts, ocean-worker.ts, presets/ | **round 2 FAIL** | reviews/core-r2.md |
| GPU FFT (TSL compute) | three/fft.ts, fft-tiers.ts, fft-worker.ts | round 3 built, **not reviewed** | reviews/fft-r1.md (r1) |
| LOD clipmap + floating origin | three/lod.ts, lod-math.ts | round 3 built + wired, **not reviewed** | reviews/lod-r2.md |
| Water material | three/material.ts, index.ts | round 2 built, **not reviewed** | reviews/material-r1.md |
| Refraction + demo seabed | three/refraction.ts, demo/seabed.ts, stubs/impl.ts | built, **never reviewed** | — |
| Foam | three/foam.ts | **round 2 FAIL (narrow)** | reviews/foam-r1.md |
| Parity probe, ?parity, ?buoy | three/parity.ts, parity-math.ts, demo/parity-demo.ts, tools/parity.mjs | built, **not reviewed** | — |
| Docs (README/TUNING/HANDOFF) | — | not started | — |

All review reports are in `docs/port/reviews/`; reviewer scratch scripts are in `docs/port/reviews/scratch/` (gitignored).
`npm test` in `world-ocean/`: 66/66 passed at the end of the session on a quiet run.

## Open work, by piece

### Core: round 3 (blockers first; see reviews/core-r2.md)
- BLOCKER: a late worker can corrupt heights by up to 7 m. A snapshot request passes an index into the shared source
  table instead of the source, so a job starting more than ~1.6 s late computes the wrong spectrum. Pass the
  source by value.
- BLOCKER: a worker crash spins the main thread forever (`ensure()`, `stepRegen`, `applyImmediate`; no `error`
  listener). Bound every wait, surface errors, fall back to the main thread.
- Mandate: tick + 100 heightAt + 10 velocityAt at 60 Hz must reach **p99 < 4 ms with workers OFF**. It is currently
  6.8–8.7 ms with workers and 11–15 ms without. Cause: sparse queries materialise 16×16 blocks. Evaluate
  Hermite-in-time per tap from cached snapshots (or 4×4 blocks), with prefill only as an optional accelerator.
- Workers never run in the browser: Vite inlines the .ts worker as a data: URL, and the demo has no COOP/COEP
  headers. Either make the browser path work (vite headers + worker bundling) or design for main-thread-only.
- Query time clamp breaks "height at time `time`": 1 s ahead is off by up to 8 m in storm. Widen the window to what
  is cached per cascade, warn on clamp, re-anchor after time jumps, consider `predictHeightAt`.
- Throughput: heightAt 0.46–0.53 M/s (spec aim ≥ 1 M/s), velocityAt 0.27–0.31 M/s. Cap the cubic refinement, keep
  the 9 m cascade bilinear / out of the solve.
- **CPU inversion bug** (found by the parity builder): heightAt does not converge near steep crests. In storm the
  returned rest point maps 0.64 m from p (height 4.21 vs true 4.82 m). Adopt the GPU probe's safeguarded Newton
  (backtracking + keep-best). `invertLocal` reports its residual off by SURFACE_PERIOD when wrapped.
- Fold-over sheet jumps: up to 2.5 m in one tick in storm (125 ppm). Prefer temporal continuity (seed from the
  last solution).
- Minor: tick after dispose() throws a TypeError, 120 s realign after a 180° wind flip, stale doc lines,
  0.63 s init.

### Foam: round 3 (reviews/foam-r1.md)
- Main gap: any history reset (first frame, anchor jump > 200 m, backwards time, step > 2 s) restarts from 0
  and **never recovers while paused** (dt = 0). Warm-start from the instantaneous source by default, clamp big
  steps to 2 s, keep the cold start only behind an explicit reference-parity option.
- No pass declares it reads `ocean.foam` (the water pass in refraction.ts:297 declares only scene.color, scene.depth,
  ocean.displacement), so a render graph may cull the foam pass. Declare the reads, including when refraction is off.
- `createOceanGpu` does not pass `assets` through to `createOceanFoam` (index.ts ~179), so the asset hook is unreachable.
- Dead `uniforms.fineInSource`; foam debug views broken (`positionNode` undefined after the LOD r3 wiring);
  frozen FOAM_PARAMS (`storageLoss`, `coldStart`) should be options; the header still says "unlit"; move
  `detailGainFor` (65k loop on the main thread at preset change) into the FFT worker.

### Parity: decisions needed from the user + fixes
Results (tools/parity.mjs, evidence docs/port/shots/parity/): GPU field vs CPU p99 < 1 % Hs on every tier and at
100 km. The strict max < 2 % Hs gate passes 8/32 (windy only). Causes:
1. The CPU inversion bug above (all storm failures).
2. The low tier draws c2 at 64² but the CPU keeps the full c2 (calm/low 3.5 % Hs). Options: accept for low, or let the
   query drop the same c2 detail.
3. **The drawn surface is view-dependent by design**: vertex footprint mips band-limit it beyond ~10–25 m, and the
   reference's far fade (200–1200 m) flattens waves (up to 2.4 m at 450 m in storm). No view-independent
   WATER_QUERY can match that. **Decide:** state the < 2 % target for the full-resolution field and near the
   camera, or change the vertex filtering/fade. Recommendation: field-level target plus a near-camera drawn
   target, documented in README.
4. The fade constants (200/1200 m) and the cubic default are duplicated in parity.ts and index.ts; centralise them.

### FFT: needs review of round 3
- Every tier draws the core's master spectrum bit-identically; low drops only c2's top (64² sub-block), and ultra adds an
  opt-in gravity-capillary band (off by default). high = medium in FFT content. Low ≈ 75 % of medium cost
  (c0/c1 can't shrink without changing gameplay heights). Catmull-Rom displacement near the camera (core query
  matches with cubic).
- Open: should "high" differ from "medium" via LOD/foam/material rather than the FFT? Tiers today: LOD density
  differs, foam history 512² on low.

### LOD: needs review of round 3
- Separate far-plane horizon closure (`lod.enableClosure`, 0/36 hidden-island cases), double-buffered tier swap
  (ultra swap frame 2.2 ms, ~50 MB buffer at maxTier ultra), float64 `clipPosition` (bit-identical at 1e5/1e7),
  per-sector frustum culling (602 k → 239 k triangles). Known limits: geomorph sliding (minor), no sphere support.

### Material: needs review of round 2
- Bit-identical shading at origin 1e5/1e7; foam/rim lit from SKY_LIGHT; no double sun dimming; motion vectors via
  `setMaterialPreviousPosition` (three's VelocityNode/positionPrevious); `?debug=rough`, `?debug=footprint`.
- Look items for the engine: distant flicker is 1.1–1.6× the reference without a LOD bias; demo seabed ~1.5× too
  bright over sand; fogDensity is read as 1/haze distance and fogColor is ignored (pass `haze: false` when the
  engine's atmosphere fogs).

### Refraction: never reviewed
- Transparent-stage `ocean.water` pass reading scene.color/scene.depth; open water is bit-identical with refraction on/off;
  +0.7 ms under 4× MSAA. TSL `select()` emits if/else, which puts derivatives in non-uniform control flow: use
  masks + mix in fragment code.

## Spec acceptance checks (§7): where each stands
1. WebGPU / r186 / no GLSL: believed OK; grep in the integration wave.
2. Core in Node: OK (`npm test`; the literal `node --test tests/` fails on Node 22.14; works on the glob form).
3. Determinism: OK (core review r2).
4. Presets Hs: OK (calm 0.54, breeze 1.46, windy 4.04 = reference look, storm 9.0 m; ×1.58 reference calibration documented).
5. Wind coupling: OK (Hs ≥ 96.6 % through 90–180° turns); realign after a 180° flip is slow.
6. CPU = GPU: **partial**; see the parity decisions above.
7. Far from origin: geometry and shading bit-identical at 1e5/1e7; parity same at 100 km.
8. Up not hardcoded: OK (rotated-frame test).
9. Motion vectors: wired and plausible; needs a reviewer pass.
10. No per-frame allocation: tick ≈ 0 B; heightAt boxes 16 B per call when the result is used (contract cost;
    heightsAt batch API is allocation-free but not on WaterQuery). Decide/document.
11. Cost table in TUNING.md: **not written**.
12. Licences: LICENSE + THIRD_PARTY_NOTICES.txt in world-ocean/; whale (CC BY 4.0) is not used by the demo.

## Useful commands (from world-ocean/)
- Tests: `npm test`
- Demo: `npm run demo` (port 5180), WebGL reference: `node tools/serve.mjs --reference --port 5173` (5173 may be
  taken by an unrelated server; use another port)
- Capture: `node tools/capture.mjs --url "http://127.0.0.1:5180/demo/?shot&t=10&scene=storm&cam=0,4.8,8,0,-16.04" --out x.png --wait-ready`
- Batch vs reference: `node tools/capture.mjs --batch tools/shots.json --target demo --base http://127.0.0.1:5180/demo/ --outdir ../docs/port/shots`
- Perf: `node tools/perf.mjs --url "<url>" --frames 600`
- FFT GPU-vs-CPU check: `node tools/fft-check.mjs`; tier switch check: `node tools/fft-tier-switch.mjs`
- Parity matrix: `node tools/parity.mjs`; foam stats: `node tools/foam-check.mjs`
- Demo flags: `?scene=golden|azure|storm`, `?preset=`, `?tier=`, `?cam=x,y,z,yaw,pitch`, `?t=`, `?shot`, `?origin=x,y,z`,
  `?parity`, `?buoy`, `?seabed=0`, `?refraction=0`, `?msaa=0`, `?debug=lod|facets|velocity|rough|footprint|foam|foamcov|fftbilinear|fftcapillary`
- Note: the reference page has a CSS overlay (style.css `body:after`) baked into the reference shots; apply it before
  brightness comparisons.
