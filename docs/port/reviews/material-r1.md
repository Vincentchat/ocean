# Water material (open-water shading), review round 1

**Verdict: FAIL** (close: colour and brightness parity is excellent, but distant specular shimmer is measurably worse than the WebGL reference, and two unit and contract choices would not hold up in the engine).

Scope: `world-ocean/three/material.ts`, its wiring in `three/index.ts`, unit handling in `demo/photo-sky.ts`, and the environment sync in `demo/demo.ts`. Out of scope: foam history, refraction, seabed and contact rims. The judgement is based on the deep-water regions.

Reviewer environment: demo on 5191, reference on 5192, headless Edge, RTX 2070, 1920x1080, high tier unless stated.

**Caveat: the code changed under me.** Other agents edited the code during the review: `material.ts`, `index.ts` and `demo.ts` at about 20:22, and `fft.ts`, `spectrum.ts` and `foam.ts` until at least 20:35. The FFT broke twice (TSL "Invalid parameter for the type uint", fft.ts:254). From about 20:30 the FFT also produced a visibly different, choppier wave field than the reference.
- **Comparison numbers** use only the batch captured at 20:10–20:20. In that batch the demo waves match the reference wave for wave.
- **Later captures** (tiers, the 100 km test, velocity) are internally consistent but cannot be compared with the reference.

## How it was checked

- **Tests:** `npm test` gives 43/45 passing. The 2 failures are core tests, out of scope: loop period 8192 s (`-0.33602 vs -0.33705`), and `heightAt` at 937 k calls/s against the 1 M target.
- **Shots:** all 27 canonical and C1–C7 x 3-scene shots are in `docs/port/shots/material-r1/`. Reference shots are in `docs/port/reference-shots/`.
- **Scratch:** scripts and evidence are in `docs/port/reviews/scratch/material-r1/`:
  - `regions.mjs`: region means.
  - `temporal.mjs`: frame-to-frame metrics.
  - `hf.mjs`: Laplacian (high-frequency) energy.
  - `crop.mjs`: zoomed crops.
  - `motion/`: frame triples and second-difference images.

## Evidence

### 1. Colour and brightness per region (sRGB 0..255, luma)

| shot | region | demo | ref | delta luma |
|---|---|---|---|---|
| 02-grazing-sun | far / mid / glitter / right deep | 143.5 / 135.3 / 121.9 / 102.1 | 144.0 / 137.1 / 121.7 / 102.3 | -0.5 / -1.8 / +0.2 / -0.2 |
| 02 | horizon band | 165.3 | 166.7 | -1.4 |
| 04-horizon | far / mid / midR | 121.5 / 95.9 / 83.1 | 122.9 / 97.9 / 83.1 | -1.4 / -2.0 / 0 |
| 06-sunset | far / mid / right | 149.0 / 131.0 / 107.0 | 149.5 / 131.8 / 101.7 | -0.5 / -0.8 / **+5.3** |
| 06 | glitter column | 172.9 (sd 43.3) | 166.0 (sd 40.6) | **+6.9** |
| C3-azure | far / mid / deep band | 113.2 / 98.2 / 62.7 | 114.5 / 99.7 / 61.5 | -1.3 / -1.5 / +1.2 |
| C3-golden | far / mid / deep band | 102.4 / 86.1 / 59.2 | 102.7 / 86.6 / 57.7 | ~0 / ~0 / +1.5 |
| C5-golden | far / mid / glitter / right | 146.7 / 121.6 / 128.3 / 73.6 | 146.6 / 122.7 / 121.8 / 69.7 | 0 / -1.1 / **+6.5** / **+3.9** |
| C5-azure | far / mid | 140.0 / 114.9 | 146.5 / 121.0 | -6.5 / -6.1 (the reference has foam smears here: out of scope) |
| C2-azure | sun path / mid | 126.5 / 126.6 | 131.8 / 135.9 | -5 / -9 (the reference has foam and seabed) |

Open deep water is within 0–2 levels almost everywhere, and the horizon band within 1.5 levels: an excellent port of body colour, Fresnel, reflection and haze.

The one systematic colour error is that the **sun glitter is too hot and too contrasty in low-sun scenes**: +6.5 to +7 luma and a higher standard deviation in the 06 and C5-golden glitter, and +4 to +5 on the side facing away from the glitter.

The storm scenes cannot be judged: the reference is dominated by foam. The demo's storm water (C2-storm far 82 vs 111) is darker only because the foam history is missing.

There are no black pixels in any of the 27 shots, and no NaNs.

### 2. Distance filtering: spatial high-frequency energy (mean |Laplacian| of luma, demo / ref)

| shot | h..h+2 % | h+2..5 % | h+5..12 % | h+12..25 % |
|---|---|---|---|---|
| 02 | 1.06 | 1.04 | 1.04 | 0.96 |
| 06-sunset | 1.05 | 1.09 | 1.09 | 1.12 |
| C3-golden | 1.09 | 1.12 | 1.12 | 1.13 |
| C2-golden | 1.08 | 1.04 | 1.11 | 1.07 |
| C2-azure | 1.07 | 1.10 | 1.11 | 0.95 |
| 04, C3-azure | 0.94–1.04 | | | |

Distant normals and glitter are 4–13 % "sharper" than the reference in every sun-facing scene. The reference filters them more.

### 3. Temporal shimmer (paused frames at t, t+1/60, t+2/60, identical cameras)

- **d1** = mean |L1−L0|.
- **d2** = mean |L0−2L1+L2|: the second temporal difference, which is flicker rather than smooth motion.
- **>24** = % of pixels with |d2| > 24 levels.

| shot | band | demo d1 / d2 / >24 | ref d1 / d2 / >24 |
|---|---|---|---|
| 06-sunset | far (h..h+3 %) | 0.97 / 0.96 / **0.19 %** | 0.75 / 0.74 / 0.01 % |
| 06 | h+3..8 % | 2.24 / 1.93 / **0.31 %** | 1.86 / 1.51 / 0.10 % |
| 06 | h+8..20 % | 4.22 / 4.01 / 1.57 % | 3.59 / 3.24 / 0.83 % |
| 06 | near | 5.47 / 6.28 / 4.91 % | 3.62 / 4.04 / 2.28 % |
| 02-grazing | far | 3.88 / 3.95 / 1.90 % | 3.36 / 3.35 / 1.19 % |
| 02 | h+3..8 % | 8.22 / 10.15 / 11.4 % | 7.29 / 8.60 / 8.1 % |
| 02 | h+8..20 % | 12.2 / 17.2 / 23.8 % | 10.8 / 14.6 / 19.1 % |
| C5 azure grazing | far | 5.88 / 6.41 / 4.34 % | 5.34 / 5.74 / 3.35 % |
| C5 | h+3..8 % | 8.32 / 10.5 / 13.1 % | 7.22 / 8.7 / 8.6 % |

- **Overall:** the demo flickers 15–30 % more on average, and has 1.3–3x more strongly flickering pixels.
- **Far sunset glitter:** about 19x more flickering pixels in the 06 far band, and 3x more in the next band.
- **Where it shows:** `motion/06-demo-d2.png` against `motion/06-ref-d2.png`.
- **Speckle rows:** zoomed crops (`crop-06-demo-dots.png`, against `crop-06-ref-horizon.png`) show rows of single-pixel dark and light speckles along distant crest lines at roughly 300–600 m. The reference has none. They change from frame to frame: this is the temporal shimmer the brief calls a major defect.

The reference interpolates FFT snapshots, which slightly lowers its d2. But d1 (plain motion) is also 10–50 % higher in the demo, and the static spatial high-frequency energy is higher too. So the excess comes from shading and filtering, not the timing scheme.

### 4. Tiers

These are measured with the later, in-flux FFT, so demo against demo only (`tier-*.png`, `tierc-*.png`). High-frequency energy relative to high, from the horizon to the near band:

| tier | ratio to high |
|---|---|
| low | 0.78 → 0.50 |
| medium | 0.95 → 0.83 |
| ultra | 1.05 → **1.48** |

- **Low** looks plasticky: broad, blobby, over-bright glitter lobes. The 128 grid on cascade c2 plus the uniform `unresolvedSlopeVariance` roughness does not reproduce the reference's mid-scale sparkle.
- **Ultra** (c2 at 512) is visibly sparklier and less filtered close-up than high. That moves it further from the reference, and adds more shimmer.
- **The 4th fine cascade (0.84 m tile) being disabled for shading:** this is the right call for reference parity. The reference has no sub-0.1 m slopes, and adding them would turn the glitter into aliasing sparkle. But it leaves an inconsistency:
  - high and ultra still pay for c3's FFT and mip chains, and the vertex shader still adds c3 displacement (σ ≈ 1.7 mm, invisible);
  - so "high" shades the same as "medium", for about 0.1 ms more GPU;
  - c3's real slope variance (≈ c2's) is thrown away instead of being folded into roughness, which is where it would help (a wider glitter).

  Either fold c3's shortening into roughness only (scalar, no normal), or stop computing c3 for tiers that do not use it. Document the choice in TUNING.

### 5. Floating origin (rule 5)

- `?origin=100000,0,100000` against origin 0, golden 06 camera: mean |Δ| 0.99, **2.3 % of pixels > 8 levels, max 151**, concentrated in the glitter (`o-diff.png`). Two captures at origin 0 differ by exactly 0, so this is not noise.
- The facets debug view differs by 19 % of pixels, so most of this is geometry at render coordinates around 1e5. The demo's `?origin` keeps the camera at world 0, so render coordinates become about 1e5. That is LOD and demo scope.
- The material adds to it: it builds V and dist from `positionWorld − cameraPosition`, a subtraction of two large f32 values. It should derive them from the view-space position (`positionView`, from the CPU-composed modelViewMatrix), which stays exact.

### 6. Motion vectors (rule 10)

- `?debug=velocity` in motion (not paused) gives smooth, coherent vectors that follow the swell, with no noise (`velocity.png`).
- `previousPositionNode` is exposed from `index.ts`. The material does not itself emit a velocity MRT: that is the engine's job through `setMaterialPreviousPosition`, so this is acceptable.
- The velocity view needed more than 25 s to leave "starting…" (async compile).

### 7. Underside (C7)

The same character as the reference: back faces with the facing fix, the sky seen through the surface, and the same dark crevice speckle the reference has. No black areas and no NaNs.

### 8. Performance

Perf runs were taken with other agents using the GPU, so they are noisy.

| configuration (azure 02 camera, high) | fps | GPU render | GPU compute | demo CPU / frame |
|---|---|---|---|---|
| demo, full | 184 | 2.10 ms | 0.82 ms | 4.28 ms mean (p95 7.4) |
| demo, `?debug=nomesh` | | 0.69 ms | | |
| demo, `?debug=facets` | | 1.23 ms | | |

- **Ocean draw:** about 1.4 ms in total, of which the water shading is about 0.9 ms over the facets baseline.
- **Reference frame time:** 3.69 ms at the default view (`docs/port/reference-shots/perf.json`). My own reference perf run reported 894 fps, which is implausible and is discarded.
- **Bottleneck:** the demo is CPU-bound (core tick, out of scope). The water shader cost is fine.

### 9. Rule compliance

- **Rule 1 (TSL only):** no GLSL, ShaderMaterial or WebGL in the material, index or photo-sky code. OK.
- **Rule 5:** periodic patterns use `tileCoord` (wrapped). See §5 for the V and dist precision issue.
- **Rule 9 (no tonemap):** linear output × preExposure, no tonemap. OK.
- **Rule 9 (units):** two problems.
  - **Foam colour does not come from the light.** `foamColor = (.3 + .7 N·L) × foamWhite`, with `foamWhite = envIntensity` (10 000 cd/m²): a constant that ignores SKY_LIGHT. Foam would glow at night, and in storm it is about 3–5x a Lambertian estimate ((E_sky + E_sun·cos)/π·0.8 ≈ 2 000 cd/m² from the demo's own SkyLight: sky 6 895 lux, sun 1 240 lux). The contact rim inherits `foamWhite`.
  - **The sun specular dimming is double-counted.** `sunSpecularScale` is derived from `ENVIRONMENT.cloudCover` (1 − 7/9 cloudCover). In an engine, cloud cover already reduces `sunIlluminance`. A partly cloudy sky would cut the glitter by up to 78 % on top of that. In the demo this is exact only because demo cloudCover = reference storminess.
- **REF_NITS mapping:** coherent overall.
  - The demo SkyLight is: azure sun 43 775 lux and sky 9 931 lux (EV100 14.1); golden 31 176 / 9 872; storm 1 240 / 6 895. Sky radiances are about 3–10 kcd/m².
  - That is about 2x dimmer than real daylight, with a plausible sun-to-sky ratio. A real engine SkyLight (100 klux) would give a sensible, slightly sun-heavier image.
  - The sun scatter terms use E/4.5 (a reference artistic gain) where the physical value is about E/π: about 30 % off physical, but consistent.
  - `fogDensity` is reinterpreted as 1 / Gaussian scale, so 0.0036/m would be a very thick exponential fog to the engine, and `fogColor` is ignored. Acceptable for the demo, but say so in the contract notes.
- **Rule 11 (inputs only via contracts):** OK, except the fallback `(sl as any)?.sunDir`, which reads a non-contract field.
- **Uniform control flow:** derivatives are at top level, and `select` is used rather than `If`. OK.
- **Textures per fragment stage:** about 10 (2 displacement + 2 normal + PMREM + foam 2 + scene color/depth + terrain). OK, at most 16.
- **No per-frame allocation in `update()`:** OK. A closure is created only when the SkyLight version changes. `photo-sky.computeSkyLight` allocates about 4 600 arrays per preset change: demo only, not per frame.

## Ranked defects

1. **Distant specular shimmer and sparkle are worse than the reference.** 1.3–3x more flickering pixels in mid and far bands (up to 19x in the sunset far band), 4–13 % more spatial high-frequency energy at distance, and single-pixel speckle rows on distant crest silhouettes. See the single biggest gap below.
2. **Low-sun glitter is too hot:** +6.5 to +7 luma and higher standard deviation in the 06 and C5-golden glitter (related to defect 1: under-filtered normals and roughness).
3. **Foam and contact white level is a constant** (`foamWhite = envIntensity`), not lit by SKY_LIGHT. This breaks rule 9 for night, overcast and storm skies.
4. **`sunSpecularScale` is tied to `ENVIRONMENT.cloudCover`:** non-physical double dimming once a real sky module drives `sunIlluminance`.
5. **V and dist come from `positionWorld − cameraPosition`** (large f32 values at render coordinates around 1e5): 2.3 % of pixels change by more than 8 levels in the 100 km origin test. Use view-space position.
6. **The tier ladder is inconsistent for shading.**
   - Ultra (c2 at 512) is 1.48x sparklier near than high: further from the reference, more shimmer.
   - High pays for c3 and never shades it; c3's slope variance is discarded instead of going into roughness.
   - Low's glitter is blobby and plasticky.
7. **Minor:**
   - `(sl as any).sunDir` non-contract fallback.
   - `fogColor` ignored.
   - `fogDensity` semantics redefined.
   - `MeshBasicNodeMaterial` keeps `fog = true`: an engine scene fog would apply on top of the built-in haze. Set `material.fog = false`, or document `haze: false` as mandatory when the engine fogs.

## Single biggest gap (instruction for the builder)

Make distant water at least as stable as the reference.

1. **Instrument it.** Add a `?debug=rough` view showing `rough`, `surf.variance`, the fragment LOD, and `sqrt(variance)`, and compare it pixel for pixel with the reference at the 06-sunset and 02 cameras. Today the demo's distant normals carry 4–13 % more high-frequency energy, and the glitter flickers 1.3–19x more.
2. **Check the mip sampling.** Confirm the normal arrays are sampled truly trilinear at the same LOD as the reference (`log2(footprint / texel)`, same footprint definition, no extra half-level from `uvOf`).
3. **Fix the speckle rows on distant crests.** They come from `dFdx` / `dFdy` of `restLocal` and `N` blowing up across crest silhouettes (2x2 quads straddling an occlusion edge). Use a robust footprint: for example, clamp the derivative-based footprint to at most a small multiple of the analytic footprint `dist × pixelAngle / max(|dot(V, up)|, ε)`, and clamp `sqrt(variance)` similarly. If needed, raise the roughness floor with distance until the far-band temporal second difference (t, t+1/60, t+2/60, `scratch/material-r1/temporal.mjs`) is at or below the reference in every band. The 06 and C5-golden glitter means must also land within ±2 levels.

## Telling images

- `docs/port/reviews/scratch/material-r1/motion/06-demo-d2.png` against `.../motion/06-ref-d2.png`: temporal flicker map.
- `docs/port/reviews/scratch/material-r1/crop-06-demo-dots.png` against `.../crop-06-ref-horizon.png`: speckle rows on far crests.
- `docs/port/shots/material-r1/02-grazing-sun.png` against `docs/port/reference-shots/02-grazing-sun.png`: open-water colour parity.

## Housekeeping

- My first motion captures were accidentally written under `world-ocean/"02 scene=…"/`, `"C5a …"/` and `"06 …"/` folders: a PowerShell variable-name clash. The orchestrator committed them in `2e0b85b`. I moved them to `scratch/material-r1/motion/`. git status now shows them as deleted, so the next commit removes them.
- Servers on 5191 and 5192 were started by me and stopped at the end.
