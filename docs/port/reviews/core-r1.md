# Review core-r1: render-agnostic core (index.ts, types.ts, spectrum.ts, cpu-fft.ts, presets/, core tests)

**Verdict: FAIL**

Reviewer: fresh context. I did not trust any comment or claim and checked everything by running code. Machine: Ryzen 7 2700X, Node 22.14.0, Windows 10. Other agents were working on the same machine at the time (browsers, other Node processes), so treat timings as ±20 %.
Scratch scripts (all runnable with `node --experimental-strip-types --no-warnings <file>` from the repo root):
`docs/port/reviews/scratch/{fidelity,omega-index,physics,blend,pop-debug,folds,bruteforce,alloc,alloc2,alloc-sites,box-experiment,tickcost,velcost,coldquery,determinism}.mjs`.

## What is good (verified)

- **The CPU FFT pipeline matches the reference GPU maths.** I emulated the reference EVOLVE, unnormalised IFFT, checkerboard and BLEND display steps in float64, using my own separate DFT (not cpu-fft.ts), at g = 9.81. A CascadeSim 256² snapshot at t = 37.3 s agrees to 6.4e-6 m (cascade 0, max |H| 4.23 m), 2.9e-6 m (cascade 1) and 7.9e-7 m (cascade 2). The height sign (−f.x), the east/south handedness and choppiness on the horizontal components only are all correct (`fidelity.mjs`).
- **h0 matches the reference at the reference gravity.** Seed 1 at g = 9.81 with the fetch refitted for 47 m: h0 matches `generateSpectra` to 1e-7 relative, the omega indices match 0/196608, and Hs is 4.0358 m (reference) vs 4.0360 m (port). The band split, the standing mix and the [h0, conj h0(−k)] layout are correct.
- **heightAt returns the height of the displaced point.** I checked this end to end with an independent drawn surface: the reference spectrum, my DFT, chained bilinear lookups at 256³, and brute-force grid inversion. Error vs heightAt: mean 2.0 mm, p99 7.2 mm, max 11 mm (0.27 % of Hs) at t = 21.7 s. At t = 5000.4 s the max is 5.9 mm. No fold points occurred in breeze (`bruteforce.mjs`).
- **Queries do not depend on query history.** World A had a clean history. World B also received random past, future and ±100 km queries plus velocity queries. Both had a wind change and a preset change. A and B are bit-identical: max difference 0 over 1000 points × 5 times (`determinism.mjs`).
- **Contract shape is correct.** id, provides, optional, phases ['simulate'], install/start; WATER_QUERY is given along up (the rotated-frame test passes, and I reran it). velocityAt is world-space with an out parameter. No three imports in the core, no performance.now/Date.now, no enum, namespace, parameter properties or decorators. All relative imports end in `.ts` and use `import type`. stubs/world-core.ts is imported in one place only (types.ts).
- **Smooth wind and preset blends.** Calm → storm via setPreset: Hs goes 0.73 → 1.11 → 6.84 → 8.97 m over about 4 s, with no pop beyond the natural per-tick change.

## Ranked defects

1. **velocityAt makes the 60 Hz tick unaffordable (engine fitness, spec §6).** The first velocityAt at each new time materialises the whole velocity frame and every position block, eagerly (`CascadeSim.materialise`, `if (withVel || this.cubic)` full fill plus the full nn loop). Measured cost: **6.1 ms per tick** for the first velocityAt, then 2.5 µs for repeats (`velcost.mjs`).
   - Tick + 100 heightAt + 10 velocityAt: mean **10.4 ms**, p95 15.3 ms, p99 20.1 ms, max 34.7 ms. 105 of 3600 ticks took longer than a whole 16.7 ms frame (`tickcost.mjs`).
   - Any buoyancy code that uses velocity for drag pays this on every step.
2. **A query at a time outside the snapshot cache stalls for 67 ms.** heightAt at an arbitrary past or future time takes **67.5 ms**; velocityAt takes 73 ms. Six 256²×3 IFFT snapshots are computed synchronously, and each such call also allocates about 1.8 KB (`coldquery.mjs`, `alloc2.mjs`).
   - The contract says "at simulation time `time`" without limits, so rewind, prediction or replay code will hitch.
   - None of this is documented as a cost.
3. **Rule 7 is violated: there is allocation per tick and per query.** Measured with the semi-space pinned (`--min/max-semi-space-size=128`, so no scavenge happens inside the window):
   - `ocean.tick` steady: **159 B/tick**. The sampling heap profiler attributes it to `differs()` and `sampleWind()` (`Math.hypot`, `wrapAngle`, `Math.min/max`) (`alloc-sites.mjs`).
   - heightAt: **16.4 B/call**, a boxed HeapNumber return.
   - velocityAt: 0.36 B/call.
   - Tick + 100 heightAt + 4 velocityAt: 1.84 KB/tick.

   `tests/core-heap.test.ts` hides this: it allows up to 2048 B/tick and up to 16.5 B/heightAt. Its new-space window method is unreliable without a fixed semi-space; without pinning I measured 2 B/call for heightAt, which is wrong.

   **Is the 16 B a violation?** By the letter of the spec, yes: "Queries must be allocation-free." In practice the contract signature (`heightAt(): number`) forces a boxed double unless the caller's JIT inlines the call. My thin-wrapper experiment did not remove it (`box-experiment.mjs`). The 16 B can be accepted only if it is documented as a contract cost. In that case also add an allocation-free batch API (`heightsAt(xyz: Float64Array, time, out: Float64Array)`) and write the number into README/TUNING. The 159 B/tick is a plain violation and is fixable: use `Math.sqrt` instead of `Math.hypot`, and keep the tick hot path free of boxed doubles.
4. **heightAt teleports between sheets where the drawn surface folds.** At 60 Hz over 10 s at 400 fixed points (`folds.mjs`, `pop-debug.mjs`):

   | Case | Folded area | Point-ticks with \|Δh\| > 0.25 m (> 15 m/s vertical) | Max jump per tick |
   |---|---|---|---|
   | storm | 0.03 % | 117 ppm | **2.02 m** |
   | breeze driven by a 25 m/s wind | 0.03 % | 54 ppm | — |

   During the 15 → 25 m/s wind test I saw a single 1/60 s tick go 2.89 → 4.69 m at a fixed point, with blend = 1 (steady state, not a crossfade bug). A floating object gets a 1–2 m vertical impulse about every 2–3 minutes per object in storm; with 100 objects that is several per second. The inversion is documented as "any of the sheets may be returned".

   Fix: choose the sheet consistently (for example the highest sheet, or warm-start from the previous rest point per caller), or clamp choppiness so storm does not fold.
5. **The test command does not work.**
   - `npm test` (`node --experimental-strip-types --test tests/`) on Node 22.14: **fails** with MODULE_NOT_FOUND for `tests`. Node treats the directory argument as a file. Node 24 is not installed here, so this is not verified, but since Node 21 positional arguments to `--test` are file/glob patterns and a bare directory is not expanded, so I expect the same failure on Node 24.
   - `node --experimental-strip-types --test` with no argument discovers the 43 tests, but **core-perf fails** ("heightAt 678154 calls/s < 1000000") because test files run concurrently.
   - `node --experimental-strip-types --test tests/core-*.test.ts` run alone: 35/35 pass in 24.4 s.

   Fix: `"test": "node --experimental-strip-types --test --test-concurrency=1 \"tests/**/*.test.ts\""` (quote the glob), and add `"engines"`.
6. **Wrong results after about 6.8 years of simulation time.** `j = Math.floor(t / dt) | 0` overflows int32 at t > 2.15e8 s. heightAt(t = 2.2e8) returns **−3.1e26 m**, and heightAt(5 + 8192·27000) returns 3.3e26 m, while the true period is 8192 s (`determinism.mjs`). NaN or Infinity time silently returns NaN. Wrap t modulo LOOP_PERIOD (the snapshot grid has to be aligned to it) or drop the `|0`.
7. **Past heights are not kept.** After a spectrum crossfade, `heightAt(p, pastTime)` no longer returns what was live then: 6 of 8 recorded heights changed by up to 3 m. The cause is that only the current from/to slots are kept. This is acceptable if documented ("valid for t ≥ last blendStart"), but the contract and README do not say so.
8. **Wind direction flip collapses the sea.** The wind is smoothed with an EMA on the wind vector, so a 180° flip sends the smoothed speed through ≈0. Hs goes 3.8 → **0.59 m (−85 %)** for about 6 s, then back up, with 6 regenerations. A 90° turn dips Hs 3.8 → 2.76 m (−27 %), because h0 lerps between spectra with different directions and the energy cancels mid-blend.

   There is no pop (max per-tick change 0.27 m vs a natural 0.13 m), but real seas do not vanish when the wind turns: the old sea should persist as swell. The wind-coupling test only covers a 57° turn and never checks Hs continuity.

   Fix: smooth speed and direction separately, and crossfade the energy (amplitude² weights or sqrt(w)/sqrt(1−w)) rather than the complex amplitudes.
9. **Preset Hs values are not physical for their stated wind and fetch.** Port Hs is about **2.04×** the 1-D JONSWAP Hs for the same U, fetch and gamma, for every preset. The factor is amplitude 1.578 × √1.757, the second term being the Tessendorf h0 + h0m variance doubling with the standing mix. Significant steepness Hs/Lp is 0.076–0.089 for all presets, about 2× real wind seas (≈0.03–0.055).

   | Preset | Hs (port) | Tp | JONSWAP Hs, same U and fetch | CERC fetch-limited | PM fully developed |
   |---|---|---|---|---|---|
   | calm | 0.82 m | 2.6 s | 0.40 m | — | — |
   | breeze | 4.04 m | 5.5 s | 1.98 m | — | — |
   | windy | 6.08 m | 6.6 s | 2.98 m | — | — |
   | storm | 9.85 m | 8.5 s | 4.82 m | 4.04 m | 16.2 m (U10 25) |

   So "storm = 9.85 m for 25 m/s over 100 km" is about 2× too high for that fetch, but below fully developed (~14–16 m). With its 8.46 s period it is far too steep.

   With the fetch range PARAM_DEFS allows (up to 500 km), the model reaches absurd values: U 25 over 2000 km gives Hs **35.5 m** with a 548 m peak in a 1024 m tile. The breeze amplitude is a reference-fidelity choice and is fine for breeze. Applying the same 1.578 artistic multiplier to calm, windy and storm while labelling them by physical U and fetch is misleading. Either use physical presets (amplitude 1 and no √2 bookkeeping error) or document the 2× factor in TUNING.
10. **The "seed 1 = reference" claim holds only near t = 0 in the shipped configuration.** The shipped config uses g = 9.80665 (spec §3), which changes **165,627 of 196,608 omega indices** (max index difference 8). The relative rms height difference from the reference waves is 0.4 % at 10 s, 2.5 % at 60 s, 24 % at 600 s and 105 % at 1 h (fully decorrelated).

    The test called "bit-for-bit" runs only at g = 9.81 with a refitted fetch, and uses a tolerance of 2e-6·peak, which is not bit-for-bit. The shipped preset fetch (45479.33 m, fitted at 9.80665) gives 46.995 m at 9.81. spectrum.ts:17 ("wave-for-wave") is false as shipped. REFERENCE.md does mention the gravity trade-off.
11. **Tick cost is a large part of a 60 Hz step even without velocity.** Measured (`tickcost.mjs`):

    | Case | mean | p50 | p95 | p99 | max | ticks > 4 ms |
    |---|---|---|---|---|---|---|
    | Tick only | 2.53 ms | 2.30 ms | 5.70 ms | 7.29 ms | 13.2 ms | 18 % |
    | + 100 heightAt | 4.40 ms | — | 8.44 ms | 11.9 ms | 21.3 ms | — |

    The built-in bench reported 4.55 ms steady when run alone. The claim of "2.1 ms steady" is optimistic. Snapshot work is budgeted per tick with a ×1.5 margin, but spikes reach 13–21 ms because `ensure()` falls back to synchronous computation.
12. **Code quality.**
    - Magic numbers without a source: pendingLead 1.6 s, switchMargin 0.4 s, `checkCounter & 7`, `rows -= 8` ("packing ≈ 8 rows"), SNAPSHOT_POOL 10, MAX_PINS 8, det > 0.1, `25 * tol2`, the 4096/16384 work units, the 16×16 blocks, DEFAULT_QUERY snapshotDt 0.4/0.25/0.1.
    - Module-global mutable scratch (`CALL`, `SAMPLE`, `GRAD`) is shared by every ocean instance and used to pass arguments. This is fragile: re-entrancy (a WIND_QUERY or perf callback that queries another ocean) silently corrupts it.
    - `for (let g = 0; g < GUESS_STEPS; g++)` (index.ts:541) shadows the gravity variable `g`.
    - The inner block at index.ts:547 is mis-indented.
    - `sampleBilinear` and `sampleBilinearGrad` are unused by the core.
    - `dispose()` releases nothing (the 3 slots × 3 cascades × 256²×4 floats plus pools stay alive while the query is referenced).
    - The arguments-through-typed-array style (ST/TS/CALL indices) makes index.ts hard to review: 830 lines with one closure holding about 60 variables.
    - There is no TypeScript type check anywhere (typescript is not installed; tsconfig says "editor only"), so `erasableSyntaxOnly`/`verbatimModuleSyntax` are never enforced.
13. **Minor.**
    - velocityAt vs finite differences during a calm→storm crossfade: max error 0.19 m/s of 8.45 m/s (2.3 %).
    - `toLocal` projects (p − anchor) onto the tangent frame at p. That is fine on flat ground, but on a sphere it is not a valid surface coordinate. The spec only asks that sphere support is not prevented; document the limitation.

## Spec checklist (scope: core)

| Item | Result |
|---|---|
| §1.3 core never imports three, runs in Node | PASS |
| §1.4 no +Y assumption | PASS (grep test + rotated-frame test, rerun) |
| §1.6 explicit time, deterministic | PASS for t < 2.15e8 s; garbage beyond (defect 6) |
| §1.7 no allocation per frame / query | **FAIL**: 159 B/tick, 16 B/heightAt, 1.8 KB per cold query (defect 3) |
| §1.8 CPU = GPU | CPU vs an independently built 256³ bilinear surface: max 0.27 % Hs. Not verified against the actual GPU (out of scope) |
| §2 erasable TS, `.ts` imports, import type, units | PASS by grep; no tsc run exists |
| §3 WorldSystem / WATER_QUERY semantics | PASS for shape and units. Folded-crest sheet jumps (defect 4) and lost history (defect 7) are undocumented |
| §6 ≥ 1 M calls/s | heightAt 1.31 M/s and velocityAt 1.23 M/s at a fixed time (alone). 0.68 M/s under the default parallel test run. The first velocityAt per tick costs 6.1 ms |
| §7.2 test command | **FAIL** (`npm test` errors on Node 22.14; default discovery fails core-perf) |
| §7.3 determinism | PASS |
| §7.4 presets Hs | PASS against its own table. The table is about 2× physical for the stated U and fetch (defect 9) |
| §7.5 wind coupling, smooth | PASS for no pop. A 180° flip drops Hs 85 % (defect 8) |
| §7.8 up not hardcoded | PASS |
| §7.10 heap flat test | Present but its thresholds legitimise the allocations (defect 3) |

## Key numbers

- Tests: core-*.test.ts run alone 35/35 pass (24.4 s). `npm test` errors (directory argument). Default discovery: 41/43, with core-perf failing under concurrency.
- Fidelity at g 9.81: field max error 6.4e-6 m, h0 1e-7 relative, omega 0 mismatches.
- Shipped g 9.80665: 84 % of omega indices differ; 2.5 % rms divergence from the reference at 60 s.
- heightAt vs independent brute force: mean 2.0 mm, max 11 mm (0.27 % Hs).
- Throughput: heightAt 1.31 M/s, velocityAt 1.23 M/s.
- First velocityAt per tick: 6.1 ms. Cold query at an arbitrary time: 67 ms.
- Tick (alone): mean 2.53 ms, p95 5.7 ms, max 13 ms. With 100 heightAt + 10 velocityAt: mean 10.4 ms, p99 20 ms.
- Allocation: tick 159 B, heightAt 16.4 B/call, velocityAt 0.36 B/call.
- Storm sheet jumps: 117 ppm of point-ticks, up to 2.02 m in one tick.
- 180° wind flip: Hs dips to 15 % of its value for about 6 s.
- Physics: Hs about 2.04× JONSWAP for the same U and fetch; Hs/Lp ≈ 0.088.

## Single biggest gap (one instruction)

**Make the velocity path lazy and bounded:** materialise velocity frames per 16×16 block on first touch, exactly like positions, and never fill whole frames inside a query. The first velocityAt per tick should drop from 6.1 ms to tens of µs. Then make the tick plus a realistic gameplay load (100 heightAt + 10 velocityAt at the tick time, 1 at the interpolated time) meet **p99 < 4 ms and zero allocation**, with the new figures in TUNING.md.
