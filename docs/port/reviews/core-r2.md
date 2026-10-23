# Review core-r2: render-agnostic core (index.ts, types.ts, spectrum.ts, cpu-fft.ts, ocean-worker.ts, presets/, core tests, benches)

**Verdict: FAIL**

Reviewer: fresh context. I verified every claim by running code; I did not trust comments or test output. Machine: Ryzen 7 2700X (8C/16T), Node 22.14.0, Windows 10. **The machine was heavily loaded by other agents.** Total CPU load sampled during each perf run was 56–95 % (a few points 31–44 %). I therefore report best-of and median, and repeat the key runs at **High process priority** (`os.setPriority` / `PriorityClass=High` on my own process only) to estimate what the code can do on a quiet machine.

Core sources were unchanged during the review (last write 21:58; review 22:15–23:10).

Scratch is in `docs/port/reviews/scratch/core-r2/`. Run each script with `node --experimental-strip-types --no-warnings <file>` from that folder:

| Script | What it checks |
|---|---|
| `tickload.mjs` | Paced 60 Hz gameplay load (env `PRIO=1`, `BREAKDOWN=1`) |
| `run-perf.ps1`, `run-perf-prio.ps1` | Repeated runs with a CPU-load sampler |
| `alloc-r2.mjs` | Allocation, pinned semi-space. Needs `--expose-gc --min-semi-space-size=128 --max-semi-space-size=128` |
| `determinism-slow.mjs` | Worker vs main thread vs deliberately slowed worker, bit for bit |
| `slowcore/` | A COPY of the core with injected worker naps, crash and tracing (env `SLOW_MS`, `SLOW_PENDING_MS`, `CRASH_AFTER`, `TRACE_RACE`). The maths are unchanged. |
| `lifecycle.mjs` | Worker lifecycle |
| `throughput.mjs` | Query throughput |
| `fwdcount.mjs` | Forward evaluations per heightAt |
| `bruteforce-cubic.mjs` + `fidelity-lite.mjs` | Independent Catmull-Rom surface parity |
| `folds-r2.mjs` | Fold-over sheet jumps |
| `presets-r2.mjs` | Preset Hs, caps, wind turns |
| `timewindow.mjs` | Cost of the query time clamp |
| `vitebuild/` | Production bundle check |

Logs sit next to the scripts. The round-1 `scratch/fidelity.mjs` no longer loads, because it imports the removed `CALL`; `fidelity-lite.mjs` is the working copy.

## What is good (verified)

- **The tick itself is now cheap.** With workers, `ocean.tick` costs **0.20–0.26 ms** on average, both on slow and fast frames (`BREAKDOWN=1`). Velocity is lazy per block: the first velocityAt after a tick fills ~35 velocity blocks for 10 random points, not whole frames.
- **Allocation (pinned semi-space, median of 5 windows, harness overhead of 6.2 B/tick measured separately):**
  - tick: **≈ 1.5 B/tick**, i.e. zero within harness noise
  - velocityAt: 0.08 B/call
  - tick + heightsAt(100) + 10 velocityAt: 22 B/tick raw, ≈ 16 B net
  - heightsAt: 3–5 B per call (not strictly zero, but negligible)
  - out-of-window query: 0.3–6 B
  - The 159 B/tick of round 1 is gone.
- **Determinism, normal timing.** I ran the worker:false main-thread path, worker:true, and a worker copy that naps randomly up to 40/60 ms before half its jobs, unpaced and paced. Over 3600 ticks with wind speed/direction changes every 4 s, three preset crossfades and one immediate `setParams`, there were **0 mismatches in 144 000 bit-for-bit height and velocity comparisons per run**, with identical blendStart and version. A 1.0 s delay on every pending-source job also gave 0 mismatches.
- **Parity vs an independent Catmull-Rom surface.** I built it from the reference spectrum (g 9.81), my own DFT, 256³ Catmull-Rom chained lookups and brute-force grid inversion (windy, 150 points), and compared heightAt against it:
  - t = 21.7 s: mean **2.23 mm**, p99 7.41 mm, max **9.79 mm (0.24 % Hs)**
  - t = 5000.4 s: mean 1.93 mm, max 7.15 mm (0.18 % Hs)
  - No folded points occurred in windy.
- **Presets match the table.** Expected Hs is calm **0.538**, breeze **1.463**, windy **4.036**, storm **8.996** m. Realised Hs (4σ of heightsAt over a 2 km grid, mean of 4 times) is 0.521 / 1.417 / 3.904 / 8.238 m (storm −8.4 %).
  - The ×2.04 factor vs 1-D JONSWAP for the same U, fetch and gamma is now documented in the `amplitude` description.
  - Caps hold: the Hs ceiling is 14.87 m (U ≥ 35 or fetch ≥ 200 km), realised 13.4 m at U 60 / 2000 km.
- **Wind turns no longer collapse the sea.**

  | Preset | 180° | 135° | 90° |
  |---|---|---|---|
  | windy, minimum Hs | 98.8 % | 98.9 % | 98.9 % |
  | storm, minimum Hs | 96.6 % | 96.6 % | 96.6 % |

  The round-1 dip was 85 %.
- **Long times are fixed.** t + 27000 loop periods agrees with t (the test passes, and I reread the code). Non-finite times throw RangeError.
- **Worker lifecycle, normal case.**
  - Node exits without `dispose()` (the workers are `unref`'d).
  - `dispose()` makes queries throw "disposed", and double dispose is fine.
  - 30 create/tick/dispose cycles: RSS 82 → 87 MB after GC. No thread or memory leak.
- **Spec greps are clean.** No three import. No `.y` used as up (only full x/y/z expressions). No enum, namespace, parameter properties or decorators. Every relative import ends in `.ts`. No `performance.now`, `Date.now` or `Math.random` in the core.
- **Tests.**
  - `npm test` run 1, under 95 % load with my scripts running in parallel: **58/60**. Failures: core-perf (0.26 M/s < 0.3 floor) and lod tier-change (out of scope).
  - Run 2, quieter: **66/66** (lod tests were added meanwhile by another agent).
  - core-perf run alone at High priority passes: 0.43–0.44 M/s.
  - The §7.2 literal command `node --experimental-strip-types --test tests/` still fails on Node 22.14 with MODULE_NOT_FOUND. `npm test` (quoted glob, concurrency 1) works.

## Ranked defects

### 1. BLOCKER: a late worker silently corrupts heights by up to 7 m (data race on the source table)

A snapshot request passes an **entry index** (`ctrl[C_ENTRY]`) into the mutable shared source table, not the source itself. When the pending source becomes active, `setActive` moves it from entry 1 to entry 0 and clears entry 1 (`setSource(1, -1, 0, 0)` → from = to = slot 0). A worker that had not yet started a pending job then reads **entry 1 = slot 0's static spectrum**. `pollRemote` accepts the result, because the job's source id (now in entry 0) is still live.

Reproduction: `slowcore` with `SLOW_PENDING_MS` (each pending-source job is delayed). Delays above `pendingLead` (1.6 s) trigger it.

| Delay per pending job | Mismatches vs the main-thread path | Max diff | First at |
|---|---|---|---|
| 1.0 s | 0 | 0 | — |
| 1.7 s | 15 375 | **6.95 m** | tick 541 |
| 2.5 s | 8 770 | **6.95 m** | tick 541 |

`TRACE_RACE=1` prints "worker computes entry 1 which is now empty (srcId −1, from 0 to 0)".

A worker 1.6 s late is not exotic. It happens on a 4-thread laptop with the engine's own workers, during a GC or swap, or with the 90 % contention seen on this machine today. The "bit-identical" test only covers fast workers.

Fix: pass the source by value in the request (id, from, to, start, duration in `ctrlF`), and have the worker echo the id it computed so the main thread rejects mismatches.

### 2. BLOCKER: a worker crash freezes the game forever

I injected a throw in the worker (`CRASH_AFTER=20`). The main thread hangs in `while (Atomics.load(ctrl, C_STATE) !== S_DONE) {}`. I killed it after 90 s; the last log was tick 100, and no error was ever surfaced.

The same unbounded spins exist in `stepRegen` (regen due) and `applyImmediate`. A late worker also stalls the main thread for the whole delay instead of computing the snapshot itself: the 2.5 s delay run took 110 s wall time for 2400 paced ticks.

There is no `error`/`exit` listener on the workers. Fix:
- Listen for `error`/`exit` and mark the remote dead.
- Bound every spin, for example "wait ≤ 1 ms, then compute locally and ignore the late result".
- Never block on a worker for correctness.

### 3. The round-1 mandate fails: tick + gameplay load p99 is 7–9 ms (target < 4 ms)

Load per paced 60 Hz tick (`tickload.mjs`): 100 heightAt + 10 velocityAt at fresh random points over a 200 m area that drifts at 8 m/s, plus 1 heightAt at the interpolated time.

| Condition | p50 | p95 | p99 | max | ticks > 4 ms |
|---|---|---|---|---|---|
| storm, workers, High priority (5 runs) | 2.0–2.4 ms | 5.9–7.2 ms | **6.8 / 7.6 / 7.7 / 8.0 / 8.7 ms** (median 7.7) | 7.7–11.6 ms | 20–34 % |
| storm, workers, batch `heightsAt`, High priority (3 runs) | — | — | 7.2 / 7.5 / 7.7 ms | — | — |
| windy, wind 15 → 25 m/s, workers, High | — | — | 8.2 ms | — | — |
| storm, normal priority, load 73–91 % | 3.5–6.2 ms | — | 25.8–36.3 ms | 47–94 ms | — |
| Builder's own `bench-tick.ts`, High priority, load 31 % (fixed points) | 1.05–1.42 ms | — | **4.9–6.0 ms** | 5.7–8.4 ms | — |

The builder's own bench fails too.

Cause (`BREAKDOWN=1`): the queries, not the tick.
- **Slow frames** (> 4 ms, 22 %, average 5.6 ms): the worker prefill was adopted for only **0.29 of 3 cascades**, so the main thread Hermite-fills **≈ 188 position blocks (≈ 48 k texels)** inside the queries, plus ~36 velocity blocks.
- **Fast frames** (2.0 ms): adopted 2.37 of 3, and still filled 49 position + 35 velocity blocks.

Random points over 200 m touch essentially every block of the 96 m and 9 m tiles every tick. Materialising frames per tick is the wrong granularity for sparse queries. See the single biggest gap.

### 4. The browser and bundled builds never get the workers, and the fallback is the 11–15 ms path

- **Production bundle.** `new URL('./ocean-worker.ts', import.meta.url)` is created outside a `new Worker(...)` literal. Vite 8 therefore inlines the **raw TypeScript source** as a `data:` URL, with relative `./cpu-fft.ts` imports (`vitebuild/dist/ocean.js`). That cannot execute in a browser. `spawnSnapshotWorker` returns a Worker object anyway, `remoteReady()` stays false, and the fallback is silent.
- **The demo is not cross-origin isolated.** The vite config sends no COOP/COEP headers, so `canUseWorkers()` is false in the only browser page that exists. The browser worker path has never run.
- **The fallback (worker:false) at High priority:** storm p50 4.8–6.5 ms, p99 **11.4 / 12.4 / 12.7 / 13.2 ms**, max 14–16 ms; windy with a wind change p99 15.0 ms. At normal priority under load: p99 24.8–41.5 ms.

Spec §6 does allow workers ("big precompute can go to a worker, pages are cross-origin isolated"). This design goes further: it **depends on three always-on worker threads every tick** (prefill and snapshots), not on one-off precompute. That is acceptable only if (a) the engine bundler path is proven and (b) the fallback still meets budget. Neither holds.

Fix:
- Use the literal `new Worker(new URL('./ocean-worker.ts', import.meta.url), { type: 'module' })` pattern, or accept an engine-supplied worker factory.
- Add a COOP/COEP demo mode that proves the browser path.
- Report `remoteReady` in `cpuStats` so a silent fallback is visible.

### 5. The query time clamp silently violates the contract and is wrong for real gameplay

WaterQuery promises "height … at simulation time `time`". Once the system has ticked, any time outside `[lastTick − 0.1, lastTick]` is silently answered at the clamped time. Measured in storm against an exact ocean (`timewindow.mjs`):

| Query time | Max error | rms error |
|---|---|---|
| +0.05 s | 0.36 m | — |
| +0.25 s | 1.68 m | — |
| +0.5 s | 4.02 m | — |
| **+1 s (projectile splash prediction)** | **8.0 m** | 2.0 m |
| +2 s | 9.7 m | — |
| −0.3 s (lag compensation) | 1.46 m | — |
| −1 s | 5.1 m | — |

Also:
- **`world.setTime(T + 20)` without a tick** (scrub, replay, a shot jump) returns **7.5 m** wrong heights at `world.time` while the GPU draws at the new time.
- **Any system ticking faster than, or before, the ocean** (120 Hz physics substeps) reads stale heights.

It is documented in the index.ts header, `SimParams.timeWindow` and PARAM_DEFS, but not in the contract and with no runtime signal. My judgement: clamping is acceptable as the *cost policy* for far times. It is not acceptable as a silent answer for near future and past times that the snapshot cache already covers.

Proposal:
- (a) Make the window **asymmetric and per cascade**. It should be exact for t ∈ [lastTick − (dt_c), lastTick + 2·dt_c − margin], where j0 − 1 … j0 + 3 are already pinned: about +1.0 s for cascade 0, +0.5 s for cascade 1, +0.25 s for cascade 2. Beyond a cascade's horizon, evaluate that cascade at its horizon only, so the small cascades degrade first.
- (b) Count clamped queries in `cpuStats` and warn once in development.
- (c) Re-anchor on a time jump. If `|t − lastTick|` is larger than 1 s on a tick-less frame, treat the next query time as a cold start (or give the ocean a `frame()` hook that follows `interpolatedTime`).
- (d) Propose a contract note upstream ("valid within [now − 0.1 s, now + H]"), plus an explicit `predictHeightAt` for long horizons.

### 6. heightAt throughput is 0.46–0.53 M/s, half the spec's ≥ 1 M/s aim, and velocityAt is 0.27–0.31 M/s

Measured at High priority, at a ticked time, 7 reps × 200 k calls:

| Config | heightAt best | heightAt median | Max diff vs default | rms diff |
|---|---|---|---|---|
| Default (cubic, 3 cascades), windy / storm | 0.497 / 0.532 M/s | 0.489 / 0.460 M/s | — | — |
| All bilinear | 0.85–0.91 M/s | — | 0.43 m | 7 cm |
| Cascade 2 dropped | 0.67–0.73 M/s | — | 0.12–0.18 m | — |
| Cubic only on cascade 0 | no faster | — | — | — |

- heightsAt runs at the same rate (0.48–0.56 M/s).
- velocityAt best is 0.30 / 0.28 M/s; round 1 measured 1.23 M/s.
- `core-perf.test.ts` lowered its floor to 0.3 M/s, so the spec's aim is not tested.

Instrumented copy (`fwdcount.mjs`): each heightAt does 2 guess steps + **3.7 bilinear forwards (3 cascades with Jacobian) + 3.1 Catmull-Rom forwards (3 cascades × 16 taps × 3 channels)**. Phase B usually runs all 3 refinement steps and then calls forward once more (`exactAt` false).

Cheap wins I expect to reach ~0.8–1 M/s:
- Cap phase B at 1–2 steps (one Newton step on cm residuals lands at mm).
- Reuse the last refinement forward instead of recomputing it.
- Take cascade 2 (9 m tile, 1–3 cm displacement) out of the iterative solve: apply its displacement once and its height at the end. The no-c2 run shows the iteration share is about 30 %.
- Use bilinear for cascade 2 (its texel is 7 cm).

Whatever is chosen, state the number in README/TUNING.

### 7. Through the contract, the realistic load still allocates 1.6 KB per tick, and the zero-allocation path is not reachable

`heightAt` costs **16.08 B per call** whenever the caller uses the result, so the realistic load is 1622 B/tick. `core-heap.test.ts` reports heightAt at **0.12 B/call** because its loop discards the result and V8 elides the box. The test hides the real cost, and its own comment contradicts it.

The allocation-free `heightsAt` exists only on the `Ocean` object. It is **not on `WaterQuery`**, so engine systems that `world.require(WATER_QUERY)` (buoyancy in another module) cannot use it. Fix:
- Measure heightAt with a consumed result.
- Propose `heightsAt` as an optional WaterQuery extension (or expose it under a second key).

### 8. Fold-over sheet jumps are still open (round-1 defect 4)

| Case | Folded area | Point-ticks with \|Δh\| > 0.25 m | Max per-tick jump |
|---|---|---|---|
| storm | 0.06 % | **125 ppm** | **2.53 m** (round 1: 2.02 m) |
| breeze driven by a 25 m/s wind | 0.06 % | 67 ppm | 0.96 m |

Nothing changed here.

### 9. Smaller

- **The sea is slow to follow a wind reversal.** At 15° per crossfade, the spectrum is still not aligned with a 180° reversal after 120 s (90°: 65 s; 135°: 93 s). It is physically defensible but sluggish for "a front passes". Make it tunable per event, or crossfade energy instead of complex amplitude so larger steps are safe.
- **Tick after `dispose()`** throws `TypeError: Cannot read properties of undefined (reading 'params')` instead of the clean "disposed" error.
- **createOcean + 30 ticks costs ~0.63 s** (30 cycles in 18.8 s): the synchronous init hitch is not documented.
- **Stale docs.**
  - types.ts:40 still says "EMA on the wind vector"; speed and direction are now smoothed separately.
  - index.ts:206 has a TAB in "at \<TAB\>ime" (`\t` in the JSDoc).
  - types.ts:97 claims "the main-thread tick drops by ~1.5 ms" with no measurement given.
- **Docs.** There is no README.md or TUNING.md in `world-ocean/`, although round 1 required the new figures in TUNING.md. The numbers in this review (worker fallback cost, time window, heightAt 16 B, throughput) are nowhere in the package docs.
- **Prefill adoption can take a frame computed from the old source** across a pending switch. `acquire` reuses invalidated entries before checking `isLazyRef`. It is harmless at w ≈ 0 (old and new are equal up to rounding), but it is a latent hazard if the switch time ever changes.

## Spec checklist (scope: core)

| Item | Result |
|---|---|
| §1.3 no three, runs in Node | PASS |
| §1.4 / §7.8 up not hardcoded | PASS (grep and rotated-frame test) |
| §1.6 / §7.3 determinism | PASS with normal timing. **FAIL** when a worker is more than 1.6 s late (defect 1) |
| §1.7 / §7.10 no allocation | tick and velocityAt PASS. heightAt 16 B/call through the contract; the batch API is not reachable through the contract (defect 7) |
| §2 erasable TS, `.ts` imports | PASS by grep (no tsc exists) |
| §3 WaterQuery semantics | **FAIL**: silent time clamp (defect 5); sheet jumps (defect 8) |
| §6 ≥ 1 M calls/s | **FAIL**: 0.46–0.53 M/s heightAt, 0.27–0.31 M/s velocityAt |
| §6 workers | Allowed in principle, but the design depends on them per tick; bundled and browser paths are broken (defect 4); crash means a hang (defect 2) |
| §7.2 tests | `npm test` 66/66 on a quieter run, 58/60 under load. The literal §7.2 command still fails on Node 22.14 |
| §7.4 presets | PASS (0.54 / 1.46 / 4.04 / 9.0 m; realised within 8.4 %) |
| §7.5 wind coupling | PASS (Hs ≥ 96.6 % through 90–180° turns; no pop) |
| Round-1 mandate p99 < 4 ms | **FAIL**: 6.8–8.7 ms with workers at High priority, 11–15 ms fallback |

## Key numbers

- Tick + 100 heightAt + 10 velocityAt (200 m, paced 60 Hz), storm, High priority:
  - workers: p50 2.0–2.4 ms, **p99 6.8–8.7 ms** (median of runs 7.7)
  - no workers: p99 11.4–13.2 ms
  - The tick alone with workers: 0.2–0.26 ms.
- Allocation: tick ≈ 0 B; heightAt **16.08 B/call** (consumed); heightsAt 3–5 B/call; velocityAt 0.08 B/call.
- heightAt **0.46–0.53 M/s** (cubic); bilinear 0.85–0.91 M/s at 0.43 m deviation; velocityAt 0.27–0.31 M/s.
- Determinism: 0 / 144 k mismatches with normal or ≤ 1 s-late workers; **15 375 mismatches, 6.95 m max, with a 1.7 s-late worker**.
- Worker crash: **permanent main-thread hang**.
- Vite production bundle: the worker is inlined as raw TS in a data: URL, so it falls back silently.
- Clamp error in storm: +1 s gives **8.0 m max** (2.0 m rms); a `setTime` jump gives 7.5 m.
- Parity vs an independent 256³ Catmull-Rom surface: max 9.8 mm (0.24 % Hs).
- Storm sheet jumps: 125 ppm, max 2.53 m/tick.
- Presets: Hs 0.538 / 1.463 / 4.036 / 8.996 m (realised 0.521 / 1.417 / 3.904 / 8.238); cap 14.87 m.
- Machine load during the measurements: 56–95 % (31 % for the builder bench run).

## Single biggest gap (one instruction)

**Stop materialising whole 16×16-block frames on the main thread for sparse gameplay queries, and make the query path independent of worker timing.** Evaluate the Hermite-in-time field per tap: the 4–16 texels a forward() touches, read from the two cached snapshots, with no frame and no block. Alternatively cut blocks to 4×4 and keep prefill only as an optional accelerator. Then:
- Bound every wait on a worker (≤ 1 ms, then compute locally).
- Pass sources by value in requests.
- Make the 100 heightAt + 10 velocityAt + tick load meet **p99 < 4 ms with workers off** at normal priority on this machine.

Today, 22 % of frames fill ≈ 188 blocks (48 k texels) inside the queries, and the fallback that every non-isolated or bundled page gets is p99 11–15 ms.
