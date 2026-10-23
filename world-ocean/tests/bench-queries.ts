// Benchmark of the CPU core: query throughput, tick cost, memory.
//   node --experimental-strip-types tests/bench-queries.ts [--json]
// Exported measure() is also used by tests/core-perf.test.ts.
import { createOcean, WIND_QUERY } from '../index.ts';
import type { PresetName } from '../index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../stubs/impl.ts';

export interface BenchResult {
  heightAtPerSec: number;
  heightAtRandomPerSec: number;
  velocityAtPerSec: number;
  tickMsSteady: number;
  tickMsRegen: number;
  tickMsMax: number;
}

function bestRate(reps: number, calls: number, fn: (i: number) => number): number {
  let best = 0, sink = 0;
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    for (let i = 0; i < calls; i++) sink += fn(i);
    const dt = (performance.now() - t0) / 1000;
    best = Math.max(best, calls / dt);
  }
  if (!Number.isFinite(sink)) throw new Error('bad sink');
  return best;
}

export function measure(preset: PresetName = 'breeze', reps = 5, calls = 200000): BenchResult {
  const shape = createFlatShape();
  const world = createStubWorld({ shape });
  const wind = createStubWind(15, -57.3, shape);
  world.provide(WIND_QUERY, wind);
  const ocean = createOcean({ preset });
  world.use(ocean); world.start();
  for (let i = 0; i < 120; i++) world.stepOnce(); // warm-up
  const q = ocean.query;
  const p = { x: 0, y: 0, z: 0 }, v = { x: 0, y: 0, z: 0 };
  const t = world.time;
  // Local area (a few hundred metres: boats around the camera) and random over 3 km.
  const heightAtPerSec = bestRate(reps, calls, (i) => { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; return q.heightAt(p, t); });
  let s = 12345;
  const heightAtRandomPerSec = bestRate(reps, calls, () => {
    s = (Math.imul(s, 1664525) + 1013904223) | 0; p.x = (s >>> 8) % 3000;
    s = (Math.imul(s, 1664525) + 1013904223) | 0; p.z = (s >>> 8) % 3000;
    return q.heightAt(p, t);
  });
  const velocityAtPerSec = bestRate(reps, calls, (i) => { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; return q.velocityAt(p, t, v).y; });
  // Tick cost (60 Hz fixed step), with 100 queries per tick so the lazy blocks are exercised.
  const tickRun = (n: number): { mean: number; max: number } => {
    let max = 0;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      const a = performance.now();
      world.stepOnce();
      max = Math.max(max, performance.now() - a);
      for (let k = 0; k < 100; k++) { p.x = (k * 7.3) % 200; p.z = (k * 3.1) % 200; q.heightAt(p, world.time); }
    }
    return { mean: (performance.now() - t0) / n, max };
  };
  const steady = tickRun(600);
  wind.set(25, 0); // regeneration (amortised) + crossfades for the next ~30 s
  const regen = tickRun(1200);
  return {
    heightAtPerSec, heightAtRandomPerSec, velocityAtPerSec,
    tickMsSteady: steady.mean, tickMsRegen: regen.mean, tickMsMax: Math.max(steady.max, regen.max),
  };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/').replace(/^\//, '')}` || process.argv[1]?.endsWith('bench-queries.ts')) {
  const json = process.argv.includes('--json');
  const r = measure();
  if (json) console.log(JSON.stringify(r));
  else {
    console.log(`heightAt (local area)   ${(r.heightAtPerSec / 1e6).toFixed(2)} M calls/s`);
    console.log(`heightAt (random 3 km)  ${(r.heightAtRandomPerSec / 1e6).toFixed(2)} M calls/s`);
    console.log(`velocityAt              ${(r.velocityAtPerSec / 1e6).toFixed(2)} M calls/s`);
    console.log(`tick (steady, incl. 100 queries)   ${r.tickMsSteady.toFixed(2)} ms`);
    console.log(`tick (wind change: regen+blend)     ${r.tickMsRegen.toFixed(2)} ms`);
    console.log(`tick max                            ${r.tickMsMax.toFixed(2)} ms`);
  }
}
