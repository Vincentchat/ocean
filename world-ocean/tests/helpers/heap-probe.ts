// Run by tests/core-heap.test.ts in a child process with --expose-gc and a PINNED semi-space
// (--min-semi-space-size=64 --max-semi-space-size=64: no scavenge inside a measurement window, so
// the growth of V8's new space between two explicit GCs is exactly the bytes allocated).
// 1) Heap growth over 1e3 ticks + 1e5 queries incl. a spectrum regeneration + crossfade.
// 2) Bytes allocated per tick / per query. Prints one JSON line.
import v8 from 'node:v8';
import { createOcean, WIND_QUERY } from '../../index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../stubs/impl.ts';

const gc = (globalThis as { gc?: () => void }).gc;
if (!gc) throw new Error('run with --expose-gc');

const shape = createFlatShape();
const world = createStubWorld({ shape });
const wind = createStubWind(15, -57.3, shape);
world.provide(WIND_QUERY, wind);
const ocean = createOcean({ preset: 'windy', windTimeConstant: 2, blendTime: 2 });
world.use(ocean);
world.start();
const q = ocean.query;
const p = { x: 0, y: 0, z: 0 };
const v = { x: 0, y: 0, z: 0 };
const pts = new Float64Array(3 * 100), hs = new Float64Array(100);
for (let k = 0; k < 100; k++) { pts[3 * k] = (k * 7.3) % 200; pts[3 * k + 2] = (k * 3.1) % 200; }

function run(ticks: number, phase: number): number {
  let acc = 0;
  for (let i = 0; i < ticks; i++) {
    if (i === 100) wind.set(phase ? 22 : 18, phase ? 30 : -20); // regeneration + crossfade inside the loop
    world.stepOnce();
    ocean.heightsAt(pts, world.time, hs);
    acc += hs[7];
    for (let k = 0; k < 10; k++) { p.x = k * 11; p.z = k * 5; acc += q.velocityAt(p, world.time, v).y; }
  }
  return acc;
}

// Warm-up (JIT, pools, every code path incl. a regeneration), then measure.
run(1000, 0); run(600, 1); for (let r = 0; r < Number(process.env.WARM ?? 0); r++) run(1000, r & 1);
gc(); gc();
const before = process.memoryUsage().heapUsed;
const acc = run(1000, 1);
gc(); gc();
const after = process.memoryUsage().heapUsed;

const newSpace = () => (v8.getHeapSpaceStatistics().find((s) => s.space_name === 'new_space') as { space_used_size: number }).space_used_size;
function bytesPer(n: number, fn: (i: number) => void): number {
  for (let i = 0; i < n; i++) fn(i);
  let best = Infinity;
  for (let r = 0; r < 3; r++) {
    gc();
    const a = newSpace();
    for (let i = 0; i < n; i++) fn(i);
    const d = newSpace() - a;
    if (d >= 0) best = Math.min(best, d / n);
  }
  return best;
}
const t = world.time;
const perHeight = bytesPer(20000, (i) => { p.x = (i * 37) % 997; p.z = (i * 11) % 911; q.heightAt(p, t); });
const perVelocity = bytesPer(20000, (i) => { p.x = (i * 37) % 997; p.z = (i * 11) % 911; q.velocityAt(p, t, v); });
const perBatchPoint = bytesPer(200, () => { ocean.heightsAt(pts, t, hs); }) / 100;
// Ticks are driven through the stub world (as in the engine); the stub world's own step allocates a
// boxed double (its time += fixedDt), measured separately and subtracted. Queries take world.time
// (an already boxed number: passing it allocates nothing).
const empty = createStubWorld({ shape });
const perStubStep = bytesPer(600, () => empty.stepOnce());
const perTickRaw = bytesPer(600, () => world.stepOnce());
const perTickLoadRaw = bytesPer(600, () => {
  world.stepOnce();
  ocean.heightsAt(pts, world.time, hs);
  for (let k = 0; k < 10; k++) { p.x = k * 11; p.z = k * 5; q.velocityAt(p, world.time, v); }
});
const perTick = perTickRaw - perStubStep, perTickLoad = perTickLoadRaw - perStubStep;
console.log(JSON.stringify({
  before, after, growth: after - before, acc: Number.isFinite(acc), version: ocean.getState().version,
  perHeight, perVelocity, perBatchPoint, perTick, perTickLoad, perStubStep,
}));
