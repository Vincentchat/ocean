// Run by tests/core-heap.test.ts in a child process with --expose-gc.
// 1) Heap growth over 1e3 ticks + 1e5 queries (heightAt + velocityAt), including a wind change
//    (spectrum regeneration + crossfade).
// 2) Bytes allocated per query / per tick, measured as the growth of V8's new space between two
//    explicit GCs (exact as long as no scavenge happens in between; the windows are small).
// Prints one JSON line.
import v8 from 'node:v8';
import { createOcean, WIND_QUERY } from '../../index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../stubs/impl.ts';

const gc = (globalThis as { gc?: () => void }).gc;
if (!gc) throw new Error('run with --expose-gc');

const shape = createFlatShape();
const world = createStubWorld({ shape });
const wind = createStubWind(15, -57.3, shape);
world.provide(WIND_QUERY, wind);
const ocean = createOcean({ preset: 'breeze', windTimeConstant: 2, blendTime: 2 });
world.use(ocean);
world.start();
const q = ocean.query;
const p = { x: 0, y: 0, z: 0 };
const v = { x: 0, y: 0, z: 0 };

function run(ticks: number, queriesPerTick: number, phase: number): number {
  let acc = 0;
  for (let i = 0; i < ticks; i++) {
    if (i === 100) wind.set(phase ? 22 : 18, phase ? 30 : -20); // regeneration + crossfade inside the loop
    world.stepOnce();
    for (let k = 0; k < queriesPerTick; k++) {
      p.x = ((i * 37 + k * 13) % 997) - 400; p.z = ((i * 11 + k * 29) % 911) - 300;
      acc += q.heightAt(p, world.time);
      if ((k & 7) === 0) acc += q.velocityAt(p, world.time, v).y;
    }
  }
  return acc;
}

// Warm-up (JIT, pools, every code path incl. a regeneration), then measure.
run(1000, 100, 0);
gc(); gc();
const before = process.memoryUsage().heapUsed;
const acc = run(1000, 100, 1);
gc(); gc();
const after = process.memoryUsage().heapUsed;

const newSpace = () => (v8.getHeapSpaceStatistics().find((s) => s.space_name === 'new_space') as v8.HeapSpaceInfo).space_used_size;
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
const perTick = bytesPer(300, () => world.stepOnce());
const perStubTick = (() => { const w2 = createStubWorld({ shape }); return bytesPer(300, () => w2.stepOnce()); })();
console.log(JSON.stringify({
  before, after, growth: after - before, acc: Number.isFinite(acc), version: ocean.getState().version,
  perHeight, perVelocity, perTick, perStubTick,
}));
