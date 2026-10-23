// Reviewer scratch: attribute per-tick / per-query allocations to source lines (V8 sampling heap profiler).
import inspector from 'node:inspector/promises';
import { createOcean, WIND_QUERY } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../../../world-ocean/stubs/impl.ts';
const shape = createFlatShape();
const world = createStubWorld({ shape });
const wind = createStubWind(15, -57.3, shape); world.provide(WIND_QUERY, wind);
const ocean = createOcean({ preset: 'breeze' }); world.use(ocean); world.start();
const info = { dt: 1 / 60, time: 0, step: 0 };
let step = 0;
function tick() { step++; info.step = step; info.time = step / 60; ocean.tick('simulate', info); }
const q = ocean.query; const p = { x: 0, y: 0, z: 0 };
const mode = process.argv[2] ?? 'tick';
function work(n) {
  let s = 0;
  if (mode === 'tick') for (let i = 0; i < n; i++) tick();
  else { const t = info.time; for (let i = 0; i < n; i++) { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; s += q.heightAt(p, t); } }
  return s;
}
for (let i = 0; i < 2000; i++) tick();
work(mode === 'tick' ? 3000 : 300000);
const session = new inspector.Session(); session.connect();
await session.post('HeapProfiler.enable');
await session.post('HeapProfiler.startSampling', { samplingInterval: 32, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
work(mode === 'tick' ? 6000 : 1000000);
const { profile } = await session.post('HeapProfiler.stopSampling');
const agg = new Map();
(function walk(node, stack) {
  const cf = node.callFrame;
  const name = `${cf.functionName || '(anon)'} ${cf.url.split('/').pop()}:${cf.lineNumber + 1}`;
  const here = stack.concat(name);
  if (node.selfSize > 0) { const k = here.slice(-3).join(' <- '); agg.set(k, (agg.get(k) ?? 0) + node.selfSize); }
  for (const c of node.children) walk(c, here);
})(profile.head, []);
const total = [...agg.values()].reduce((a, b) => a + b, 0);
console.log(`mode ${mode}: sampled total ${total} B`);
for (const [k, v] of [...agg.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12)) console.log(`  ${v} B  ${k}`);
