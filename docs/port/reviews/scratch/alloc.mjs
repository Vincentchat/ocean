// Reviewer scratch: bytes allocated per call, measured as new-space growth over 1e6 calls with a semi-space
// large enough that no scavenge happens inside the window (run with --expose-gc --max-semi-space-size=256).
import v8 from 'node:v8';
import { createOcean, WIND_QUERY, WATER_QUERY } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../../../world-ocean/stubs/impl.ts';
const gc = globalThis.gc;
const shape = createFlatShape();
const world = createStubWorld({ shape });
const wind = createStubWind(15, -57.3, shape); world.provide(WIND_QUERY, wind);
const ocean = createOcean({ preset: 'breeze' }); world.use(ocean); world.start();
for (let i = 0; i < 300; i++) world.stepOnce();
const q = world.require(WATER_QUERY);
const p = { x: 0, y: 0, z: 0 }, v = { x: 0, y: 0, z: 0 };
const ns = () => v8.getHeapSpaceStatistics().find((s) => s.space_name === 'new_space').space_used_size;
const used = () => { const h = v8.getHeapStatistics(); return h.used_heap_size; };
function measure(label, n, fn) {
  for (let r = 0; r < 3; r++) fn(Math.min(n, 200000)); // warm-up / JIT
  gc(); gc();
  const a = ns(), u0 = used();
  fn(n);
  const b = ns(), u1 = used();
  console.log(`${label}: new-space ${((b - a) / n).toFixed(2)} B/call, total heap ${((u1 - u0) / n).toFixed(2)} B/call (n=${n})`);
}
const t = world.time;
let sink = 0;
// 1) heightAt from a hot loop (caller optimised, call may be inlined)
measure('heightAt, hot loop, result summed', 1e6, (n) => { for (let i = 0; i < n; i++) { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; sink += q.heightAt(p, t); } });
// 2) heightAt through a megamorphic / non-inlinable caller: a different function per object
function viaCall(fnq, pp, tt) { return fnq.heightAt(pp, tt); }
measure('heightAt via helper', 1e6, (n) => { for (let i = 0; i < n; i++) { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; sink += viaCall(q, p, t); } });
// 3) heightAt result stored to a Float64Array (typical buoyancy code)
const store = new Float64Array(1);
measure('heightAt -> Float64Array', 1e6, (n) => { for (let i = 0; i < n; i++) { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; store[0] = q.heightAt(p, t); } });
// 4) velocityAt
measure('velocityAt', 1e6, (n) => { for (let i = 0; i < n; i++) { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; sink += q.velocityAt(p, t, v).y; } });
// 5) ticks (no queries), ticks with 100 heightAt + 4 velocityAt at the tick time (velocity frame per tick)
measure('tick (no queries)', 3000, (n) => { for (let i = 0; i < n; i++) world.stepOnce(); });
measure('tick + 100 heightAt + 4 velocityAt', 3000, (n) => { for (let i = 0; i < n; i++) { world.stepOnce(); for (let k = 0; k < 100; k++) { p.x = k * 3.1; p.z = k * 1.7; sink += q.heightAt(p, world.time); } for (let k = 0; k < 4; k++) sink += q.velocityAt(p, world.time, v).x; } });
// 6) during regeneration + crossfade
wind.set(22, 30);
measure('tick during wind change (regen + crossfade)', 3000, (n) => { for (let i = 0; i < n; i++) world.stepOnce(); });
// 7) queries at time far from the ticks (past / future): synchronous snapshot computation
let tt = 1000;
measure('heightAt at a new far time each call (cache miss)', 200, (n) => { for (let i = 0; i < n; i++) { tt += 7.77; sink += q.heightAt(p, tt); } });
console.log('sink', Number.isFinite(sink));
