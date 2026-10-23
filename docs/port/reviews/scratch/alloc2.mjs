// Reviewer scratch: cleaner allocation measurement (local accumulators, ocean.tick called directly).
import v8 from 'node:v8';
import { createOcean, WIND_QUERY } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../../../world-ocean/stubs/impl.ts';
const gc = globalThis.gc;
const shape = createFlatShape();
const world = createStubWorld({ shape });
const wind = createStubWind(15, -57.3, shape); world.provide(WIND_QUERY, wind);
const ocean = createOcean({ preset: 'breeze' }); world.use(ocean); world.start();
const info = { dt: 1 / 60, time: 0, step: 0 };
let step = 0;
function tick() { step++; info.step = step; info.time = step / 60; ocean.tick('simulate', info); }
for (let i = 0; i < 300; i++) tick();
const q = ocean.query;
const p = { x: 0, y: 0, z: 0 }, v = { x: 0, y: 0, z: 0 };
const ns = () => v8.getHeapSpaceStatistics().find((s) => s.space_name === 'new_space').space_used_size;
const out = new Float64Array(1);
function measure(label, n, fn) {
  for (let r = 0; r < 3; r++) out[0] += fn(Math.min(n, 100000));
  gc(); gc();
  const a = ns();
  out[0] += fn(n);
  const b = ns();
  console.log(`${label}: ${((b - a) / n).toFixed(2)} B/call (n=${n})`);
}
function hLoop(n) { const t = info.time; let s = 0; for (let i = 0; i < n; i++) { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; s += q.heightAt(p, t); } return s; }
function vLoop(n) { const t = info.time; let s = 0; for (let i = 0; i < n; i++) { p.x = (i * 0.37) % 500; p.z = (i * 0.11) % 300; s += q.velocityAt(p, t, v).y; } return s; }
function tLoop(n) { for (let i = 0; i < n; i++) tick(); return 0; }
function tqLoop(n) { let s = 0; for (let i = 0; i < n; i++) { tick(); const t = info.time; for (let k = 0; k < 100; k++) { p.x = k * 3.1; p.z = k * 1.7; s += q.heightAt(p, t); } for (let k = 0; k < 4; k++) s += q.velocityAt(p, t, v).x; } return s; }
// interpolated render time between ticks too (2 times per tick, like gameplay at tick time + render at interpolatedTime)
function tqiLoop(n) { let s = 0; for (let i = 0; i < n; i++) { tick(); const t = info.time; for (let k = 0; k < 50; k++) { p.x = k * 3.1; p.z = k * 1.7; s += q.heightAt(p, t); s += q.heightAt(p, t - 0.5 / 60); } } return s; }
measure('heightAt (local accumulator)', 1e6, hLoop);
measure('velocityAt', 1e6, vLoop);
measure('ocean.tick alone (steady)', 3000, tLoop);
measure('tick + 100 heightAt + 4 velocityAt', 3000, tqLoop);
measure('tick + 50 heightAt@t + 50 heightAt@t-dt/2', 3000, tqiLoop);
let far = 1000;
function farLoop(n) { let s = 0; for (let i = 0; i < n; i++) { far += 7.77; s += q.heightAt(p, far); } return s; }
measure('heightAt at a new far time (synchronous snapshots)', 100, farLoop);
