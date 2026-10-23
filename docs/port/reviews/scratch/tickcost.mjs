// Reviewer scratch: per-tick CPU cost distribution at a 60 Hz fixed step (ocean.tick + the queries gameplay makes).
import { createOcean, WIND_QUERY } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../../../world-ocean/stubs/impl.ts';
const shape = createFlatShape();
function scenario(label, perTick, opts = {}) {
  const world = createStubWorld({ shape });
  const wind = createStubWind(15, -57.29578, shape); world.provide(WIND_QUERY, wind);
  const ocean = createOcean({ preset: opts.preset ?? 'breeze' }); world.use(ocean); world.start();
  const q = ocean.query; const p = { x: 0, y: 0, z: 0 }, v = { x: 0, y: 0, z: 0 };
  for (let i = 0; i < 600; i++) { world.stepOnce(); perTick(q, p, v, world.time, i); }
  if (opts.change) opts.change(ocean, wind);
  const N = opts.ticks ?? 3600; const ms = new Float64Array(N); let sink = 0;
  for (let i = 0; i < N; i++) {
    const a = performance.now();
    world.stepOnce(); sink += perTick(q, p, v, world.time, i);
    ms[i] = performance.now() - a;
  }
  const s = Array.from(ms).sort((x, y) => x - y);
  const mean = s.reduce((x, y) => x + y, 0) / N;
  const over = (th) => s.filter((x) => x > th).length;
  console.log(`${label}: mean ${mean.toFixed(2)} ms, p50 ${s[N >> 1].toFixed(2)}, p95 ${s[Math.floor(N * 0.95)].toFixed(2)}, p99 ${s[Math.floor(N * 0.99)].toFixed(2)}, max ${s[N - 1].toFixed(2)} ms; ticks > 4 ms: ${over(4)}, > 8 ms: ${over(8)}, > 16.7 ms: ${over(16.7)} (of ${N})`);
  return sink;
}
const none = () => 0;
const boats = (q, p, v, t) => { let s = 0; for (let k = 0; k < 100; k++) { p.x = (k * 7.3) % 200; p.z = (k * 3.1) % 200; s += q.heightAt(p, t); } return s; };
const boatsVel = (q, p, v, t) => { let s = boats(q, p, v, t); for (let k = 0; k < 10; k++) { p.x = k * 11; p.z = k * 5; s += q.velocityAt(p, t, v).y; } return s; };
// gameplay at tick time + render-side probe at the interpolated time (half a step back) each tick
const boatsVelRender = (q, p, v, t) => boatsVel(q, p, v, t) + q.heightAt(p, t - 0.5 / 60);
const which = process.argv[2] ?? 'all';
scenario('tick only (steady)', none);
scenario('tick + 100 heightAt', boats);
scenario('tick + 100 heightAt + 10 velocityAt', boatsVel);
scenario('tick + 100 heightAt + 10 velocityAt + 1 heightAt at interpolated time', boatsVelRender);
scenario('tick + 100 heightAt, storm', boats, { preset: 'storm' });
scenario('wind change 15 -> 25 m/s (regen + crossfades), tick + 100 heightAt + 10 velocityAt', boatsVel, { change: (o, w) => w.set(25, 0), ticks: 3600 });
scenario('setPreset calm->storm, tick + 100 heightAt', boats, { preset: 'calm', change: (o) => o.setPreset('storm'), ticks: 1200 });
