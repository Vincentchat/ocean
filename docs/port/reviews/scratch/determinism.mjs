// Reviewer scratch: determinism independent of query history; past/future queries; velocity during a crossfade.
import { createOcean, WIND_QUERY } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../../../world-ocean/stubs/impl.ts';
const shape = createFlatShape();
function make() {
  const world = createStubWorld({ shape });
  const wind = createStubWind(15, -57.29578, shape); world.provide(WIND_QUERY, wind);
  const ocean = createOcean({ preset: 'breeze', seed: 1 }); world.use(ocean); world.start();
  return { world, wind, ocean };
}
const A = make(), B = make();
const p = { x: 0, y: 0, z: 0 }, v = { x: 0, y: 0, z: 0 };
let rs = 99;
const rnd = () => { rs = (Math.imul(rs, 1664525) + 1013904223) >>> 0; return rs / 4294967296; };
const recorded = []; // heights recorded in A "live" at some tick times
for (let i = 0; i < 60 * 40; i++) {
  if (i === 600) { A.wind.set(22, 30); B.wind.set(22, 30); }
  if (i === 1500) { A.ocean.setPreset('storm'); B.ocean.setPreset('storm'); }
  A.world.stepOnce(); B.world.stepOnce();
  // B gets a hostile query history: random past / future times, velocities, far points
  if (i % 20 === 0) for (let k = 0; k < 2; k++) {
    p.x = (rnd() - 0.5) * 2e5; p.z = (rnd() - 0.5) * 2e5;
    const tt = B.world.time + (rnd() - 0.5) * 400;
    B.ocean.query.heightAt(p, tt); if (k === 0) B.ocean.query.velocityAt(p, tt, v);
  }
  if (i % 300 === 150) { p.x = 123.4; p.z = -56.7; recorded.push([A.world.time, A.ocean.query.heightAt(p, A.world.time)]); }
}
// Compare A vs B at the current time and at nearby past/future times
let maxd = 0, cnt = 0;
for (const dt of [0, -0.37, -3.1, 2.2, 15.5]) {
  for (let k = 0; k < 200; k++) {
    p.x = ((k * 0.618) % 1) * 3000; p.z = ((k * 0.7548) % 1) * 3000;
    const t = A.world.time + dt;
    const a = A.ocean.query.heightAt(p, t), b = B.ocean.query.heightAt(p, t);
    maxd = Math.max(maxd, Math.abs(a - b)); cnt++;
    A.ocean.query.velocityAt(p, t, v); const va = v.y; B.ocean.query.velocityAt(p, t, v);
    maxd = Math.max(maxd, Math.abs(va - v.y));
  }
}
console.log(`A (clean history) vs B (hostile random past/future/far queries): max |difference| = ${maxd} over ${cnt} points x 5 times`);
console.log(`A === B state versions: ${A.ocean.getState().version} / ${B.ocean.getState().version}, blendStart ${A.ocean.getState().blendStart} / ${B.ocean.getState().blendStart}`);
// Past queries after spectrum changes: does heightAt(p, pastTime) still return what was live at that time?
p.x = 123.4; p.z = -56.7;
for (const [t, h] of recorded) {
  const now = A.ocean.query.heightAt(p, t);
  console.log(`  t=${t.toFixed(2)} s: live ${h.toFixed(4)} m, re-queried later ${now.toFixed(4)} m ${Math.abs(now - h) > 1e-6 ? '<-- history not kept' : ''}`);
}
// Velocity check by finite differences of the rest point's trajectory, during an active crossfade with choppiness change
{
  const world = createStubWorld({ shape });
  const ocean = createOcean({ preset: 'calm' }); world.use(ocean); world.start();
  for (let i = 0; i < 60; i++) world.stepOnce();
  ocean.setPreset('storm');
  let t = 0;
  for (let i = 0; i < 60 * 12; i++) { world.stepOnce(); const st = ocean.getState(); if (st.fromSlot !== st.toSlot && ocean.blendAt(world.time) > 0.3 && ocean.blendAt(world.time) < 0.7) { t = world.time; break; } }
  const r = { east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 }, a = { ...r }, b = { ...r };
  let maxErr = 0, maxV = 0;
  for (let k = 0; k < 200; k++) {
    const pe = ((k * 0.618) % 1) * 1000, pn = ((k * 0.7548) % 1) * 1000;
    ocean.invertLocal(pe, pn, t, r);
    const h = 1e-3;
    ocean.sampleRest(r.east, r.north, t + h, a); ocean.sampleRest(r.east, r.north, t - h, b);
    p.x = pe; p.z = -pn; ocean.query.velocityAt(p, t, v);
    const fe = (a.east - b.east) / (2 * h), fn = (a.north - b.north) / (2 * h), fu = (a.up - b.up) / (2 * h);
    maxErr = Math.max(maxErr, Math.abs(fe - v.x), Math.abs(fn + v.z), Math.abs(fu - v.y));
    maxV = Math.max(maxV, Math.hypot(v.x, v.y, v.z));
  }
  console.log(`velocityAt during calm->storm crossfade (blend ${ocean.blendAt(t).toFixed(2)}, chop ${ocean.choppinessAt(t).toFixed(2)}): max |v - FD| ${maxErr.toFixed(4)} m/s, max |v| ${maxV.toFixed(2)} m/s`);
}
// Robustness: NaN / Infinity / negative / very large times
{
  const o = createOcean({ shape });
  for (const t of [NaN, Infinity, -1e3, 1e7, 2.2e8, 3e8]) {
    let r; try { r = o.query.heightAt({ x: 1, y: 0, z: 2 }, t); } catch (e) { r = 'throws ' + e.message; }
    console.log(`  heightAt(t=${t}) = ${r}`);
  }
  console.log(`  heightAt(t=5) vs heightAt(t=5+8192*27000): ${o.query.heightAt({ x: 1, y: 0, z: 2 }, 5)} / ${o.query.heightAt({ x: 1, y: 0, z: 2 }, 5 + 8192 * 27000)}`);
}

