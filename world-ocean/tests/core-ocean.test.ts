// world/ocean system: determinism, presets (Hs), wind coupling + crossfade (no pop), rotated up,
// velocity, arbitrary times, WorldSystem wiring.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createOcean, WATER_QUERY, WIND_QUERY, PRESETS, PRESET_NAMES } from '../index.ts';
import type { RestSample, Vec3 } from '../index.ts';
import {
  createFlatShape, createRotatedFlatShape, quatFromAxisAngle, createStubWorld, createStubWind,
} from '../stubs/impl.ts';

const shape = createFlatShape();
const rest = (): RestSample => ({ east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 });

function probes(count: number, seed: number, span: number): Float64Array {
  let s = seed >>> 0;
  const out = new Float64Array(2 * count);
  for (let i = 0; i < out.length; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; out[i] = (s / 4294967296 - 0.5) * span; }
  return out;
}

test('WorldSystem shape: id, provides WATER_QUERY, optional keys, simulate phase; install provides the query', () => {
  const ocean = createOcean({ preset: 'windy', seed: 1 });
  assert.equal(ocean.id, 'world/ocean');
  assert.deepEqual(ocean.provides!.map((k) => k.id), ['WATER_QUERY']);
  assert.deepEqual(ocean.optional!.map((k) => k.id), ['WIND_QUERY', 'ENVIRONMENT', 'TERRAIN_QUERY', 'TERRAIN_GPU', 'SKY_LIGHT']);
  assert.deepEqual(ocean.phases, ['simulate']);
  const world = createStubWorld({ shape });
  world.use(ocean); world.start();
  const q = world.require(WATER_QUERY);
  assert.equal(q, ocean.query);
  assert.ok(Number.isFinite(q.heightAt({ x: 10, y: 0, z: 5 }, world.time)));
});

test('determinism: same seed + same time -> identical heights, independent of history; other seed differs', () => {
  const a = createOcean({ shape, seed: 1 });
  const b = createOcean({ shape, seed: 1 });
  const c = createOcean({ shape, seed: 2 });
  const world = createStubWorld({ shape });
  world.use(a); world.start();
  for (let i = 0; i < 90; i++) world.stepOnce(); // a has ticked (prefetched snapshots), b never has
  const t = world.time;
  const pts = probes(200, 7, 3000);
  const p: Vec3 = { x: 0, y: 0, z: 0 };
  const v = { x: 0, y: 0, z: 0 };
  b.query.velocityAt(p, t, v); // different query history in b (velocity frame first)
  let diffC = 0;
  for (let i = 0; i < pts.length; i += 2) {
    p.x = pts[i]; p.z = pts[i + 1];
    const ha = a.query.heightAt(p, t), hb = b.query.heightAt(p, t), hc = c.query.heightAt(p, t);
    assert.equal(ha, hb, `point ${i / 2}`);
    diffC = Math.max(diffC, Math.abs(ha - hc));
  }
  assert.ok(diffC > 0.1, 'a different seed must give a different sea');
  // An older time again (cache miss, recomputed) gives the same numbers as before.
  const h0 = a.query.heightAt({ x: 3, y: 0, z: 4 }, 0.5);
  a.query.heightAt({ x: 3, y: 0, z: 4 }, 7.25);
  a.query.heightAt({ x: 3, y: 0, z: 4 }, 9.5);
  assert.equal(a.query.heightAt({ x: 3, y: 0, z: 4 }, 0.5), h0);
});

test('presets: realised Hs (4 sigma of heightAt over a 2 km grid) within 15 % of the target table', () => {
  let prev = 0;
  for (const name of PRESET_NAMES) {
    const ocean = createOcean({ shape, preset: name });
    let s = 0, s2 = 0, n = 0;
    for (let i = 0; i < 4000; i++) {
      const e = ((i * 0.6180339887) % 1) * 2048, nn = ((i * 0.7548776662) % 1) * 2048;
      const h = ocean.heightAtLocal(e, nn, 20);
      s += h; s2 += h * h; n++;
    }
    const sigma = Math.sqrt(s2 / n - (s / n) * (s / n));
    const hs = 4 * sigma;
    const target = PRESETS[name].target.hs;
    assert.ok(Math.abs(hs - target) / target < 0.15, `${name}: realised Hs ${hs.toFixed(2)} m vs target ${target} m`);
    assert.ok(Math.abs(ocean.expectedHs() - target) / target < 0.02);
    assert.ok(hs > prev, `${name} rougher than the previous preset`);
    prev = hs;
  }
});

test('wind coupling: WIND_QUERY drives the spectrum, regenerates + crossfades with no pop', () => {
  const world = createStubWorld({ shape });
  const wind = createStubWind(15, -57.29578, shape); // = the breeze preset's wind (-1 rad)
  world.provide(WIND_QUERY, wind);
  const ocean = createOcean({ preset: 'windy', blendTime: 4, windTimeConstant: 4 });
  world.use(ocean); world.start();
  const st = ocean.getState();
  const v0 = st.version;
  for (let i = 0; i < 120; i++) world.stepOnce();
  assert.equal(st.version, v0, 'matching wind must not regenerate the spectrum');
  const hs0 = ocean.expectedHs();
  const h0Before = st.slots[st.toSlot].h0[0].slice();

  const pts = probes(48, 11, 1500);
  const last = new Float64Array(pts.length / 2);
  const p: Vec3 = { x: 0, y: 0, z: 0 };
  // Track fixed REST points (sampleRest: the surface point itself), so the measure is the sea's motion
  // and the crossfade, not the (separate, documented) sheet choice of heightAt where crests fold.
  const rs = { east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 };
  const sample = (out: Float64Array) => {
    for (let i = 0; i < out.length; i++) { out[i] = ocean.sampleRest(pts[2 * i], pts[2 * i + 1], world.time, rs).up; }
  };
  const cur = new Float64Array(last.length);
  // Natural per-tick change before the wind changes (reference for "no pop").
  sample(last);
  let steadyMax = 0;
  for (let i = 0; i < 240; i++) {
    world.stepOnce(); sample(cur);
    for (let k = 0; k < cur.length; k++) steadyMax = Math.max(steadyMax, Math.abs(cur[k] - last[k]));
    last.set(cur);
  }
  // Wind rises to 25 m/s and turns towards east.
  wind.set(25, 0);
  let crossMax = 0, sawBlend = false, minBlendSeen = 1;
  for (let i = 0; i < 60 * 40; i++) {
    world.stepOnce(); sample(cur);
    for (let k = 0; k < cur.length; k++) crossMax = Math.max(crossMax, Math.abs(cur[k] - last[k]));
    last.set(cur);
    if (st.fromSlot !== st.toSlot) { sawBlend = true; minBlendSeen = Math.min(minBlendSeen, st.blend); }
  }
  assert.ok(sawBlend, 'a crossfade happened');
  assert.ok(minBlendSeen < 0.1, 'crossfades start from weight ~0');
  assert.ok(st.version > v0, 'spectrum regenerated (GPU must re-upload)');
  assert.ok(ocean.params.windSpeed > 20, `smoothed wind ${ocean.params.windSpeed}`);
  assert.ok(Math.abs(ocean.params.windDirection) < 0.2, 'direction follows the wind');
  assert.ok(ocean.expectedHs() > hs0 * 1.3, 'stronger wind -> higher waves');
  assert.notDeepEqual(st.slots[st.toSlot].h0[0], h0Before, 'h0 changed');
  // No pop: the largest per-tick change while crossfading stays within 2x the largest natural
  // per-tick change of the (rougher) final sea. An instant switch would be ~Hs-sized.
  let finalMax = 0;
  for (let i = 0; i < 240; i++) {
    world.stepOnce(); sample(cur);
    for (let k = 0; k < cur.length; k++) finalMax = Math.max(finalMax, Math.abs(cur[k] - last[k]));
    last.set(cur);
  }
  const bound = 2 * Math.max(steadyMax, finalMax);
  assert.ok(crossMax <= bound, `max per-tick change ${crossMax.toFixed(4)} m > ${bound.toFixed(4)} m`);
  assert.ok(crossMax < 0.05 * ocean.expectedHs(), `per-tick change ${crossMax} is a pop`);
});

test('setPreset crossfades (calm -> storm) and immediate switches at once', () => {
  const world = createStubWorld({ shape });
  const ocean = createOcean({ preset: 'calm' });
  world.use(ocean); world.start();
  for (let i = 0; i < 30; i++) world.stepOnce();
  const hsCalm = ocean.expectedHs();
  ocean.setPreset('storm');
  for (let i = 0; i < 60 * 12; i++) world.stepOnce();
  const st = ocean.getState();
  assert.equal(st.blend, 1, 'crossfade finished');
  assert.equal(st.slots[st.toSlot].params.windSpeed, PRESETS.storm.params.windSpeed);
  assert.ok(ocean.expectedHs() > 5 * hsCalm);
  ocean.setPreset('windy', true);
  assert.equal(ocean.getState().slots[ocean.getState().toSlot].params.windSpeed, 15);
});

test('rotated-up world: identical results in local terms (no hardcoded +Y)', () => {
  const q = quatFromAxisAngle({ x: 1, y: 0.3, z: -0.4 }, 1.1);
  const pivot = { x: 1234.5, y: -50, z: 321 };
  const rot = createRotatedFlatShape(q, { pivot });
  const flat = createOcean({ shape, seed: 3, preset: 'windy' });
  const tilted = createOcean({ shape: rot, seed: 3, preset: 'windy', anchor: pivot });
  const t = 4.2;
  const pts = probes(100, 5, 800);
  const local: Vec3 = { x: 0, y: 0, z: 0 }, world: Vec3 = { x: 0, y: 0, z: 0 };
  const vf = { x: 0, y: 0, z: 0 }, vr = { x: 0, y: 0, z: 0 };
  const fr = { east: { x: 0, y: 0, z: 0 }, north: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 0 } };
  rot.tangentFrame(pivot, fr);
  for (let i = 0; i < pts.length; i += 2) {
    // canonical flat coordinates: x = east, y = altitude, z = -north
    local.x = pts[i]; local.y = 1.5; local.z = -pts[i + 1];
    rot.localToWorld(local, world);
    const hf = flat.query.heightAt(local, t), hr = tilted.query.heightAt(world, t);
    assert.ok(Math.abs(hf - hr) < 1e-6, `height ${hf} vs ${hr}`);
    flat.query.velocityAt(local, t, vf);
    tilted.query.velocityAt(world, t, vr);
    const ve = vr.x * fr.east.x + vr.y * fr.east.y + vr.z * fr.east.z;
    const vn = vr.x * fr.north.x + vr.y * fr.north.y + vr.z * fr.north.z;
    const vu = vr.x * fr.up.x + vr.y * fr.up.y + vr.z * fr.up.z;
    assert.ok(Math.abs(ve - vf.x) < 1e-6 && Math.abs(vn + vf.z) < 1e-6 && Math.abs(vu - vf.y) < 1e-6, 'velocity in local terms');
  }
});

test('far from the anchor: the surface repeats every 9216 m (lcm of the tiles), precision kept at 100 km', () => {
  const ocean = createOcean({ shape });
  const t = 3;
  for (const [e, n] of [[12.3, -40.1], [700.5, 333.25]]) {
    const h = ocean.heightAtLocal(e, n, t);
    const far = ocean.heightAtLocal(e + 9216 * 11, n - 9216 * 11, t); // ~101 km away
    assert.ok(Math.abs(h - far) < 1e-6, `${h} vs ${far}`);
  }
});

test('velocityAt = time derivative of the displaced surface point (analytic vs finite difference)', () => {
  const ocean = createOcean({ shape, preset: 'windy' });
  const a = rest(), b = rest(), c = rest();
  const pts = probes(60, 9, 1000);
  const t = 7.3, h = 1e-3;
  let maxErr = 0, maxV = 0;
  for (let i = 0; i < pts.length; i += 2) {
    ocean.sampleRest(pts[i], pts[i + 1], t, a);
    ocean.sampleRest(pts[i], pts[i + 1], t - h, b);
    ocean.sampleRest(pts[i], pts[i + 1], t + h, c);
    const fe = (c.east - b.east) / (2 * h), fn = (c.north - b.north) / (2 * h), fu = (c.up - b.up) / (2 * h);
    maxErr = Math.max(maxErr, Math.abs(fe - a.vEast), Math.abs(fn - a.vNorth), Math.abs(fu - a.vUp));
    maxV = Math.max(maxV, Math.hypot(a.vEast, a.vNorth, a.vUp));
  }
  assert.ok(maxV > 0.5, `waves move (${maxV} m/s)`);
  assert.ok(maxErr < 0.01 * maxV, `velocity error ${maxErr} m/s`);
  // velocityAt(p) is the velocity of the point found by heightAt's inversion
  const p = { x: 0, y: 0, z: 0 }, v = { x: 0, y: 0, z: 0 }, r = rest();
  ocean.sampleRest(100, 200, t, a);
  p.x = a.east; p.z = -a.north;
  ocean.query.velocityAt(p, t, v);
  ocean.invertLocal(a.east, a.north, t, r);
  // (the inversion stops at a 4 mm residual: velocities agree to ~|grad v| x 4 mm)
  assert.ok(Math.abs(v.y - a.vUp) < 0.03 && Math.abs(v.x - a.vEast) < 0.03 && Math.abs(-v.z - a.vNorth) < 0.03, `,, vs ,,`);
});

test('heights at arbitrary times: Hermite between snapshots matches exact evaluation (< 0.5 % Hs)', () => {
  const t = 13.37;
  const herm = createOcean({ shape });
  // snapshotDt = t puts t exactly on a snapshot: exact spectral evaluation
  const exact = createOcean({ shape, query: [
    { size: 256, snapshotDt: t, filter: 'cubic' }, { size: 256, snapshotDt: t, filter: 'cubic' }, { size: 128, snapshotDt: t, filter: 'cubic' }] });
  const a = rest(), b = rest();
  const pts = probes(500, 21, 2000);
  let mx = 0;
  for (let i = 0; i < pts.length; i += 2) {
    herm.sampleRest(pts[i], pts[i + 1], t, a); exact.sampleRest(pts[i], pts[i + 1], t, b);
    mx = Math.max(mx, Math.abs(a.up - b.up), Math.abs(a.east - b.east), Math.abs(a.north - b.north));
  }
  assert.ok(mx < 0.005 * exact.expectedHs(), `Hermite error ${mx} m`);
});

test('loop period: the sea repeats exactly after 8192 s', () => {
  const ocean = createOcean({ shape });
  for (const [e, n] of [[1, 2], [345, -77]]) {
    const a = ocean.heightAtLocal(e, n, 2.4);
    const b = ocean.heightAtLocal(e, n, 2.4 + 8192);
    assert.ok(Math.abs(a - b) < 1e-5, `${a} vs ${b}`);
  }
});

test('focus point: wind is sampled 10 m above sea level at the focus', () => {
  const world = createStubWorld({ shape });
  const seen: Vec3[] = [];
  world.provide(WIND_QUERY, { at(p, out) { if (seen.length < 4) seen.push({ x: p.x, y: p.y, z: p.z }); out.x = 15 * Math.cos(-1); out.y = 0; out.z = -15 * Math.sin(-1); return out; } });
  const ocean = createOcean({});
  ocean.setFocus({ x: 500, y: 30, z: -20 });
  world.use(ocean); world.start();
  world.stepOnce();
  assert.ok(seen.length > 0);
  for (const s of seen) { assert.equal(s.x, 500); assert.equal(s.y, 10); assert.equal(s.z, -20); }
  assert.ok(Math.abs(ocean.params.windDirection + 1) < 1e-9, 'wind (cos -1, sin -1) in east/north = -1 rad');
});

test('perf sink receives begin/end markers around the simulate tick', () => {
  const calls: string[] = [];
  const world = createStubWorld({ shape });
  const ocean = createOcean({ perf: { begin(m, p) { calls.push(`b:${m}:${p}`); }, end() { calls.push('e'); } } });
  world.use(ocean); world.start();
  world.stepOnce(); world.stepOnce();
  assert.deepEqual(calls, ['b:world/ocean:sim', 'e', 'b:world/ocean:sim', 'e']);
});
