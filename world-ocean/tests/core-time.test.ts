// Query-time policy, cost bounds, long times, the batch API, wind reversal and parameter caps.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createOcean, WIND_QUERY } from '../index.ts';
import type { Vec3 } from '../index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../stubs/impl.ts';
import { PRESETS } from '../index.ts';
import { expectedHs } from '../spectrum.ts';

const shape = createFlatShape();

function tickedOcean(opts: Record<string, unknown> = {}) {
  const world = createStubWorld({ shape });
  const ocean = createOcean({ preset: 'windy', ...opts });
  world.use(ocean); world.start();
  for (let i = 0; i < 120; i++) world.stepOnce();
  return { world, ocean };
}

test('query time window: times are clamped to [lastTick - timeWindow, lastTick], never compute snapshots', () => {
  const { world, ocean } = tickedOcean();
  const st: number[] = [];
  const before = ocean.cpuStats(st).slice();
  const p: Vec3 = { x: 12, y: 0, z: -7 };
  const T = world.time, w = ocean.sim.timeWindow;
  const far = ocean.query.heightAt(p, T + 1000);
  assert.equal(far, ocean.query.heightAt(p, T));
  assert.equal(ocean.query.heightAt(p, T - 1e6), ocean.query.heightAt(p, T - w));
  // inside the window: exact and cached (the interpolated render time)
  ocean.query.heightAt(p, T - 0.5 / 60);
  const v = { x: 0, y: 0, z: 0 };
  ocean.query.velocityAt(p, T - 0.5 / 60, v);
  const after = ocean.cpuStats(st);
  for (let c = 0; c < 3; c++) assert.equal(after[5 * c + 1], before[5 * c + 1], `cascade ${c}: no synchronous snapshot`);
});

test('timeWindow = Infinity: any time is exact (synchronous cost), and differs from the clamp', () => {
  const { world, ocean } = tickedOcean({ timeWindow: Infinity });
  const p: Vec3 = { x: 12, y: 0, z: -7 };
  const exact = ocean.query.heightAt(p, world.time + 37.3);
  const ref = createOcean({ shape, preset: 'windy' }); // never ticked: exact everywhere
  assert.equal(exact, ref.query.heightAt(p, world.time + 37.3));
});

test('non-finite times throw RangeError (query and tick)', () => {
  const { world, ocean } = tickedOcean();
  const p: Vec3 = { x: 0, y: 0, z: 0 };
  assert.throws(() => ocean.query.heightAt(p, NaN), RangeError);
  assert.throws(() => ocean.query.heightAt(p, Infinity), RangeError);
  assert.throws(() => ocean.query.velocityAt(p, -Infinity, { x: 0, y: 0, z: 0 }), RangeError);
  assert.throws(() => ocean.tick!('simulate', { dt: 1 / 60, time: NaN, step: 0 }), RangeError);
  assert.ok(Number.isFinite(ocean.query.heightAt(p, world.time)));
});

test('very long times stay exact (no int32 overflow): t and t + 27000 loop periods agree', () => {
  const ocean = createOcean({ shape, preset: 'windy' });
  for (const [e, n] of [[3, 4], [500, -120]]) {
    const a = ocean.heightAtLocal(e, n, 5);
    const b = ocean.heightAtLocal(e, n, 5 + 8192 * 27000); // 2.2e8 s, beyond int32 snapshot indices
    const c = ocean.heightAtLocal(e, n, 5 + 8192 * 1e6);   // 8.2e9 s
    assert.ok(Math.abs(a) < 20 && Math.abs(a - b) < 1e-4 && Math.abs(a - c) < 1e-3, `${a} ${b} ${c}`);
  }
});

test('heightsAt (allocation-free batch) equals heightAt', () => {
  const { world, ocean } = tickedOcean();
  const n = 50, pts = new Float64Array(3 * n), out = new Float64Array(n);
  for (let i = 0; i < n; i++) { pts[3 * i] = i * 13.7 - 300; pts[3 * i + 1] = 2; pts[3 * i + 2] = (i * 7.1) % 200; }
  ocean.heightsAt(pts, world.time, out);
  const p: Vec3 = { x: 0, y: 0, z: 0 };
  for (let i = 0; i < n; i++) {
    p.x = pts[3 * i]; p.y = pts[3 * i + 1]; p.z = pts[3 * i + 2];
    assert.equal(out[i], ocean.query.heightAt(p, world.time));
  }
});

test('lazy velocity: the first velocityAt after a tick fills only the blocks it touches', () => {
  const { world, ocean } = tickedOcean();
  const st: number[] = [];
  const v = { x: 0, y: 0, z: 0 };
  world.stepOnce();
  const a = ocean.cpuStats(st).slice();
  ocean.query.velocityAt({ x: 10, y: 0, z: 20 }, world.time, v);
  const b = ocean.cpuStats(st);
  for (let c = 0; c < 3; c++) {
    assert.ok(b[5 * c + 4] - a[5 * c + 4] <= 4 * 8, `cascade ${c}: ${b[5 * c + 4] - a[5 * c + 4]} velocity blocks`);
    assert.equal(b[5 * c + 1], a[5 * c + 1], 'no synchronous snapshot');
  }
});

/** Realised Hs (4 sigma over a 400-point grid) at the current tick. */
function realisedHs(ocean: ReturnType<typeof createOcean>, t: number, pts: Float64Array, out: Float64Array): number {
  ocean.heightsAt(pts, t, out);
  let s = 0, s2 = 0;
  for (let i = 0; i < out.length; i++) { s += out[i]; s2 += out[i] * out[i]; }
  const m = s / out.length;
  return 4 * Math.sqrt(s2 / out.length - m * m);
}

for (const turn of [180, 90]) {
  test(`wind turning ${turn} deg: the sea turns gradually, Hs stays >= 85 % (no collapse)`, () => {
    const world = createStubWorld({ shape });
    const wind = createStubWind(15, -57.29578, shape);
    world.provide(WIND_QUERY, wind);
    const ocean = createOcean({ preset: 'windy' });
    world.use(ocean); world.start();
    const n = 1600, pts = new Float64Array(3 * n), out = new Float64Array(n);
    for (let i = 0; i < n; i++) { pts[3 * i] = (i % 40) * 26.3; pts[3 * i + 2] = Math.floor(i / 40) * 26.3; }
    for (let i = 0; i < 60; i++) world.stepOnce();
    const hs0 = realisedHs(ocean, world.time, pts, out);
    wind.set(15, -57.29578 + turn);
    let minHs = Infinity, maxStep = 0;
    const st = ocean.getState();
    let lastDir = st.slots[st.toSlot].params.windDirection, versions = st.version;
    for (let i = 0; i < 60 * 60; i++) {
      world.stepOnce();
      if (i % 30 === 0) minHs = Math.min(minHs, realisedHs(ocean, world.time, pts, out));
      const d = st.slots[st.toSlot].params.windDirection;
      let dd = Math.abs(d - lastDir); if (dd > Math.PI) dd = 2 * Math.PI - dd;
      maxStep = Math.max(maxStep, dd); lastDir = d;
    }
    assert.ok(st.version > versions, 'the spectrum followed the wind');
    assert.ok(maxStep <= (ocean.sim.maxTurnPerBlendDeg * Math.PI) / 180 + 1e-9, `spectrum turned ${(maxStep * 180 / Math.PI).toFixed(1)} deg at once`);
    assert.ok(Math.abs(ocean.params.windSpeed - 15) < 0.5, `smoothed wind speed ${ocean.params.windSpeed} (never through 0)`);
    console.log(`  ${turn} deg turn: Hs ${hs0.toFixed(2)} m -> min ${minHs.toFixed(2)} m over 60 s`);
    assert.ok(minHs >= 0.85 * hs0, `Hs dipped to ${(minHs / hs0 * 100).toFixed(0)} %`);
  });
}

test('caps: fetch / wind speed / peak wavelength are bounded (Hs <= ~15 m)', () => {
  const p = { ...PRESETS.storm.params, windSpeed: 60, fetch: 2e6 };
  const hs = expectedHs(p, 9.80665);
  assert.ok(hs < 15.5, `Hs ${hs}`);
});

test('snapshot workers: bit-identical heights with and without workers', () => {
  const a = tickedOcean({ worker: true }), b = tickedOcean({ worker: false });
  for (let i = 0; i < 200; i++) { a.world.stepOnce(); b.world.stepOnce(); }
  const st: number[] = [];
  a.ocean.cpuStats(st);
  console.log(`  worker-computed snapshots: ${st[2] + st[7] + st[12]} of ${st[0] + st[5] + st[10]}`);
  const p: Vec3 = { x: 0, y: 0, z: 0 }, va = { x: 0, y: 0, z: 0 }, vb = { x: 0, y: 0, z: 0 };
  for (let i = 0; i < 200; i++) {
    p.x = i * 17.3 - 1500; p.z = (i * 7.9) % 900;
    assert.equal(a.ocean.query.heightAt(p, a.world.time), b.ocean.query.heightAt(p, b.world.time));
    a.ocean.query.velocityAt(p, a.world.time, va); b.ocean.query.velocityAt(p, b.world.time, vb);
    assert.ok(va.x === vb.x && va.y === vb.y && va.z === vb.z);
  }
  a.ocean.dispose!(); b.ocean.dispose!();
});