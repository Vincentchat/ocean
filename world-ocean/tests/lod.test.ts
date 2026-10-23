// LOD pure-math tests (three/lod-math.ts: no three import).
//   node --experimental-strip-types --test tests/lod.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  LOD_KIND_GRID, LOD_TIER_TABLE, buildLodGeometry, createLodState, emulateVertex, lodDrawCount, lodGeometrySteps, lodParams,
  tileAnchor, updateLodState, wrapPeriod, type EmulatedVertex, type LodGeometryData, type LodParams, type LodState,
  type LodTier,
} from '../three/lod-math.ts';
import { createFloatingOrigin } from '../stubs/impl.ts';

const TIERS = Object.keys(LOD_TIER_TABLE) as LodTier[];
const geoms = new Map<LodTier, { p: LodParams; g: LodGeometryData }>();
function geom(tier: LodTier) {
  let e = geoms.get(tier);
  if (!e) { const p = lodParams(tier); e = { p, g: buildLodGeometry(p) }; geoms.set(tier, e); }
  return e;
}

// Deterministic PRNG for the random feet.
let seed = 12345;
const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);

test('geometry: counts, index range, draw ranges', () => {
  for (const tier of TIERS) {
    const { p, g } = geom(tier);
    assert.equal(g.index.length, g.triangleCount * 3);
    let max = 0;
    for (let i = 0; i < g.index.length; i++) if (g.index[i] > max) max = g.index[i];
    assert.ok(max < g.vertexCount, `${tier}: index out of range`);
    assert.equal(g.levelIndexEnd.length, p.levels);
    for (let l = 1; l < p.levels; l++) assert.ok(g.levelIndexEnd[l] > g.levelIndexEnd[l - 1]);
    // Drawn vertices at the reference default camera height (4.8 m).
    const st = createLodState();
    updateLodState(st, p, 0, 8, 4.8);
    const drawnVerts = 3 * 8 * (p.halfCells + p.margin) + g.levelVertexCount.slice(0, st.drawnLevels).reduce((a, b) => a + b, 0);
    console.log(`  ${tier}: M=${p.halfCells} levels=${p.levels} vertices=${g.vertexCount} triangles=${g.triangleCount}` +
      ` | h=4.8 m: s0=${st.spacing0} m, drawn levels=${st.drawnLevels}, drawn vertices~${drawnVerts}, drawn triangles=${lodDrawCount(st, g) / 3}`);
  }
});

test('tier change: time-sliced builder == one-shot builder, and no slice is long', () => {
  for (const tier of TIERS) {
    const p = lodParams(tier);
    const ref = geom(tier).g;
    const it = lodGeometrySteps(p);
    let r = it.next(), slices = 0, worst = 0;
    while (!r.done) {
      const t0 = performance.now();
      r = it.next();
      worst = Math.max(worst, performance.now() - t0);
      slices++;
    }
    const g = r.value;
    assert.deepEqual(Array.from(g.levelIndexEnd), Array.from(ref.levelIndexEnd));
    assert.ok(Buffer.compare(Buffer.from(g.index.buffer), Buffer.from(ref.index.buffer)) === 0, 'index differs');
    assert.ok(Buffer.compare(Buffer.from(g.packed.buffer), Buffer.from(ref.packed.buffer)) === 0, 'packed differs');
    console.log(`  ${tier}: ${slices} slices, longest single step ${worst.toFixed(3)} ms (budget per frame: 2 ms)`);
    assert.ok(worst < 4, `a single build step took ${worst} ms`);
  }
});

test('finite far plane: drawn rings never reach past maxReach (rings dropped, coverage yields)', () => {
  for (const tier of TIERS) {
    const { p } = geom(tier);
    for (const far of [3000, 5000, 20000]) {
      for (const h of [1.2, 4.8, 60, 400]) {
        const st = createLodState();
        updateLodState(st, p, 123.4, -567.8, h, true, 0.6 * far);
        const reach = (p.halfCells + p.margin) * st.spacing0 * 2 ** (st.drawnLevels - 1);
        assert.ok(st.drawnLevels === 1 || reach <= 0.6 * far, `${tier} far=${far} h=${h}: reach ${reach}`);
        assert.ok(reach * Math.SQRT2 < 0.995 * 0.999 * far, `${tier} far=${far} h=${h}: ring corner ${reach * Math.SQRT2} beyond the horizon closure`);
      }
    }
  }
});

/** Exact key of a float32 pair. */
const key = (u: number, v: number) => `${u},${v}`;

function emulateLevel(st: LodState, p: LodParams, g: LodGeometryData, level: number, cb: (gx: number, gy: number, e: EmulatedVertex) => void) {
  const e: EmulatedVertex = { qu: 0, qv: 0, pu: 0, pv: 0, y: 0, morph: 0 };
  for (let i = 0; i < g.vertexCount; i++) {
    const lk = g.packed[i * 3 + 2];
    if (lk % 4 !== LOD_KIND_GRID || (lk - (lk % 4)) / 4 !== level) continue;
    const gx = g.packed[i * 3], gy = g.packed[i * 3 + 1];
    emulateVertex(st, p, gx, gy, lk, 18000, e);
    cb(gx, gy, e);
  }
}

const FEET: [number, number, number][] = [];
for (const base of [0, 1e5, 1234567.891, 1e7]) {
  for (const h of [0.3, 1.2, 4.8, 60, 400, 500]) FEET.push([base + rnd() * 2000 - 1000, -base + rnd() * 2000 - 1000, h]);
}

test('watertight: every level boundary has exactly the parent ring vertices (bit-exact float32)', () => {
  for (const tier of TIERS) {
    const { p, g } = geom(tier);
    for (const [fu, fv, h] of FEET) {
      const st = createLodState();
      updateLodState(st, p, fu, fv, h);
      for (let l = 0; l < st.drawnLevels - 1; l++) {
        // Parent hole bounds (exact float32 values).
        const Pu = Math.fround(st.level[(l + 1) * 4]), Pv = Math.fround(st.level[(l + 1) * 4 + 1]);
        const half = Math.fround((p.halfCells / 2) * st.level[(l + 1) * 4 + 2]);
        const minU = Math.fround(Pu - half), maxU = Math.fround(Pu + half), minV = Math.fround(Pv - half), maxV = Math.fround(Pv + half);
        const inner = new Set<string>();
        emulateLevel(st, p, g, l, (_gx, _gy, e) => {
          assert.ok(e.qu >= minU && e.qu <= maxU && e.qv >= minV && e.qv <= maxV, 'inner level escapes the hole');
          if (e.qu === minU || e.qu === maxU || e.qv === minV || e.qv === maxV) inner.add(key(e.pu, e.pv));
        });
        const ring = new Set<string>();
        emulateLevel(st, p, g, l + 1, (gx, gy, e) => {
          if (Math.max(Math.abs(gx), Math.abs(gy)) === p.halfCells / 2) {
            assert.equal(e.morph, 0, 'ring inner edge must not morph');
            ring.add(key(e.pu, e.pv));
          }
        });
        assert.equal(ring.size, 4 * p.halfCells, 'ring inner loop size');
        const missing = [...inner].filter((k) => !ring.has(k));
        const extra = [...ring].filter((k) => !inner.has(k));
        assert.equal(missing.length + extra.length, 0,
          `${tier} foot=(${fu},${fv}) h=${h} level ${l}: ${missing.length} inner-only, ${extra.length} ring-only boundary vertices (crack / T-junction)`);
      }
    }
  }
});

test('no swimming: every vertex sits on its world lattice (s_l; 2 s_l once fully morphed), at 1e5..1e7 m', () => {
  const { p, g } = geom('medium');
  for (const [fu, fv, h] of FEET) {
    const st = createLodState();
    updateLodState(st, p, fu, fv, h);
    for (let l = 0; l < st.drawnLevels; l++) {
      const s = st.level[l * 4 + 2];
      emulateLevel(st, p, g, l, (gx, gy, e) => {
        const wu = st.baseU + e.qu, wv = st.baseV + e.qv; // float64 world lattice coordinates
        if (e.morph > 0 && e.morph < 1) return; // geomorph band: slides continuously with camera distance (by design)
        assert.ok(Number.isInteger(wu / s) && Number.isInteger(wv / s), `off-lattice vertex level ${l} at foot ${fu}`);
        if (e.morph === 1) {
          assert.ok(Number.isInteger(wu / (2 * s)) && Number.isInteger(wv / (2 * s)), 'fully morphed vertex off the parent lattice');
        }
        void gx; void gy;
      });
    }
  }
});

test('non-snapped camera move keeps every vertex world position on the same lattice (pan)', () => {
  const { p, g } = geom('high');
  const a = createLodState(), b = createLodState();
  const fu = 1e5 + 3.0, fv = -2e5 + 7.0, h = 4.8; // on a level-0 centre (multiple of 2 s0)
  updateLodState(a, p, fu, fv, h);
  updateLodState(b, p, fu + 0.0371, fv - 0.0213, h); // < s0 / 2: no recentring of level 0
  assert.equal(a.spacing0, b.spacing0);
  // Unmorphed vertices of level 0 exist in both with identical world positions.
  const wa = new Map<string, number>();
  emulateLevel(a, p, g, 0, (gx, gy, e) => { if (e.morph === 0) wa.set(`${gx},${gy}`, 1); void e; });
  let common = 0;
  emulateLevel(b, p, g, 0, (gx, gy, e) => {
    if (e.morph !== 0 || !wa.has(`${gx},${gy}`)) return;
    common++;
  });
  assert.ok(common > 1000);
  // World positions are B + q: B and C_l unchanged by a sub-snap move, q identical, only phi changes.
  assert.equal(a.baseU, b.baseU); assert.equal(a.baseV, b.baseV);
  for (let l = 0; l < a.drawnLevels; l++) assert.equal(a.level[l * 4], b.level[l * 4]);
  assert.notEqual(a.phiU, b.phiU);
});

test('tile anchors: renderPos + wrapped(period) == world mod period at 1e5..1e7 m (flat)', () => {
  const east = { x: 1, y: 0, z: 0 }, south = { x: 0, y: 0, z: 1 };
  const wrapped = { x: 0, y: 0, z: 0 };
  const { p } = geom('high');
  for (const big of [1e5, 101376, 1234567.25, 9999360, 1e7 + 0.375]) {
    for (const P of [1024, 96, 9]) {
      const origin = createFloatingOrigin({ x: big, y: 0, z: -big * 0.7 });
      // Camera world position near the origin (within the demo's 2 km rebase distance).
      const camX = origin.origin.x + 813.25 + rnd(), camZ = origin.origin.z - 1500.5 + rnd();
      const st = createLodState();
      updateLodState(st, p, camX, camZ, 4.8);
      origin.wrapped(P, wrapped);
      const footRenderX = camX - origin.origin.x, footRenderZ = camZ - origin.origin.z;
      const au = tileAnchor(east, footRenderX + (st.baseU - st.footU), st.baseU, wrapped, P);
      const av = tileAnchor(south, footRenderZ + (st.baseV - st.footV), st.baseV, wrapped, P);
      const eu = wrapPeriod(st.baseU, P), ev = wrapPeriod(st.baseV, P);
      const du = Math.abs(au - eu), dv = Math.abs(av - ev);
      assert.ok(Math.min(du, P - du) < 1e-9 && Math.min(dv, P - dv) < 1e-9, `anchor error ${du} ${dv} at ${big}, P=${P}`);
    }
  }
});

test('tile coordinate of a fixed world point is stable (float32) across camera moves and origin shifts', () => {
  const east = { x: 1, y: 0, z: 0 };
  const wrapped = { x: 0, y: 0, z: 0 };
  const { p, g } = geom('high');
  const P = 9;
  const pointWorldU = 1e7 + 1003.125; // a lattice point of level <= 3 (multiple of 1/8 m)
  const results: number[] = [];
  for (const [originX, camOff] of [[1e7, 1000.3], [1e7 + 1777.77, -774.0], [1e7 - 50, 1053.9]] as [number, number][]) {
    const origin = createFloatingOrigin({ x: originX, y: 0, z: 0 });
    const camU = origin.origin.x + camOff;
    const st = createLodState();
    updateLodState(st, p, camU, 0, 4.8);
    origin.wrapped(P, wrapped);
    const anchor = Math.fround(tileAnchor(east, camU - origin.origin.x + (st.baseU - st.footU), st.baseU, wrapped, P));
    const q = Math.fround(pointWorldU - st.baseU); // what the vertex computes for that lattice point (exact)
    results.push(wrapPeriod(Math.fround(anchor + q), P));
    void g;
  }
  const exact = wrapPeriod(pointWorldU, P);
  for (const r of results) {
    const d = Math.abs(r - exact);
    assert.ok(Math.min(d, P - d) < 2e-4, `tile coordinate error ${d} m`);
  }
});

test('tile anchors: non-axis-aligned frame falls back to the float64 lattice coordinate', () => {
  const a = Math.SQRT1_2;
  const axis = { x: a, y: 0, z: a };
  const r = tileAnchor(axis, 12.5, 1e7 + 3.25, { x: 1, y: 2, z: 3 }, 96);
  assert.equal(r, wrapPeriod(1e7 + 3.25, 96));
});

test('update is allocation-free (heap stays flat)', () => {
  const { p } = geom('ultra');
  const st = createLodState();
  for (let i = 0; i < 1000; i++) updateLodState(st, p, i * 0.37, -i * 0.11, 4.8 + (i % 50));
  // Background GC of earlier tests' big arrays makes single windows noisy: take the best of 3 windows.
  let grown = Infinity;
  for (let w = 0; w < 3; w++) {
    const before = process.memoryUsage().heapUsed;
    for (let i = 0; i < 300000; i++) updateLodState(st, p, 1e7 + i * 0.37, -i * 0.11, 4.8 + (i % 500), true, 0.6 * 5000);
    grown = Math.min(grown, process.memoryUsage().heapUsed - before);
  }
  assert.ok(grown < 2e5, `heap grew ${grown} bytes over 300k updates`);
});
