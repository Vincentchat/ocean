// Reviewer scratch: crossfade smoothness. Worst cases: preset jump calm -> storm, wind direction flip 180 deg.
import { createOcean, WIND_QUERY } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../../../world-ocean/stubs/impl.ts';
const shape = createFlatShape();

// Spatial Hs at time t: 4 sigma of heightAt over a fixed 1.5 km quasi-random set (1500 points).
const NP = 1500;
const pe = new Float64Array(NP), pn = new Float64Array(NP);
for (let i = 0; i < NP; i++) { pe[i] = ((i * 0.6180339887) % 1) * 1500; pn[i] = ((i * 0.7548776662) % 1) * 1500; }
function hsAt(ocean, t) { let s = 0, s2 = 0; for (let i = 0; i < NP; i++) { const h = ocean.heightAtLocal(pe[i], pn[i], t); s += h; s2 += h * h; } const m = s / NP; return 4 * Math.sqrt(s2 / NP - m * m); }

function run(label, setup, change, seconds) {
  const world = createStubWorld({ shape });
  const { ocean, wind } = setup(world);
  for (let i = 0; i < 120; i++) world.stepOnce();
  const M = 64; const last = new Float64Array(M), cur = new Float64Array(M);
  const sample = (out) => { for (let k = 0; k < M; k++) out[k] = ocean.heightAtLocal(pe[k * 7] , pn[k * 7], world.time); };
  sample(last);
  let steady = 0;
  for (let i = 0; i < 300; i++) { world.stepOnce(); sample(cur); for (let k = 0; k < M; k++) steady = Math.max(steady, Math.abs(cur[k] - last[k])); last.set(cur); }
  const hs0 = hsAt(ocean, world.time);
  change(ocean, wind);
  let maxStep = 0, maxStepT = 0, hsMin = Infinity, hsMinT = 0, hsMax = 0, regenEvents = 0;
  let lastVersion = ocean.getState().version;
  const trace = [];
  for (let i = 0; i < seconds * 60; i++) {
    world.stepOnce(); sample(cur);
    for (let k = 0; k < M; k++) { const d = Math.abs(cur[k] - last[k]); if (d > maxStep) { maxStep = d; maxStepT = world.time; } }
    last.set(cur);
    const st = ocean.getState();
    if (st.version !== lastVersion) { regenEvents++; lastVersion = st.version; }
    if (i % 30 === 0) {
      const hs = hsAt(ocean, world.time);
      if (hs < hsMin) { hsMin = hs; hsMinT = world.time; }
      hsMax = Math.max(hsMax, hs);
      if (i % 120 === 0) trace.push(`${(world.time).toFixed(0)}s:${hs.toFixed(2)}${st.fromSlot !== st.toSlot ? '*' : ''}`);
    }
  }
  let finalSteady = 0; sample(last);
  for (let i = 0; i < 300; i++) { world.stepOnce(); sample(cur); for (let k = 0; k < M; k++) finalSteady = Math.max(finalSteady, Math.abs(cur[k] - last[k])); last.set(cur); }
  const hs1 = hsAt(ocean, world.time);
  console.log(`\n${label}`);
  console.log(`  Hs before ${hs0.toFixed(2)} m, after ${hs1.toFixed(2)} m, min during ${hsMin.toFixed(2)} m @${hsMinT.toFixed(1)} s, max ${hsMax.toFixed(2)} m; spectrum regenerations ${regenEvents}`);
  console.log(`  max per-tick height change: steady-before ${steady.toFixed(4)} m, during ${maxStep.toFixed(4)} m @${maxStepT.toFixed(1)} s, steady-after ${finalSteady.toFixed(4)} m`);
  console.log(`  Hs trace (* = crossfading): ${trace.join(' ')}`);
}

const noWind = (preset) => (world) => { const ocean = createOcean({ preset }); world.use(ocean); world.start(); return { ocean, wind: null }; };
const withWind = (speed, dirDeg, extra = {}) => (world) => {
  const wind = createStubWind(speed, dirDeg, shape); world.provide(WIND_QUERY, wind);
  const ocean = createOcean({ preset: 'breeze', ...extra }); world.use(ocean); world.start(); return { ocean, wind };
};
const which = process.argv[2] ?? 'all';
if (which === 'all' || which === 'preset') {
  run('preset jump calm -> storm (setPreset, default crossfade)', noWind('calm'), (o) => o.setPreset('storm'), 20);
  run('preset jump storm -> calm (setPreset, default crossfade)', noWind('storm'), (o) => o.setPreset('calm'), 20);
}
if (which === 'all' || which === 'flip') {
  run('wind direction flip 180 deg at 15 m/s (default windTimeConstant 8 s)', withWind(15, -57.29578), (o, w) => w.set(15, 122.70422), 60);
  run('wind direction turn 90 deg at 15 m/s', withWind(15, -57.29578), (o, w) => w.set(15, 32.70422), 40);
  run('wind 15 -> 25 m/s same direction', withWind(15, -57.29578), (o, w) => w.set(25, -57.29578), 60);
}
