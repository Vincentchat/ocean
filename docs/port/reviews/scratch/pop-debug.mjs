// Reviewer scratch: find the per-tick height jump during wind 15 -> 25 m/s and log the crossfade state.
import { createOcean, WIND_QUERY } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../../../../world-ocean/stubs/impl.ts';
const shape = createFlatShape();
const world = createStubWorld({ shape });
const wind = createStubWind(15, -57.29578, shape); world.provide(WIND_QUERY, wind);
const speed = Number(process.argv[2] ?? 25);
const ocean = createOcean({ preset: 'breeze' }); world.use(ocean); world.start();
const M = 64;
const pe = new Float64Array(M), pn = new Float64Array(M);
for (let k = 0; k < M; k++) { const i = k * 7; pe[k] = ((i * 0.6180339887) % 1) * 1500; pn[k] = ((i * 0.7548776662) % 1) * 1500; }
for (let i = 0; i < 420; i++) world.stepOnce();
wind.set(speed, -57.29578);
const last = new Float64Array(M), cur = new Float64Array(M);
const sample = (out, t) => { for (let k = 0; k < M; k++) out[k] = ocean.heightAtLocal(pe[k], pn[k], t); };
sample(last, world.time);
const st = ocean.getState();
const events = [];
let prev = '';
let steady = 0.13;
for (let i = 0; i < 60 * 70; i++) {
  world.stepOnce(); sample(cur, world.time);
  let mx = 0, mk = 0;
  for (let k = 0; k < M; k++) { const d = Math.abs(cur[k] - last[k]); if (d > mx) { mx = d; mk = k; } }
  const desc = `from ${st.fromSlot} to ${st.toSlot} pend ${st.pendingSlot} start ${st.blendStart.toFixed(2)} dur ${st.blendDuration} ver ${st.version} U ${st.windSpeed.toFixed(2)}`;
  if (desc !== prev) { events.push(`t=${world.time.toFixed(3)} ${desc}`); prev = desc; }
  if (mx > 2 * steady) {
    // Is the jump real? Evaluate the point at t-dt and t again on a FRESH query path (time far away first to flush the cache).
    const t = world.time, t0 = t - 1 / 60;
    ocean.heightAtLocal(0, 0, t + 100);
    const a = ocean.heightAtLocal(pe[mk], pn[mk], t0), b = ocean.heightAtLocal(pe[mk], pn[mk], t);
    // finer time steps across the tick
    let fine = 0, fprev = a;
    for (let s = 1; s <= 20; s++) { const h = ocean.heightAtLocal(pe[mk], pn[mk], t0 + s * (1 / 60) / 20); fine = Math.max(fine, Math.abs(h - fprev)); fprev = h; }
    events.push(`  JUMP t=${t.toFixed(3)} point ${mk} (${pe[mk].toFixed(1)}, ${pn[mk].toFixed(1)}): ${last[mk].toFixed(3)} -> ${cur[mk].toFixed(3)} (recomputed ${a.toFixed(3)} -> ${b.toFixed(3)}, max sub-step change ${fine.toFixed(3)}) blend ${ocean.blendAt(t).toFixed(3)} chop ${ocean.choppinessAt(t).toFixed(3)}`);
  }
  last.set(cur);
}
console.log(events.join('\n'));
