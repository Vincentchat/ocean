// Per-tick CPU cost of the ocean at a 60 Hz fixed step with a gameplay load, PACED in real time
// (each tick starts 16.67 ms after the previous one, as in a game; the snapshot workers then run in
// parallel with the frame as designed). Unpaced loops run simulated time many times faster than real
// time and measure worker starvation instead (use --unpaced to see that).
//   node --experimental-strip-types tests/bench-tick.ts [--seconds 30] [--unpaced] [--no-worker]
import { createOcean, WIND_QUERY } from '../index.ts';
import type { PresetName } from '../index.ts';
import { createFlatShape, createStubWorld, createStubWind } from '../stubs/impl.ts';

const args = process.argv.slice(2);
const seconds = Number(args[args.indexOf('--seconds') + 1] || 30);
const paced = !args.includes('--unpaced');
const worker = !args.includes('--no-worker');
const shape = createFlatShape();

function scenario(label: string, preset: PresetName, change: ((o: ReturnType<typeof createOcean>, w: ReturnType<typeof createStubWind>) => void) | null) {
  const world = createStubWorld({ shape });
  const wind = createStubWind(15, -57.29578, shape);
  world.provide(WIND_QUERY, wind);
  const ocean = createOcean({ preset, worker });
  world.use(ocean); world.start();
  const q = ocean.query;
  const p = { x: 0, y: 0, z: 0 }, v = { x: 0, y: 0, z: 0 };
  const pts = new Float64Array(300), hs = new Float64Array(100);
  for (let k = 0; k < 100; k++) { pts[3 * k] = (k * 7.3) % 200; pts[3 * k + 2] = (k * 3.1) % 200; }
  const frame = 1000 / 60;
  const load = (): number => {
    let s = 0;
    for (let k = 0; k < 100; k++) { p.x = (k * 7.3) % 200; p.z = (k * 3.1) % 200; s += q.heightAt(p, world.time); }
    for (let k = 0; k < 10; k++) { p.x = k * 11; p.z = k * 5; s += q.velocityAt(p, world.time, v).y; }
    return s + q.heightAt(p, world.time - 0.5 / 60); // render-side probe at the interpolated time
  };
  const run = (n: number, record: Float64Array | null): number => {
    let sink = 0, next = performance.now();
    for (let i = 0; i < n; i++) {
      if (paced) { while (performance.now() < next) { /* wait for the next 60 Hz slot */ } next += frame; }
      const a = performance.now();
      world.stepOnce();
      sink += load();
      if (record) record[i] = performance.now() - a;
    }
    return sink;
  };
  run(paced ? 180 : 600, null); // warm-up (JIT, workers, prefetch)
  if (change) change(ocean, wind);
  const N = Math.round(seconds * 60), ms = new Float64Array(N);
  run(N, ms);
  const s = Array.from(ms).sort((x, y) => x - y);
  const mean = s.reduce((x, y) => x + y, 0) / N;
  const st: number[] = []; ocean.cpuStats(st);
  console.log(`${label}: mean ${mean.toFixed(2)} ms, p50 ${s[N >> 1].toFixed(2)}, p95 ${s[Math.floor(N * 0.95)].toFixed(2)}, ` +
    `p99 ${s[Math.floor(N * 0.99)].toFixed(2)}, max ${s[N - 1].toFixed(2)} ms; > 4 ms: ${s.filter((x) => x > 4).length}/${N}; ` +
    `sync snapshots ${st[1] + st[6] + st[11]}, worker snapshots ${st[2] + st[7] + st[12]}`);
  ocean.dispose!();
}

console.log(`${paced ? 'paced 60 Hz' : 'UNPACED'}, workers ${worker ? 'on' : 'off'}, ${seconds} s per scenario; load = tick + 100 heightAt + 10 velocityAt + 1 heightAt at the interpolated time`);
scenario('windy (reference), steady', 'windy', null);
scenario('storm, steady', 'storm', null);
scenario('wind 15 -> 25 m/s (regeneration + crossfades)', 'windy', (o, w) => w.set(25, 0));
scenario('setPreset calm -> storm', 'calm', (o) => o.setPreset('storm'));
