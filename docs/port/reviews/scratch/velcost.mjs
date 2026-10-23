import { createOcean } from '../../../../world-ocean/index.ts';
import { createFlatShape, createStubWorld } from '../../../../world-ocean/stubs/impl.ts';
const shape = createFlatShape(); const world = createStubWorld({ shape });
const ocean = createOcean({}); world.use(ocean); world.start();
const q = ocean.query; const p = { x: 3, y: 0, z: 4 }, v = { x: 0, y: 0, z: 0 };
for (let i = 0; i < 600; i++) { world.stepOnce(); q.velocityAt(p, world.time, v); }
let a = 0, b = 0; const N = 1200;
for (let i = 0; i < N; i++) { const t0 = performance.now(); world.stepOnce(); const t1 = performance.now(); q.velocityAt(p, world.time, v); const t2 = performance.now(); a += t1 - t0; b += t2 - t1; }
console.log(`tick ${(a / N).toFixed(2)} ms, first velocityAt after the tick ${(b / N).toFixed(2)} ms`);
let c = 0; for (let i = 0; i < N; i++) { const t0 = performance.now(); q.velocityAt(p, world.time, v); c += performance.now() - t0; }
console.log(`repeat velocityAt same time ${(c / N * 1000).toFixed(2)} us`);
const os = await import('node:os'); console.log('load', os.loadavg(), 'cpus', os.cpus().length, os.cpus()[0].model);
