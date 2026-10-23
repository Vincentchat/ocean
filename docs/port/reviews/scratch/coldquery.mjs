import { createOcean } from '../../../../world-ocean/index.ts';
import { createFlatShape } from '../../../../world-ocean/stubs/impl.ts';
const o = createOcean({ shape: createFlatShape() }); const p = { x: 1, y: 0, z: 2 }, v = { x: 0, y: 0, z: 0 };
o.query.heightAt(p, 1);
let t = 100, a = 0; const N = 30;
for (let i = 0; i < N; i++) { t += 37.1; const t0 = performance.now(); o.query.heightAt(p, t); a += performance.now() - t0; }
let b = 0; for (let i = 0; i < N; i++) { t += 37.1; const t0 = performance.now(); o.query.velocityAt(p, t, v); b += performance.now() - t0; }
console.log(`cold heightAt at an arbitrary time: ${(a / N).toFixed(1)} ms; cold velocityAt: ${(b / N).toFixed(1)} ms`);
