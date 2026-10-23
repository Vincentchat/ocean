// Reviewer scratch: is the 16 B/call of heightAt avoidable? Thin inlinable wrapper over a non-inlined worker
// that writes the result into a Float64Array slot, vs returning the double from the big function.
import v8 from 'node:v8';
import { createOcean } from '../../../../world-ocean/index.ts';
import { createFlatShape } from '../../../../world-ocean/stubs/impl.ts';
const ocean = createOcean({ shape: createFlatShape() });
const q = ocean.query;
const R = new Float64Array(1);
// worker: calls the real heightAt (big, not inlined) and stores into R (the boxed return is consumed inside worker
// only if worker is optimised with heightAt inlined - it is not, so emulate a real big worker instead):
function bigWorker(x, z, t) { // deliberately big
  let s = 0; for (let i = 0; i < 4; i++) { s += Math.sin(x * (i + 1) + t) * Math.cos(z * (i + 2)); s += Math.sqrt(Math.abs(s) + x * x * 1e-6 + z * z * 1e-7 + i); s *= 0.5; }
  for (let i = 0; i < 4; i++) { s += Math.atan2(x + i, z - i) * 1e-3; s += Math.log1p(Math.abs(s)) * 1e-3; s -= Math.exp(-Math.abs(x) * 1e-3 - i) * 1e-4; }
  return s;
}
function bigWorkerInto(x, z, t) { let s = 0; for (let i = 0; i < 4; i++) { s += Math.sin(x * (i + 1) + t) * Math.cos(z * (i + 2)); s += Math.sqrt(Math.abs(s) + x * x * 1e-6 + z * z * 1e-7 + i); s *= 0.5; } for (let i = 0; i < 4; i++) { s += Math.atan2(x + i, z - i) * 1e-3; s += Math.log1p(Math.abs(s)) * 1e-3; s -= Math.exp(-Math.abs(x) * 1e-3 - i) * 1e-4; } R[0] = s; }
function bigWorker2(x, z, t) { return bigWorker(x, z, t); }
const ns = () => v8.getHeapSpaceStatistics().find((s) => s.space_name === 'new_space').space_used_size;
function m(label, fn) { fn(200000); fn(200000); globalThis.gc(); const a = ns(); fn(1000000); console.log(`${label}: ${((ns() - a) / 1e6).toFixed(2)} B/call`); }
const p = { x: 0, y: 0, z: 0 };
m('real heightAt', (n) => { let s = 0; for (let i = 0; i < n; i++) { p.x = i * 0.37 % 500; s += q.heightAt(p, 3); } return s; });
m('big function returning a double', (n) => { let s = 0; for (let i = 0; i < n; i++) s += bigWorker(i * 0.37 % 500, 1, 3); return s; });
const thin = (x, z, t) => { bigWorkerInto(x, z, t); return R[0]; };
m('thin wrapper -> worker writes Float64Array, wrapper returns R[0]', (n) => { let s = 0; for (let i = 0; i < n; i++) s += thin(i * 0.37 % 500, 1, 3); return s; });

const A = new Float64Array(3);
function workerNoArgs() { const x = A[0], z = A[1], t = A[2]; let s = 0; for (let i = 0; i < 4; i++) { s += Math.sin(x * (i + 1) + t) * Math.cos(z * (i + 2)); s += Math.sqrt(Math.abs(s) + x * x * 1e-6 + z * z * 1e-7 + i); s *= 0.5; } for (let i = 0; i < 4; i++) { s += Math.atan2(x + i, z - i) * 1e-3; s += Math.log1p(Math.abs(s)) * 1e-3; s -= Math.exp(-Math.abs(x) * 1e-3 - i) * 1e-4; } R[0] = s; }
const thin2 = (x, z, t) => { A[0] = x; A[1] = z; A[2] = t; workerNoArgs(); return R[0]; };
m('thin wrapper (doubles via Float64Array) -> no-arg worker', (n) => { let s = 0; for (let i = 0; i < n; i++) s += thin2(i * 0.37 % 500, 1, 3.5); return s; });
const pt = { x: 0, y: 0, z: 0 };
const thin3 = (pp, t) => { A[0] = pp.x; A[1] = pp.z; A[2] = t; workerNoArgs(); return R[0]; };
m('heightAt-shaped thin wrapper (p object, double time)', (n) => { let s = 0; for (let i = 0; i < n; i++) { pt.x = i * 0.37 % 500; s += thin3(pt, 3.5 + i * 1e-9); } return s; });
m('baseline: loop + p.x store + s += R[0]', (n) => { let s = 0; for (let i = 0; i < n; i++) { pt.x = i * 0.37 % 500; s += R[0]; } return s; });
m('baseline: no-arg worker call only', (n) => { for (let i = 0; i < n; i++) { workerNoArgs(); } return 0; });
function hloop(n) { let s = 0; for (let i = 0; i < n; i++) { pt.x = i * 0.37 % 500; s += q.heightAt(pt, 3); } return s; }
m('real heightAt, named fn', hloop);
