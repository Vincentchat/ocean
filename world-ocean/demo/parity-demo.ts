// DEMO ONLY: ?parity (GPU drawn heights vs CPU WATER_QUERY) and ?buoy (floating test objects driven by the CPU
// query). Loaded by demo.ts only when one of the flags is present.
//
// ?parity[=N]   N probe points (default 256) around the camera foot, radius 1 .. 450 m (log-spaced, golden-angle
//               azimuths). Each frame the GPU probe (three/parity.ts) solves the drawn surface at the points; the CPU
//               reference (ocean.heightsAt) is evaluated at dispatch with the same points and simulation time; the
//               readback lands one frame later and is compared. PASS = both gates < 2 % Hs (max |err|):
//                 A  GPU field (full-resolution GPU surface: LOD 0 Catmull-Rom, no fade) vs CPU, all points;
//                 B  DRAWN surface (vertex footprint mips + far fade, exactly as the vertex shader) vs CPU where the
//                    vertex shader uses the CPU's filter for c0 + c1 (pure Catmull-Rom: footprint <= 0.375 m).
//               Further out the drawn surface is band-limited by the vertex footprint mips (25-200 m) and flattened by
//               the far fade (> 200 m): reported per band, not gated (view-dependent by design; see the report).
//               window.__parity: live aggregate since the last reset() (tools/parity.mjs).
// ?buoy         5 floating test objects: height from WATER_QUERY.heightAt at the interpolated (draw) time, tilt from
//               the heightAt slope, the last one drifts with WATER_QUERY.velocityAt (spring-anchored). A small GPU
//               probe reads the DRAWN height under each buoy: the overlay shows the CPU - drawn gap per buoy.
import { CylinderGeometry, Group, Mesh, MeshBasicNodeMaterial, SphereGeometry, Vector3 } from 'three/webgpu';
import { dot, max, normalWorld, uniform, vec3 } from 'three/tsl';
import { createOceanHeightProbe, type OceanHeightProbe, type OceanProbeResult } from '../three/parity.ts';
import type { OceanGpu, Pass, PassContext } from '../three/index.ts';
import type { FloatingOrigin, Vec3, WorldShape } from '../stubs/world-core.ts';

export interface ParityDemoOptions {
  parity: boolean;
  buoy: boolean;
  /** ?parity=N */
  points?: number;
  renderer: any;
  gpu: OceanGpu;
  ocean: any;
  origin: FloatingOrigin;
  shape: WorldShape;
  camera: any;
  scene: any;
  runner: { add(p: Pass): void };
  preExposure: any;
  refNits: number;
  sunDir: Readonly<Vec3>;
  cubic: boolean;
  /** Initial camera world position + yaw (rad) for the buoy layout. */
  camWorld: Readonly<Vec3>;
  yaw: number;
}

function makeOverlay(): HTMLElement {
  const el = document.createElement('pre');
  el.id = 'parity-overlay';
  el.style.cssText = 'position:fixed;top:12px;right:12px;margin:0;padding:8px 10px;background:#071d24dd;border:1px solid #2b4b52;' +
    'border-radius:8px;font:12px/1.35 ui-monospace,Consolas,monospace;color:#d6e6e1;z-index:5;pointer-events:none;white-space:pre';
  document.body.appendChild(el);
  return el;
}

/** Running error statistics (allocation-free after construction). */
function createAgg(cap: number) {
  const abs = new Float32Array(cap);
  const a = { n: 0, max: 0, sumSq: 0, sum: 0, kept: 0, abs };
  return {
    a,
    reset() { a.n = 0; a.max = 0; a.sumSq = 0; a.sum = 0; a.kept = 0; },
    add(err: number) {
      const e = err < 0 ? -err : err;
      a.n++; a.sumSq += err * err; a.sum += err;
      if (e > a.max) a.max = e;
      if (a.kept < cap) abs[a.kept++] = e; else abs[(a.n * 2654435761 >>> 0) % cap] = e; // reservoir-ish
    },
    summary(hs: number) {
      if (a.n === 0) return { n: 0, max: 0, p99: 0, rms: 0, mean: 0, maxPctHs: 0, p99PctHs: 0, rmsPctHs: 0 };
      const s = abs.subarray(0, a.kept).slice().sort();
      const p99 = s[Math.min(s.length - 1, Math.floor(0.99 * s.length))];
      const rms = Math.sqrt(a.sumSq / a.n);
      return { n: a.n, max: a.max, p99, rms, mean: a.sum / a.n, maxPctHs: (100 * a.max) / hs, p99PctHs: (100 * p99) / hs, rmsPctHs: (100 * rms) / hs };
    },
  };
}

export async function attachParityDemo(o: ParityDemoOptions) {
  const { gpu, ocean, origin, shape, camera, runner } = o;
  const overlay = makeOverlay();
  const camR = new Vector3();
  const camRv: Vec3 = { x: 0, y: 0, z: 0 };
  const camW: Vec3 = { x: 0, y: 0, z: 0 };
  const footW: Vec3 = { x: 0, y: 0, z: 0 };
  const frame = { east: { x: 1, y: 0, z: 0 }, north: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 } };
  const cameraFoot = () => {
    camR.setFromMatrixPosition(camera.matrixWorld);
    camRv.x = camR.x; camRv.y = camR.y; camRv.z = camR.z;
    origin.toWorld(camRv, camW);
    shape.atAltitude(camW, 0, footW);
    shape.tangentFrame(footW, frame);
  };
  const hs = () => ocean.expectedHs() as number;
  const text = { parity: '', buoy: '' };
  let lastText = 0;
  const showText = () => {
    const now = performance.now();
    if (now - lastText < 250) return;
    lastText = now;
    overlay.textContent = [text.parity, text.buoy].filter(Boolean).join('\n\n');
  };

  // ---------------------------------------------------------------------------------------------- ?parity
  if (o.parity) {
    const N = Math.max(8, Math.min(4096, o.points ?? 256));
    const R0 = 1, R1 = 450, GOLDEN = Math.PI * (3 - Math.sqrt(5));
    const offE = new Float64Array(N), offS = new Float64Array(N), radius = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const r = R0 * Math.pow(R1 / R0, i / (N - 1));
      radius[i] = r;
      offE[i] = r * Math.cos(i * GOLDEN); offS[i] = r * Math.sin(i * GOLDEN);
    }
    const probe: OceanHeightProbe = createOceanHeightProbe({
      renderer: o.renderer, fft: gpu.fft, lod: gpu.lod, origin, shape, capacity: N, cubic: o.cubic,
      reference: (pts, n, t, out) => { ocean.heightsAt(pts, t, out, n); },
    });
    // Statistics. Gate A (spec check 6 "GPU heights = CPU heights"): the full-resolution GPU surface (LOD 0, no fade)
    // vs the CPU at every point. Gate B (the floating-object view): the DRAWN surface vs the CPU where the vertex
    // shader samples cascades 0 and 1 with the CPU's filter (Catmull-Rom at texture LOD 0 for c1, i.e. vertex
    // footprint <= CUBIC_FP m). Elsewhere the drawn surface is band-limited on purpose (vertex footprint mips) and
    // flattened beyond 200 m (far fade): reported per distance band, not gated.
    const NEAR = 25;
    const CUBIC_FP = 96 / 256 + 1e-6; // c1 texture LOD 0 (= one c1 texel): pure Catmull-Rom for c0 and c1
    const CAP = 1 << 18;
    const field = createAgg(CAP), dCubic = createAgg(CAP), dNear = createAgg(CAP), dMid = createAgg(CAP), dFar = createAgg(CAP), dFaded = createAgg(CAP);
    const st = {
      readbacks: 0, latencyMax: 0, latencySum: 0, residualMax: 0, timeMin: Infinity, timeMax: -Infinity,
      sumRef: 0, sumRef2: 0, nRef: 0, worst: { err: 0, x: 0, z: 0, dist: 0, fp: 0, t: 0, cpu: 0, gpu: 0 },
    };
    const reset = () => {
      field.reset(); dCubic.reset(); dNear.reset(); dMid.reset(); dFar.reset(); dFaded.reset();
      st.readbacks = 0; st.latencyMax = 0; st.latencySum = 0; st.residualMax = 0; st.timeMin = Infinity; st.timeMax = -Infinity;
      st.sumRef = 0; st.sumRef2 = 0; st.nRef = 0; st.worst.err = 0;
    };
    const cm = (x: number) => (x * 100).toFixed(2);
    probe.onResult = (r: OceanProbeResult) => {
      st.readbacks++;
      st.latencySum += r.latencyFrames;
      if (r.latencyFrames > st.latencyMax) st.latencyMax = r.latencyFrames;
      if (r.time < st.timeMin) st.timeMin = r.time;
      if (r.time > st.timeMax) st.timeMax = r.time;
      for (let i = 0; i < r.count; i++) {
        const cpu = r.reference[i];
        if (!(cpu === cpu)) continue;
        const err = r.drawn[i] - cpu, ef = r.field[i] - cpu;
        st.sumRef += cpu; st.sumRef2 += cpu * cpu; st.nRef++;
        field.add(ef);
        const ae = ef < 0 ? -ef : ef;
        if (ae > st.worst.err) {
          const w = st.worst;
          w.err = ae; w.x = r.points[3 * i]; w.z = r.points[3 * i + 2]; w.dist = radius[i]; w.fp = r.footprint[i]; w.t = r.time; w.cpu = cpu; w.gpu = r.field[i];
        }
        if (r.footprint[i] <= CUBIC_FP && r.fade[i] >= 0.99999) { dCubic.add(err); if (r.residual[i] > st.residualMax) st.residualMax = r.residual[i]; }
        if (r.fade[i] < 0.99999) dFaded.add(err);
        else if (radius[i] < NEAR) dNear.add(err);
        else if (radius[i] < 100) dMid.add(err);
        else dFar.add(err);
      }
      const H = hs();
      const a = field.summary(H), b = dCubic.summary(H), nr = dNear.summary(H);
      const pa = a.n > 0 && a.maxPctHs < 2, pb = b.n > 0 && b.maxPctHs < 2;
      text.parity =
        `PARITY  ${N} pts, ${R0}..${R1} m   ${pa && pb ? 'PASS' : 'FAIL'}\n` +
        `Hs ${H.toFixed(2)} m  tier ${gpu.fft.tier}  readbacks ${st.readbacks}  latency ${(st.latencySum / Math.max(1, st.readbacks)).toFixed(1)} fr\n` +
        `GPU field vs CPU (all)   max ${cm(a.max)} cm (${a.maxPctHs.toFixed(2)} % Hs)  p99 ${cm(a.p99)}  rms ${cm(a.rms)}  ${pa ? 'PASS' : 'FAIL'}\n` +
        `drawn vs CPU, cubic zone max ${cm(b.max)} cm (${b.maxPctHs.toFixed(2)} % Hs)  p99 ${cm(b.p99)}  rms ${cm(b.rms)}  ${pb ? 'PASS' : 'FAIL'}\n` +
        `  <${NEAR} m              max ${cm(nr.max)} cm (${nr.maxPctHs.toFixed(2)} % Hs)  rms ${cm(nr.rms)}\n` +
        `  ${NEAR}-100 m (mips)      max ${cm(dMid.a.max)} cm  rms ${cm(Math.sqrt(dMid.a.sumSq / Math.max(1, dMid.a.n)))}\n` +
        `  100-200 m (mips)     max ${cm(dFar.a.max)} cm  rms ${cm(Math.sqrt(dFar.a.sumSq / Math.max(1, dFar.a.n)))}\n` +
        `  >200 m (far fade)    max ${cm(dFaded.a.max)} cm  rms ${cm(Math.sqrt(dFaded.a.sumSq / Math.max(1, dFaded.a.n)))}\n` +
        `inversion residual (cubic zone) max ${(st.residualMax * 1000).toFixed(2)} mm`;
      showText();
    };
    // Wrapper pass: fill the points (camera foot + offsets, float64 world) then run the probe.
    const pass: Pass = {
      desc: probe.pass.desc,
      execute(ctx: PassContext) {
        cameraFoot();
        const E = frame.east, Nn = frame.north, p = probe.points;
        for (let i = 0; i < N; i++) {
          const e = offE[i], s = offS[i];
          p[3 * i] = footW.x + e * E.x - s * Nn.x;
          p[3 * i + 1] = footW.y + e * E.y - s * Nn.y;
          p[3 * i + 2] = footW.z + e * E.z - s * Nn.z;
        }
        probe.count = N;
        probe.pass.execute(ctx);
      },
    };
    await probe.ready;
    runner.add(pass);
    (window as any).__parity = {
      probe,
      reset,
      get ready() { return st.readbacks > 0; },
      get readbacks() { return st.readbacks; },
      summary() {
        const H = hs();
        const a = field.summary(H), b = dCubic.summary(H);
        const m = st.nRef ? st.sumRef / st.nRef : 0;
        const hsMeasured = st.nRef ? 4 * Math.sqrt(Math.max(0, st.sumRef2 / st.nRef - m * m)) : 0;
        return {
          hs: H, hsMeasured, tier: gpu.fft.tier, pendingTier: gpu.fft.pendingTier, points: N, near: NEAR, cubicFootprint: CUBIC_FP, readbacks: st.readbacks,
          latencyMean: st.latencySum / Math.max(1, st.readbacks), latencyMax: st.latencyMax, timeMin: st.timeMin, timeMax: st.timeMax,
          residualMaxM: st.residualMax, stats: { ...probe.stats },
          field: a, drawnCubic: b, drawnNear: dNear.summary(H), drawnMid: dMid.summary(H), drawnFar: dFar.summary(H), drawnFaded: dFaded.summary(H),
          worstField: { ...st.worst }, passField: a.n > 0 && a.maxPctHs < 2, passDrawnCubic: b.n > 0 && b.maxPctHs < 2,
          pass: a.n > 0 && a.maxPctHs < 2 && b.n > 0 && b.maxPctHs < 2,
        };
      },
    };
  }

  // ---------------------------------------------------------------------------------------------- ?buoy
  if (o.buoy) {
    const query = ocean.query; // = world.require(WATER_QUERY)
    const sun = uniform(new Vector3(o.sunDir.x, o.sunDir.y, o.sunDir.z));
    const mat = (r: number, g: number, b: number) => {
      const m = new MeshBasicNodeMaterial();
      const lit = max(dot(normalWorld, sun), 0).mul(0.7).add(0.3);
      m.colorNode = vec3(r, g, b).mul(lit).mul(o.refNits * 0.35).mul(o.preExposure);
      return m;
    };
    const hullGeo = new CylinderGeometry(0.55, 0.45, 0.5, 32);
    const mastGeo = new CylinderGeometry(0.035, 0.035, 1.4, 8);
    const topGeo = new SphereGeometry(0.12, 16, 8);
    const ringGeo = new CylinderGeometry(0.565, 0.565, 0.04, 32); // black waterline band at hull mid-height
    const hullMat = mat(1.0, 0.38, 0.05), mastMat = mat(0.9, 0.9, 0.9), topMat = mat(1, 0.05, 0.05), ringMat = mat(0.02, 0.02, 0.02);
    // Layout relative to the initial camera: ahead (fwd) and to the side (right), metres.
    const fwdE = Math.sin(o.yaw), fwdS = -Math.cos(o.yaw); // yaw 0 looks towards -Z (= north): east 0, south -1
    const rightE = Math.cos(o.yaw), rightS = Math.sin(o.yaw);
    const LAYOUT = [[4.5, -1.2], [8, 2.2], [13, -3.5], [22, 3.5], [34, -1]];
    const B = LAYOUT.length;
    const anchorW = new Float64Array(B * 3), posW = new Float64Array(B * 3);
    for (let k = 0; k < B; k++) {
      const [f, r] = LAYOUT[k];
      const e = o.camWorld.x + f * fwdE + r * rightE, s = o.camWorld.z + f * fwdS + r * rightS;
      anchorW[3 * k] = e; anchorW[3 * k + 1] = 0; anchorW[3 * k + 2] = s; // flat demo world: east = x, south = z
    }
    posW.set(anchorW);
    const groups: any[] = [];
    for (let k = 0; k < B; k++) {
      const g = new Group();
      g.name = `demo.buoy${k}`;
      const hull = new Mesh(hullGeo, hullMat);
      const ring = new Mesh(ringGeo, ringMat);
      const mast = new Mesh(mastGeo, mastMat); mast.position.y = 0.95;
      const top = new Mesh(topGeo, topMat); top.position.y = 1.7;
      g.add(hull, ring, mast, top);
      o.scene.add(g);
      groups.push(g);
    }
    const p: Vec3 = { x: 0, y: 0, z: 0 }, v: Vec3 = { x: 0, y: 0, z: 0 };
    const up = new Vector3(), nrm = new Vector3(0, 1, 0);
    const cpuH = new Float64Array(B);
    // GPU probe under each buoy; the CPU reference is evaluated at dispatch with the same points and time.
    const probe = createOceanHeightProbe({
      renderer: o.renderer, fft: gpu.fft, lod: gpu.lod, origin, shape, capacity: 8, cubic: o.cubic,
      reference: (pts, n, t, out) => { ocean.heightsAt(pts, t, out, n); },
    });
    const gap = new Float64Array(B);
    const hist = { n: 0, max: 0 };
    probe.onResult = (r) => {
      const H = hs();
      let line = `BUOYS (CPU heightAt @ draw time; gap = CPU - drawn, GPU probe)\n`;
      for (let k = 0; k < r.count; k++) {
        const c = r.reference[k];
        gap[k] = c - r.drawn[k];
        const a = Math.abs(gap[k]);
        hist.n++; if (a > hist.max) hist.max = a;
        line += `  #${k}  h ${c.toFixed(3)} m  gap ${(gap[k] * 100).toFixed(2)} cm\n`;
      }
      line += `  max gap ${(hist.max * 100).toFixed(2)} cm (${((100 * hist.max) / H).toFixed(2)} % Hs) over ${hist.n} samples`;
      text.buoy = line;
      showText();
    };
    let lastT = NaN;
    const pass: Pass = {
      desc: { id: 'demo.buoys', module: 'demo', stage: 'compute', reads: ['ocean.parity'], writes: ['demo.buoys'] },
      execute(ctx: PassContext) {
        const t = ctx.time;
        const dt = lastT === lastT ? Math.min(Math.max(t - lastT, 0), 0.1) : 0;
        lastT = t;
        for (let k = 0; k < B; k++) {
          let x = posW[3 * k], z = posW[3 * k + 2];
          if (k === B - 1 && dt > 0) {
            // drifter: water-particle horizontal velocity + a weak spring back to its anchor
            p.x = x; p.y = 0; p.z = z;
            query.velocityAt(p, t, v);
            x += (v.x - 0.15 * (x - anchorW[3 * k])) * dt;
            z += (v.z - 0.15 * (z - anchorW[3 * k + 2])) * dt;
            posW[3 * k] = x; posW[3 * k + 2] = z;
          }
          p.x = x; p.y = 0; p.z = z;
          const h = query.heightAt(p, t);
          cpuH[k] = h;
          // tilt: surface normal from the heightAt slope over the hull size (+-0.5 m)
          p.x = x + 0.5; const hx1 = query.heightAt(p, t);
          p.x = x - 0.5; const hx0 = query.heightAt(p, t);
          p.x = x; p.z = z + 0.5; const hz1 = query.heightAt(p, t);
          p.z = z - 0.5; const hz0 = query.heightAt(p, t);
          nrm.set(-(hx1 - hx0), 2 * 0.5, -(hz1 - hz0)).normalize();
          const g = groups[k];
          g.position.set(x - origin.origin.x, h - origin.origin.y, z - origin.origin.z);
          up.set(0, 1, 0);
          g.quaternion.setFromUnitVectors(up, nrm);
          probe.points[3 * k] = x; probe.points[3 * k + 1] = 0; probe.points[3 * k + 2] = z;
        }
        probe.count = B;
        probe.pass.execute(ctx);
      },
    };
    await probe.ready;
    runner.add(pass);
    (window as any).__buoy = {
      probe,
      get positions() { return Array.from(posW); },
      get heights() { return Array.from(cpuH); },
      get gaps() { return Array.from(gap); },
      get maxGap() { return hist.max; },
      reset() { hist.n = 0; hist.max = 0; },
    };
  }
}
