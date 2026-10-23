// Reviewer probe (lod-r1): drives the demo in one page so LOD state (hysteresis) persists between shots.
//   node probe.mjs <experiment> [baseUrl]
import { mkdirSync } from 'node:fs';
import { launch, watchPage } from 'file:///C:/Users/Vincent/Documents/Claude/ocean/world-ocean/tools/lib/browser.mjs';

const exp = process.argv[2];
const base = process.argv[3] ?? 'http://127.0.0.1:5184/demo/';
const out = new URL('./probe-out/', import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1');
mkdirSync(out, { recursive: true });
const { browser, context } = await launch({ width: 1920, height: 1080 });
const page = await context.newPage();
watchPage(page, { label: exp });
async function open(q) {
  await page.goto(base + '?' + q, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__shotReady === true || !!window.__demoFatal, null, { timeout: 120000, polling: 100 });
}
const frames = (n = 3) => page.evaluate((n) => new Promise((r) => { let k = 0; const f = () => (++k >= n ? r() : requestAnimationFrame(f)); requestAnimationFrame(f); }), n);
async function shot(name) { await frames(4); await page.screenshot({ path: out + name + '.png' }); console.log('wrote', name); }
const ev = (fn, arg) => page.evaluate(fn, arg);
const stats = () => ev(() => { const s = window.__demo.gpu.lod.stats(); return { s0: s.spacing0, levels: s.drawnLevels, tris: s.drawnTriangles, skirt: s.skirtRadius, reach: s.ringReach }; });

try {
  if (exp === 'altpop') {
    // Altitude sweep across the up-switch threshold (high: 8.56 m) and back down (7.46 m).
    for (const dbg of ['facets', '']) {
      await open(`shot&t=30&scene=azure&cam=0,8.0,8,0,-25${dbg ? '&debug=' + dbg : ''}`);
      const tag = dbg || 'water';
      const ys = [8.0, 8.3, 8.5, 8.55, 8.6, 8.7, 8.0, 7.6, 7.5, 7.45, 7.4];
      for (let i = 0; i < ys.length; i++) {
        await ev((y) => { window.__demo.freeCam.y = y; }, ys[i]);
        await frames(3);
        console.log(tag, ys[i], JSON.stringify(await stats()));
        await shot(`altpop-${tag}-${String(i).padStart(2, '0')}-y${ys[i]}`);
      }
    }
  } else if (exp === 'altpop2') {
    // Identical camera at y = 8.0, two hysteresis states: s0 = 1/8 (from above) vs s0 = 1/16 (came from 7.4 m).
    for (const dbg of ['facets', '']) {
      const tag = dbg || 'water';
      await open(`shot&t=30&scene=azure&cam=0,8.0,8,0,-25${dbg ? '&debug=' + dbg : ''}`);
      console.log(tag, 'A', JSON.stringify(await stats()));
      await shot(`altpop2-${tag}-A`);
      await ev(() => { window.__demo.freeCam.y = 7.3; }); await frames(3);
      await ev(() => { window.__demo.freeCam.y = 8.0; }); await frames(3);
      console.log(tag, 'B', JSON.stringify(await stats()));
      await shot(`altpop2-${tag}-B`);
    }
  } else if (exp === 'swim') {
    // Same view, LOD foot offset (east) only: differences = everything the LOD changes on screen when the camera moves.
    for (const dbg of ['facets', '']) {
      const tag = dbg || 'water';
      await open(`shot&t=30&scene=azure&cam=0,4.8,8,0,-16.04${dbg ? '&debug=' + dbg : ''}`);
      for (const off of [0, 0.02, 0.1, 0.3, 1.0, 3.0]) {
        await ev((o) => { window.__demo.gpu.lod.debug.footOffsetU = o; }, off);
        await frames(3);
        await shot(`swim-${tag}-off${off}`);
      }
      await ev(() => { window.__demo.gpu.lod.debug.noSnap = true; window.__demo.gpu.lod.debug.footOffsetU = 0.1; });
      await shot(`swim-${tag}-nosnap-off0.1`);
      await ev(() => { window.__demo.gpu.lod.debug.footOffsetU = 0; });
      await shot(`swim-${tag}-nosnap-off0`);
    }
  } else if (exp === 'move') {
    // Real camera motion sequence: strafe east 0.05 m/frame; frames saved for flicker inspection.
    await open('shot&t=30&scene=azure&cam=0,4.8,8,0,-16.04&debug=facets');
    for (let i = 0; i < 6; i++) {
      await ev((x) => { window.__demo.freeCam.x = x; }, i * 0.25);
      await frames(2);
      await shot(`move-facets-${i}`);
    }
  } else if (exp === 'tier') {
    await open('shot&t=30&scene=azure&cam=0,4.8,8,0,-16.04');
    const r = [];
    for (const t of ['ultra', 'low', 'medium', 'high', 'ultra', 'high']) {
      const res = await ev(async (t) => {
        const d = window.__demo;
        const t0 = performance.now();
        d.gpu.setTier(t);
        const t1 = performance.now();
        // the next 3 frame intervals
        const iv = [];
        let last = performance.now();
        await new Promise((res) => { let k = 0; const f = (now) => { iv.push(+(now - last).toFixed(1)); last = now; if (++k >= 4) res(); else requestAnimationFrame(f); }; requestAnimationFrame(f); });
        const s = d.gpu.lod.stats();
        return { tier: t, setTierMs: +(t1 - t0).toFixed(1), nextFrameIntervals: iv, verts: s.vertices, tris: s.triangles, drawnTris: s.drawnTriangles };
      }, t);
      console.log(JSON.stringify(res));
      r.push(res);
    }
    // altitude change cost (no geometry rebuild expected)
  } else if (exp === 'far') {
    for (const far of [20000, 5000, 3000, 1e6]) {
      await open('shot&t=30&scene=azure&cam=0,4.8,8,90,0&debug=lod');
      await ev((f) => { const c = window.__demo.camera; c.far = f; c.updateProjectionMatrix(); }, far);
      await frames(3);
      console.log(far, JSON.stringify(await stats()));
      await shot(`far-${far}-lod`);
    }
    await open('shot&t=30&scene=azure&cam=0,4.8,8,90,0');
    await ev(() => { const c = window.__demo.camera; c.far = 5000; c.updateProjectionMatrix(); });
    await shot('far-5000-water');
    await ev(() => { const c = window.__demo.camera; c.far = Infinity; c.updateProjectionMatrix(); });
    await frames(3);
    console.log('Infinity', JSON.stringify(await stats()));
    await shot('far-inf-water');
  } else if (exp === 'far2') {
    for (const [far, dbg] of [[20000, 'lod'], [5000, 'lod'], [5000, ''], [Infinity, 'lod']]) {
      await open(`shot&t=30&scene=azure&cam=0,400,8,0,-3${dbg ? '&debug=' + dbg : ''}`);
      await ev((f) => { const c = window.__demo.camera; c.far = f === null ? Infinity : f; c.updateProjectionMatrix(); }, far === Infinity ? null : far);
      await frames(3);
      console.log(far, dbg, JSON.stringify(await stats()));
      await shot(`far2-h400-${far}-${dbg || 'water'}`);
    }
  } else if (exp === 'info') {
    await open('shot&t=30&scene=storm&cam=0,4.8,8,0,-16.04');
    const r = await ev(async () => {
      const d = window.__demo, info = d.renderer.info;
      const before = { calls: info.render.calls, drawCalls: info.render.drawCalls, triangles: info.render.triangles };
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const meshes = []; d.scene.traverse((o) => { if (o.isMesh) meshes.push({ name: o.name, frustumCulled: o.frustumCulled, visible: o.visible, drawRange: o.geometry?.drawRange?.count }); });
      return { render: { calls: info.render.calls, drawCalls: info.render.drawCalls, frameCalls: info.render.frameCalls, triangles: info.render.triangles }, meshes };
    });
    console.log(JSON.stringify(r));
  } else if (exp === 'origin') {
    await open('shot&t=30&scene=storm&cam=0,4.8,8,0,-16.04');
    await shot('origin-base');
    // live shift of the floating origin with the camera kept at the same WORLD position
    await ev(() => window.__demo.origin.setOrigin(100000, 0, 100000));
    await shot('origin-live-100k');
    console.log(JSON.stringify(await ev(() => ({ cam: window.__demo.camera.position.toArray(), origin: window.__demo.origin.origin }))));
    await ev(() => window.__demo.origin.setOrigin(-2e6, 0, 3e6));
    await shot('origin-live-2e6');
    await ev(() => window.__demo.origin.setOrigin(0, 0, 0));
    await shot('origin-back-0');
  } else if (exp === 'alloc') {
    await open('shot&t=30&scene=azure&cam=0,4.8,8,0,-16.04');
    const r = await ev(async () => {
      const d = window.__demo, lod = d.gpu.lod, cam = d.camera;
      for (let i = 0; i < 2000; i++) lod.update(cam);
      const m0 = performance.memory?.usedJSHeapSize;
      const t0 = performance.now();
      for (let i = 0; i < 200000; i++) { cam.matrixWorld.elements[12] = i * 0.01; lod.update(cam); }
      const t1 = performance.now();
      const m1 = performance.memory?.usedJSHeapSize;
      cam.updateMatrixWorld();
      return { usPerUpdate: ((t1 - t0) / 200000 * 1000).toFixed(2), heapDelta: m1 - m0 };
    });
    console.log(JSON.stringify(r));
  }
} finally {
  await browser.close();
}
