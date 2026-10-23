#!/usr/bin/env node
// Foam verification driver (three/foam.ts) on the demo's ?debug=foam view, headless Edge with hardware WebGPU.
//   node tools/foam-check.mjs --base http://127.0.0.1:5189 [--tiers low,medium,high,ultra] [--scene storm]
//        [--cam 0,60,8,0,-42.97] [--t 30] [--bench-frames 600] [--extra foamfine] [--no-bench] [--out file.json]
// Per tier:
//   stats  deterministic ?shot (6 s fixed-step warm-up, then paused at t): the foam history buffer is read back
//          and summarised (mean, p50/p90/p99, fraction above the thresholds where the lace starts to show).
//   bench  free-running page (not paused, uncapped): GPU time of the foam dispatch itself, read per dispatch from
//          three's compute timestamp query pool, averaged over N dispatches (plus the per-frame total of all
//          compute dispatches for context).
import { writeFileSync } from 'node:fs';
import { ADAPTER_PROBE, isSoftwareAdapter, launch, parseArgs, watchPage } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const base = opts.base ?? 'http://127.0.0.1:5189';
const tiers = String(opts.tiers ?? 'low,medium,high,ultra').split(',');
const scene = opts.scene ?? 'storm';
const cam = opts.cam ?? '0,60,8,0,-42.97';
const t = Number(opts.t ?? 30);
const benchFrames = Number(opts['bench-frames'] ?? 600);
const extra = opts.extra ? `,${opts.extra}` : '';

// Stats run vsynced (the demo's fixed-count warm-up then spans 6 real seconds, so the FFT's ~1 s async compile
// does not eat the warm-up); the bench runs uncapped in a second browser.
const { browser, context } = await launch({ width: 1920, height: 1080 });
const bench = opts['no-bench'] ? null : await launch({ width: 1920, height: 1080, extraArgs: ['--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const out = { date: new Date().toISOString(), scene, cam, t, extra: opts.extra ?? '', adapter: null, tiers: {} };
let exit = 0;
const summarise = `(async () => {
  const f = window.__foam.foam;
  const h = await f.debug.readHistory();
  const s = Array.from(h).sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  let sum = 0, a03 = 0, a0625 = 0; for (const v of h) { sum += v; if (v > 0.3) a03++; if (v > 0.625) a0625++; }
  return { res: f.debug.resolution(), mean: sum / h.length, p50: q(0.5), p90: q(0.9), p99: q(0.99), max: s[s.length - 1],
    over03: a03 / h.length, over0625: a0625 / h.length, steps: f.debug.steps, patternLoaded: f.debug.patternLoaded(),
    detailGain: f.uniforms.detailGain.value, fineInSource: f.uniforms.fineInSource.value,
    memoryMB: f.debug.memoryBytes() / 1048576 };
})()`;
try {
  for (const tier of tiers) {
    const res = {};
    const page = await context.newPage();
    const watch = watchPage(page, { label: `foam-${tier}` });
    await page.goto(`${base}/demo/?shot&t=${t}&scene=${scene}&cam=${cam}&tier=${tier}&debug=foam${extra}`, { waitUntil: 'load', timeout: 120000 });
    if (!out.adapter) {
      const a = await page.evaluate(ADAPTER_PROBE);
      if (!a.ok || isSoftwareAdapter(a)) throw new Error(`no hardware WebGPU adapter: ${JSON.stringify(a)}`);
      out.adapter = a;
    }
    await page.waitForFunction(() => window.__shotReady === true || !!window.__demoFatal, null, { timeout: 120000, polling: 100 });
    res.stats = await page.evaluate(summarise);
    await page.close();
    if (watch.webgpuErrors.length) { exit = 1; res.errors = watch.webgpuErrors; }
    if (!opts['no-bench']) {
      const bp = await bench.context.newPage();
      const bw = watchPage(bp, { label: `bench-${tier}` });
      await bp.goto(`${base}/demo/?t=${t}&scene=${scene}&cam=${cam}&tier=${tier}&debug=foam${extra}`, { waitUntil: 'load', timeout: 120000 });
      await bp.waitForFunction(() => window.__foam && window.__perf && window.__perf.count > 120 && window.__foam.foam.debug.steps > 60, null, { timeout: 120000 });
      // Per-dispatch GPU time of the foam kernel from three's compute timestamp pool (uids 'c:<n>:<nodeId>:f<frame>'),
      // collected every frame while the demo keeps resolving the compute queries.
      res.bench = await bp.evaluate(async (frames) => {
        const be = window.__demo.renderer.backend;
        const id = window.__foam.foam.debug.kernelId();
        const tag = `:${id}:f`;
        const seen = new Set(), foam = [], perFrame = new Map();
        await new Promise((done) => {
          const poll = () => {
            const pool = be.timestampQueryPool && be.timestampQueryPool.compute;
            if (pool) for (const [uid, d] of pool.timestamps) {
              if (seen.has(uid)) continue;
              seen.add(uid);
              const f = uid.slice(uid.lastIndexOf(':f') + 2);
              perFrame.set(f, (perFrame.get(f) ?? 0) + d);
              if (uid.includes(tag)) foam.push(d);
            }
            if (foam.length >= frames) done(); else requestAnimationFrame(poll);
          };
          poll();
        });
        const s = foam.slice().sort((a, b) => a - b);
        const all = [...perFrame.values()];
        return { foamMs: foam.reduce((a, b) => a + b, 0) / foam.length, p50: s[s.length >> 1], p95: s[Math.floor(s.length * 0.95)], n: foam.length,
          allComputeMs: all.reduce((a, b) => a + b, 0) / Math.max(all.length, 1), kernelId: id };
      }, benchFrames);
      await bp.close();
      if (bw.webgpuErrors.length) { exit = 1; res.benchErrors = bw.webgpuErrors; }
    }
    out.tiers[tier] = res;
    console.error(`[foam-check] ${tier}: res ${res.stats.res} gain ${res.stats.detailGain.toFixed(4)} mean ${res.stats.mean.toFixed(4)} p90 ${res.stats.p90.toFixed(3)} over.3 ${(100 * res.stats.over03).toFixed(2)} % over.625 ${(100 * res.stats.over0625).toFixed(2)} %` +
      (res.bench ? `  foam GPU ${res.bench.foamMs.toFixed(3)} ms (p50 ${res.bench.p50.toFixed(3)} p95 ${res.bench.p95.toFixed(3)}, n ${res.bench.n}; all compute ${res.bench.allComputeMs.toFixed(3)} ms/frame)` : ''));
  }
} catch (e) {
  console.error(`foam-check: ${e.message || e}`);
  exit = 1;
} finally {
  await browser.close();
  if (bench) await bench.browser.close();
}
const json = JSON.stringify(out, null, 2);
console.log(json);
if (opts.out) writeFileSync(opts.out, json + '\n');
process.exit(exit);
