#!/usr/bin/env node
// CPU = GPU parity matrix (spec section 7 checks 6 + 7) on the demo's ?parity mode, headless Edge, hardware WebGPU.
//   node tools/parity.mjs [--base http://127.0.0.1:5197] [--presets calm,breeze,windy,storm]
//        [--tiers low,medium,high,ultra] [--origins "0,0,0;100000,0,100000"] [--cam 3,12,-5,30,-25]
//        [--points 256] [--readbacks 90] [--extra "&cpufilter=bilinear"] [--out file.json]
// Per case: load ?parity, wait until the tier is live and readbacks flow, reset the statistics, collect --readbacks
// readbacks (each = all points at one simulation time, the sim keeps running), print max / p99 / rms in cm and
// % of Hs of  A: GPU field (full-resolution GPU surface: LOD 0 Catmull-Rom, no far fade) - CPU WATER_QUERY, all
// points (1-450 m);  B: DRAWN surface (vertex footprint mips + far fade, as the vertex shader) - CPU where the vertex
// shader uses the CPU's filter for cascades 0 + 1 (pure Catmull-Rom: footprint <= 0.375 m); and the drawn error per distance band\n// (< 25 m, 25-100 m, 100-200 m, > 200 m: view-dependent, not gated).
// The camera keeps the same WORLD position for every origin (?origin only moves the floating origin).
// Exit code 1 when any case fails (A or B max >= 2 % Hs), has WebGPU errors, or produced no data.
import { writeFileSync } from 'node:fs';
import { ADAPTER_PROBE, isSoftwareAdapter, launch, parseArgs, watchPage } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const base = opts.base ?? 'http://127.0.0.1:5197';
const presets = String(opts.presets ?? 'calm,breeze,windy,storm').split(',');
const tiers = String(opts.tiers ?? 'low,medium,high,ultra').split(',');
const origins = String(opts.origins ?? '0,0,0;100000,0,100000').split(';');
const cam = opts.cam ?? '3,12,-5,30,-25';
const points = Number(opts.points ?? 256);
const readbacks = Number(opts.readbacks ?? 90);
const extra = opts.extra ?? '';

const { browser, context } = await launch({ width: 1280, height: 720 });
const out = { date: new Date().toISOString(), cam, points, readbacks, extra, adapter: null, cases: [] };
let exit = 0;
const pad = (s, n) => String(s).padEnd(n);
const lpad = (s, n) => String(s).padStart(n);
const COLS = [
  ['preset', 7], ['tier', 6], ['origin', 15], ['Hs m', 5],
  ['A:field max cm', 14], ['%Hs', 5], ['p99', 5], ['rms', 5],
  ['B:drawn cubic max', 17], ['%Hs', 5], ['rms', 5],
  ['<25 m', 6], ['25-100', 7], ['100-200', 7], ['>200', 6], ['res mm', 6], ['lat', 4],
];
console.log(COLS.map(([h, n], i) => (i < 3 ? pad(h, n) : lpad(h, n))).join(' ') + '  result   (cm unless noted; <25 / 25-100 / 100-200 / >200 m = drawn max per distance band, not gated)');
try {
  for (const preset of presets) for (const tier of tiers) for (const origin of origins) {
    const url = `${base}/demo/?parity=${points}&preset=${preset}&tier=${tier}&cam=${cam}&origin=${origin}&seabed=0${extra}`;
    let rec, watch, page;
    // one retry: a dev-server reload (another builder saving a file) mid-run is not a parity failure
    for (let attempt = 0; attempt < 2; attempt++) {
    if (page) await page.close();
    page = await context.newPage();
    watch = watchPage(page, { label: `${preset}/${tier}/${origin}`, quiet: true });
    rec = { preset, tier, origin, url, attempt };
    try {
      await page.goto(url, { waitUntil: 'load', timeout: 120000 });
      if (!out.adapter) {
        const a = await page.evaluate(ADAPTER_PROBE);
        if (!a.ok || isSoftwareAdapter(a)) throw new Error(`no hardware WebGPU adapter: ${JSON.stringify(a)}`);
        out.adapter = a;
      }
      await page.waitForFunction(() => !!window.__demoFatal || (window.__parity && window.__parity.readbacks > 10 && !window.__parity.summary().pendingTier), null, { timeout: 120000, polling: 100 });
      const fatal = await page.evaluate(() => window.__demoFatal ?? null);
      if (fatal) throw new Error(fatal);
      await page.evaluate(() => window.__parity.reset());
      await page.waitForFunction((n) => window.__parity.readbacks >= n, readbacks, { timeout: 120000, polling: 100 });
      Object.assign(rec, await page.evaluate(() => window.__parity.summary()));
    } catch (e) {
      rec.error = String(e?.message ?? e);
    }
    if (!rec.error && !watch.webgpuErrors.length) break;
    }
    if (watch.webgpuErrors.length) rec.webgpuErrors = watch.webgpuErrors.slice(0, 5);
    await page.close();
    const ok = !rec.error && !rec.webgpuErrors && rec.pass === true;
    if (!ok) exit = 1;
    out.cases.push(rec);
    const a = rec.field, b = rec.drawnCubic;
    if (a && b) {
      const c = (x) => (x * 100).toFixed(2);
      const v = [preset, tier, origin, rec.hs.toFixed(2), c(a.max), a.maxPctHs.toFixed(2), c(a.p99), c(a.rms), c(b.max), b.maxPctHs.toFixed(2), c(b.rms),
        c(rec.drawnNear.max), c(rec.drawnMid.max), c(rec.drawnFar.max), (rec.drawnFaded.max * 100).toFixed(1), (rec.residualMaxM * 1000).toFixed(2), rec.latencyMean.toFixed(1)];
      console.log(v.map((s, i) => (i < 3 ? pad(s, COLS[i][1]) : lpad(s, COLS[i][1]))).join(' ') + `  ${ok ? 'PASS' : 'FAIL'}${rec.webgpuErrors ? ' (webgpu errors)' : ''}`);
    } else {
      console.log(`${pad(preset, 7)} ${pad(tier, 6)} ${pad(origin, 15)} ERROR ${rec.error ?? ''} ${rec.webgpuErrors ? rec.webgpuErrors[0] : ''}`);
    }
  }
} finally {
  await browser.close();
}
if (opts.out) writeFileSync(opts.out, JSON.stringify(out, null, 2));
const fails = out.cases.filter((c) => !(c.pass === true && !c.error && !c.webgpuErrors)).length;
console.log(`\n${out.cases.length - fails}/${out.cases.length} cases pass (A: max |GPU field (LOD 0, no fade) - CPU| < 2 % Hs at all points; B: max |drawn - CPU| < 2 % Hs where the vertex shader samples c0 + c1 with pure Catmull-Rom).`);
process.exit(exit);
