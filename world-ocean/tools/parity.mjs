#!/usr/bin/env node
// CPU = GPU parity matrix (spec section 7 checks 6 + 7) on the demo's ?parity mode, headless Edge, hardware WebGPU.
//   node tools/parity.mjs [--base http://127.0.0.1:5197] [--presets calm,breeze,windy,storm]
//        [--tiers low,medium,high,ultra] [--origins "0,0,0;100000,0,100000"] [--cam 3,12,-5,30,-25]
//        [--points 256] [--readbacks 90] [--extra "&cpufilter=bilinear"] [--out file.json]
// Per case: load ?parity, wait until the tier is live and readbacks flow, reset the statistics, collect --readbacks
// readbacks (each = all points at one simulation time, the sim keeps running), print max / p99 / rms of
// (drawn GPU height - CPU WATER_QUERY height) over the non-faded points (< 200 m), in cm and % of Hs.
// The camera keeps the same WORLD position for every origin (?origin only moves the floating origin).
// Exit code 1 when any case fails (max >= 2 % Hs), has WebGPU errors, or produced no data.
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
console.log(`${pad('preset', 7)} ${pad('tier', 7)} ${pad('origin', 15)} ${lpad('Hs m', 6)} ${lpad('max cm', 8)} ${lpad('max %Hs', 8)} ${lpad('p99 cm', 7)} ${lpad('rms cm', 7)} ${lpad('near30 max', 10)} ${lpad('faded max', 9)} ${lpad('field>200', 9)} ${lpad('res mm', 7)} ${lpad('lat', 4)}  result`);
try {
  for (const preset of presets) for (const tier of tiers) for (const origin of origins) {
    const page = await context.newPage();
    const watch = watchPage(page, { label: `${preset}/${tier}/${origin}`, quiet: true });
    const url = `${base}/demo/?parity=${points}&preset=${preset}&tier=${tier}&cam=${cam}&origin=${origin}&seabed=0${extra}`;
    const rec = { preset, tier, origin, url };
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
    if (watch.webgpuErrors.length) rec.webgpuErrors = watch.webgpuErrors.slice(0, 5);
    await page.close();
    const ok = !rec.error && !rec.webgpuErrors && rec.pass === true;
    if (!ok) exit = 1;
    out.cases.push(rec);
    const g = rec.gate;
    if (g) {
      console.log(`${pad(preset, 7)} ${pad(tier, 7)} ${pad(origin, 15)} ${lpad(rec.hs.toFixed(2), 6)} ${lpad((g.max * 100).toFixed(2), 8)} ${lpad(g.maxPctHs.toFixed(2), 8)} ${lpad((g.p99 * 100).toFixed(2), 7)} ${lpad((g.rms * 100).toFixed(2), 7)} ${lpad((rec.near30.max * 100).toFixed(2), 10)} ${lpad((rec.faded.max * 100).toFixed(1), 9)} ${lpad((rec.fieldFaded.max * 100).toFixed(1), 9)} ${lpad((rec.residualMaxM * 1000).toFixed(2), 7)} ${lpad(rec.latencyMean.toFixed(1), 4)}  ${ok ? 'PASS' : 'FAIL'}${rec.webgpuErrors ? ' (webgpu errors)' : ''}`);
    } else {
      console.log(`${pad(preset, 7)} ${pad(tier, 7)} ${pad(origin, 15)} ERROR ${rec.error ?? ''} ${rec.webgpuErrors ? rec.webgpuErrors[0] : ''}`);
    }
  }
} finally {
  await browser.close();
}
if (opts.out) writeFileSync(opts.out, JSON.stringify(out, null, 2));
const fails = out.cases.filter((c) => !(c.pass === true && !c.error && !c.webgpuErrors)).length;
console.log(`\n${out.cases.length - fails}/${out.cases.length} cases pass (gate: max |drawn - CPU| < 2 % Hs over points < 200 m; faded / field>200 = metres beyond the vertex far fade, informational).`);
process.exit(exit);
