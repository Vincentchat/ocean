#!/usr/bin/env node
// Tier-switch capture: paused demo shot, screenshot, switch the FFT tier only (gpu.fft.setTier) in the same page,
// screenshot again. With tier invariance only fine detail may change (compare with tools/imgdiff.mjs).
//   node tools/fft-tier-switch.mjs --base http://127.0.0.1:5182 --from medium --to low [--cam 0,4.8,8,0,-16]
//        [--query "&scene=storm&debug=facets"] [--out dir] [--label name] [--cap (also enable the capillary band)]
import { launch, parseArgs } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const base = opts.base ?? 'http://127.0.0.1:5182';
const from = opts.from ?? 'medium', to = opts.to ?? 'low';
const cam = opts.cam ?? '0,4.8,8,0,-16';
const out = opts.out ?? '.';
const label = opts.label ?? 'switch';
const { browser, context } = await launch({ width: 1600, height: 900 });
try {
  const page = await context.newPage();
  page.on('pageerror', (e) => console.error(`[pageerror] ${e}`));
  await page.goto(`${base}/demo/?shot&t=30&cam=${cam}&tier=${from}${opts.query ?? ''}`, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction(() => window.__shotReady === true, null, { timeout: 120000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${out}/${label}-${from}.png` });
  await page.evaluate(([t, cap]) => { const f = window.__demo.gpu.fft; if (cap) f.setCapillary(true); f.setTier(t); }, [to, !!opts.cap]);
  await page.waitForFunction(([t, cap]) => { const f = window.__demo.gpu.fft; return f.tier === t && !f.pendingTier && (!cap || f.capillary); }, [to, !!opts.cap], { timeout: 120000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${out}/${label}-${to}.png` });
  console.log(JSON.stringify({ a: `${out}/${label}-${from}.png`, b: `${out}/${label}-${to}.png` }));
} finally { await browser.close(); }
