#!/usr/bin/env node
// Debug helper: open a URL (default fft-debug), wait, print console, #stats, #results and window.__fftCheck.
//   node tools/fft-probe.mjs --url <url> [--wait 15] [--shot file.png] [--eval 'js expression run after 6 s']
import { launch, parseArgs } from './lib/browser.mjs';
const opts = parseArgs(process.argv.slice(2));
const url = opts.url ?? 'http://127.0.0.1:5182/demo/fft-debug.html?tier=medium&check';
const { browser, context } = await launch({ width: 1600, height: 900 });
try {
  const page = await context.newPage();
  page.on('console', (m) => console.log(`[console.${m.type()}] ${m.text().slice(0, 2000)}`));
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.stack || e}`));
  await page.goto(url, { waitUntil: 'load', timeout: 120000 });
  if (opts.eval) { await page.waitForTimeout(6000); console.log('[eval]', JSON.stringify(await page.evaluate(opts.eval))); }
  await page.waitForTimeout(Number(opts.wait ?? 15) * 1000);
  const r = await page.evaluate(() => ({
    stats: document.getElementById('stats')?.textContent, results: document.getElementById('results')?.textContent,
    check: window.__fftCheck ? JSON.stringify({ done: window.__fftCheck.done, error: window.__fftCheck.error, n: window.__fftCheck.tests.length }) : null,
  }));
  console.log(JSON.stringify(r, null, 1));
  if (opts.shot) await page.screenshot({ path: opts.shot });
} finally { await browser.close(); }
