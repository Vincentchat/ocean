#!/usr/bin/env node
// Screenshot harness: headless Edge (hardware WebGPU on the RTX 2070) via playwright-core.
//
// Single shot:
//   node tools/capture.mjs --url "http://127.0.0.1:5180/demo/?shot&t=10&cam=0,4.8,8,0,-16" --out shot.png
//        [--width 1920 --height 1080] [--wait-ready] [--eval "<js run in the page before the screenshot>"]
//        [--timeout 90000] [--headed] [--allow-errors]
// Batch:
//   node tools/capture.mjs --batch tools/shots.json --base http://127.0.0.1:5180/demo/ --outdir ../docs/port/shots
//        [--target demo|reference] [--only name1,name2] [--set canonical|reference-md]
//   shots.json: [{ "name", "query" | "url", "demoQuery"?, "referenceQuery"?, "width"?, "height"? }, ...]
//   The page URL is base + "?" + (targetQuery ?? query). Output: <outdir>/<name>.png
//
// --wait-ready waits for window.__shotReady === true (demo ?shot and the reference ?shot hook both set it).
// Console errors / warnings from the page are printed. Exit code 1 on WebGPU errors / page errors / uncaught
// exceptions, on a software (SwiftShader/WARP) adapter, or when the ready flag never appears.
import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { ADAPTER_PROBE, isSoftwareAdapter, launch, parseArgs, watchPage } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const width = Number(opts.width ?? 1920), height = Number(opts.height ?? 1080);
const timeout = Number(opts.timeout ?? 120000);

function fail(msg) {
  console.error(`capture: ${msg}`);
  process.exitCode = 1;
}

async function captureOne(context, { url, out, waitReady, evalJs, w = width, h = height, label }) {
  const page = await context.newPage();
  await page.setViewportSize({ width: w, height: h });
  const watch = watchPage(page, { label });
  const t0 = Date.now();
  let ok = true;
  try {
    await page.goto(url, { waitUntil: 'load', timeout });
    const adapter = await page.evaluate(ADAPTER_PROBE);
    if (!adapter.ok) throw new Error(`no WebGPU adapter: ${adapter.error}`);
    if (isSoftwareAdapter(adapter)) throw new Error(`software WebGPU adapter: ${JSON.stringify(adapter)}`);
    if (waitReady) {
      await page.waitForFunction(() => window.__shotReady === true || !!window.__demoFatal, null, { timeout, polling: 100 });
      const fatalMsg = await page.evaluate(() => window.__demoFatal || null);
      if (fatalMsg) throw new Error(`page reported fatal: ${fatalMsg}`);
    } else {
      await page.waitForTimeout(Number(opts.delay ?? 3000));
    }
    if (evalJs) {
      const r = await page.evaluate(evalJs);
      if (r !== undefined) console.log(`[${label}] eval -> ${typeof r === 'string' ? r : JSON.stringify(r)}`);
    }
    // Two more animation frames so the last presented frame is the settled one.
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    mkdirSync(dirname(resolve(out)), { recursive: true });
    await page.screenshot({ path: out, type: 'png' });
    console.log(`[${label}] wrote ${out} (${w}x${h}, ${((Date.now() - t0) / 1000).toFixed(1)} s, adapter ${adapter.vendor}/${adapter.architecture} ${adapter.description || ''})`);
  } catch (e) {
    ok = false;
    fail(`[${label}] ${e.message || e}`);
  }
  if (watch.webgpuErrors.length && !opts['allow-errors']) {
    ok = false;
    fail(`[${label}] ${watch.webgpuErrors.length} WebGPU/page error(s):\n  ${watch.webgpuErrors.join('\n  ')}`);
  }
  await page.close();
  return ok;
}

const mode = opts.headed ? 'headed' : 'headless';
const { browser, context } = await launch({ mode, width, height });
try {
  if (opts.batch) {
    const list = JSON.parse(readFileSync(opts.batch, 'utf8'));
    const shots = Array.isArray(list) ? list : list.shots;
    const only = opts.only ? new Set(String(opts.only).split(',')) : null;
    const target = opts.target ?? 'demo';
    const base = opts.base;
    const outdir = opts.outdir ?? '.';
    if (!base) throw new Error('--batch needs --base <page url>');
    let failures = 0;
    const set = opts.set ? new Set(String(opts.set).split(',')) : null;
    for (const s of shots) {
      if (only && !only.has(s.name)) continue;
      if (set && !set.has(s.set)) continue;
      const q = s[`${target}Query`] ?? s.query ?? '';
      const url = s.url ?? `${base}${base.includes('?') ? '&' : '?'}${q}`;
      const ok = await captureOne(context, {
        url, out: resolve(outdir, `${s.name}.png`), waitReady: !opts['no-wait-ready'], evalJs: s.eval ?? opts.eval,
        w: s.width ?? width, h: s.height ?? height, label: s.name,
      });
      if (!ok) failures++;
    }
    if (failures) fail(`${failures} shot(s) failed`);
  } else {
    if (!opts.url || !opts.out) throw new Error('usage: --url <url> --out <png> [--wait-ready] [--eval js] | --batch shots.json --base url --outdir dir');
    await captureOne(context, { url: opts.url, out: opts.out, waitReady: !!opts['wait-ready'], evalJs: opts.eval, label: 'shot' });
  }
} catch (e) {
  fail(e.message || String(e));
} finally {
  await browser.close();
}
