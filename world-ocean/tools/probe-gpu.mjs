// Prints the WebGPU adapter Edge exposes under several launch configurations.
// Usage: node tools/probe-gpu.mjs [--mode headless|headed] [--args "--flag1 --flag2"]
import { chromium } from 'playwright-core';
import { ADAPTER_PROBE, WEBGPU_ARGS, edgePath, isSoftwareAdapter, parseArgs } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));

async function probe(name, headless, args) {
  let browser;
  try {
    browser = await chromium.launch({ executablePath: edgePath(), headless, args });
    const page = await browser.newPage();
    // Serve a blank page on an https origin so navigator.gpu is available (secure context).
    await page.route('https://probe.invalid/**', (r) => r.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>probe</title>' }));
    await page.goto('https://probe.invalid/');
    const info = await page.evaluate(ADAPTER_PROBE);
    const verdict = !info.ok ? 'NO ADAPTER' : isSoftwareAdapter(info) ? 'SOFTWARE' : 'HARDWARE';
    console.log(JSON.stringify({ config: name, verdict, ...info }));
    return { verdict, info };
  } catch (e) {
    console.log(JSON.stringify({ config: name, verdict: 'LAUNCH FAILED', error: String(e.message || e) }));
    return { verdict: 'LAUNCH FAILED' };
  } finally {
    await browser?.close();
  }
}

if (opts.args || opts.mode) {
  const args = opts.args ? String(opts.args).split(/\s+/).filter(Boolean) : WEBGPU_ARGS;
  await probe('custom', opts.mode !== 'headed', opts.mode === 'headed' ? [...args, '--window-position=-32000,-32000'] : args);
} else {
  await probe('headless:default-flags-only(--enable-unsafe-webgpu)', true, ['--enable-unsafe-webgpu']);
  await probe('headless:harness-flags', true, WEBGPU_ARGS);
  await probe('headless:harness-flags+vulkan', true, [...WEBGPU_ARGS, '--enable-features=Vulkan', '--use-webgpu-adapter=default']);
  await probe('headed-offscreen:harness-flags', false, [...WEBGPU_ARGS, '--window-position=-32000,-32000']);
}
