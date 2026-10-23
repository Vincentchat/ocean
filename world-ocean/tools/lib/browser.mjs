// Shared browser launcher for the capture / perf harness.
// Drives Microsoft Edge (Chromium) through playwright-core with WebGPU enabled on the real GPU.
import { existsSync } from 'node:fs';
import { chromium } from 'playwright-core';

export const EDGE_PATHS = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
];

/**
 * Flags that give a hardware (NVIDIA, D3D12 backend) WebGPU adapter in headless Edge on Windows.
 * Verified with tools/probe-gpu.mjs (see README of the tools / final report).
 */
export const WEBGPU_ARGS = [
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features', // exposes adapter.info.description + timestamp queries without quantization
  '--enable-dawn-features=allow_unsafe_apis',
  '--ignore-gpu-blocklist',
  '--enable-gpu',
  '--use-angle=d3d11',
  '--disable-gpu-sandbox',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows',
  '--force-color-profile=srgb',
  '--hide-scrollbars',
  '--mute-audio',
];

export function edgePath() {
  const p = process.env.EDGE_PATH || EDGE_PATHS.find((x) => existsSync(x));
  if (!p) throw new Error('Microsoft Edge not found; set EDGE_PATH');
  return p;
}

/** Launch Edge. mode: 'headless' (new headless, default) | 'headed' (window parked off-screen). */
export async function launch({ mode = 'headless', width = 1920, height = 1080, extraArgs = [] } = {}) {
  const args = [...WEBGPU_ARGS, ...extraArgs];
  if (mode === 'headed') args.push('--window-position=-32000,-32000', `--window-size=${width},${height + 140}`);
  const browser = await chromium.launch({
    executablePath: edgePath(),
    headless: mode !== 'headed',
    args,
  });
  const context = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: 1,
  });
  return { browser, context };
}

/** Run in the page: returns adapter info (or an error string). */
export const ADAPTER_PROBE = `(async () => {
  if (!navigator.gpu) return { ok: false, error: 'navigator.gpu missing' };
  const a = await navigator.gpu.requestAdapter();
  if (!a) return { ok: false, error: 'requestAdapter returned null' };
  const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
  return { ok: true, vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description,
    isFallbackAdapter: a.isFallbackAdapter ?? i.isFallbackAdapter ?? null,
    features: [...a.features].filter((f) => f === 'timestamp-query' || f === 'shader-f16' || f === 'float32-filterable') };
})()`;

/** True when the adapter is a software rasteriser (SwiftShader / WARP / Basic Render Driver). */
export function isSoftwareAdapter(info) {
  const s = `${info.vendor} ${info.architecture} ${info.description} ${info.device}`.toLowerCase();
  return !!info.isFallbackAdapter || /swiftshader|warp|basic render|microsoft basic|llvmpipe|google/.test(s) && !/nvidia|amd|intel/.test(s);
}

/** Attach console / error forwarding. Returns a collector of problems. */
export function watchPage(page, { label = 'page', quiet = false } = {}) {
  const problems = [];
  const webgpuErrors = [];
  const WEBGPU_RE = /webgpu|wgsl|gpuvalidationerror|gpuoutofmemory|device (was )?lost|createrenderpipeline|createcomputepipeline|invalid (bindgroup|texture|buffer|shadermodule|pipeline)|tsl|nodematerial/i;
  page.on('console', (msg) => {
    const type = msg.type();
    if (type !== 'error' && type !== 'warning') return;
    const text = msg.text();
    problems.push({ type, text });
    if (!quiet) console.log(`[${label}] console.${type}: ${text}`);
    if (type === 'error' && WEBGPU_RE.test(text)) webgpuErrors.push(text);
  });
  page.on('pageerror', (err) => {
    const text = String(err?.stack || err);
    problems.push({ type: 'pageerror', text });
    if (!quiet) console.log(`[${label}] pageerror: ${text}`);
    webgpuErrors.push(text);
  });
  page.on('requestfailed', (req) => {
    const text = `${req.url()} ${req.failure()?.errorText ?? ''}`;
    problems.push({ type: 'requestfailed', text });
    if (!quiet) console.log(`[${label}] request failed: ${text}`);
  });
  return { problems, webgpuErrors };
}

/** Minimal argv parser: --key value, --flag. */
export function parseArgs(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('--')) out[key] = true;
      else { out[key] = next; i++; }
    } else out._.push(a);
  }
  return out;
}
