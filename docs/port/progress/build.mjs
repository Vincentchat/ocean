// Builds docs/port/progress/index.html from docs/port/STATUS.json.
// Images referenced in STATUS.json (paths relative to repo root) are downscaled to JPEG data URIs via headless Edge.
// Usage: node docs/port/progress/build.mjs
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../../..');
const status = JSON.parse(readFileSync(resolve(root, 'docs/port/STATUS.json'), 'utf8'));
const require = createRequire(resolve(root, 'world-ocean/package.json'));

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

async function encodeImages(paths) {
  const out = new Map();
  const todo = paths.filter((p) => p && existsSync(resolve(root, p)));
  if (!todo.length) return out;
  const { chromium } = require('playwright-core');
  const browser = await chromium.launch({ executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', headless: true });
  const page = await browser.newPage();
  for (const p of todo) {
    const b64 = readFileSync(resolve(root, p)).toString('base64');
    const mime = p.endsWith('.png') ? 'image/png' : 'image/jpeg';
    const data = await page.evaluate(async ({ b64, mime }) => {
      const img = new Image();
      img.src = `data:${mime};base64,${b64}`;
      await img.decode();
      const w = Math.min(960, img.naturalWidth), h = Math.round(img.naturalHeight * (w / img.naturalWidth));
      const c = document.createElement('canvas'); c.width = w; c.height = h;
      c.getContext('2d').drawImage(img, 0, 0, w, h);
      return c.toDataURL('image/jpeg', 0.82);
    }, { b64, mime });
    out.set(p, data);
  }
  await browser.close();
  return out;
}

const shots = status.shots ?? [];
const imgs = await encodeImages(shots.flatMap((s) => [s.webgpu, s.reference]));
const img = (p, alt) => (imgs.has(p) ? `<img src="${imgs.get(p)}" alt="${esc(alt)}">` : `<div class="noimg">not captured yet</div>`);

const statusClass = (s) => ({ done: 'ok', pass: 'ok', passed: 'ok', building: 'run', reviewing: 'run', 'in progress': 'run', blocked: 'bad', fail: 'bad', failed: 'bad' })[String(s).toLowerCase()] ?? 'idle';
const pill = (s) => `<span class="pill ${statusClass(s)}">${esc(s)}</span>`;

const html = `<title>Ocean WebGPU Port</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&family=Fraunces:opsz,wght@9..144,600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: one column log-book — live ocean comparison first, then pieces, perf, checks, issues. */
:root {
  --bg: #f3f6f7; --panel: #ffffff; --fg: #15232b; --muted: #5b6d77; --line: #d5dfe3; --accent: #0b6e7f;
  --ok: #1d7a46; --run: #a76a00; --bad: #b3261e; --idle: #6b7a83;
  --display: "Fraunces", Georgia, serif; --body: "Instrument Sans", "Segoe UI", system-ui, sans-serif; --mono: "JetBrains Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d171c; --panel: #13222a; --fg: #e3edf0; --muted: #93a8b2; --line: #24363f; --accent: #4fc3d4;
  --ok: #5cc98a; --run: #e2a53b; --bad: #f2756b; --idle: #8a9aa3; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0d171c; --panel: #13222a; --fg: #e3edf0; --muted: #93a8b2; --line: #24363f; --accent: #4fc3d4;
  --ok: #5cc98a; --run: #e2a53b; --bad: #f2756b; --idle: #8a9aa3; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }
main { max-width: 1180px; margin: 0 auto; padding-block: 28px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 36px; }
h1 { font: 600 clamp(28px, 4vw, 40px)/1.1 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 20px/1.2 var(--display); margin: 0 0 12px; }
.eyebrow { font: 600 12px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
.sub { color: var(--muted); max-width: 70ch; margin: 8px 0 0; }
.meta { font: 12px var(--mono); color: var(--muted); margin-top: 6px; }
.shots { display: grid; gap: 22px; }
.shot figure { margin: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.shot figcaption { grid-column: 1 / -1; color: var(--muted); font-size: 13px; }
.shot .lab { font: 600 11px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
.shot img, .noimg { width: 100%; aspect-ratio: 16/9; object-fit: cover; border-radius: 6px; border: 1px solid var(--line); background: var(--panel); }
.noimg { display: grid; place-items: center; color: var(--muted); font: 12px var(--mono); }
.tbl { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font: 600 11px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
tr:last-child td { border-bottom: 0; }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pill { display: inline-block; font: 600 11px var(--mono); padding: 2px 8px; border-radius: 999px; border: 1px solid currentColor; white-space: nowrap; }
.pill.ok { color: var(--ok); } .pill.run { color: var(--run); } .pill.bad { color: var(--bad); } .pill.idle { color: var(--idle); }
ul.issues { margin: 0; padding-left: 18px; display: grid; gap: 6px; }
.sev { font: 600 11px var(--mono); text-transform: uppercase; margin-right: 6px; }
.sev.high { color: var(--bad); } .sev.medium { color: var(--run); } .sev.low { color: var(--idle); }
.log { font: 13px/1.5 var(--mono); color: var(--muted); display: grid; gap: 4px; }
.log b { color: var(--fg); font-weight: 600; }
@media (max-width: 640px) { .shot figure { grid-template-columns: 1fr; } }
</style>
<main>
<header>
  <div class="eyebrow">@dreamp/world-ocean · three r186 · WebGPU + TSL</div>
  <h1>Ocean WebGPU Port</h1>
  <p class="sub">${esc(status.summary ?? '')}</p>
  <div class="meta">Phase: ${esc(status.phase ?? '—')} · updated ${esc(status.updated ?? '')}</div>
</header>

<section><h2>Rendered ocean: WebGPU port vs WebGL reference</h2>
<div class="shots">${shots.map((s) => `<div class="shot"><figure>
  <div><div class="lab">WebGPU port${s.tier ? ' · ' + esc(s.tier) : ''}</div>${img(s.webgpu, 'WebGPU ' + s.label)}</div>
  <div><div class="lab">WebGL reference</div>${img(s.reference, 'WebGL ' + s.label)}</div>
  <figcaption><b>${esc(s.label)}</b>${s.caption ? ' — ' + esc(s.caption) : ''}</figcaption></figure></div>`).join('') || '<div class="noimg">No captures yet</div>'}</div>
</section>

<section><h2>Pieces</h2><div class="tbl"><table>
<tr><th>Piece</th><th>Status</th><th>Review rounds</th><th>Latest verdict</th></tr>
${(status.pieces ?? []).map((p) => `<tr><td><b>${esc(p.name)}</b>${p.notes ? `<div class="meta">${esc(p.notes)}</div>` : ''}</td><td>${pill(p.status)}</td><td class="num">${esc(p.rounds ?? 0)}</td><td>${esc(p.verdict ?? '')}</td></tr>`).join('')}
</table></div></section>

<section><h2>Performance (1920×1080, RTX 2070)</h2><div class="tbl"><table>
<tr><th>Build</th><th>GPU ms</th><th>CPU ms</th><th>FPS</th><th>Memory</th><th>Notes</th></tr>
${(status.perf ?? []).map((r) => `<tr><td><b>${esc(r.name)}</b></td><td class="num">${esc(r.gpuMs ?? '—')}</td><td class="num">${esc(r.cpuMs ?? '—')}</td><td class="num">${esc(r.fps ?? '—')}</td><td class="num">${esc(r.mem ?? '—')}</td><td>${esc(r.notes ?? '')}</td></tr>`).join('')}
</table></div></section>

<section><h2>Acceptance checks (spec §7)</h2><div class="tbl"><table>
<tr><th>#</th><th>Check</th><th>State</th><th>Evidence</th></tr>
${(status.checks ?? []).map((c) => `<tr><td class="num">${esc(c.id)}</td><td>${esc(c.name)}</td><td>${pill(c.status)}</td><td>${esc(c.notes ?? '')}</td></tr>`).join('')}
</table></div></section>

<section><h2>Outstanding issues</h2>
${(status.issues ?? []).length ? `<ul class="issues">${status.issues.map((i) => `<li><span class="sev ${esc(i.severity)}">${esc(i.severity)}</span>${esc(i.text)}</li>`).join('')}</ul>` : '<p class="sub">None open.</p>'}
</section>

<section><h2>Log</h2><div class="log">${(status.log ?? []).slice().reverse().map((l) => `<div><b>${esc(l.time)}</b> ${esc(l.text)}</div>`).join('')}</div></section>
</main>`;

writeFileSync(resolve(here, 'index.html'), html);
console.log('wrote', resolve(here, 'index.html'), (html.length / 1024).toFixed(0) + ' KB');
