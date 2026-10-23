// Crack / gap sweep: one demo page with ?debug=lod (magenta background, back faces culled). Moves the camera
// to N pseudo-random poses (altitudes 0.5..500 m, random pitch/yaw, non-snapped positions), screenshots, and counts
// magenta pixels below the horizon (+2 px). Any > 0 means a crack / hole / horizon gap.
//   node sweep.mjs --url "http://127.0.0.1:5181/demo/?shot&t=10&debug=lod&tier=high" [--n 40] [--outdir dir]
import { writeFileSync, mkdirSync } from 'node:fs';
import { launch, parseArgs, watchPage } from 'file:///C:/Users/Vincent/Documents/Claude/ocean/world-ocean/tools/lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const N = Number(opts.n ?? 40);
const W = 1920, H = 1080;
const { browser, context } = await launch({ width: W, height: H });
let seed = 777;
const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
try {
  const page = await context.newPage();
  const watch = watchPage(page, { label: 'sweep' });
  await page.goto(opts.url, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction(() => window.__shotReady === true, null, { timeout: 120000 });
  const ana = await context.newPage();
  let worst = 0, total = 0;
  const rows = [];
  for (let i = 0; i < N; i++) {
    const alt = i < 4 ? [0.5, 4.8, 60, 500][i] : Math.exp(Math.log(0.5) + rnd() * (Math.log(500) - Math.log(0.5)));
    const pitch = -(rnd() ** 1.5) * 89.5; // degrees
    const yaw = rnd() * 360;
    const x = (rnd() - 0.5) * 3000 + rnd() * 0.37, z = (rnd() - 0.5) * 3000 + rnd() * 0.71;
    await page.evaluate(({ x, alt, z, yaw, pitch }) => {
      const d = window.__demo; const o = d.origin.origin;
      d.freeCam.x = o.x + x; d.freeCam.y = o.y + alt; d.freeCam.z = o.z + z;
      d.freeCam.yaw = yaw * Math.PI / 180; d.freeCam.pitch = pitch * Math.PI / 180;
      return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(r))));
    }, { x, alt, z, yaw, pitch });
    if (opts.verbose) console.log('pose', i, alt.toFixed(2), pitch.toFixed(1), yaw.toFixed(1));
    const png = await page.screenshot({ type: 'png' });
    const horizonY = Math.ceil(H / 2 - Math.tan(-pitch * Math.PI / 180) / Math.tan(Math.PI / 6) * (H / 2)) + 2;
    const r = await ana.evaluate(async ({ b64, from }) => {
      const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = `data:image/png;base64,${b64}`; });
      const c = new OffscreenCanvas(img.width, img.height); const x = c.getContext('2d'); x.drawImage(img, 0, 0);
      const d = x.getImageData(0, 0, img.width, img.height).data;
      let n = 0, first = null;
      for (let y = Math.max(0, from); y < img.height; y++) for (let xx = 0; xx < img.width; xx++) {
        const k = (y * img.width + xx) * 4;
        if (d[k] > 180 && d[k + 1] < 60 && d[k + 2] > 180) { n++; if (!first) first = [xx, y]; }
      }
      return { n, first };
    }, { b64: png.toString('base64'), from: horizonY });
    rows.push({ i, alt: +alt.toFixed(2), pitch: +pitch.toFixed(1), yaw: +yaw.toFixed(1), x: +x.toFixed(2), z: +z.toFixed(2), magenta: r.n, first: r.first });
    total += r.n; worst = Math.max(worst, r.n);
    if (r.n > 0 && opts.outdir) { mkdirSync(opts.outdir, { recursive: true }); writeFileSync(`${opts.outdir}/sweep-fail-${i}.png`, png); }
  }
  for (const r of rows) if (r.magenta > 0) console.log('FAIL', JSON.stringify(r));
  console.log(JSON.stringify({ poses: N, posesWithMagenta: rows.filter((r) => r.magenta > 0).length, totalMagentaPixels: total, worst, errors: watch.webgpuErrors.length }));
} finally {
  await browser.close();
}
