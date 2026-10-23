#!/usr/bin/env node
// Compare two PNGs (same size) in a headless browser canvas.
//   node tools/imgdiff.mjs a.png b.png [--out diff.png] [--gain 8]
// Prints JSON: mean absolute difference per channel (0..255), max, % of pixels differing by > 8 levels,
// and mean luminance of each image. --out writes |a-b| * gain as a PNG.
import { readFileSync, writeFileSync } from 'node:fs';
import { launch, parseArgs } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const [a, b] = opts._;
if (!a || !b) { console.error('usage: node tools/imgdiff.mjs a.png b.png [--out diff.png] [--gain 8]'); process.exit(2); }
const toUrl = (p) => `data:image/png;base64,${readFileSync(p).toString('base64')}`;
const { browser, context } = await launch({ width: 64, height: 64 });
try {
  const page = await context.newPage();
  const r = await page.evaluate(async ({ ua, ub, gain, wantDiff }) => {
    const load = (u) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = u; });
    const [ia, ib] = await Promise.all([load(ua), load(ub)]);
    if (ia.width !== ib.width || ia.height !== ib.height) return { error: `size mismatch ${ia.width}x${ia.height} vs ${ib.width}x${ib.height}` };
    const w = ia.width, h = ia.height;
    const read = (img) => { const c = new OffscreenCanvas(w, h); const x = c.getContext('2d'); x.drawImage(img, 0, 0); return x.getImageData(0, 0, w, h).data; };
    const da = read(ia), db = read(ib);
    let sum = 0, max = 0, over = 0, la = 0, lb = 0;
    const out = wantDiff ? new ImageData(w, h) : null;
    for (let i = 0; i < da.length; i += 4) {
      let pmax = 0;
      for (let k = 0; k < 3; k++) {
        const d = Math.abs(da[i + k] - db[i + k]);
        sum += d; if (d > max) max = d; if (d > pmax) pmax = d;
        if (out) out.data[i + k] = Math.min(255, d * gain);
      }
      if (out) out.data[i + 3] = 255;
      if (pmax > 8) over++;
      la += 0.2126 * da[i] + 0.7152 * da[i + 1] + 0.0722 * da[i + 2];
      lb += 0.2126 * db[i] + 0.7152 * db[i + 1] + 0.0722 * db[i + 2];
    }
    const n = w * h;
    let diffUrl = null;
    if (out) { const c = new OffscreenCanvas(w, h); c.getContext('2d').putImageData(out, 0, 0); const blob = await c.convertToBlob({ type: 'image/png' }); diffUrl = await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(blob); }); }
    return { width: w, height: h, meanAbsDiff: sum / (n * 3), maxDiff: max, pctPixelsOver8: (100 * over) / n, meanLumA: la / n, meanLumB: lb / n, diffUrl };
  }, { ua: toUrl(a), ub: toUrl(b), gain: Number(opts.gain ?? 8), wantDiff: !!opts.out });
  if (r.diffUrl && opts.out) writeFileSync(opts.out, Buffer.from(r.diffUrl.split(',')[1], 'base64'));
  delete r.diffUrl;
  for (const k of Object.keys(r)) if (typeof r[k] === 'number') r[k] = Math.round(r[k] * 1000) / 1000;
  console.log(JSON.stringify({ a, b, ...r }, null, 2));
  if (r.error) process.exitCode = 1;
} finally {
  await browser.close();
}
