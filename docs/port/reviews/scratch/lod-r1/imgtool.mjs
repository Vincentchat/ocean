// Scratch image helper (headless browser canvas).
//   node imgtool.mjs crop in.png out.png x y w h [scale=1] [stretch=0]
//   node imgtool.mjs magenta in.png [rowFrom=0]       -> count of magenta pixels at/below rowFrom
//   node imgtool.mjs shiftdiff a.png b.png dx dy       -> compare a(x,y) with b(x+dx,y+dy) on the overlap
import { readFileSync, writeFileSync } from 'node:fs';
import { launch } from 'file:///C:/Users/Vincent/Documents/Claude/ocean/world-ocean/tools/lib/browser.mjs';

const [op, ...args] = process.argv.slice(2);
const toUrl = (p) => `data:image/png;base64,${readFileSync(p).toString('base64')}`;
const { browser, context } = await launch({ width: 64, height: 64 });
try {
  const page = await context.newPage();
  const r = await page.evaluate(async ({ op, args, urls }) => {
    const load = (u) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = u; });
    const imgs = await Promise.all(urls.map(load));
    const read = (img) => { const c = new OffscreenCanvas(img.width, img.height); const x = c.getContext('2d'); x.drawImage(img, 0, 0); return x.getImageData(0, 0, img.width, img.height); };
    if (op === 'crop') {
      const [x, y, w, h, scale = 1, stretch = 0] = args.map(Number);
      const src = read(imgs[0]);
      const out = new ImageData(w * scale, h * scale);
      let lo = 255, hi = 0;
      if (stretch) for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) for (let k = 0; k < 3; k++) { const v = src.data[((y + j) * src.width + x + i) * 4 + k]; lo = Math.min(lo, v); hi = Math.max(hi, v); }
      for (let j = 0; j < h * scale; j++) for (let i = 0; i < w * scale; i++) {
        const si = ((y + Math.floor(j / scale)) * src.width + x + Math.floor(i / scale)) * 4, di = (j * w * scale + i) * 4;
        for (let k = 0; k < 3; k++) { const v = src.data[si + k]; out.data[di + k] = stretch ? Math.round((v - lo) * 255 / Math.max(hi - lo, 1)) : v; }
        out.data[di + 3] = 255;
      }
      const c = new OffscreenCanvas(w * scale, h * scale); c.getContext('2d').putImageData(out, 0, 0);
      const blob = await c.convertToBlob({ type: 'image/png' });
      const url = await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(blob); });
      return { url, lo, hi };
    }
    if (op === 'magenta') {
      const from = Number(args[0] ?? 0);
      const d = read(imgs[0]);
      let n = 0, minY = 1e9, maxY = -1; const samples = [];
      for (let y = from; y < d.height; y++) for (let x = 0; x < d.width; x++) {
        const i = (y * d.width + x) * 4;
        if (d.data[i] > 180 && d.data[i + 1] < 60 && d.data[i + 2] > 180 && Math.abs(d.data[i] - d.data[i + 2]) < 40) {
          n++; minY = Math.min(minY, y); maxY = Math.max(maxY, y); if (samples.length < 10) samples.push([x, y]);
        }
      }
      return { magentaPixels: n, minY, maxY, samples, rowFrom: from };
    }
    if (op === 'shiftdiff') {
      const [dx, dy] = args.map(Number);
      const a = read(imgs[0]), b = read(imgs[1]);
      let sum = 0, over = 0, cnt = 0, max = 0;
      for (let y = Math.max(0, -dy); y < Math.min(a.height, b.height - dy); y++) for (let x = Math.max(0, -dx); x < Math.min(a.width, b.width - dx); x++) {
        const ia = (y * a.width + x) * 4, ib = ((y + dy) * b.width + x + dx) * 4;
        let pm = 0;
        for (let k = 0; k < 3; k++) { const v = Math.abs(a.data[ia + k] - b.data[ib + k]); sum += v; pm = Math.max(pm, v); }
        if (pm > 8) over++; max = Math.max(max, pm); cnt++;
      }
      return { meanAbsDiff: sum / (cnt * 3), pctPixelsOver8: 100 * over / cnt, maxDiff: max, pixels: cnt };
    }
    return { error: 'unknown op' };
  }, { op, args: op === 'crop' ? args.slice(2) : op === 'shiftdiff' ? args.slice(2) : args.slice(1), urls: op === 'shiftdiff' ? [toUrl(args[0]), toUrl(args[1])] : [toUrl(args[0])] });
  if (op === 'crop') { writeFileSync(args[1], Buffer.from(r.url.split(',')[1], 'base64')); delete r.url; }
  console.log(JSON.stringify(r));
} finally {
  await browser.close();
}
