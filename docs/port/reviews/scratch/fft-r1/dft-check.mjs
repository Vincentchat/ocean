// Reviewer's independent check (fft-r1): GPU f32 FFT fields vs a brute-force f64 direct DFT computed from the core's
// h0 slots with dispersion recomputed here (not the core's omega table), at large times.
// node dft-check.mjs <base> <tier> <preset> <t>
import { launch } from '../../../../../world-ocean/tools/lib/browser.mjs';
const [base = 'http://127.0.0.1:5185', tier = 'high', preset = 'storm', tArg = '4321.123'] = process.argv.slice(2);
const { browser, context } = await launch({ width: 800, height: 600 });
try {
  const page = await context.newPage();
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(`${base}/demo/fft-debug.html?tier=${tier}&preset=${preset}&t=${tArg}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__fft && window.__fft.fft.pendingTier === null, null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  const res = await page.evaluate(async (T) => {
    const { fft, ocean } = window.__fft;
    const st = ocean.getState();
    const gb = await fft.debug.readFields('base');
    const gd = await fft.debug.readFields('detail');
    const nb = fft.debug.groupSize('base'), nd = fft.debug.groupSize('detail');
    const chop = ocean.choppinessAt(T);
    const g = st.gravity;
    const Q = 2 * Math.PI / 8192;
    const out = { chop, g, fromSlot: st.fromSlot, toSlot: st.toSlot, hs: ocean.expectedHs(), rows: [], omegaMismatch: [] };
    const N = 256;
    for (let c = 0; c < 4; c++) {
      const def = fft.debug.cascades[c];
      const h0 = c < 3 ? st.slots[st.toSlot].h0[c] : fft.debug.fineH0(st.toSlot);
      const dk = 2 * Math.PI / def.domain;
      const n = c < 2 ? nb : nd;
      const field = c < 2 ? gb : gd;
      const layer = c & 1;
      if (c === 3 && fft.tier !== 'high' && fft.tier !== 'ultra') continue;
      // omega recomputed in f64 (reference: round(sqrt(g(k+1e-4))/q)*q), compare with core table for c<3
      let mism = 0;
      const om = new Float64Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const kx = (x - 128) * dk, kz = (y - 128) * dk, k = Math.hypot(kx, kz);
        const m = Math.round(Math.sqrt(g * (k + 1e-4)) / Q);
        om[y * N + x] = m * Q;
        if (c < 3 && st.omegaIndex[c][y * N + x] !== m) mism++;
      }
      out.omegaMismatch.push(mism);
      // sample texels
      const samples = [[0, 0], [17, 203 % n],[n - 1, 5], [n / 2 + 3, n / 2 - 7], [101 % n, 33 % n], [n - 9, n - 2]];
      let mu = 0, me = 0, mn = 0, amp = 0;
      for (const [px, py] of samples) {
        const a = px * def.domain / n, b = py * def.domain / n;
        let H = 0, E = 0, S = 0;
        // band of the GPU grid: for n<256 only the centred sub-block; n>256 zero-padded (same as 256)
        const lim = Math.min(n, N) / 2;
        for (let y = 0; y < N; y++) {
          const iy = y - 128; if (iy < -lim || iy >= lim) continue;
          if (n < N && iy === -lim) continue; // zeroed Nyquist line
          for (let x = 0; x < N; x++) {
            const ix = x - 128; if (ix < -lim || ix >= lim) continue;
            if (n < N && ix === -lim) continue;
            const i = y * N + x;
            const kx = ix * dk, kz = iy * dk;
            const kl = Math.hypot(kx, kz) + 1e-4;
            const th = ((om[i] * T) % (2 * Math.PI));
            const cs = Math.cos(th), sn = Math.sin(th);
            // ht = h0 e^{-i th} + h0m e^{+i th}
            const hr = h0[4 * i] * cs + h0[4 * i + 1] * sn + h0[4 * i + 2] * cs - h0[4 * i + 3] * sn;
            const hi = h0[4 * i + 1] * cs - h0[4 * i] * sn + h0[4 * i + 3] * cs + h0[4 * i + 2] * sn;
            const ph = kx * a + kz * b;
            const ec = Math.cos(ph), es = Math.sin(ph);
            const re = hr * ec - hi * es, im = hr * es + hi * ec;
            H += re; E += im * kx / kl; S += im * kz / kl;
          }
        }
        const up = -H, east = chop * E, north = -chop * S;
        const o = (layer * n * n + py * n + px) * 4;
        mu = Math.max(mu, Math.abs(field[o] - up)); me = Math.max(me, Math.abs(field[o + 1] - east)); mn = Math.max(mn, Math.abs(field[o + 2] - north));
        amp = Math.max(amp, Math.abs(up));
      }
      out.rows.push({ c, n, maxErrUp: mu, maxErrEast: me, maxErrNorth: mn, maxAbsUp: amp });
    }
    return out;
  }, Number(tArg));
  console.log(JSON.stringify(res, null, 1));
} finally { await browser.close(); }
