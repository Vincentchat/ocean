// Reviewer scratch: preset Hs vs real sea states.
import { expectedHs, peakPeriod, peakWavelength, peakOmega, cascadePower, CASCADES } from '../../../../world-ocean/spectrum.ts';
import { PRESETS, PRESET_NAMES } from '../../../../world-ocean/presets/index.ts';
const g = 9.80665;
// 1-D JONSWAP (Hasselmann 1973) with the same alpha / fetch law, integrated numerically: physical Hs.
function jonswapHs(U, F, gamma) {
  const wp = peakOmega(U, F, g);
  const alpha = 0.076 * Math.pow((U * wp) / (22 * g), 0.66);
  let m0 = 0;
  for (let w = 0.01; w < 40; w += 0.0005) {
    const s = w < wp ? 0.07 : 0.09;
    const r = Math.exp(-((w - wp) ** 2) / (2 * s * s * wp * wp));
    m0 += alpha * g * g / w ** 5 * Math.exp(-1.25 * (wp / w) ** 4) * gamma ** r * 0.0005;
  }
  return 4 * Math.sqrt(m0);
}
console.log('preset | U m/s | F km | Tp s | Lp m | Hs port m | Hs/amp m | JONSWAP Hs (same U,F) | CERC fetch-limited Hs | PM fully dev. Hs | steepness Hs/Lp');
for (const name of PRESET_NAMES) {
  const p = PRESETS[name].params;
  const hs = expectedHs(p, g);
  const tp = peakPeriod(p.windSpeed, p.fetch, g), lp = peakWavelength(p.windSpeed, p.fetch, g);
  const js = jonswapHs(p.windSpeed, p.fetch, p.gamma);
  const cerc = 0.0016 * p.windSpeed * Math.sqrt(p.fetch / g);   // g Hs/U^2 = 0.0016 sqrt(g F / U^2)
  const U195 = p.windSpeed * Math.pow(19.5 / 10, 1 / 7);
  const pm = 0.21 * U195 * U195 / g;
  console.log(`${name} | ${p.windSpeed} | ${(p.fetch / 1000).toFixed(1)} | ${tp.toFixed(2)} | ${lp.toFixed(1)} | ${hs.toFixed(2)} | ${(hs / p.amplitude).toFixed(2)} | ${js.toFixed(2)} | ${cerc.toFixed(2)} | ${pm.toFixed(2)} | ${(hs / lp).toFixed(3)}`);
}
// Variance bookkeeping: sum P vs expected variance factor
const p = PRESETS.storm.params;
let sumP = 0; for (const c of CASCADES) sumP += cascadePower({ ...p, amplitude: 1 }, c, g);
console.log(`storm: 4 sqrt(sum P) (amp 1) = ${(4 * Math.sqrt(sumP)).toFixed(2)} m; with (1+(1-st)^2) factor = ${(4 * Math.sqrt(sumP * (1 + (1 - p.standing) ** 2))).toFixed(2)} m; x amp = ${expectedHs(p, g).toFixed(2)} m`);
// fully developed PM peak cap check
for (const U of [5, 15, 25]) console.log(`U ${U}: fully developed Tp (PM cap) = ${(2 * Math.PI / (0.855 * g / U)).toFixed(2)} s; fetch-limited Tp at preset fetch shown above`);
// 'storm' Hs if fully developed with this model:
console.log(`model Hs, U 25, fetch 2000 km (fully developed): ${expectedHs({ ...PRESETS.storm.params, fetch: 2e6 }, g).toFixed(2)} m (amp 1.578), Tp ${peakPeriod(25, 2e6, g).toFixed(2)} s, peak ${peakWavelength(25, 2e6, g).toFixed(0)} m vs 1024 m tile`);
