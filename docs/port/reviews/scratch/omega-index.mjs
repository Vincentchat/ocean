import * as ref from '../../../../src/spectral-ocean.js';
import { omegaIndices, CASCADES, OMEGA_UNIT } from '../../../../world-ocean/spectrum.ts';
const s = ref.generateSpectra();
for (const g of [9.81, 9.80665]) { let bad = 0, maxd = 0;
for (let c = 0; c < 3; c++) { const om = omegaIndices(CASCADES[c], g, new Float32Array(65536)); const f = s.cascades[c].frequency;
 for (let i = 0; i < 65536; i++) { const d = Math.abs(Math.round(f[i] / OMEGA_UNIT) - om[i]); if (d) bad++; maxd = Math.max(maxd, d); } }
console.log(`g=${g}: index mismatches ${bad}/196608, max index diff ${maxd}`); }
