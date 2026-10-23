// Presets: calm / breeze / windy / storm. 'windy' = the WebGL reference's load-time scene, exactly.
//
// Calibration: every preset keeps the reference's amplitude x1.578 (its default 'storm' page preset:
// wave 1.5 x ((5 + 12.5 wind) / 15)^0.33), because that IS the visual target, and the spectrum
// counts the Tessendorf h0 + h0m pair (variance x (1 + (1 - standing)^2) = x1.76). Together Hs is
// about 2x the open-ocean JONSWAP Hs for the same wind and fetch (and Hs / peak wavelength ~0.08,
// steeper than real wind seas). The names describe the look (calm ~0.5 m, breeze ~1.5 m, windy ~4 m,
// storm ~9 m); wind speed and fetch were chosen with this calibration to land there.
import type { OceanPreset, PresetName } from '../types.ts';
import { calm } from './calm.ts';
import { breeze } from './breeze.ts';
import { windy } from './windy.ts';
import { storm } from './storm.ts';

export { calm, breeze, windy, storm };

export const PRESETS: Readonly<Record<PresetName, OceanPreset>> = Object.freeze({ calm, breeze, windy, storm });
export const PRESET_NAMES: readonly PresetName[] = Object.freeze(['calm', 'breeze', 'windy', 'storm'] as PresetName[]);

/**
 * The reference page's own scene presets (src/ocean.js `presets`) expressed on the 'windy'
 * spectrum: they only change the amplitude, amp = wave * ((5 + 12.5 wind) / 15)^0.33.
 * Use createOcean({ preset: 'windy', amplitude: REFERENCE_SCENE_AMPLITUDE.golden }) to compare.
 */
export const REFERENCE_SCENE_AMPLITUDE = Object.freeze({
  golden: 0.9 * Math.pow((5 + 12.5 * 0.6) / 15, 0.33),  // 0.847 (Evening calm), Hs ~2.17 m
  azure: 1.0 * Math.pow((5 + 12.5 * 0.8) / 15, 0.33),   // 1.000 (Blue hour), Hs ~2.56 m
  storm: 1.5 * Math.pow((5 + 12.5 * 1.0) / 15, 0.33),   // 1.578 (After the rain, default at load), Hs ~4.04 m
});
