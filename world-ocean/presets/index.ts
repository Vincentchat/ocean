// Presets: calm / breeze / windy / storm. 'breeze' = the WebGL reference's default scene.
import type { OceanPreset, PresetName } from '../types.ts';
import { calm } from './calm.ts';
import { breeze } from './breeze.ts';
import { windy } from './windy.ts';
import { storm } from './storm.ts';

export { calm, breeze, windy, storm };

export const PRESETS: Readonly<Record<PresetName, OceanPreset>> = Object.freeze({ calm, breeze, windy, storm });
export const PRESET_NAMES: readonly PresetName[] = Object.freeze(['calm', 'breeze', 'windy', 'storm'] as PresetName[]);

/**
 * The reference page's own scene presets (src/ocean.js `presets`) expressed on the 'breeze'
 * spectrum: they only change the amplitude, amp = wave * ((5 + 12.5 wind) / 15)^0.33.
 * Use createOcean({ preset: 'breeze', amplitude: REFERENCE_SCENE_AMPLITUDE.golden }) to compare.
 */
export const REFERENCE_SCENE_AMPLITUDE = Object.freeze({
  golden: 0.9 * Math.pow((5 + 12.5 * 0.6) / 15, 0.33),  // 0.847 (Evening calm), Hs ~2.17 m
  azure: 1.0 * Math.pow((5 + 12.5 * 0.8) / 15, 0.33),   // 1.000 (Blue hour), Hs ~2.56 m
  storm: 1.5 * Math.pow((5 + 12.5 * 1.0) / 15, 0.33),   // 1.578 (After the rain, default at load), Hs ~4.04 m
});
