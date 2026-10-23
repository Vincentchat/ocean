import type { OceanPreset } from '../types.ts';
import { REFERENCE_AMPLITUDE } from '../spectrum.ts';

/**
 * Moderate breeze (8 m/s, 20 km fetch).
 * Hs includes the reference's amplitude calibration x1.578 (see presets/index.ts).
 */
export const breeze: OceanPreset = Object.freeze({
  name: 'breeze',
  params: Object.freeze({
    windSpeed: 8,         // m/s at 10 m
    windDirection: -1,      // rad, towards (CCW from east): the reference's 1 rad towards +X..+Z
    fetch: 20000,           // m
    gamma: 3.3,             // JONSWAP peak enhancement (reference)
    spread: 1.57,           // reference
    standing: 0.13,         // reference
    choppiness: 1.5,        // reference
    amplitude: REFERENCE_AMPLITUDE, // 1.5783: reference calibration (wave 1.5 x ((5+12.5*1)/15)^0.33)
  }),
  target: Object.freeze({ hs: 1.46, peakPeriod: 3.38, peakWavelength: 17.9, reason: 'Moderate breeze, coastal fetch: short wind waves with occasional whitecaps (JONSWAP Hs ~0.7 m).' }),
}) as OceanPreset;
