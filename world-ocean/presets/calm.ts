import type { OceanPreset } from '../types.ts';
import { REFERENCE_AMPLITUDE } from '../spectrum.ts';

/**
 * Light wind (4 m/s, 10 km fetch).
 * Hs includes the reference's amplitude calibration x1.578 (see presets/index.ts).
 */
export const calm: OceanPreset = Object.freeze({
  name: 'calm',
  params: Object.freeze({
    windSpeed: 4,         // m/s at 10 m
    windDirection: -1,      // rad, towards (CCW from east): the reference's 1 rad towards +X..+Z
    fetch: 10000,           // m
    gamma: 3.3,             // JONSWAP peak enhancement (reference)
    spread: 1.57,           // reference
    standing: 0.13,         // reference
    choppiness: 1.5,        // reference
    amplitude: REFERENCE_AMPLITUDE, // 1.5783: reference calibration (wave 1.5 x ((5+12.5*1)/15)^0.33)
  }),
  target: Object.freeze({ hs: 0.54, peakPeriod: 2.13, peakWavelength: 7.1, reason: 'Light air over a bay: small, short ripples and wavelets (open-ocean JONSWAP Hs ~0.27 m).' }),
}) as OceanPreset;
