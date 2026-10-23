import type { OceanPreset } from '../types.ts';
import { REFERENCE_AMPLITUDE } from '../spectrum.ts';

/**
 * Light wind (5 m/s, 15 km fetch). Brackets breeze from below.
 */
export const calm: OceanPreset = Object.freeze({
  name: 'calm',
  params: Object.freeze({
    windSpeed: 5,         // m/s at 10 m
    windDirection: -1,      // rad, towards (CCW from east): the reference's 1 rad towards +X..+Z
    fetch: 15000,           // m
    gamma: 3.3,             // JONSWAP peak enhancement (reference)
    spread: 1.57,           // reference
    standing: 0.13,         // reference
    choppiness: 1.5,        // reference
    amplitude: REFERENCE_AMPLITUDE, // 1.5783: reference default scene (wave 1.5 x ((5+12.5*1)/15)^0.33)
  }),
  target: Object.freeze({ hs: 0.82, peakPeriod: 2.63, peakWavelength: 10.8, reason: 'Light wind over a short fetch: small, short, wind-aligned waves.' }),
}) as OceanPreset;
