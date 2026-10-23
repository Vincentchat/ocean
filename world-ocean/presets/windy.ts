import type { OceanPreset } from '../types.ts';
import { REFERENCE_AMPLITUDE } from '../spectrum.ts';

/**
 * 20 m/s over 60 km.
 */
export const windy: OceanPreset = Object.freeze({
  name: 'windy',
  params: Object.freeze({
    windSpeed: 20,         // m/s at 10 m
    windDirection: -1,      // rad, towards (CCW from east): the reference's 1 rad towards +X..+Z
    fetch: 60000,           // m
    gamma: 3.3,             // JONSWAP peak enhancement (reference)
    spread: 1.57,           // reference
    standing: 0.13,         // reference
    choppiness: 1.5,        // reference
    amplitude: REFERENCE_AMPLITUDE, // 1.5783: reference default scene (wave 1.5 x ((5+12.5*1)/15)^0.33)
  }),
  target: Object.freeze({ hs: 6.08, peakPeriod: 6.62, peakWavelength: 68.5, reason: 'Stronger wind over a longer fetch: longer, higher waves.' }),
}) as OceanPreset;
