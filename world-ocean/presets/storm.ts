import type { OceanPreset } from '../types.ts';
import { REFERENCE_AMPLITUDE } from '../spectrum.ts';

/**
 * 25 m/s over 100 km (peak 112 m, still ~9 wavelengths per 1024 m tile).
 */
export const storm: OceanPreset = Object.freeze({
  name: 'storm',
  params: Object.freeze({
    windSpeed: 25,         // m/s at 10 m
    windDirection: -1,      // rad, towards (CCW from east): the reference's 1 rad towards +X..+Z
    fetch: 100000,           // m
    gamma: 3.3,             // JONSWAP peak enhancement (reference)
    spread: 1.57,           // reference
    standing: 0.13,         // reference
    choppiness: 1.5,        // reference
    amplitude: REFERENCE_AMPLITUDE, // 1.5783: reference default scene (wave 1.5 x ((5+12.5*1)/15)^0.33)
  }),
  target: Object.freeze({ hs: 9.85, peakPeriod: 8.46, peakWavelength: 111.7, reason: 'Gale over a long fetch: long swell-like waves, heavy chop.' }),
}) as OceanPreset;
