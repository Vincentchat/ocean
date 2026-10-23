import type { OceanPreset } from '../types.ts';
import { REFERENCE_AMPLITUDE } from '../spectrum.ts';

/**
 * Gale (25 m/s, 85 km fetch).
 * Hs includes the reference's amplitude calibration x1.578 (see presets/index.ts).
 */
export const storm: OceanPreset = Object.freeze({
  name: 'storm',
  params: Object.freeze({
    windSpeed: 25,         // m/s at 10 m
    windDirection: -1,      // rad, towards (CCW from east): the reference's 1 rad towards +X..+Z
    fetch: 85000,           // m
    gamma: 3.3,             // JONSWAP peak enhancement (reference)
    spread: 1.57,           // reference
    standing: 0.13,         // reference
    choppiness: 1.5,        // reference
    amplitude: REFERENCE_AMPLITUDE, // 1.5783: reference calibration (wave 1.5 x ((5+12.5*1)/15)^0.33)
  }),
  target: Object.freeze({ hs: 9.00, peakPeriod: 8.01, peakWavelength: 100.2, reason: 'Gale over a long fetch: long heavy waves, heavy chop (JONSWAP Hs ~4.4 m).' }),
}) as OceanPreset;
