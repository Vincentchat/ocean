import type { OceanPreset } from '../types.ts';
import { REFERENCE_AMPLITUDE } from '../spectrum.ts';

/**
 * Reference default: reproduces src/spectral-ocean.js OCEAN_PARAMETERS (U 15 m/s, peak wavelength 47 m via fetch 45.479 km) at the reference page default amplitude 1.578 (Hs 4.04 m; the reference doc rounds to 4.07).
 */
export const breeze: OceanPreset = Object.freeze({
  name: 'breeze',
  params: Object.freeze({
    windSpeed: 15,         // m/s at 10 m
    windDirection: -1,      // rad, towards (CCW from east): the reference's 1 rad towards +X..+Z
    fetch: 45479.33,           // m
    gamma: 3.3,             // JONSWAP peak enhancement (reference)
    spread: 1.57,           // reference
    standing: 0.13,         // reference
    choppiness: 1.5,        // reference
    amplitude: REFERENCE_AMPLITUDE, // 1.5783: reference default scene (wave 1.5 x ((5+12.5*1)/15)^0.33)
  }),
  target: Object.freeze({ hs: 4.04, peakPeriod: 5.49, peakWavelength: 47.0, reason: 'The reference default scene (After the rain / storm preset at load): U 15 m/s, peak 47 m, amplitude 1.578. 1:1 with the WebGL reference.' }),
}) as OceanPreset;
