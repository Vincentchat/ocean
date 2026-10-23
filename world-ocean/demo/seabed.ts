// DEMO ONLY: the reference's procedural seabed (src/seabed.js): shallow sand shelf, 26 rocks, refracted-sun lighting
// with absorption and analytic caustics. It is NOT part of the ocean module (spec 4: "Whale, seabed ... demo only;
// seabed later reads TERRAIN_QUERY"). The ocean only sees it through the contracts:
//   - as opaque geometry in scene.color / scene.depth (the scene pass draws this mesh), and
//   - as TERRAIN_QUERY (seabed.query.heightAt) and TERRAIN_GPU (stubs/impl.ts createStubTerrainGrid over the query).
// The ocean never imports this file.
//
// Geometry: 200 m x 200 m log-spaced grid centred at world (0, 8) (x, z), 512 segments, heights from the same
// sand + mound + rock functions as the reference (CPU, once). Positions are stored in the seabed's own frame (world
// metres relative to the seabed anchor, the world origin); the mesh is placed at anchor - floatingOrigin, so nothing
// on the GPU is a raw world position and an origin shift does not move it on screen.
// Shading: the reference fragment shader term by term (TSL), output = radiance (reference units) x REF_NITS x
// preExposure, like the sky stand-in (no tonemap here; the demo tonemap pass does it).
import { BufferAttribute, BufferGeometry, Mesh, MeshBasicNodeMaterial, Vector3 } from 'three/webgpu';
import {
  Fn, abs, clamp, dFdx, dFdy, dot, exp, float, floor, fract, ivec2, length, max, min, mix, normalLocal, normalize,
  positionLocal, refract, select, sin, smoothstep, uint, uniform, uvec2, vec2, vec3,
} from 'three/tsl';
import type { FloatingOrigin, TerrainQuery, Vec3 } from '../stubs/world-core.ts';

export const SEABED_EXTENT = 200;
export const SEABED_SEGMENTS = 512;
export const SEABED_CENTER_Z = 8;
/** cx, cz, rx, rz, peak, angle (src/seabed.js). */
export const SEABED_ROCKS: readonly (readonly number[])[] = [
  [-8.1, 0.8, 3.45, 2.16, 2.08, 0.46], [-10.5, -0.5, 2.45, 1.63, 1.53, -0.29], [-6.0, -1.3, 2.4, 1.34, 1.54, -0.63],
  [-11.8, 1.8, 1.35, 1.04, 0.83, 0.17], [3.4, -8.8, 3.2, 2.4, 2.18, -0.34], [5.5, -10.2, 2.34, 1.68, 1.76, 0.73],
  [1.0, -9.6, 2.26, 1.22, 1.21, -0.18], [4.1, -6.4, 1.74, 1.13, 1.22, 0.84], [7.2, -11.1, 1.48, 1.04, 0.69, -0.12],
  [12.0, 3.2, 3.2, 2.16, 1.81, -0.68], [14.5, 1.9, 2.23, 1.77, 1.36, 0.12], [10.2, 4.9, 1.84, 1.2, 1.12, -0.25],
  [15.6, 4.7, 1.33, 0.96, 0.65, 0.81], [-11.2, -12.8, 3.15, 1.9, 1.83, -0.71], [-13.9, -11.9, 2.56, 1.25, 1.2, 0.28],
  [-9.1, -14.4, 1.94, 1.27, 0.94, -0.29], [-15.7, -13.6, 1.4, 1.07, 0.81, -0.57], [-3.6, 19.0, 3.03, 1.99, 1.69, 0.64],
  [-6.0, 18.2, 2.0, 1.43, 1.35, -0.18], [-1.4, 20.3, 1.84, 1.06, 0.83, -0.46], [17.1, 23.4, 3.16, 1.75, 1.4, 0.82],
  [19.1, 24.6, 2.3, 1.56, 1.01, -0.37], [-19.8, -29.2, 4.5, 2.65, 2.08, 0.64], [-16.9, -27.3, 2.2, 1.24, 1.07, -0.43],
  [-29.4, 11.7, 3.15, 2.26, 1.58, -0.18], [32.7, -41.1, 4.15, 2.46, 1.85, 0.36],
];
/** Ground height reported outside the seabed mesh (no ground: open, deep ocean). */
export const NO_GROUND = -1e4;

// ------------------------------------------------------------------------------------------------ CPU (heights)
function hash(x: number, z: number) {
  let h = (Math.imul(x | 0, 0x9e3779b9) ^ Math.imul(z | 0, 0x85ebca6b)) >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d); h ^= h >>> 15; h = Math.imul(h, 0x846ca68b); h ^= h >>> 16;
  return ((h >>> 9) + 0.5) / 8388608;
}
const smooth = (t: number) => t * t * (3 - 2 * t);
const sstep = (a: number, b: number, x: number) => smooth(Math.max(0, Math.min(1, (x - a) / (b - a))));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
function noise(x: number, z: number) {
  const ix = Math.floor(x), iz = Math.floor(z), tx = smooth(x - ix), tz = smooth(z - iz);
  return lerp(lerp(hash(ix, iz), hash(ix + 1, iz), tx), lerp(hash(ix, iz + 1), hash(ix + 1, iz + 1), tx), tz);
}
function sandHeight(x: number, z: number) {
  const qx = x * 0.92 + 4.5 * (noise(x * 0.025 + 19, z * 0.025 - 7) - 0.5);
  const qz = (z - SEABED_CENTER_Z) * 1.035 + 4 * (noise(x * 0.022 - 4, z * 0.022 + 13) - 0.5);
  const r = Math.hypot(qx, qz);
  const depth = 8.68 + 0.022 * r + 20 * sstep(14, 102, r);
  const broad = 1.25 * (noise(x * 0.037 + 12.6, z * 0.037 - 8.2) - 0.5);
  const medium = 0.39 * (noise(0.093 * x + 0.027 * z - 6, -0.027 * x + 0.093 * z + 17) - 0.5);
  const fine = 0.085 * (noise(0.34 * x - 0.13 * z + 13, 0.13 * x + 0.34 * z - 3) - 0.5);
  const m1 = 4.8 * Math.exp(-((x + 8) ** 2 / 59 + (z + 1) ** 2 / 43));
  const m2 = 3.7 * Math.exp(-((x - 11) ** 2 / 89 + (z - 4) ** 2 / 67));
  const m3 = 0.98 * Math.exp(-((x - 1) ** 2 / 174 + (z + 34) ** 2 / 76));
  const m4 = 2.75 * Math.exp(-((x - 3.8) ** 2 / 66 + (z + 8.4) ** 2 / 46));
  const m5 = 1.45 * Math.exp(-((x + 11.4) ** 2 / 52 + (z + 12.8) ** 2 / 49));
  const m6 = 1.65 * Math.exp(-((x + 3.6) ** 2 / 65 + (z - 19) ** 2 / 48));
  return -depth + broad + medium + fine + m1 + m2 + m3 + m4 + m5 + m6;
}
/** Rock height (m) at (x, z) (allocation-free). */
function rockSurface(x: number, z: number) {
  let height = 0;
  for (let i = 0; i < SEABED_ROCKS.length; i++) {
    const [cx, cz, rx, rz, peak, angle] = SEABED_ROCKS[i];
    const dx = x - cx, dz = z - cz;
    if (Math.abs(dx) > rx + rz || Math.abs(dz) > rx + rz) continue;
    const c = Math.cos(angle), s = Math.sin(angle);
    const u = (c * dx + s * dz) / rx, v = (-s * dx + c * dz) / rz;
    const erosion = 0.19 * (noise(x * 0.31 + 41, z * 0.31 - 21) - 0.5);
    const q = u * u + v * v + erosion;
    const body = Math.pow(Math.max(1 - q, 0), 1.12);
    const n = noise(x * 0.62 + i * 17, z * 0.62 - i * 11);
    height = Math.max(height, peak * body * (0.84 + 0.22 * n));
  }
  return height;
}
/** Seabed height (m above sea level) at world (x, z), anchored at the world origin (reference coordinates). */
export function seabedHeight(x: number, z: number) {
  return sandHeight(x, z) + rockSurface(x, z);
}

function seabedCoords(segments: number, extent: number) {
  const coords = new Float64Array(segments + 1);
  const innerScale = 8, logarithm = Math.log(1 + (extent * 0.5) / innerScale);
  for (let i = 0; i <= segments; i++) {
    const u = (i * 2) / segments - 1;
    coords[i] = Math.sign(u) * innerScale * Math.expm1(Math.abs(u) * logarithm);
  }
  return coords;
}

export function createSeabedGeometry(segments = SEABED_SEGMENTS, extent = SEABED_EXTENT) {
  const count = (segments + 1) ** 2;
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const indices = new Uint32Array(segments * segments * 6);
  const coords = seabedCoords(segments, extent);
  for (let iz = 0; iz <= segments; iz++) {
    for (let ix = 0; ix <= segments; ix++) {
      const i = iz * (segments + 1) + ix, x = coords[ix], z = coords[iz] + SEABED_CENTER_Z;
      positions[i * 3] = x; positions[i * 3 + 1] = seabedHeight(x, z); positions[i * 3 + 2] = z;
    }
  }
  let k = 0;
  for (let iz = 0; iz < segments; iz++) {
    for (let ix = 0; ix < segments; ix++) {
      const a = iz * (segments + 1) + ix, b = a + 1, c = a + segments + 1, d = c + 1;
      indices[k++] = a; indices[k++] = c; indices[k++] = b; indices[k++] = b; indices[k++] = c; indices[k++] = d;
    }
  }
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const j of [a, b, c]) { normals[j] += nx; normals[j + 1] += ny; normals[j + 2] += nz; }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const l = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= l; normals[i + 1] /= l; normals[i + 2] /= l;
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(positions, 3));
  g.setAttribute('normal', new BufferAttribute(normals, 3));
  g.setIndex(new BufferAttribute(indices, 1));
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

// ------------------------------------------------------------------------------------------------ GPU (shading)
const hashGpu = Fn(([p]: any[]) => {
  const q = uvec2(ivec2(p));
  const h0 = q.x.mul(uint(0x9e3779b9)).bitXor(q.y.mul(uint(0x85ebca6b))).toVar();
  h0.assign(h0.bitXor(h0.shiftRight(uint(16))));
  h0.assign(h0.mul(uint(0x7feb352d)));
  h0.assign(h0.bitXor(h0.shiftRight(uint(15))));
  h0.assign(h0.mul(uint(0x846ca68b)));
  h0.assign(h0.bitXor(h0.shiftRight(uint(16))));
  return float(h0.shiftRight(uint(9))).add(0.5).mul(1 / 8388608);
}).setLayout({ name: 'seabedHash', type: 'float', inputs: [{ name: 'p', type: 'vec2' }] });

const noiseGpu = Fn(([p]: any[]) => {
  const i = floor(p);
  const f0 = fract(p);
  const f = f0.mul(f0).mul(f0.mul(-2).add(3));
  return mix(mix(hashGpu(i), hashGpu(i.add(vec2(1, 0))), f.x), mix(hashGpu(i.add(vec2(0, 1))), hashGpu(i.add(vec2(1, 1))), f.x), f.y);
}).setLayout({ name: 'seabedNoise', type: 'float', inputs: [{ name: 'p', type: 'vec2' }] });

/** Footprint-filtered noise (reference seabedFilteredNoise); derivatives at the call site (uniform control flow). */
function filteredNoise(p: any) {
  const footprint = max(length(dFdx(p)), length(dFdy(p)));
  const resolved = exp(footprint.mul(footprint).mul(-1.1));
  return select(resolved.lessThan(0.015), float(0.5), mix(float(0.5), noiseGpu(p), resolved));
}

function rockMask(p: any) {
  const erosion = noiseGpu(p.mul(0.31).add(vec2(41, -21))).sub(0.5).mul(0.19);
  let q: any = float(1e5);
  for (const [x, z, rx, rz, , a] of SEABED_ROCKS) {
    const c = Math.cos(a), t = Math.sin(a);
    const d = p.sub(vec2(x, z));
    const v = vec2(dot(d, vec2(c / rx, t / rx)), dot(d, vec2(-t / rz, c / rz)));
    q = min(q, dot(v, v).add(erosion));
  }
  return float(1).sub(smoothstep(0.72, 1.15, q));
}

const LENSES: readonly (readonly number[])[] = [
  [0.9578, 0.2874, 2.71, 0.21, 0.81], [-0.342, 0.9397, 3.83, 0.16, 3.21], [0.7431, -0.6691, 4.93, 0.12, 1.63],
  [-0.8829, -0.4695, 1.93, 0.17, 5.13], [0.1392, 0.9903, 6.17, 0.075, 4.02], [0.6157, 0.788, 3.19, 0.1, 2.36],
];

function causticFocus(p0: any, depth: any, time: any) {
  const drift = vec2(0.025, -0.017).mul(time);
  const warp = vec2(noiseGpu(p0.mul(0.39).add(drift).add(vec2(4.1, 19.7))), noiseGpu(p0.mul(0.43).sub(drift).add(vec2(13.3, 1.2))));
  const p = p0.add(warp.sub(0.5).mul(2.4));
  let H: any = vec3(0);
  for (const [dx, dy, k, amp, phase] of LENSES) {
    const angle = dot(p, vec2(dx, dy)).mul(k).sub(time.mul(Math.sqrt(9.81 * k) * 0.64)).add(phase);
    const fp = max(abs(dFdx(angle)), abs(dFdy(angle)));
    const h = sin(angle).mul(amp).mul(exp(fp.mul(fp).mul(-0.55)));
    H = H.add(vec3(dx * dx, dx * dy, dy * dy).mul(h));
  }
  const lens = min(depth, 9).mul(0.31);
  const det = abs(H.x.mul(lens).add(1).mul(H.z.mul(lens).add(1)).sub(lens.mul(lens).mul(H.y).mul(H.y)));
  return clamp(float(0.83).div(max(det, 0.28)), 0.62, 2.25);
}

export interface DemoSeabedOptions {
  origin: FloatingOrigin;
  /** TSL preExposure node (shared with the rest of the demo). */
  preExposure: any;
  /** cd/m^2 per reference unit. */
  refNits: number;
  segments?: number;
}

export interface DemoSeabed {
  readonly mesh: any;
  /** TERRAIN_QUERY: ground height at a world point (m above sea level), NO_GROUND outside the seabed mesh. */
  readonly query: TerrainQuery;
  readonly uniforms: { time: any; sunDir: any; solar: any; storm: any; caustics: any };
  /** Copy the frame state (allocation-free): sim time (s), sun direction (towards the sun), reference solar colour (RU), storminess 0..1. */
  update(time: number, sunDir: Readonly<Vec3>, solar: { r: number; g: number; b: number }, storminess: number): void;
  dispose(): void;
}

export function createDemoSeabed(o: DemoSeabedOptions): DemoSeabed {
  const geometry = createSeabedGeometry(o.segments ?? SEABED_SEGMENTS);
  const time = uniform(0);
  const sunDir = uniform(new Vector3(-0.513, 0.766, -0.387));
  const solar = uniform(new Vector3(1, 0.95, 0.84));
  const storm = uniform(0);
  const caustics = uniform(0.55);             // reference uSeabedCaustics (ocean.js:433)
  const IOR = 1.31;
  const ABSORPTION = vec3(0.047, 0.016, 0.009); // uSeabedLightAbsorption, 1/m
  const AMBIENT = vec3(0.38, 0.455, 0.49);      // uSeabedAmbient, RU

  const p = positionLocal;                      // seabed frame (reference coordinates), fragment varying
  const N = normalize(normalLocal);
  const depth = max(p.y.negate(), 0);
  const uvw = p.xz;
  const warp = vec2(noiseGpu(uvw.mul(0.052).add(vec2(12.3, 4.7))), noiseGpu(uvw.mul(0.047).sub(vec2(1.9, 7.8))));
  const sandPatch = filteredNoise(uvw.mul(0.21).add(warp.mul(1.1)));
  const grain = filteredNoise(uvw.mul(14.7).add(vec2(9.1, 3.6)));
  const mineral = filteredNoise(uvw.mul(2.43).sub(warp.mul(2.3)));
  const sand = mix(vec3(0.54, 0.505, 0.407), vec3(0.72, 0.684, 0.565), sandPatch.mul(0.6).add(0.28)).mul(grain.mul(0.07).add(0.96));
  const stoneTexture = filteredNoise(uvw.mul(1.71).add(warp)).mul(0.7).add(mineral.mul(0.3));
  const stone = mix(vec3(0.024, 0.029, 0.027), vec3(0.078, 0.083, 0.074), stoneTexture);
  const rock = smoothstep(0.08, 0.84, rockMask(uvw));
  const albedo = mix(sand, stone, rock);
  const sun = normalize(sunDir);
  const sunUnderwater = refract(sun.negate(), vec3(0, 1, 0), 1 / IOR).negate();
  const incidence = max(dot(N, sunUnderwater), 0);
  const cosU = max(sunUnderwater.y, 0.28);
  const sunPath = depth.div(cosU);
  const sunTransmission = exp(ABSORPTION.negate().mul(sunPath));
  const caustic = causticFocus(uvw.sub(sunUnderwater.xz.mul(depth).div(cosU)), depth, time);
  const causticStrength = caustics.mul(exp(depth.mul(-0.032))).mul(float(1).sub(storm)).mul(float(1).sub(rock.mul(0.45)));
  const focusing = float(1).add(causticStrength.mul(caustic.sub(1)));
  const direct = solar.mul(incidence).mul(1.1).mul(float(1).sub(storm.mul(0.91))).mul(sunTransmission).mul(focusing);
  const ambient = AMBIENT.mul(mix(float(1), float(0.6), storm)).mul(exp(vec3(-0.017, -0.007, -0.004).mul(depth)));
  const occlusion = mix(float(1), float(0.9), rock.mul(float(1).sub(clamp(N.y, 0, 1))));
  const radiance = albedo.mul(ambient.add(direct)).mul(occlusion);

  const material = new MeshBasicNodeMaterial();
  material.name = 'demo.seabed';
  material.colorNode = max(radiance, vec3(0)).mul(o.refNits).mul(o.preExposure);
  const mesh = new Mesh(geometry, material);
  mesh.name = 'demo.seabed';
  mesh.frustumCulled = false;
  // Anchor = world origin (reference coordinates): render position = anchor - floating origin.
  const place = () => mesh.position.set(-o.origin.origin.x, -o.origin.origin.y, -o.origin.origin.z);
  place();
  const off = o.origin.onShift(place);

  // TERRAIN_QUERY (flat demo world: x east, y up, z south; the seabed is anchored at the world origin).
  const xs = seabedCoords(o.segments ?? SEABED_SEGMENTS, SEABED_EXTENT);
  const xMin = xs[0], xMax = xs[xs.length - 1];
  const query: TerrainQuery = {
    heightAt(pw) {
      const x = pw.x, z = pw.z;
      if (x < xMin || x > xMax || z - SEABED_CENTER_Z < xMin || z - SEABED_CENTER_Z > xMax) return NO_GROUND;
      return seabedHeight(x, z);
    },
  };
  return {
    mesh,
    query,
    uniforms: { time, sunDir, solar, storm, caustics },
    update(t, sd, sc, st) {
      time.value = t;
      (sunDir.value as Vector3).set(sd.x, sd.y, sd.z);
      (solar.value as Vector3).set(sc.r, sc.g, sc.b);
      storm.value = st;
    },
    dispose() { off(); geometry.dispose(); material.dispose(); mesh.removeFromParent(); },
  };
}
