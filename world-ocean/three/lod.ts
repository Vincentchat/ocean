// Ocean mesh LOD: camera-centred geometry clipmap, snapped per level, geomorphed, drawn camera-relative.
// Isolated so a future world/lod-tree can replace it: the material only talks to `lod.nodes`.
//
// API
//   const lod = createOceanLod({ tier: 'high', shape, origin, material? , coverage? });
//   scene.add(lod.mesh);                         // ONE draw call (merged indexed mesh), frustumCulled = false
//   material.positionNode = lod.nodes.localPos.add(displacementLocal);   // REQUIRED (see below)
//   per frame (after the camera moved, before rendering):  lod.update(camera);   // allocation-free
//   lod.setTier('low');  lod.dispose();
//
// lod.nodes (TSL, vertex stage; wrap in varying(...) to use per fragment):
//   localPos          vec3  undisplaced vertex position in the mesh's LOCAL frame, metres, float32, small:
//                           x = east, y = up, z = south (= -north), origin = the camera's foot point at sea level.
//                           The mesh is placed at the foot (render space) and rotated by the shape's tangent
//                           frame there, so "up" is local +Y for every shape — displace along local +Y.
//                           (On the flat shape local == world axes.)
//   previousLocalPos  vec3  where the same world point was in LAST frame's local frame (undisplaced): add last
//                           frame's displacement to build the motion-vector previous position.
//   tileCoord(P)      vec2  periodic sampling coordinate for an FFT tile of period P metres, in tangent axes
//                           (east, south) = (world x, world z) on the flat shape: (world mod P) + small lattice
//                           offset. Built as anchor_P + q, anchor_P = (B_render + origin.wrapped(P)) mod P on the
//                           CPU in float64, q = the vertex's exact float32 lattice offset from B (B = lattice
//                           anchor near the camera). Never a raw world position on the GPU. uv = tileCoord(P) / P
//                           with repeat wrapping. Values are stable to < 1e-4 m when the camera or origin moves.
//   distanceToCamera  float 3D distance camera -> undisplaced vertex, metres (displacement fade, LOD).
//   morphFactor       float 0..1 geomorph weight towards the next coarser level.
//   cellSize          float local grid spacing (m) including the morph (footprint for explicit texture LOD).
//   isSkirt           float 1 on the horizon skirt (flat annulus + wall up to eye level), 0 on the grid.
//   level             float level index (0 = finest), -1 on the skirt (debug views).
//   cameraLocal       vec3  camera position in the local frame = (0, altitude, 0) (view vectors in local space).
//
// Geometry: the 'position' attribute holds PACKED GRID DATA (gx, gy, level*4+kind), not positions — a material
// without `positionNode = lod.nodes.localPos ...` draws garbage. See three/lod-math.ts for the clipmap model
// (snapping, morph, clamp, bit-exact level boundaries) and tests/lod.test.ts for its proofs.
//
// Floating origin: update() reads the camera's render position (camera.matrixWorld), converts to world with
// origin.toWorld (float64), and recomputes everything from world lattice coordinates, so an origin shift changes
// nothing on screen. origin.onShift re-places the mesh and the wrapped anchors immediately (no one-frame jump).
//
// Up from the shape: foot = shape.atAltitude(camera, 0), frame = shape.tangentFrame(foot). Nothing assumes +Y.
// Sphere: the grid is laid flat in the tangent plane at the foot (curvature not applied yet).
import { BufferAttribute, BufferGeometry, Matrix4, Mesh, Sphere, Vector2, Vector3, Vector4 } from 'three/webgpu';
import {
  abs, clamp, float, floor, int, length, max, normalize, positionGeometry, select, uniform, uniformArray, vec2, vec3,
} from 'three/tsl';
import type { FloatingOrigin, TangentFrame, Vec3, WorldShape } from '../stubs/world-core.ts';
import {
  MAX_LOD_LEVELS, buildLodGeometry, createLodState, lodDrawCount, lodParams, tileAnchor, updateLodState,
  type LodGeometryData, type LodParams, type LodState, type LodTier,
} from './lod-math.ts';

export type { LodTier } from './lod-math.ts';

export interface OceanLodOptions {
  tier: LodTier;
  shape: WorldShape;
  origin: FloatingOrigin;
  /** Material for the mesh (its positionNode must be built from lod.nodes.localPos). Can be set later. */
  material?: any;
  /** Rings reach at least this distance (m) before the skirt takes over. Default 2400 (reference mesh radius). */
  coverage?: number;
}

export interface OceanLodNodes {
  readonly localPos: any;
  readonly previousLocalPos: any;
  tileCoord(period: number): any;
  readonly distanceToCamera: any;
  readonly morphFactor: any;
  readonly cellSize: any;
  readonly isSkirt: any;
  /** Debug: level index of the vertex (float), -1 on the skirt. */
  readonly level: any;
  /** Camera position in the mesh's local frame: (0, altitude, 0). */
  readonly cameraLocal: any;
}

export interface OceanLodStats {
  tier: LodTier;
  vertices: number;
  triangles: number;
  drawnTriangles: number;
  drawnLevels: number;
  /** Finest drawn spacing, m. */
  spacing0: number;
  /** Max-norm reach of the last drawn ring, m (skirt beyond). */
  ringReach: number;
  skirtRadius: number;
}

export interface OceanLod {
  readonly mesh: any;
  readonly nodes: OceanLodNodes;
  readonly params: LodParams;
  readonly state: Readonly<LodState>;
  /** Allocation-free. Call once per frame after the camera's matrixWorld is current. */
  update(camera: any, origin?: FloatingOrigin): void;
  setTier(tier: LodTier): void;
  stats(out?: OceanLodStats): OceanLodStats;
  /** Debug/test hooks: offset the LOD's notion of the camera foot (m, east/south), disable snapping. */
  readonly debug: { footOffsetU: number; footOffsetV: number; noSnap: boolean };
  dispose(): void;
}

function makeGeometry(data: LodGeometryData): BufferGeometry {
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(data.packed, 3));
  g.setIndex(new BufferAttribute(data.index, 1));
  // Positions are packed grid data: give a huge bounding volume so nothing culls / computes from them.
  g.boundingSphere = new Sphere(new Vector3(), 1e9);
  g.name = 'ocean.lod';
  return g;
}

export function createOceanLod(opts: OceanLodOptions): OceanLod {
  const { shape } = opts;
  let origin = opts.origin;
  let params = lodParams(opts.tier, { coverage: opts.coverage });
  let data = buildLodGeometry(params);
  let geometry = makeGeometry(data);
  const state = createLodState();
  const debug = { footOffsetU: 0, footOffsetV: 0, noSnap: false };

  // ------------------------------------------------------------------------------------ uniforms
  const levelVecs: Vector4[] = [];
  for (let i = 0; i <= MAX_LOD_LEVELS; i++) levelVecs.push(new Vector4(0, 0, 1, 0));
  const uLevels = uniformArray(levelVecs, 'vec4');       // per level: (C_l - B) (m), spacing (m), hasParent
  const uPhi = uniform(new Vector2());                   // foot - B (m)
  const uDFoot = uniform(new Vector2());                 // foot - previous foot (m)
  const uLast = uniform(0);                              // index of the last drawn level (skirt attaches there)
  const uMorphA = uniform(params.morphStart);
  const uMorphB = uniform(params.morphEnd);
  const uHalfHole = uniform(params.halfCells / 2);
  const uSkirtRadius = uniform(10000);                   // m
  const uCamHeight = uniform(0);                         // camera altitude above the foot (m)

  // ------------------------------------------------------------------------------------ vertex nodes
  // Mirrors lod-math.ts emulateVertex op for op (bit-exact level boundaries rely on the op order).
  const packed = positionGeometry;
  const lk = packed.z;
  const levelAttr = floor(lk.mul(0.25));
  const kind = lk.sub(levelAttr.mul(4));
  const skirt = kind.greaterThan(0.5);
  const lvlF = select(skirt, uLast, levelAttr);
  const A = uLevels.element(int(lvlF));
  const Pa = uLevels.element(int(lvlF.add(1)));
  const s = A.z;
  const g = packed.xy;
  const q0 = A.xy.add(g.mul(s));
  const p0 = q0.sub(uPhi);
  const dmax = max(abs(p0.x), abs(p0.y));
  const morph = clamp(dmax.div(s).sub(uMorphA).div(uMorphB.sub(uMorphA)), 0, 1).mul(A.w);
  const odd = g.sub(floor(g.mul(0.5)).mul(2));
  const q1 = q0.sub(odd.mul(s).mul(morph));
  const half = uHalfHole.mul(Pa.z);
  const q2 = select(A.w.greaterThan(0.5), clamp(q1, Pa.xy.sub(half), Pa.xy.add(half)), q1);
  const pg = q2.sub(uPhi);
  const outer = kind.greaterThan(1.5);
  const pOut = normalize(pg).mul(uSkirtRadius);
  const pl = select(outer, pOut, pg);
  const ql = select(outer, pOut.add(uPhi), q2);
  const y = select(kind.greaterThan(2.5), uCamHeight, float(0));
  const localPos = vec3(pl.x, y, pl.y);
  const previousLocalPos = localPos.add(vec3(uDFoot.x, 0, uDFoot.y));
  const distanceToCamera = length(vec3(pl.x, y.sub(uCamHeight), pl.y));
  const cellSize = select(skirt, uSkirtRadius.mul(0.05), s.mul(morph.add(1)));
  const isSkirt = select(skirt, float(1), float(0));
  const level = select(skirt, float(-1), levelAttr);
  const cameraLocal = vec3(0, uCamHeight, 0);

  const periods: number[] = [];
  const anchors: any[] = [];
  const tileNodes: any[] = [];
  function tileCoord(period: number) {
    if (!(period > 0)) throw new RangeError('tileCoord: period must be > 0');
    let i = periods.indexOf(period);
    if (i < 0) {
      i = periods.length;
      periods.push(period);
      anchors.push(uniform(new Vector2()));
      tileNodes.push(vec2(anchors[i]).add(ql));
      if (state.hasPrev) updateAnchors();
    }
    return tileNodes[i];
  }

  const nodes: OceanLodNodes = { localPos, previousLocalPos, tileCoord, distanceToCamera, morphFactor: morph, cellSize, isSkirt, level, cameraLocal };

  const mesh = new Mesh(geometry, opts.material);
  mesh.name = 'ocean.lod';
  mesh.frustumCulled = false; // camera-centred and displaced: always visible (the reference does the same)
  mesh.matrixAutoUpdate = true;

  // ------------------------------------------------------------------------------------ per-frame update
  const camR: Vec3 = { x: 0, y: 0, z: 0 };
  const camW: Vec3 = { x: 0, y: 0, z: 0 };
  const footW: Vec3 = { x: 0, y: 0, z: 0 };
  const footR: Vec3 = { x: 0, y: 0, z: 0 };
  const wrappedTmp: Vec3 = { x: 0, y: 0, z: 0 };
  const south: Vec3 = { x: 0, y: 0, z: 0 };
  const frame: TangentFrame = { east: { x: 1, y: 0, z: 0 }, north: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 } };
  const vE = new Vector3(), vU = new Vector3(), vS = new Vector3();
  const basis = new Matrix4();
  let lastFar = 20000;

  function updateAnchors() {
    const E = frame.east;
    const bDotE = footR.x * E.x + footR.y * E.y + footR.z * E.z + (state.baseU - state.footU);
    const bDotS = footR.x * south.x + footR.y * south.y + footR.z * south.z + (state.baseV - state.footV);
    for (let i = 0; i < periods.length; i++) {
      const P = periods[i];
      origin.wrapped(P, wrappedTmp);
      (anchors[i].value as Vector2).set(
        tileAnchor(E, bDotE, state.baseU, wrappedTmp, P),
        tileAnchor(south, bDotS, state.baseV, wrappedTmp, P),
      );
    }
  }

  function updateFromRender(advance: boolean) {
    origin.toWorld(camR, camW);
    const h = shape.altitude(camW);
    shape.atAltitude(camW, 0, footW);
    shape.tangentFrame(footW, frame);
    const E = frame.east, N = frame.north, U = frame.up;
    south.x = -N.x; south.y = -N.y; south.z = -N.z;
    const fu = footW.x * E.x + footW.y * E.y + footW.z * E.z + debug.footOffsetU;
    const fv = footW.x * south.x + footW.y * south.y + footW.z * south.z + debug.footOffsetV;
    state.noSnap = debug.noSnap;
    updateLodState(state, params, fu, fv, h, advance);

    origin.toRender(footW, footR);
    footR.x += E.x * debug.footOffsetU + south.x * debug.footOffsetV;
    footR.y += E.y * debug.footOffsetU + south.y * debug.footOffsetV;
    footR.z += E.z * debug.footOffsetU + south.z * debug.footOffsetV;
    mesh.position.set(footR.x, footR.y, footR.z);
    vE.set(E.x, E.y, E.z); vU.set(U.x, U.y, U.z); vS.set(south.x, south.y, south.z);
    basis.makeBasis(vE, vU, vS);
    mesh.quaternion.setFromRotationMatrix(basis);

    const lv = state.level;
    for (let l = 0; l <= params.levels; l++) levelVecs[l].set(lv[l * 4], lv[l * 4 + 1], lv[l * 4 + 2], lv[l * 4 + 3]);
    (uPhi.value as Vector2).set(state.phiU, state.phiV);
    (uDFoot.value as Vector2).set(state.dFootU, state.dFootV);
    uLast.value = state.drawnLevels - 1;
    uCamHeight.value = h;
    const reach = (params.halfCells + params.margin) * state.spacing0 * 2 ** (state.drawnLevels - 1);
    // Skirt edge inside the far plane (wall top at eye level projects onto the horizon line).
    uSkirtRadius.value = Math.max(0.9 * lastFar, 1.5 * Math.SQRT2 * reach);
    geometry.setDrawRange(0, lodDrawCount(state, data));
    updateAnchors();
  }

  function update(camera: any, org?: FloatingOrigin) {
    if (org) origin = org;
    const e = camera.matrixWorld.elements;
    camR.x = e[12]; camR.y = e[13]; camR.z = e[14];
    if (typeof camera.far === 'number' && camera.far > 0) lastFar = camera.far;
    updateFromRender(true);
  }

  // Origin shift: the camera's render position moves by -delta; re-place the mesh + anchors right away.
  const onShift = (delta: Readonly<Vec3>) => {
    camR.x -= delta.x; camR.y -= delta.y; camR.z -= delta.z;
    if (state.hasPrev) updateFromRender(false);
  };
  let unsubscribe = origin.onShift(onShift);

  return {
    mesh,
    nodes,
    get params() { return params; },
    state,
    update,
    debug,
    setTier(tier: LodTier) {
      if (tier === params.tier) return;
      params = lodParams(tier, { coverage: opts.coverage });
      data = buildLodGeometry(params);
      const old = geometry;
      geometry = makeGeometry(data);
      mesh.geometry = geometry;
      old.dispose();
      uMorphA.value = params.morphStart;
      uMorphB.value = params.morphEnd;
      uHalfHole.value = params.halfCells / 2;
      state.log2Spacing = NaN;
      if (state.hasPrev) updateFromRender(false);
    },
    stats(out?: OceanLodStats): OceanLodStats {
      const o = out ?? ({} as OceanLodStats);
      o.tier = params.tier;
      o.vertices = data.vertexCount;
      o.triangles = data.triangleCount;
      o.drawnTriangles = lodDrawCount(state, data) / 3;
      o.drawnLevels = state.drawnLevels;
      o.spacing0 = state.spacing0;
      o.ringReach = (params.halfCells + params.margin) * state.spacing0 * 2 ** (state.drawnLevels - 1);
      o.skirtRadius = uSkirtRadius.value as number;
      return o;
    },
    dispose() {
      unsubscribe();
      unsubscribe = () => {};
      mesh.removeFromParent();
      geometry.dispose();
    },
  };
}
