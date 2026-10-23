// Ocean mesh LOD: camera-centred geometry clipmap, snapped per level, geomorphed, drawn camera-relative.
// Isolated so a future world/lod-tree can replace it: the material only talks to `lod.nodes`.
//
// API
//   const lod = createOceanLod({ tier: 'high', shape, origin, material? , coverage? });
//   scene.add(lod.mesh);                         // ONE draw call (merged indexed mesh), frustumCulled = false
//   material.positionNode = lod.nodes.localPos.add(displacementLocal);   // REQUIRED (see below)
//   per frame (after the camera moved, before rendering):  lod.update(camera);   // allocation-free; reads the
//                                                camera's matrixWorld and far (far only caps the ring count)
//   lod.setTier('low');   // non-blocking: built in 2 ms slices inside update(), old tier drawn until the swap
//   lod.dispose();
//
// lod.nodes (TSL, vertex stage; wrap in varying(...) to use per fragment):
//   localPos          vec3  undisplaced vertex position in the mesh's LOCAL frame, metres, float32, small:
//                           x = east, y = up, z = south (= -north), origin = the camera's foot point at sea level.
//                           The mesh is placed at the foot (render space) and rotated by the shape's tangent
//                           frame there, so "up" is local +Y for every shape — displace along local +Y.
//                           (On the flat shape local == world axes.)
//   previousLocalPos  vec3  where the same world point was in LAST frame's local frame (undisplaced), using last
//                           frame's foot AND last frame's geomorph; the horizon closure is camera-attached (= localPos).
//                           Add last frame's displacement to build the motion-vector previous position.
//   tileCoord(P)      vec2  periodic sampling coordinate for an FFT tile of period P metres, in tangent axes
//                           (east, south) = (world x, world z) on the flat shape: (world mod P) + small lattice
//                           offset. Built as anchor_P + q, anchor_P = (B_render + origin.wrapped(P)) mod P on the
//                           CPU in float64, q = the vertex's exact float32 lattice offset from B (B = lattice
//                           anchor near the camera). Never a raw world position on the GPU. uv = tileCoord(P) / P
//                           with repeat wrapping. Values are stable to < 1e-4 m when the camera or origin moves.
//   distanceToCamera  float 3D distance camera -> undisplaced vertex, metres (displacement fade, LOD).
//   morphFactor       float 0..1 geomorph weight towards the next coarser level.
//   cellSize          float local grid spacing (m) including the morph (footprint for explicit texture LOD).
//   isSkirt           float 1 on the horizon closure, 0 on the grid. The closure is a flat sea-level annulus from
//                           the last ring out to view depth 0.995 far, then a strip up to EYE LEVEL at view depth
//                           0.999 far (= the true horizon line; a positionNode cannot emit w = 0, so points at infinity
//                           are mapped onto the far plane). Computed per rendering camera in the shader from cameraFar
//                           and the view matrix (far = Infinity / huge -> 1e6 m). It covers up to the horizon for any
//                           far plane and hides nothing nearer than 0.995 far.
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
// Sphere (DESIGN LIMIT, not just 'curvature missing'): the snapping lattice lives in the tangent frame at the camera
// foot. On a sphere that frame rotates as the camera moves, so vertices would swim; a sphere needs the lattice anchored
// to a fixed per-region frame (or a cube-sphere quadtree) plus curvature. tileAnchor's non-axis-aligned path wraps the
// float64 lattice coordinate (equivalent value) instead of origin.wrapped.
//
// tileCoord(P) creates its uniform/node on first call: call it at material-build time, never per frame.
import { BufferAttribute, BufferGeometry, Matrix4, Mesh, Sphere, Vector2, Vector3, Vector4 } from 'three/webgpu';
import {
  abs, cameraFar, clamp, float, floor, int, length, max, min, cameraViewMatrix, modelWorldMatrix, normalize, positionGeometry, select,
  uniform, uniformArray, vec2, vec3, vec4,
} from 'three/tsl';
import type { FloatingOrigin, TangentFrame, Vec3, WorldShape } from '../stubs/world-core.ts';
import {
  MAX_LOD_LEVELS, buildLodGeometry, createLodState, lodGeometrySteps, lodDrawCount, lodParams, tileAnchor, updateLodState,
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
  /** Tier being built in the background (old tier still drawn), or null. */
  pendingTier: LodTier | null;
}

export interface OceanLod {
  readonly mesh: any;
  readonly nodes: OceanLodNodes;
  readonly params: LodParams;
  readonly state: Readonly<LodState>;
  /** Allocation-free. Call once per frame after the camera's matrixWorld is current. */
  update(camera: any, origin?: FloatingOrigin): void;
  /** Starts building the tier in time slices inside update(); the old tier stays drawn until the atomic swap. */
  setTier(tier: LodTier): void;
  readonly pendingTier: LodTier | null;
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
  // Horizon closure (skirt kinds 2, 3), per rendering camera, independent of the CPU: the outer loop is put ON the
  // camera's far plane (view depth 0.999 far; 1e6 m when far is infinite / huge). Kind 3 sits at eye level, so it
  // projects exactly onto the true horizon line (a point at infinity mapped onto the far plane, since a positionNode
  // cannot emit w = 0); kind 2 is the flat sea-level annulus edge at 0.995 of that depth. The closure therefore only
  // occupies the last 0.5 % of the view depth range: it hides nothing nearer than 0.995 far.
  const outer = kind.greaterThan(1.5);
  const top = kind.greaterThan(2.5);
  const dirH = normalize(pg);
  // (cameraViewMatrix * modelWorldMatrix inline: the shared modelViewMatrix node gets emitted as a variable assigned
  // inside whichever branch references it first, leaving it unset for the other vertices.)
  const vdir = cameraViewMatrix.mul(modelWorldMatrix.mul(vec4(dirH.x, 0, dirH.y, 0))).xyz;
  const cosView = max(vdir.z.negate(), 0.2); // cos(angle to the view axis); directions > 78 deg off-axis are off-screen
  // View depth of local point (dirH * t, y) = t * cosView - (y - h) * upView.z  ->  solve t for the target depth.
  const upViewZ = cameraViewMatrix.mul(modelWorldMatrix.mul(vec4(0, 1, 0, 0))).z;
  const depthFar = min(cameraFar.mul(0.999), 1e6);
  const depthTarget = select(top, depthFar, depthFar.mul(0.995));
  const yOffset = select(top, float(0), uCamHeight.negate()); // vertex height - camera height
  const tOut = max(depthTarget.add(yOffset.mul(upViewZ)), depthTarget.mul(0.5)).div(cosView);
  const pOut = dirH.mul(tOut);
  const pl = select(outer, pOut, pg);
  const ql = select(outer, pOut.add(uPhi), q2);
  const y = select(top, uCamHeight, float(0));
  const localPos = vec3(pl.x, y, pl.y);
  // Previous frame (motion vectors): the same WORLD point in last frame's local frame. Grid vertices: lattice point
  // minus last frame's foot, with LAST frame's morph (morph depends on the camera). Horizon closure: camera-attached
  // (at infinity), so its previous local position is its current one.
  const phiPrev = uPhi.sub(uDFoot);
  const p0Prev = q0.sub(phiPrev);
  const morphPrev = clamp(max(abs(p0Prev.x), abs(p0Prev.y)).div(s).sub(uMorphA).div(uMorphB.sub(uMorphA)), 0, 1).mul(A.w);
  const q1Prev = q0.sub(odd.mul(s).mul(morphPrev));
  const q2Prev = select(A.w.greaterThan(0.5), clamp(q1Prev, Pa.xy.sub(half), Pa.xy.add(half)), q1Prev);
  const pPrev = q2Prev.sub(phiPrev);
  const previousLocalPos = select(outer, localPos, vec3(pPrev.x, 0, pPrev.y));
  const distanceToCamera = length(vec3(pl.x, y.sub(uCamHeight), pl.y));
  const cellSize = select(skirt, distanceToCamera.mul(0.05), s.mul(morph.add(1)));
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
    // Rings must stay well inside the far plane (the horizon closure lives at 0.995..0.999 far): limit the max-norm
    // reach to 0.6 far (corner 0.85 far). Infinite / huge far: no limit.
    updateLodState(state, params, fu, fv, h, advance, Number.isFinite(lastFar) ? 0.6 * lastFar : Infinity);

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
    geometry.setDrawRange(0, lodDrawCount(state, data));
    updateAnchors();
  }

  // ---------------------------------------------------------------------------------- tier change (no stall)
  // The new tier's geometry is generated in time slices (BUILD_BUDGET_MS per update) while the old tier stays on
  // screen, then swapped atomically (geometry + morph uniforms + params in the same update).
  const BUILD_BUDGET_MS = 2;
  let pending: { params: LodParams; it: Generator<void, LodGeometryData, void> } | null = null;
  function stepPending() {
    if (!pending) return;
    const t0 = performance.now();
    let r = pending.it.next();
    while (!r.done && performance.now() - t0 < BUILD_BUDGET_MS) r = pending.it.next();
    if (!r.done) return;
    params = pending.params;
    pending = null;
    data = r.value;
    const old = geometry;
    geometry = makeGeometry(data);
    mesh.geometry = geometry;
    old.dispose();
    uMorphA.value = params.morphStart;
    uMorphB.value = params.morphEnd;
    uHalfHole.value = params.halfCells / 2;
    state.log2Spacing = NaN;
  }

  function update(camera: any, org?: FloatingOrigin) {
    if (org) origin = org;
    const e = camera.matrixWorld.elements;
    camR.x = e[12]; camR.y = e[13]; camR.z = e[14];
    // camera.far only limits how many rings are drawn (they must stay inside the far plane); the horizon closure
    // itself reads the rendering camera's far in the shader (cameraFar), so other cameras/passes stay correct.
    lastFar = typeof camera.far === 'number' && camera.far > 0 ? camera.far : Infinity;
    stepPending();
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
      if (pending) { if (pending.params.tier === tier) return; pending = null; }
      if (tier === params.tier) return;
      const p = lodParams(tier, { coverage: opts.coverage });
      pending = { params: p, it: lodGeometrySteps(p) };
    },
    get pendingTier() { return pending ? pending.params.tier : null; },
    stats(out?: OceanLodStats): OceanLodStats {
      const o = out ?? ({} as OceanLodStats);
      o.tier = params.tier;
      o.vertices = data.vertexCount;
      o.triangles = data.triangleCount;
      o.drawnTriangles = lodDrawCount(state, data) / 3;
      o.drawnLevels = state.drawnLevels;
      o.spacing0 = state.spacing0;
      o.ringReach = (params.halfCells + params.margin) * state.spacing0 * 2 ** (state.drawnLevels - 1);
      o.pendingTier = pending ? pending.params.tier : null;
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
