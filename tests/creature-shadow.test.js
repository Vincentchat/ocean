import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import * as THREE from "three";
import {
  CREATURE_CYCLE_SECONDS,
  blendTailStroke,
  createTailStrokeControls,
  createCreatureShadow,
  loadWhaleAsset,
  prepareSwimmingClip,
  sampleCreatureMotion,
} from "../src/creature-shadow.js";

const close = (actual, expected, tolerance = 1e-5) =>
  assert(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test("swimming clip leaves joint strokes but removes mechanical whole-body pitch", () => {
  const wholeBody = new THREE.QuaternionKeyframeTrack(
      "locator3.quaternion",
      [0, 1],
      [0, 0, 0, 1, 0.02, 0, 0, 0.9998],
    ),
    tail = new THREE.QuaternionKeyframeTrack(
      "locator4.quaternion",
      [0, 1],
      [0, 0, 0, 1, 0, 0.2, 0, 0.9799],
    ),
    prepared = prepareSwimmingClip(
      new THREE.AnimationClip("Swimming", 1, [wholeBody, tail]),
    );
  assert.deepEqual(prepared.tracks.map((track) => track.name), ["locator4.quaternion"]);
});

test("tail stroke keeps linked joints but reduces the asset's exaggerated amplitude", () => {
  const root = new THREE.Group(),
    locator4 = new THREE.Bone(),
    locator5 = new THREE.Bone(),
    locator6 = new THREE.Bone();
  locator4.name = "locator4";
  locator5.name = "locator5";
  locator6.name = "locator6";
  root.add(locator4);
  locator4.add(locator5);
  locator5.add(locator6);
  const controls = createTailStrokeControls(THREE, root),
    fullAngle = Math.PI / 3;
  for (const control of controls)
    control.joint.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), fullAngle);
  blendTailStroke(controls, 0.5);
  for (const control of controls) {
    const angle = 2 * Math.acos(control.joint.quaternion.w);
    close(angle, fullAngle * 0.5, 1e-5);
  }
  const amplitudes = [0, 2, 4, 6, 8].map(
    (time) => sampleCreatureMotion(time).tailAmplitude,
  );
  assert(Math.min(...amplitudes) >= 0.42);
  assert(Math.max(...amplitudes) <= 0.68);
});

function fakeWhaleGltf() {
  const geometry = new THREE.BoxGeometry(3, 2, 10, 2, 2, 4);
  geometry.computeTangents();
  const count = geometry.attributes.position.count,
    skinIndices = new Uint16Array(count * 4),
    skinWeights = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) skinWeights[i * 4] = 1;
  geometry.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(skinIndices, 4));
  geometry.setAttribute("skinWeight", new THREE.Float32BufferAttribute(skinWeights, 4));
  const albedo = new THREE.DataTexture(new Uint8Array([90, 110, 120, 255]), 1, 1),
    normal = new THREE.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1),
    orm = new THREE.DataTexture(new Uint8Array([255, 130, 0, 255]), 1, 1);
  for (const texture of [albedo, normal, orm]) texture.needsUpdate = true;
  const material = new THREE.MeshStandardMaterial({
      map: albedo,
      normalMap: normal,
      roughnessMap: orm,
      metalnessMap: orm,
    }),
    bone = new THREE.Bone(),
    skeleton = new THREE.Skeleton([bone]),
    mesh = new THREE.SkinnedMesh(geometry, material),
    scene = new THREE.Group();
  bone.name = "WhaleBone";
  mesh.add(bone);
  mesh.bind(skeleton);
  scene.add(mesh);
  const track = new THREE.QuaternionKeyframeTrack(
      "WhaleBone.quaternion",
      [0, 2, 4],
      [0, 0, 0, 1, 0, 0.25, 0, 0.9682458, 0, 0, 0, 1],
    ),
    clip = new THREE.AnimationClip("Swimming", 4, [track]);
  return { scene, animations: [clip], resources: { geometry, albedo, normal, orm } };
}

function fakeLoadOptions(gltf = fakeWhaleGltf()) {
  return {
    fetchImpl: async () => ({
      ok: true,
      arrayBuffer: async () => new ArrayBuffer(8),
    }),
    loader: { parseAsync: async () => gltf },
    gltf,
  };
}

test("creature motion is finite, periodic, offshore, and visible immediately", () => {
  const start = sampleCreatureMotion(0),
    early = sampleCreatureMotion(3),
    repeated = sampleCreatureMotion(CREATURE_CYCLE_SECONDS);
  for (const pose of [start, early, repeated])
    for (const value of Object.values(pose)) assert(Number.isFinite(value));
  assert(Math.abs(start.x - repeated.x) < 1e-9);
  assert(Math.abs(start.y - repeated.y) < 1e-9);
  assert(start.x > -18 && start.x < -10);
  assert(early.x > -10 && early.x < 0);
  assert(start.z < -30 && early.z < -30);
  assert(start.y <= -1.56 && early.y <= -1.56);
  assert(start.opacity > 0.99);
  assert.equal(start.breathPhase, 0);
  assert.equal(start.breath, 0);
});

test("creature fades before the periodic position wraps", () => {
  const wrapTime = CREATURE_CYCLE_SECONDS * 0.62,
    beforeWrap = sampleCreatureMotion(wrapTime - 0.02),
    afterWrap = sampleCreatureMotion(wrapTime + 0.02);
  assert(beforeWrap.opacity < 0.001);
  assert(afterWrap.opacity < 0.001);
});

test("asset loader forwards bytes and rejects unavailable responses", async () => {
  let parsed;
  const result = await loadWhaleAsset({
    url: "./whale.glb",
    fetchImpl: async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(12) }),
    loader: {
      async parseAsync(buffer, base) {
        parsed = { bytes: buffer.byteLength, base };
        return { scene: new THREE.Group(), animations: [] };
      },
    },
  });
  assert(result.scene.isGroup);
  assert.equal(parsed.bytes, 12);
  assert.match(parsed.base, /whale\.glb$/);
  await assert.rejects(
    loadWhaleAsset({
      fetchImpl: async () => ({ ok: false, status: 404 }),
      loader: { parseAsync() {} },
    }),
    /404/,
  );
});

test("late parsed whale resources are disposed after the load timeout", async () => {
  const gltf = fakeWhaleGltf();
  let geometryDisposals = 0,
    textureDisposals = 0;
  gltf.resources.geometry.addEventListener("dispose", () => geometryDisposals++);
  gltf.resources.albedo.addEventListener("dispose", () => textureDisposals++);
  await assert.rejects(
    loadWhaleAsset({
      timeoutMs: 5,
      fetchImpl: async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(8) }),
      loader: {
        parseAsync: async () => {
          await new Promise((resolve) => setTimeout(resolve, 15));
          return gltf;
        },
      },
    }),
    /timed out/,
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(geometryDisposals, 1);
  assert.equal(textureDisposals, 1);
});

test("loaded whale uses cloned skeletons, deterministic animation, and toggle wake", async () => {
  const options = fakeLoadOptions(),
    uniforms = {},
    creature = createCreatureShadow(THREE, uniforms, "", options);
  await creature.ready;
  await Promise.resolve();
  assert.equal(creature.loaded, true);
  const displayBone = creature.object.getObjectByName("WhaleBone"),
    refractedBone = creature.refractedObject.getObjectByName("WhaleBone");
  assert(displayBone?.isBone && refractedBone?.isBone);
  assert.notEqual(displayBone, refractedBone);
  assert.equal(creature.object.visible, false);
  assert.equal(creature.refractedObject.visible, true);
  creature.update(2);
  const first = displayBone.quaternion.clone();
  creature.update(2);
  assert(displayBone.quaternion.equals(first));
  assert(refractedBone.quaternion.equals(first));
  creature.update(4);
  assert(!displayBone.quaternion.equals(first));
  creature.update(4, 1 / 60);
  const movingVelocity = uniforms.uCreatureVelocity.value.clone();
  assert(movingVelocity.length() > 0);
  creature.update(4, 0);
  assert(
    uniforms.uCreatureVelocity.value.equals(movingVelocity),
    "pausing at the same time must not jump released breath back to the emitter",
  );
  creature.setEnabled(false);
  assert.equal(uniforms.uCreatureWake.value, 0);
  assert.equal(creature.object.visible, false);
  assert.equal(creature.refractedObject.visible, false);
  creature.setEnabled(true);
  assert.equal(uniforms.uCreatureWake.value, 1);
  assert.equal(creature.object.visible, false);
  assert.equal(creature.refractedObject.visible, true);
  creature.dispose();
});

test("display and refraction share source resources and dispose them once", async () => {
  const options = fakeLoadOptions(),
    creature = createCreatureShadow(THREE, {}, "", options);
  await creature.ready;
  await Promise.resolve();
  const displayMesh = creature.object.getObjectByProperty("isSkinnedMesh", true),
    refractedMesh = creature.refractedObject.getObjectByProperty("isSkinnedMesh", true);
  assert.equal(displayMesh.geometry, refractedMesh.geometry);
  assert.notEqual(displayMesh.material, refractedMesh.material);
  assert.notEqual(displayMesh.skeleton, refractedMesh.skeleton);
  assert.equal(displayMesh.material.uniforms.uWhaleAlbedo.value, options.gltf.resources.albedo);
  let geometryDisposals = 0,
    textureDisposals = 0,
    skeletonTextureDisposals = 0;
  options.gltf.resources.geometry.addEventListener("dispose", () => geometryDisposals++);
  options.gltf.resources.albedo.addEventListener("dispose", () => textureDisposals++);
  for (const mesh of [displayMesh, refractedMesh]) {
    mesh.skeleton.computeBoneTexture();
    mesh.skeleton.boneTexture.addEventListener(
      "dispose",
      () => skeletonTextureDisposals++,
    );
  }
  creature.dispose();
  creature.dispose();
  assert.equal(geometryDisposals, 1);
  assert.equal(textureDisposals, 1);
  assert.equal(skeletonTextureDisposals, 2);
});

test("failed whale load keeps groups empty and wake disabled", async () => {
  const warning = console.warn;
  console.warn = () => {};
  try {
    const uniforms = {},
      creature = createCreatureShadow(THREE, uniforms, "", {
        fetchImpl: async () => ({ ok: false, status: 503 }),
      });
    await assert.rejects(creature.ready, /503/);
    await Promise.resolve();
    assert.equal(creature.loaded, false);
    assert.equal(creature.object.children.length, 0);
    assert.equal(creature.refractedObject.children.length, 0);
    assert.equal(uniforms.uCreatureWake.value, 0);
    creature.dispose();
  } finally {
    console.warn = warning;
  }
});

test("bundled GLB preserves attribution, skin, animation, and embedded maps", () => {
  const file = fs.readFileSync(new URL("../public/blue-whale.glb", import.meta.url)),
    jsonLength = file.readUInt32LE(12),
    json = JSON.parse(file.subarray(20, 20 + jsonLength).toString().replace(/\0+$/, ""));
  assert.equal(json.asset.extras.title, "Blue Whale - Textured");
  assert.match(json.asset.extras.author, /Bohdan Lvov/);
  assert.match(json.asset.extras.license, /CC-BY-4.0/);
  assert.equal(json.skins.length, 1);
  assert.equal(json.animations[0].name, "Swimming");
  assert.deepEqual(
    json.images.map((image) => image.mimeType),
    ["image/png", "image/png", "image/png"],
  );
});
