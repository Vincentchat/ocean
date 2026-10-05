import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import {
  backtrackEmissionPosition,
  createWhaleBreath,
} from "../src/whale-breath.js";

test("released mist stays in world space as its emitter advances", () => {
  const output = { x: 0, y: 0, z: 0 };
  backtrackEmissionPosition(
    { x: 12, y: 3, z: -4 },
    { x: 2, y: 0, z: 0 },
    1,
    output,
  );
  assert.deepEqual(output, { x: 10, y: 3, z: -4 });
});

test("breath installs shared controls and updates time without replacing shared values", () => {
  const pose = { value: new THREE.Vector4(3, -1.5, -40, 0.8) },
    wake = { value: 1 },
    uniforms = { uCreaturePose: pose, uCreatureWake: wake },
    breath = createWhaleBreath(THREE, uniforms);
  assert.equal(uniforms.uCreaturePose, pose);
  assert.equal(uniforms.uCreatureWake, wake);
  assert.equal(uniforms.uCreatureBreath.value, 0);
  assert.equal(uniforms.uCreatureBreathPhase.value, 0);
  breath.update(12.5);
  assert.equal(uniforms.uTime.value, 12.5);
  breath.update(Number.NaN);
  assert.equal(uniforms.uTime.value, 0);
  breath.setEnabled(false);
  assert.equal(breath.object.visible, false);
  assert.equal(breath.object.material.uniforms.uBreathEnabled.value, 0);
  breath.dispose();
});

test("emission uses the shared rigid creature frame for the blowhole anchor", () => {
  const buoyancy = { value: new THREE.DataTexture() },
    ready = { value: 1 },
    uniforms = { uCreatureBuoyancy: buoyancy, uCreatureBuoyancyReady: ready },
    first = createWhaleBreath(THREE, uniforms),
    second = createWhaleBreath(THREE, {}),
    seeds = first.object.geometry.getAttribute("aBreathSeed").array,
    repeated = second.object.geometry.getAttribute("aBreathSeed").array,
    kinds = first.object.geometry.getAttribute("aBreathKind").array;
  assert.equal(first.object.geometry.instanceCount, 38);
  assert.deepEqual(seeds, repeated);
  assert([...seeds].every((value) => Number.isFinite(value) && value >= 0 && value < 1));
  assert.equal([...kinds].filter((value) => value === 1).length, 10);
  assert.equal(first.object.material.uniforms.uCreatureBuoyancy, buoyancy);
  assert.equal(first.object.material.uniforms.uCreatureBuoyancyReady, ready);
  assert.match(first.object.material.vertexShader, /creatureApplyFramePoint\(unframedAnchor,origin,heading\)/);
  assert.match(first.object.material.vertexShader, /creatureSurfaceHeight\(anchor\.xz\)/);
  assert.match(first.object.material.vertexShader, /texture2D\(uCreatureBuoyancy,vec2\(\.75,\.5\)\)/);
  assert.match(first.object.material.vertexShader, /backtrackCreatureMotion/);
  assert.doesNotMatch(first.object.material.vertexShader, /sin\(uTime\*\.31\)/);
  assert.doesNotMatch(first.object.material.vertexShader, /breathFloatHeight/);
  assert.match(first.object.material.vertexShader, /uCreatureBreathPhase-birth/);
  assert.match(first.object.material.vertexShader, /age\*2\.30/);
  assert.match(first.object.material.vertexShader, /\.065,\.14/);
  assert.match(first.object.material.vertexShader, /uCreatureWake\*uCreaturePose\.w/);
  first.dispose();
  second.dispose();
});

test("geometry, material, and owned fallback field are disposed exactly once", () => {
  const breath = createWhaleBreath(THREE, {}),
    geometry = breath.object.geometry,
    material = breath.object.material,
    fallback = material.uniforms.uFieldLarge.value;
  let geometryDisposals = 0,
    materialDisposals = 0,
    textureDisposals = 0;
  geometry.addEventListener("dispose", () => geometryDisposals++);
  material.addEventListener("dispose", () => materialDisposals++);
  fallback.addEventListener("dispose", () => textureDisposals++);
  breath.dispose();
  breath.dispose();
  assert.equal(geometryDisposals, 1);
  assert.equal(materialDisposals, 1);
  assert.equal(textureDisposals, 1);
});
