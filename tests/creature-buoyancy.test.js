import test from "node:test";
import assert from "node:assert/strict";
import {
  applyCreatureFramePoint,
  createCreatureBuoyancy,
  createBuoyancyState,
  deriveBuoyancyTarget,
  stepBuoyancyState,
} from "../src/creature-buoyancy.js";
import * as THREE from "three";
import { sampleCreatureMotion } from "../src/creature-shadow.js";

const close = (actual, expected, tolerance = 1e-4) =>
  assert(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test("constant water plane produces heave without pitch or roll", () => {
  const target = deriveBuoyancyTarget({
    center: 2,
    bow: 2,
    stern: 2,
    port: 2,
    starboard: 2,
  });
  close(target.heave, 1.5);
  close(target.pitch, 0);
  close(target.roll, 0);
});

test("bow and port wave heights tilt body points toward the raised water", () => {
  const target = deriveBuoyancyTarget({
    center: 0,
    bow: 1,
    stern: -1,
    port: 0.4,
    starboard: -0.4,
  });
  assert(target.pitch > 0 && target.pitch <= 0.08);
  assert(target.roll < 0 && target.roll >= -0.06);
  const origin = { x: 0, y: 0, z: 0 },
    heading = { x: 1, y: 0 },
    bow = applyCreatureFramePoint({ x: 8, y: 0, z: 0 }, origin, heading, target),
    port = applyCreatureFramePoint({ x: 0, y: 0, z: 2 }, origin, heading, target);
  assert(bow.y > 0);
  assert(port.y > 0);
});

test("damped buoyancy is frozen at dt zero and nearly frame-rate independent", () => {
  const frozen = createBuoyancyState({ heave: 0.25, pitch: 0.01, roll: -0.02 });
  const snapshot = { ...frozen };
  stepBuoyancyState(frozen, { heave: 1, pitch: 0.06, roll: 0.04 }, 0);
  assert.deepEqual(frozen, snapshot);

  const target = { heave: 1.4, pitch: 0.075, roll: -0.05 },
    at30 = createBuoyancyState(),
    at120 = createBuoyancyState();
  for (let i = 0; i < 60; i++) stepBuoyancyState(at30, target, 1 / 30);
  for (let i = 0; i < 240; i++) stepBuoyancyState(at120, target, 1 / 120);
  close(at30.heave, at120.heave, 2e-4);
  close(at30.pitch, at120.pitch, 2e-4);
  close(at30.roll, at120.roll, 2e-4);
  assert(at30.heave > 0.8 && at30.heave < target.heave);
});

test("trajectory heading follows its derivative and propulsion speed varies smoothly", () => {
  const samples = [];
  for (const time of [1, 3, 5, 7, 9, 11, 13]) {
    const pose = sampleCreatureMotion(time),
      before = sampleCreatureMotion(time - 0.001),
      after = sampleCreatureMotion(time + 0.001),
      dx = after.x - before.x,
      dz = after.z - before.z,
      length = Math.hypot(dx, dz),
      tangentX = dx / length,
      tangentZ = dz / length;
    close(Math.cos(pose.yaw), tangentX, 2e-3);
    close(-Math.sin(pose.yaw), tangentZ, 2e-3);
    assert(pose.speed > 0);
    samples.push(pose.speed);
  }
  assert(Math.max(...samples) / Math.min(...samples) > 1.12);
  for (let time = 0.1; time < 30; time += 0.2) {
    const a = sampleCreatureMotion(time),
      b = sampleCreatureMotion(time + 0.01);
    assert(b.animationTime > a.animationTime);
  }
});

function uniforms() {
  return {
    uTime: { value: 0 },
    uWave: { value: 1.5 },
    uWind: { value: 1 },
    uWavePhase: { value: new Float32Array(7) },
    uGridRadialStep: { value: Math.log(9601) / 384 },
    uGridAngularStep: { value: (Math.PI * 2) / 512 },
    uCreaturePose: { value: new THREE.Vector4(0, -1.5, -40, 1) },
    uCreatureHeading: { value: new THREE.Vector2(1, 0) },
  };
}

function renderer({ fail = false, float = true } = {}) {
  let target = { name: "original" },
    renders = 0;
  const renderTargets = new Set();
  return {
    extensions: { has: () => float },
    getRenderTarget: () => target,
    getActiveCubeFace: () => 0,
    getActiveMipmapLevel: () => 0,
    setRenderTarget(value) {
      target = value;
      if (value?.isWebGLRenderTarget) renderTargets.add(value);
    },
    render() {
      renders++;
      if (fail) throw new Error("injected buoyancy failure");
    },
    async readRenderTargetPixelsAsync(_target, _x, _y, _w, _h, output) {
      output.set([0.6, 0.03, -0.02, 0.9, 0.2, 0.01, -0.01, 0.045]);
    },
    stats() {
      return { target, renders, renderTargets: [...renderTargets] };
    },
  };
}

test("GPU buoyancy ping-pongs state, restores the target, and freezes at dt zero", async () => {
  const shared = uniforms(),
    fakeRenderer = renderer(),
    original = fakeRenderer.getRenderTarget(),
    buoyancy = createCreatureBuoyancy(THREE, fakeRenderer, shared, "", {
      forceGpu: true,
    });
  const firstTexture = shared.uCreatureBuoyancy.value;
  buoyancy.update(1, 1 / 60);
  assert.equal(fakeRenderer.stats().renders, 1);
  assert.notEqual(shared.uCreatureBuoyancy.value, firstTexture);
  assert.equal(fakeRenderer.getRenderTarget(), original);
  buoyancy.update(1, 0);
  assert.equal(fakeRenderer.stats().renders, 1);
  const debug = await buoyancy.debugRead(1);
  debug.offset.forEach((value, index) => close(value, [0.6, 0.03, -0.02][index]));
  debug.velocity.forEach((value, index) => close(value, [0.2, 0.01, -0.01][index]));
  close(debug.targetHeave, 0.9);
  close(debug.targetPitch, 0.045);
  let activeTargetDisposals = 0;
  fakeRenderer.stats().renderTargets[0].addEventListener(
    "dispose",
    () => activeTargetDisposals++,
  );
  buoyancy.dispose();
  buoyancy.dispose();
  assert.equal(activeTargetDisposals, 1);
});

test("GPU failure releases targets and continues with finite CPU buoyancy", () => {
  const shared = uniforms(),
    fakeRenderer = renderer({ fail: true }),
    buoyancy = createCreatureBuoyancy(THREE, fakeRenderer, shared, "", {
      forceGpu: true,
    });
  buoyancy.update(1, 1 / 60);
  assert.equal(buoyancy.mode, "cpu");
  buoyancy.update(1.1, 1 / 60);
  assert.equal(shared.uCreatureBuoyancyReady.value, 1);
  const values = shared.uCreatureBuoyancy.value.image.data;
  assert(values.every(Number.isFinite));
  buoyancy.dispose();
  buoyancy.dispose();
});
