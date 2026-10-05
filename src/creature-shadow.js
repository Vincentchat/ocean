import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { clone as cloneSkeleton } from "three/addons/utils/SkeletonUtils.js";
import { CREATURE_FRAME_GLSL } from "./creature-buoyancy.js";

export const CREATURE_CYCLE_SECONDS = 55;
export const WHALE_ASSET_URL = "./blue-whale.glb";

function smoothstep(a, b, value) {
  const t = Math.max(0, Math.min(1, (value - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function smoothstepDerivative(a, b, value) {
  const t = (value - a) / (b - a);
  return t > 0 && t < 1 ? (6 * t * (1 - t)) / (b - a) : 0;
}

export function sampleCreatureMotion(time, output = {}) {
  const safeTime = Number.isFinite(time) ? time : 0,
    cyclePosition = safeTime / CREATURE_CYCLE_SECONDS + 0.38,
    cycleIndex = Math.floor(cyclePosition),
    phase = cyclePosition - cycleIndex,
    beatCount = 6,
    beatAngle = phase * Math.PI * 2 * beatCount,
    propulsion = 0.22,
    progress =
      phase +
      (propulsion / (Math.PI * 2 * beatCount)) * (1 - Math.cos(beatAngle)),
    progressRate = 1 + propulsion * Math.sin(beatAngle),
    pathT = Math.max(0, Math.min(1, (progress - 0.2) / 0.62)),
    pathSmooth = pathT * pathT * (3 - 2 * pathT),
    pathDerivative =
      progress > 0.2 && progress < 0.82
        ? (6 * pathT * (1 - pathT)) / 0.62
        : 0,
    dx = 128 * progressRate,
    dz = 14 * pathDerivative * progressRate;
  output.x = -64 + progress * 128;
  const rise = smoothstep(0.32, 0.46, phase),
    fall = smoothstep(0.7, 0.84, phase),
    surfacing = rise * (1 - fall),
    surfacingRate =
      smoothstepDerivative(0.32, 0.46, phase) * (1 - fall) -
      rise * smoothstepDerivative(0.7, 0.84, phase);
  output.y = -1.7 + surfacing * 0.14;
  output.z = -49 + pathSmooth * 14;
  output.yaw = -Math.atan2(dz, dx);
  output.roll = 0;
  output.speed = Math.hypot(dx, dz) / CREATURE_CYCLE_SECONDS;
  output.tailAmplitude = 0.55 + 0.13 * Math.sin(beatAngle);
  output.velocityX = dx / CREATURE_CYCLE_SECONDS;
  output.velocityY = (surfacingRate * 0.14) / CREATURE_CYCLE_SECONDS;
  output.velocityZ = dz / CREATURE_CYCLE_SECONDS;
  output.animationTime =
    (cycleIndex + progress) * beatCount * 4.125;
  output.scale = 1;
  output.breathPhase = 0;
  output.breath = 0;
  output.opacity =
    smoothstep(0.025, 0.09, phase) * (1 - smoothstep(0.91, 0.975, phase));
  return output;
}

export function installCreatureUniforms(THREE, uniforms) {
  if (!uniforms.uTime) uniforms.uTime = { value: 0 };
  if (!uniforms.uCreaturePose)
    uniforms.uCreaturePose = { value: new THREE.Vector4(-15.36, -1.7, -38, 0) };
  if (!uniforms.uCreatureHeading)
    uniforms.uCreatureHeading = { value: new THREE.Vector2(1, 0) };
  if (!uniforms.uCreatureVelocity)
    uniforms.uCreatureVelocity = { value: new THREE.Vector3() };
  if (!uniforms.uCreatureWake) uniforms.uCreatureWake = { value: 0 };
  if (!uniforms.uCreatureBreath) uniforms.uCreatureBreath = { value: 0 };
  if (!uniforms.uCreatureBreathPhase)
    uniforms.uCreatureBreathPhase = { value: 0 };
  return uniforms;
}

export async function loadWhaleAsset({
  url = WHALE_ASSET_URL,
  timeoutMs = 8000,
  fetchImpl = globalThis.fetch,
  loader = new GLTFLoader(),
} = {}) {
  if (typeof fetchImpl !== "function") throw new Error("Whale asset fetch unavailable");
  const controller = new AbortController();
  let timedOut = false, timeoutId;
  const loading = (async () => {
      const response = await fetchImpl(url, { signal: controller.signal });
      if (!response.ok) throw new Error(`Whale asset unavailable (${response.status})`);
      const buffer = await response.arrayBuffer();
      const base = typeof document === "undefined" ? "http://localhost/" : document.baseURI;
      return loader.parseAsync(buffer, new URL(url, base).href);
    })(),
    timeout = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        timedOut = true;
        controller.abort();
        reject(new Error("Whale asset timed out"));
      }, timeoutMs);
    });
  loading.then(
    (gltf) => {
      if (timedOut) disposeGltfResources(gltf);
    },
    () => {},
  );
  try {
    return await Promise.race([loading, timeout]);
  } finally {
    clearTimeout(timeoutId);
    if (!timedOut) controller.abort();
  }
}

function disposeGltfResources(gltf) {
  const geometries = new Set(), textures = new Set(), materials = new Set();
  gltf?.scene?.traverse((node) => {
    if (!node.isMesh) return;
    geometries.add(node.geometry);
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (!material) continue;
      materials.add(material);
      for (const key of ["map", "normalMap", "roughnessMap", "metalnessMap", "aoMap"])
        if (material[key]) textures.add(material[key]);
    }
  });
  for (const geometry of geometries) geometry.dispose();
  for (const texture of textures) texture.dispose();
  for (const material of materials) material.dispose();
}

const WHALE_VERTEX = `
#include <common>
#include <skinning_pars_vertex>
uniform vec4 uCreaturePose;
uniform vec2 uCreatureHeading;
${CREATURE_FRAME_GLSL}
varying vec2 vWhaleUv;
varying vec3 vWhaleWorld,vWhaleNormal,vWhaleTangent,vWhaleBitangent;
void main() {
  vWhaleUv=uv;
  #include <beginnormal_vertex>
  #include <skinbase_vertex>
  #include <skinnormal_vertex>
  #include <begin_vertex>
  #include <skinning_vertex>
  vec4 world=modelMatrix*vec4(transformed,1.);
  world.xyz=creatureApplyFramePoint(world.xyz,uCreaturePose.xyz,uCreatureHeading);
  vWhaleWorld=world.xyz;
  vec3 worldNormal=creatureApplyFrameVector(normalize(mat3(modelMatrix)*objectNormal),uCreatureHeading);
  vec3 worldTangent=creatureApplyFrameVector(normalize(mat3(modelMatrix)*objectTangent),uCreatureHeading);
  vWhaleNormal=normalize(worldNormal);
  vWhaleTangent=normalize(worldTangent);
  vWhaleBitangent=normalize(cross(vWhaleNormal,vWhaleTangent)*tangent.w);
  gl_Position=projectionMatrix*viewMatrix*world;
}`;

export function prepareSwimmingClip(clip) {
  const tracks = clip.tracks.filter(
    (track) => !/(^|\/)locator3\.(quaternion|rotation)$/.test(track.name),
  );
  return new clip.constructor(clip.name, clip.duration, tracks, clip.blendMode);
}

export function createTailStrokeControls(THREE, root) {
  return ["locator4", "locator5", "locator6"]
    .map((name) => root.getObjectByName(name))
    .filter(Boolean)
    .map((joint) => ({
      joint,
      rest: joint.quaternion.clone(),
      animated: new THREE.Quaternion(),
    }));
}

export function blendTailStroke(controls, amplitude) {
  const weight = Math.max(0, Math.min(1, amplitude));
  for (const control of controls) {
    control.animated.copy(control.joint.quaternion);
    control.joint.quaternion.slerpQuaternions(
      control.rest,
      control.animated,
      weight,
    );
  }
}

const FALLBACK_LIGHTING = `
uniform float uEnvironmentReady;
vec3 environmentReflection(vec3 direction,float roughness){return mix(vec3(.018,.027,.032),vec3(.16,.22,.25),max(direction.y,0.))*(1.-roughness*.35);}
vec3 environmentDiffuse(){return vec3(.11,.15,.17);}
vec3 sunlight(){return vec3(.7,.78,.8);}
vec3 sunDir(){return normalize(vec3(-.5,.65,-.4));}
float sunVisibility(){return 1.;}
vec3 horizonColor(){return vec3(.12,.18,.2);}
float storminess(){return 1.;}
vec3 tonemap(vec3 x){x=max(x,vec3(0));return pow(clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.),vec3(1./2.2));}
`;

function whaleFragment(lightingShader) {
  return `${lightingShader || FALLBACK_LIGHTING}
${lightingShader ? "" : "uniform vec4 uCreaturePose;"}
uniform sampler2D uWhaleAlbedo,uWhaleNormalMap,uWhaleOrm;
uniform float uCreatureDisplay,uHasNormalMap,uHasOrm;
varying vec2 vWhaleUv;
varying vec3 vWhaleWorld,vWhaleNormal,vWhaleTangent,vWhaleBitangent;
vec3 creatureAnalyticSky(vec3 direction){
 float h=clamp(direction.y*.5+.5,0.,1.);
 return mix(horizonColor()*.55,vec3(.035,.075,.12),h)+sunlight()*pow(max(dot(normalize(direction),sunDir()),0.),28.)*.08;
}
void main(){
 vec4 texel=texture2D(uWhaleAlbedo,vWhaleUv);
 vec3 N=normalize(vWhaleNormal);
 if(uHasNormalMap>.5){
  vec3 mapN=texture2D(uWhaleNormalMap,vWhaleUv).xyz*2.-1.;
  N=normalize(mat3(normalize(vWhaleTangent),normalize(vWhaleBitangent),N)*mapN);
 }
 vec3 orm=uHasOrm>.5?texture2D(uWhaleOrm,vWhaleUv).rgb:vec3(1.,.46,0.);
 float roughness=clamp(orm.g*.62+.14,.28,.72);
 vec3 V=normalize(cameraPosition-vWhaleWorld),R=reflect(-V,N);
 float NoV=max(dot(N,V),.001),fresnel=.022+.978*pow(1.-NoV,5.);
 vec3 environment=uEnvironmentReady>.5?environmentReflection(R,roughness):creatureAnalyticSky(R)*(1.-roughness*.3);
 vec3 diffuse=uEnvironmentReady>.5?environmentDiffuse():mix(vec3(.07,.10,.13),horizonColor(),.28);
 vec3 L=sunDir(),H=normalize(V+L);
 float luminance=dot(texel.rgb,vec3(.2126,.7152,.0722));
 vec3 albedo=mix(vec3(luminance),texel.rgb,.34)*.76;
 float NoL=max(dot(N,L),0.);
 float wet=pow(max(dot(N,H),0.),mix(82.,18.,roughness))*(.08+.12*NoV);
 vec3 direct=sunlight()*NoL*.34*sunVisibility()*(1.-storminess()*.42);
 vec3 radiance=albedo*(.3+diffuse*.9+direct)*mix(.72,1.,orm.r);
 radiance+=environment*(.045+fresnel*.42);
 vec3 coatEnvironment=uEnvironmentReady>.5?environmentReflection(R,.08):creatureAnalyticSky(R)*.97;
 float coatFresnel=.02+.98*pow(1.-NoV,5.);
 radiance+=coatEnvironment*coatFresnel*.38;
 radiance+=sunlight()*wet*sunVisibility()*(1.-storminess()*.55);
 float haze=.07*smoothstep(30.,75.,length(cameraPosition-vWhaleWorld));
 radiance=mix(radiance,horizonColor()*.2,haze);
 vec3 color=uCreatureDisplay>.5?tonemap(radiance):radiance;
 gl_FragColor=vec4(color,texel.a*uCreaturePose.w);
}`;
}

function makeWhaleMaterial(THREE, source, sharedUniforms, lightingShader, display) {
  const white = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
  white.needsUpdate = true;
  const material = new THREE.ShaderMaterial({
    uniforms: {
      ...sharedUniforms,
      uWhaleAlbedo: { value: source.map || white },
      uWhaleNormalMap: { value: source.normalMap || white },
      uWhaleOrm: { value: source.metalnessMap || source.roughnessMap || white },
      uHasNormalMap: { value: source.normalMap ? 1 : 0 },
      uHasOrm: { value: source.metalnessMap || source.roughnessMap ? 1 : 0 },
      uCreatureDisplay: { value: display ? 1 : 0 },
    },
    vertexShader: WHALE_VERTEX,
    fragmentShader: whaleFragment(lightingShader),
    defines: { USE_TANGENT: "" },
    transparent: true,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  material.userData.fallbackTexture = white;
  return material;
}

function normalizeWhale(THREE, scene, targetLength = 24) {
  scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(scene),
    size = box.getSize(new THREE.Vector3()),
    center = box.getCenter(new THREE.Vector3()),
    centered = new THREE.Group(),
    oriented = new THREE.Group();
  centered.add(scene);
  centered.position.copy(center).multiplyScalar(-1);
  oriented.add(centered);
  oriented.rotation.y = Math.PI / 2;
  oriented.scale.setScalar(targetLength / Math.max(size.x, size.y, size.z));
  return oriented;
}

function configureWhaleClone(THREE, sourceScene, sharedUniforms, lightingShader, display, resources) {
  const scene = cloneSkeleton(sourceScene);
  scene.traverse((node) => {
    if (!node.isMesh) return;
    resources.geometries.add(node.geometry);
    if (node.isSkinnedMesh && node.skeleton)
      resources.skeletons.add(node.skeleton);
    const sources = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of sources) {
      if (material) resources.sourceMaterials.add(material);
      for (const key of ["map", "normalMap", "roughnessMap", "metalnessMap", "aoMap"])
        if (material?.[key]) resources.textures.add(material[key]);
    }
    const replacements = sources.map((material) => {
      const next = makeWhaleMaterial(THREE, material, sharedUniforms, lightingShader, display);
      resources.materials.add(next);
      return next;
    });
    node.material = Array.isArray(node.material) ? replacements : replacements[0];
    node.frustumCulled = false;
  });
  return normalizeWhale(THREE, scene);
}

export function createCreatureShadow(
  THREE,
  sharedUniforms = {},
  lightingShader = "",
  options = {},
) {
  installCreatureUniforms(THREE, sharedUniforms);
  const object = new THREE.Group(),
    refractedObject = new THREE.Group(),
    resources = {
      geometries: new Set(),
      textures: new Set(),
      materials: new Set(),
      sourceMaterials: new Set(),
      skeletons: new Set(),
    },
    mixers = [],
    pose = {},
    ready = loadWhaleAsset(options),
    state = { enabled: true, loaded: false, disposed: false };
  object.name = "Offshore blue whale";
  refractedObject.name = "Refracted offshore blue whale";
  object.renderOrder = refractedObject.renderOrder = 2;
  object.visible = refractedObject.visible = false;

  function applyPose(group) {
    group.position.set(pose.x, pose.y, pose.z);
    group.rotation.set(0, pose.yaw, pose.roll);
  }
  function update(time, dt) {
    const safeTime = Number.isFinite(time) ? time : 0;
    sampleCreatureMotion(safeTime, pose);
    sharedUniforms.uTime.value = safeTime;
    applyPose(object);
    applyPose(refractedObject);
    sharedUniforms.uCreaturePose.value.set(pose.x, pose.y, pose.z, pose.opacity);
    sharedUniforms.uCreatureHeading.value.set(Math.cos(pose.yaw), -Math.sin(pose.yaw));
    sharedUniforms.uCreatureVelocity.value.set(
      pose.velocityX,
      pose.velocityY,
      pose.velocityZ,
    );
    for (const control of mixers) {
      control.mixer.setTime(pose.animationTime);
      blendTailStroke(control.tails, pose.tailAmplitude);
    }
    const visible = state.loaded && state.enabled && pose.opacity > 0.001;
    object.visible = false;
    refractedObject.visible = visible;
    sharedUniforms.uCreatureBreath.value = visible ? pose.breath : 0;
    sharedUniforms.uCreatureBreathPhase.value = visible ? pose.breathPhase : 0;
  }

  ready
    .then((gltf) => {
      if (state.disposed) {
        disposeGltfResources(gltf);
        return;
      }
      const displayWhale = configureWhaleClone(THREE, gltf.scene, sharedUniforms, lightingShader, true, resources),
        refractedWhale = configureWhaleClone(THREE, gltf.scene, sharedUniforms, lightingShader, false, resources);
      object.add(displayWhale);
      refractedObject.add(refractedWhale);
      if (gltf.animations?.length) {
        const swimming = prepareSwimmingClip(gltf.animations[0]);
        for (const root of [displayWhale, refractedWhale]) {
          const mixer = new THREE.AnimationMixer(root);
          mixer.clipAction(swimming).play();
          mixers.push({
            mixer,
            tails: createTailStrokeControls(THREE, root),
          });
        }
      }
      state.loaded = true;
      sharedUniforms.uCreatureWake.value = state.enabled ? 1 : 0;
      update(sharedUniforms.uTime.value);
      options.onReady?.();
    })
    .catch((error) => {
      sharedUniforms.uCreatureWake.value = 0;
      options.onError?.(error);
      if (typeof window !== "undefined")
        console.warn("Blue whale asset unavailable; continuing without creature", error);
    });
  update(0);

  return {
    object,
    refractedObject,
    ready,
    update,
    get loaded() {
      return state.loaded;
    },
    setEnabled(enabled) {
      state.enabled = Boolean(enabled);
      sharedUniforms.uCreatureWake.value = state.loaded && state.enabled ? 1 : 0;
      update(sharedUniforms.uTime.value);
    },
    dispose() {
      if (state.disposed) return;
      state.disposed = true;
      sharedUniforms.uCreatureWake.value = 0;
      sharedUniforms.uCreatureBreath.value = 0;
      sharedUniforms.uCreatureBreathPhase.value = 0;
      for (const control of mixers) control.mixer.stopAllAction();
      for (const skeleton of resources.skeletons) skeleton.dispose();
      for (const geometry of resources.geometries) geometry.dispose();
      for (const texture of resources.textures) texture.dispose();
      for (const material of resources.materials) {
        material.userData.fallbackTexture?.dispose();
        material.dispose();
      }
      for (const material of resources.sourceMaterials) material.dispose();
      object.clear();
      refractedObject.clear();
    },
  };
}
