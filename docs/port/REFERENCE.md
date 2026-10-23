# REFERENCE — what the WebGPU port must reproduce

Analysis of the WebGL2 ocean (`src/`, three 0.180, fork of noxellab/nagi-ocean-sim) for the `@dreamp/world-ocean`
port. Line refs are `file:line` in the original repo. Numbers marked **(computed)** were computed by running
`generateSpectra()` from `src/spectral-ocean.js` in Node; everything else is read straight from the code.

> Reading guide: §0 is the summary, §1 to §3 are the spec, §5 is the reviewer checklist.

---

## 0. Ten facts that matter most

| # | Fact | Where |
|---|---|---|
| 1 | The real ocean is **3 FFT cascades, 256², tiles 1024 m / 96 m / 9 m**, JONSWAP spectrum with a *fixed* wind of 15 m/s, peak wavelength 47 m, direction 1 rad. The UI "wind" slider does **not** change the spectrum. It only scales the amplitude by `((5+12.5·wind)/15)^0.33`. | spectral-ocean.js:4-14, ocean.js:1120-1125 |
| 2 | **Height is negated** (`-f.x`) relative to the horizontal Tessendorf displacement. That is what makes the crests sharp and the troughs round. Choppiness is 1.5 and is multiplied by the same amplitude, so the waves get steeper when the Wave slider goes up. | spectral-ocean.js:173 |
| 3 | The FFT is not run every frame. **Two snapshots are computed at `rate` Hz (60/30/20 by quality) and lerped** by a per-frame "blend" pass. That pass also computes the normals, the Jacobian and the foam mask, and writes 2 MRT RGBA16F targets **with mipmaps**. | spectral-ocean.js:170-179, 362-392 |
| 4 | Cascades are **chained**. The smaller tile is sampled at a position offset by the larger tile's horizontal displacement: `q = rest + D_large.xz`, `fine at q + D_small.xz`. | ocean.js:170-174, 176-182 |
| 5 | Every texture fetch uses an **explicit LOD from the footprint**: `lod = log2(footprint / (domain/256))`. The vertex shader uses the grid spacing as the footprint; the fragment shader uses `dFdx/dFdy` of the rest position. When LOD < 0.5 on the High quality the sample is **bicubic B-spline**. The shortening of mip-filtered normals is fed back as **roughness (Toksvig-like)**. | ocean.js:159-169, 175-182, 277 |
| 6 | **Whitecaps** come from a 400 m × 400 m camera-anchored **foam history texture** (1024²/512²/256²). Each frame: `history = mix(source, old, exp(-dt/1.48 s))`. The source is the Jacobian compression plus a downwind-slope term. The foam is then thresholded against a **1024² foam pattern PNG** (R = breaking lace stretched 1.9× along the wind, G = surface lace). | ocean.js:355-367, 299-317 |
| 7 | Shading terms: GGX sun specular (clamped to 22, ×4.5), split-sum env-BRDF Fresnel (F0 = 0.018, IOR 1.31), PMREM reflection at **roughness ≥ 0.24** (+0.22 constant blur), screen-space **refraction of a seabed pre-pass** with Beer-Lambert absorption, crest translucency, foam that is **grey and unlit by sun colour**, shoreline/contact foam, and a **very strong Gaussian distance haze** (63 % at 280 m) to the env colour at the horizon. | ocean.js:271-353 |
| 8 | The output is a **hand-written ACES-fit (Narkowicz) + `pow(1/2.2)`** applied inside each material, with `outputColorSpace = Linear`. Radiance units are display-referred (sky ≈ 0..1, sun ≈ 10). There is no exposure. | ocean.js:187, 353, 614 |
| 9 | The mesh is a **static polar grid** with exponential rings out to 2400 m, centred at (0, 0, 8), and is **not camera-following**. The camera is clamped to within 150 m. Displacement fades out between 200 m and 1200 m from the camera. | ocean.js:642-673, 218 |
| 10 | Surprises: (a) the **capillary layers and Gerstner waves are fallbacks only**; in the normal spectral path the ripple branch is empty (ocean.js:231). (b) There is **no CPU height query** that matches the FFT: the "64² CPU FFT" is a capillary *slope* fallback, and buoyancy samples the GPU FFT in a shader. (c) There is **no underwater shading**: the sea is drawn `DoubleSide` with the same shader. (d) The foam pattern does not advect (`uFoamOffset` is computed but never used). | — |

Computed sea state (amp = Wave · ((5+12.5·Wind)/15)^0.33):

| Preset | wave | wind | amp | **Hs (computed)** |
|---|---|---|---|---|
| golden (Evening calm) | 0.9 | 0.6 | 0.847 | **2.18 m** |
| azure (Blue hour) | 1.0 | 0.8 | 1.000 | **2.58 m** |
| storm (After the rain, **default at load**) | 1.5 | 1.0 | 1.578 | **4.07 m** |

---

## 1. Simulation pipeline

### 1.1 Spectrum (CPU, once at start) — `spectral-ocean.js:4-137`

Constants (`OCEAN_PARAMETERS`, :4-14):

| name | value | meaning |
|---|---|---|
| gravity | 9.81 | m/s² (the port's spec says to use `shape.surfaceGravity` = 9.80665; the difference is negligible but changes the phases, so state it) |
| windSpeed U | 15 m/s | only used in α |
| windDirection | 1 rad | wind vector = (cos 1, sin 1) = **(0.540302306, 0.841470985)** in (x, z) |
| peakWavelength | 47 m | ω_p = √(2πg/47) = 1.1452 rad/s, **T_p = 5.49 s**, c_p = 8.57 m/s |
| gamma | 3.3 | JONSWAP peak enhancement |
| spread | 1.57 | directional spreading scale |
| standing | 0.13 | opposing-wave fraction |
| choppiness | 1.5 | (hard-coded again as `chop = 1.5`, :185) |
| period | 8192 s | dispersion quantised so the whole ocean loops every 8192 s |
| FFT_SIZE | 256 | |
| DOMAINS | [1024, 96, 9] m | |

**Dispersion** (:29-34): `ω(k) = round(√(g·(k+1e-4)) / q) · q`, where `q = 2π/8192`.

**Spectral density** `P(kx,kz)` (:35-81). With k = max(|k|, 1e-4), ω = √(gk), ω_p as above, r = ω/ω_p:
```
α      = 0.076 · (max(U,0.1)·ω_p / (22 g))^0.66
σ      = ω<ω_p ? 0.07 : 0.09
peak   = exp(-(ω-ω_p)² / (2σ²ω_p² + 1e-4))
E(ω)   = α g²/ω⁵ · exp(-1.25/r⁴) · γ^peak · (0.5·√(g/k)/k)        // JONSWAP · dω/dk / k
q      = clamp((kx cos1 + kz sin1)/k, -1, 1)
s      = max(0.5, spread·9.77·r^(ω<ω_p ? 5 : -2.5))
D      = (√((q+1)/2))^(2s) · √(s+0.25)/(2√π)                         // cos^2s(θ/2)
         · (0.07+0.93·standing + (1-(0.07+0.93·standing))·(q+1)/2)    // = 0.1909 + 0.8091·(q+1)/2
band   = smooth(lower/1.5, lower·1.5, k) · (1 - smooth(upper/1.5, upper·1.5, k))
P      = E · D · (2π/domain)² · band
```

**Cascade bands** (:87-135): `upper = π·256/domain/1.5` (Nyquist/1.5); `lower` of a cascade = `upper` of the previous one (the first is 1e-9). The edges are smoothstepped by ×/÷1.5, so neighbouring bands overlap.

| cascade | domain | dk (rad/m) | k band | λ band | cell | height σ **(computed)** | slope var **(computed)** | strongest bin |
|---|---|---|---|---|---|---|---|---|
| Large | 1024 m | 0.00614 | 0 → 0.524 | ∞ → 12 m | 4 m | 0.630 m | 0.0154 | λ ≈ 45 m, T 5.4 s |
| Small | 96 m | 0.0654 | 0.524 → 5.585 | 12 → 1.125 m | 0.375 m | 0.134 m | 0.0226 | λ ≈ 9.7 m |
| Fine | 9 m | 0.698 | 5.585 → 59.6 | 1.125 → 0.105 m | 0.035 m | 0.0115 m | 0.0208 | λ ≈ 0.95 m |
| total | | | | | | **0.644 m → Hs 2.575 m @amp 1** | 0.0589 (rms slope 0.24) | |

**Random amplitudes** (:96-111): PRNG = mulberry32-style `randomGenerator(71831 + band·13007)`, consumed in row-major (y outer, x inner) order, two draws per bin: `radius = √(-2 ln max(u1,1e-4))`, `phase = 2π u2`, `h = radius·e^{i·phase}·√(P/2)`. `kx = (x - 128)·dk`, `kz = (y - 128)·dk` (centred spectrum).
**Standing mix** (:112-120): `h0(k) = 0.935·raw(k) + 0.065·conj(raw(-k))` (a = 1 - standing/2, b = standing/2; index -k = ((N-y)%N, (N-x)%N)).
Texture `initial` RGBA32F 256²: (Re h0, Im h0, kx, kz). Texture `frequency` R32F: ω(k).
To be **bit-exact with the reference, port this PRNG and the loop order exactly** (the port is free to re-seed, but then the shots will not match the reference shots wave-for-wave).

### 1.2 Per-frame GPU work for the spectral ocean — `spectral-ocean.js:313-392`

**Time base**: the CPU rotates `h0` by `e^{-iω·base}` with `base = floor(t/120)·120` whenever the base changes (float precision). The GPU evolves with `t - base` (:313-329). The WebGPU port can do the same, or keep the phase in f32 with `ω·t mod 2π` computed in f64 on the CPU. Never send a raw large `t` to the GPU.

**Snapshot scheme** (:362-392): `step = 1/clamp(rate,10,60)`. When `t > nextTime`, compute one new FFT snapshot at `nextTime = max(first+step, t+step/2)`. If time jumps (>0.25 s or backwards), recompute both. Every frame, the blend pass lerps the two snapshots with `α = (t-first)/(next-first)`. Rate is 60 (High), 30 (Balanced), 20 (Light). **In the port, with compute being cheap, evaluate directly at `interpolatedTime` every frame** (better quality, and it gives exact CPU/GPU agreement). Keep the lerp scheme only if it is needed for the tier budget.

Passes per cascade per snapshot (each a full-screen 256² draw, RGBA32F, nearest):

| pass | shader | math |
|---|---|---|
| evolve (1) | EVOLVE_FRAGMENT :139-147 | output column x stores the input at bit-reversed x (reverses the rows for the horizontal FFT). `e = e^{-iωt'}`, `h = h0(k)·e + conj(h0(-k))·conj(e)`. Output **packs two complex IFFTs into one RGBA**: `xy = h·(1 + kx/k)`, `zw = (-i·h)·kz/k` (written as `vec2(h.y,-h.x)*kz/k`), k = \|k\|+1e-4. |
| FFT horizontal (4) | FFT_FRAGMENT :148-169 | Stockham-like radix-2 butterflies, **two stages per pass** (`uPair`, stage = 1,3,5,7); twiddle `e^{+2πij/span}` (inverse DFT); acts on both complex pairs. |
| FFT vertical (4) | same | first pass reads its input with bit-reversed y (`uReorderY`); the last pass multiplies by `(-1)^(x+y)` (`uFinalize`) to undo the centred spectrum, and writes to `snapshots[i]`. |

After the IFFT, per texel: `.x = h(x)` (real), `.y = Dx = IFFT(-i kx/k h)`, `.z = Dz = IFFT(-i kz/k h)`, `.w ≈ 0`. With no 1/N normalisation, amplitudes are as generated.

**Blend pass** (BLEND_FRAGMENT :170-179), 1 per cascade per frame, MRT → `output` (2 × RGBA16F 256², **LinearMipmapLinear, Repeat, generateMipmaps = true**):
```glsl
// spectral-ocean.js:173-178
vec3 displacement(ivec2 p){p=(p+ivec2(uSize))%uSize;vec4 f=mix(texelFetch(uField0,p,0),texelFetch(uField1,p,0),uAlpha);
  return vec3(f.y*uChop,-f.x,f.z*uChop)*uAmplitude;}                       // (Dx·1.5, -h, Dz·1.5)·amp
 cell=uDomain/256; dx=(d(p+x)-d(p-x))/(2cell); dz=(d(p+z)-d(p-z))/(2cell);
 tx=vec3(1,0,0)+dx, tz=vec3(0,0,1)+dz, n=normalize(cross(tz,tx));
 float eigen=.5*(tx.x+tz.z-sqrt(max((tx.x-tz.z)*(tx.x-tz.z)+4.*tx.z*tz.x,0.)));   // min eigenvalue of the 2×2 Jacobian
 float compression=clamp(1.-eigen,0.,1.)*smoothstep(0.,.2,dot(n.xz,vec2(.540302306,.841470985)));
 field=vec4(d.y,d.x,d.z,0.);              // tex0: (height, dispX, dispZ, 0)
 normalFoam=vec4(n*.5+.5,1.-compression); // tex1: (packed normal, 1-compression)
```
* `uAmplitude = wave·((5+12.5·wind)/15)^0.33` (ocean.js:1123-1124). It scales the height **and** the horizontal displacement, so the steepness and the Jacobian scale with it.
* The compression is **masked to slopes whose normal tilts downwind** (the front faces of waves travelling with the wind).
* The normal is per cascade (finite differences at that cascade's cell size, chop included).

### 1.3 How the cascades are combined

**Vertex** (ocean.js:212-220), spectral path:
```glsl
float radius=length(position.xz-vec2(0,8));
float spacing=max((radius+.25)*uGridRadialStep,radius*uGridAngularStep);   // grid-cell size = footprint
vec4 field=spectralField(position.xz,spacing);
p=position+vec3(field.y,field.x,field.z);
float farFade=1.-smoothstep(200.,1200.,length(position.xz-cameraPosition.xz));
p=mix(position,p,farFade);
```
`spectralField` (ocean.js:170-174) **chains** the cascades:
```glsl
vec4 a=smoothSpectral(uFieldLarge,rest/1024.+.5,1024.,footprint);
vec2 q=rest+a.yz; vec4 b=smoothSpectral(uFieldSmall,q/96.+.5,96.,footprint);
vec4 c=smoothSpectral(uFieldFine,(q+b.yz)/9.+.5,9.,footprint); return a+b+c;
```
`smoothSpectral` (:159-169): `lod = max(log2(footprint/(domain/256)), 0)`. If `uDetail` (not Light) and lod < 0.5, it does a **4-tap bicubic B-spline** (the GPU Gems 2 fast bicubic using bilinear taps), blended to trilinear by `smoothstep(0,.5,lod)`; otherwise it does `textureLod(source, uv, lod)`. UVs are `rest/domain + 0.5` with Repeat wrap (texture x ↔ world x, texture y ↔ world z).

**Fragment** normals (`spectralSurface`, :176-182), with footprint = `max(|dFdx(vRest)|, |dFdy(vRest)|)`:
```glsl
a=normalAt(uNormalLarge,rest,1024.,fp); da=smoothSpectral(uFieldLarge,...); q=rest+da.yz;
b=normalAt(uNormalSmall,q,96.,fp);      db=smoothSpectral(uFieldSmall,...);
c=normalAt(uNormalFine,q+db.yz,9.,fp);
normal=normalize(na+nb+nc-vec3(0,2,0));                       // sum of the three slopes
compression=max(3.-a.a-b.a-c.a,0.);
variance=clamp(Σ max(1-length(n_i),0),0,1);                   // normal shortening from mips → roughness
```
`normalAt` is trilinear with the same explicit LOD (no bicubic). **Fragment normals are not faded by farFade.** Distance attenuation comes only from the mip LOD (the footprint grows) and from the haze.

### 1.4 Mesh / LOD — ocean.js:642-673

| quality | rings | segments | vertices | triangles | radial step | angular step |
|---|---|---|---|---|---|---|
| high (default if innerWidth > 700) | 512 | 768 | 393,217 | 785,664 | ln(9601)/512 = 0.01791 | 2π/768 = 0.00818 |
| balanced | 384 | 512 | 196,609 | 392,704 | 0.02388 | 0.01227 |
| light | 256 | 384 | 98,305 | 196,224 | 0.03582 | 0.01636 |

* Polar grid: centre vertex plus rings at `r_i = 0.25·(e^{(i/rings)·ln 9601} − 1)`, so 0 → **2400 m**. Centred at **(x 0, z 8)**, **static in the world**, frustumCulled = false, `side: DoubleSide`, 1 draw call.
* Cell size at radius r ≈ (r+0.25)·radialStep (High: 0.18 m at 10 m, 1.8 m at 100 m, 18 m at 1 km). The vertex displacement LOD uses this, so the fine cascade is averaged out far away instead of aliasing.
* No snapping, no camera following. The free camera is limited to a radius of 150 m and y ∈ [-50, 400] (camera/free-camera.js), so the static grid always works.
* Horizon: the grid ends at 2400 m. Fog is ≈100 % past ~600 m (see §2.11). The sky dome (r = 3000, follows the camera) fills the rest. Far plane 5000.
* **Port requirement**: camera-centred, snapped LOD (spec §5). Reproduce the *effective* sampling: grid cell ≈ 1.8 % of the distance and footprint-based LOD for the displacement, with the displacement fading to flat between 200 m and 1200 m.

### 1.5 Foam history pass — ocean.js:355-367, 763-936

| item | value |
|---|---|
| target | 2 ping-pong RGBA16F (RGBA8 fallback with `uFoamScale = 2.1` + dither), Linear filter, no mips |
| size | High 1024², Balanced 512², Light 256²; world window **400 m × 400 m** → cell 0.39 / 0.78 / 1.56 m |
| anchor | the camera's look ray hits y = 0 (if dir.y < -0.01), offset clamped to ±152 m from the camera, **snapped to the cell size**; reset (cleared) if the anchor jumps > 200 m |
| scroll | the previous texture is sampled at `vUv + delta/400` (texels outside → 0) |
| source | `foamSurface()` at **LOD 0** of all 3 normal textures at the texel's world (rest) position |
| update | `src = .99·smoothstep(.15,.5,compression) + 1.10·sqrt(max(dot(n.xz, windDir),0))`; `retention = exp(-min(dt,.05)/1.48)`; `history = mix(src, old, retention)` |
| channel G | whale heave packed as `heave/8 + .5` (demo only) |
| when paused | runs only if the anchor moved (`uSimDt = 0`, so history = old) |

So foam is an **exponential moving average with τ = 1.48 s** toward the instantaneous source, not a max-accumulator. Streaks persist about 1.5 s to 3 s behind breaking crests. The `sqrt(downwind slope)` term gives a broad faint foam on the lee faces even without Jacobian folding.

### 1.6 Capillary layer (fallback only)

* `capillary-ocean.js`: 2 GPU FFT bands (256² over 17.82 m, 128² over 2.173 m), dispersion `√(9.81k + 7.4e-5·k³)` (gravity-capillary), output slopes (sx, sz, sx²+sz²), sampled 6× with rotations in the fragment shader (ocean.js:246-258).
* `cpu-capillary-ocean.js`: the same at 64² (9.113 m / 2.173 m) in JS, with a CPU mip chain uploaded as half floats.
* **Created only if the spectral ocean fails** (ocean.js:741). In the normal path `slopes = 0` and `unresolved = 0`. The fine 9 m cascade (down to λ 0.105 m) is the small-scale detail. **The port does not need capillaries for parity.** (Optional extra detail; if added, it must not change the look at the reference cameras.)

### 1.7 Gerstner fallback (not to port)

7 fixed waves (λ 31, 17.3, 9.8, 6.1, 3.7, 2.2, 1.37 m; steepness .25…, ocean.js:198-208), phases advanced on the CPU (ocean-motion.js). This is a fallback only. The spec forbids a Gerstner-only tier.

### 1.8 Buoyancy / height query in the reference

`creatureSurfaceHeight` (creature-buoyancy.js:190-207) runs **on the GPU**: 3 fixed-point iterations `rest -= (rest + D(rest)) - world` using `spectralField(rest, footprint 4 m)`, so the result is low-passed (LOD 0 / 3.4 / 6.8 for the 3 cascades). The result is written to a 2×1 RGBA32F target, consumed in shaders, and read back (debug only) with `readRenderTargetPixelsAsync`. The CPU path uses Gerstner only. **The port's CPU FFT `heightAt` is new work.** For parity use the same inversion (3 iterations) but at full resolution.

---

## 2. Shading — water fragment shader (ocean.js:228-354)

Notation: `L = sunDir()`, `V = normalize(cam - P)`, `storm = max(mood-1,0)`.

### 2.1 Sun and light helpers (shaderCommon)
| helper | code | line |
|---|---|---|
| sun direction | `vec3(-.79863551*cos(e), sin(e), -.60181502*cos(e))`, e = elevation (rad). Fixed azimuth: horizontal dir (-0.7986, -0.6018), **53° left of the default view direction (-Z)** | :104 |
| sun colour (env ready) | `uSolarColor = (1, .58+.39w, .3+.62w)`, w = clamp((e-.08)/.7, 0, 1) → 10°: (1, .633, .384); 38°: (1, .905, .816); 50°: (1, .97, .92) | atmosphere.js:26-29 |
| sun visibility | env alpha at sunDir = `1 - storm·.97` | :113, atmosphere.js:21 |
| env diffuse | `prefilteredEnvironment(up, 1.0)` (PMREM, roughness 1) | :135-138 |

### 2.2 Normal
```glsl
vec3 N=normalize(surfaceNormal+vec3(-slopes.x,0.,-slopes.y)*surfaceNormal.y);   // slopes=0 in spectral path
N=normalize(N+V*max(.025-dot(N,V),0.));                                           // :273 keep N facing the viewer
vec3 R=normalize(reflect(-V,N)); float NoV=max(dot(N,V),.015);                    // :274
```

### 2.3 Roughness (from slope variance)
```glsl
float variance=max(dot(dFdx(N),dFdx(N)),dot(dFdy(N),dFdy(N)));                   // screen-space normal variation
float rough=clamp(.018+.8*spectralVariance+.8*(sqrt(1.+slopeVariance)-1.)+.8*sqrt(variance),.018,.8);  // :277
```
(`slopeVariance = 0` in the spectral path; `spectralVariance` = mip shortening from §1.3.) In WGSL `dpdx` must be in uniform control flow: compute N before any non-uniform branch.

### 2.4 Sun specular (GGX)
```glsl
float a2=pow(max(rough,.12),4.); float D=a2/(PI*pow(NoH*NoH*(a2-1.)+1.,2.));
float k=pow(max(rough,.12)+1.,2.)*.125; float G=NoV/(NoV*(1.-k)+k)*NoL/(NoL*(1.-k)+k);
float sunF=.0180094+.9819906*pow(1.-max(dot(V,H),0.),5.);
float spec=D*G*sunF/(4.*NoV+.04);                                                 // :278-281
color+=sunlight()*min(spec,22.)*(4.5-storminess()*3.5)*solarVisibility;           // :298
```
The roughness floor of 0.12 gives the **sun glitter path** (broad and sparkly). The cap is 22 × 4.5 ≈ 99 (pre-tonemap). Under foam the specular is removed again: `color -= …*(1-clamp(1-2·foam,0,1))` (:334).

### 2.5 Reflection
```glsl
reflected=environmentReflection(R,min(uSkyReflectionBlur+rough,1.));   // uSkyReflectionBlur = 0.22   :295
```
Roughness passed is ≥ 0.238 > 0.10, so it is always the **PMREM** branch (:132): `prefiltered(R, ρ) · uEnvironmentIntensity(1)`. The reflections are therefore always noticeably blurred, and the sharp sparkle comes only from the GGX sun. Rays below the horizon hit the env cube's lower hemisphere, which the atmosphere bakes as `sky·F + deep·(1-F)` (atmosphere.js:20).

### 2.6 Fresnel (split-sum env BRDF, Karis mobile approximation)
```glsl
float environmentFresnel(float NoV,float roughness){                      // :114-118
 vec4 r=roughness*vec4(-1.,-.0275,-.572,.022)+vec4(1.,.0425,1.04,-.04);
 float a=min(r.x*r.x,exp2(-9.28*NoV))*r.x+r.y;vec2 brdf=vec2(-1.04,1.04)*a+r.zw;
 return clamp(.0180094*brdf.x+brdf.y,0.,1.);}
vec3 color=water*(1.-integratedF)+reflected*integratedF;                    // :297  (F0 = 0.018 ⇔ IOR 1.31)
```

### 2.7 Water body: refraction, absorption, scatter
```glsl
vec3 deep=vec3(.0159963,.0612461,.0998987);
vec3 refractionNormal=normalize(mix(surfaceNormal,N,.22)); (+ same V-facing fix)        // :286
vec3 bottom=transmittedScene(vWorld,refractionNormal,V,opticalPath,refractionValid);    // :287
vec3 sigma=vec3(.29613827,.10461648,.09530747); vec3 extinction=exp(-sigma*opticalPath);// per metre
vec3 mediumLight=diffuseSky*.8+sunlight()*solarVisibility*(.12+.18*NoL);
vec3 water=bottom*extinction*refractionValid+deep*mediumLight*(1.-extinction*refractionValid);
```
`transmittedScene` (refraction.js:7-31): reads the **seabed pre-pass** (HalfFloat colour + depth texture, full drawing-buffer size).
* `opticalPath` = **view-depth difference** (background eye-Z − water eye-Z), clamped to 0..120; it is 32 when the pre-pass is missing or the pixel is sky.
* UV offset = view-space xy of `refract(-V,N,1/1.31) - refract(-V,up,1/1.31)` × `uRefractionStrength 0.25` × `smoothstep(0,4,thickness)` × `clamp(6/waterZ,.15,1)`, soft-limited `offset *= .035/(.035+|offset|)`, and faded within 5.5 % of the screen edge.
* 2 iterations of depth-continuity checks (tolerance .45+.12·thickness), then a fallback to the unrefracted uv if the sample is in front of the water.

### 2.8 Crest translucency (subsurface glow)
```glsl
float crest=smoothstep(-.30,.85,vFoam.z/max(uWave,.1));            // vFoam.z = displaced height (m)   :283
vec3 transmissionTint=vec3(.0159963,.1356333,.0908417);             // green-teal
float backLight=pow(max(dot(V,-normalize(L+N*.4)),0.),2.);
water+=transmissionTint*crest*(.12+.6*backLight)*(diffuseSky*.4+sunlight()*solarVisibility*2.2);  // :292-293
```
Visible as teal-green crests when looking toward the sun.

### 2.9 Foam (whitecaps)
```glsl
windDir=(.540302306,.841470985); crossWind=(-windDir.y,windDir.x);
foamWorld=vRest;                                  // REST position: pattern is fixed on the undisplaced grid
foamCoord=(foamWorld-uFoamAnchor)/400.+.5;
edge=1-smoothstep(152,200,|foamWorld-uFoamAnchor|) per axis;      // foam is 0 outside the 400 m window
history=foamHistory.r*uFoamScale*foamEdge;
breakingUV=(windDir*dot(foamWorld,windDir)*.53+crossWind*dot(foamWorld,crossWind))/9.1;   // 9.1 m × 17.2 m tile
red=texture(uFoamPattern,breakingUV).r;  surfaceRed=texture(uFoamPattern,foamWorld/30.).g;  // trilinear, repeat
float threshold=1.-1.6*max(history,0.);
float breaking=clamp(red*smoothstep(threshold,threshold+.6,red)*.6,0.,1.);
float surfaceFoam=surfaceRed*smoothstep(.79,.94,surfaceRed)*.3;    // sparse static foam everywhere
foam=clamp(breaking+surfaceFoam+wakeFoam+contactFoam,0.,1.);
vec3 foamColor=vec3(.3+.7*clamp(dot(surfaceNormal,L),0.,1.));      // GREY, not tinted by the sun or sky
color=mix(color,foamColor,foam);                                   // :299-336
```
Foam pattern (`public/foam-pattern.png`, 1024², RGB8, NoColorSpace, mipmapped, repeat, made by tools/generate-foam.py): R = "breaking" bubble lace, mean 0.657, std 0.0965; G = "surface" lace, mean 0.635, std 0.155; B = 0. Max breaking coverage is about 0.6. If the PNG fails to load there is a procedural Voronoi-lace fallback (`foamFilteredLace`, :75-100); the port may skip it.

### 2.10 Shoreline / contact (water over rocks)
(:337-348) where the background depth exists and thickness = bgZ − waterZ:
`edgeFade = 1-smoothstep(0, 1.975, thickness)` blends toward the **raw scene colour** (soft intersection). Contact foam `smoothstep(thr, thr+.15, pattern.r at p/18) * .2` uses colour **(.846873,.947307,.982251)**, with `thr = mix(1, d+.23, smoothstep(0,.3,d))` and `d = thickness/3.95`. Visible as a white rim around the rocks.

### 2.11 Distance haze
```glsl
float haze=1.-exp(-pow(dist/mix(280.,220.,storminess()),2.));                     // :350
vec3 hazeDirection=normalize(vec3(P.x-cam.x,dist*.002,P.z-cam.z));                 // just above the horizon
vec3 hazeColor=environmentSample(hazeDirection,0.).rgb;   // raw env cube LOD0 (analytic: horizonColor())
color=mix(color,hazeColor,haze);
```
Haze is 12 % at 100 m, 40 % at 200 m, 63 % at 280 m, 96 % at 500 m (storm: 63 % at 220 m). It is **Gaussian, not exponential**, and **fully hides the grid edge and far aliasing**. The port's fog from `SkyLight.fogDensity` must be able to reproduce this curve in the demo.

### 2.12 Output / colour
`gl_FragColor = vec4(tonemap(color),1.)` with
`tonemap(x) = pow(clamp(x(2.51x+.03)/(x(2.43x+.59)+.14),0,1), 1/2.2)` (:187) and `renderer.outputColorSpace = LinearSRGBColorSpace` (:614). This is a **pure 2.2 gamma, not the sRGB OETF**, and there is no exposure. The sky, seabed display, whale and breath all tonemap the same way in their own shaders. **Port**: the water writes linear `color × preExposure`. The demo's tonemap pass must apply exactly this curve (exposure 1) to compare with the reference shots.

### 2.13 Other things the reference does / does not do
* **Underwater**: no special path. The camera can go to y = -50; the sea is DoubleSide with the same shader (the "sky" seen from below is the reflection math with N flipped toward V by the 0.025 fix). The seabed display mesh is drawn tonemapped. There is no underwater fog or Snell's window. (Underwater reference videos exist in `references/underwater references (for later)/`: later work.)
* **No dithering** of the output (only foam history quantisation noise in the RGBA8 fallback).
* **MSAA**: `antialias: true` on the default framebuffer. Pixel ratio cap: High 1.75, Balanced 1.25 (1 on mobile), Light 0.8.
* Draw order: sky (renderOrder -1, depthWrite off) → seabed display (0) → sea (1, ocean.js:963) → whale (2) → breath (4). The refraction pre-pass renders `seabed.scene` (linear radiance seabed + refracted whale) before the main render.

### 2.14 Environment / sky (demo-side stand-in, not part of the ocean module)
* Photo sky: `sky-base.jpg` + `sky-gain.jpg` (4096×2048 equirect, gain-map HDR: `code=(sdr+1)·2^(15.99·gain·0.424)-1`, then an sRGB-ish decode; photo-sky.js:10-25), sun core removed. **It is warped so the photo's sun (elevation 47.9°) sits at the requested elevation**: rotate xz by 3.093747 rad, then `q.y *= tan(.835763)/max(tan(sun),.025)`, ×0.7, sunset tint `(1.28,.75,.48)` when the sun is low, storm desaturation, horizon haze band (photo-sky.js:78-89, atmosphere.js:10-22).
* The env cube is captured from that function (HalfFloat cube 1024/512/256/128 by quality and canvas size, mipmapped, one face per frame while rebuilding, cross-faded A/B over 0.3 s) → PMREM (256 when the canvas is > 1000 px wide, otherwise 128) for reflection and diffuse. `sky-radiance.bin` = 1024×512 RGBA16F fallback radiance.
* Sun disc on the sky dome: `+ sunlight·4.5·smoothstep(.999955,.999985,dot(rd,L))·(1-storm·.97)` (angular radius ≈ 0.31°).

### 2.15 Seabed (demo-side; what the refraction shows)
`seabed.js`: 200 m × 200 m grid centred at (0, 8), 512 segments (513² = 263,169 vertices, 524,288 triangles, rendered **twice**: pre-pass and display), log-spaced vertices. Sand depth `-(8.68 + .022r + 20·smoothstep(14,102,r))` + noise + 6 mounds + 26 rocks (up to ~-1.7 m deep near (-8, 1)). Lighting: sand/stone albedo, sun refracted (IOR 1.31), absorption `(.047,.016,.009)`/m, ambient `(.38,.455,.49)`, **caustics** (6 analytic lens waves, Jacobian focus, strength `uSeabedCaustics = 0.55`, fade `exp(-.032 depth)`, off in storm). This is what gives the **turquoise shallows and dark rocks** in the screenshots. The port reads `TERRAIN_QUERY`/scene.depth; the demo must recreate this seabed (or an equivalent) or the near-field colour will not match.

---

## 3. Scene presets and user parameters

### 3.1 Presets (ocean.js:368-372; UI names from en.js)
| key | UI name (EN / JA) | wave | wind | sun elevation | mood | storm = max(mood-1,0) | amp | Hs |
|---|---|---|---|---|---|---|---|---|
| golden | Evening calm / 夕凪 | 0.9 | 0.6 | 10° | 0 | 0 | 0.847 | 2.18 m |
| azure | Blue hour / 青の時間 | 1.0 | 0.8 | 50° | 1 | 0 | 1.000 | 2.58 m |
| storm | After the rain / 雨のあと | 1.5 | 1.0 | 38° | 2 | 1 | 1.578 | 4.07 m |

At load the preset is **storm** (ocean.js:377). Mood is not a slider: it follows the preset and is smoothed. Note that "Blue hour" actually has a high sun (50°) and blue sky, and "Evening calm" is the sunset look (sun at 10°, warm, low sun glitter).

What mood/storm changes: sky colour and desaturation (photo-sky.js:85), sun visibility `1-.97·storm` (so the sun specular is nearly gone in storm), spec gain `4.5 → 1.0`, haze distance `280 → 220 m`, seabed caustics off, analytic-sky cloud cover.

### 3.2 Sliders (index.html:101-126). All are smoothed per frame with `v += (target-v)·(1-e^{-3dt})` (ocean.js:1061-1093)
| id | label | min | max | step | default (storm) | effect |
|---|---|---|---|---|---|---|
| wave | Wave height | 0.25 | 1.5 | 0.05 | 1.5 | linear amplitude of all cascades, including horizontal displacement, so also steepness and foam; also normalises the crest-glow height |
| wind | Wind strength | 0 | 1 | 0.05 | 1.0 | amplitude × ((5+12.5w)/15)^0.33 (0.70× at 0 … 1.05× at 1). **No change to the spectrum shape, direction or foam threshold.** |
| light | Sun height | 2° | 55° | 1° | 38° | sun elevation: direction, colour, sky warp, glitter |

### 3.3 Other toggles
| control | values | effect |
|---|---|---|
| quality button | balanced / high / light (default high if innerWidth > 700) | mesh rings/segments, FFT snapshot rate 30/60/20 Hz, `uDetail` (bicubic) 1/1/0, foam 512/1024/256, pixel ratio cap 1.25/1.75/0.8, env cube size |
| pause (Space) | | freezes time (foam history held) |
| shadow | whale on/off | whale, wake foam, breath |
| camera | drag yaw ×0.0025/px, pitch ×0.002/px clamped ±1.52; WASD/QE fly; speed level 1-5 (6 m/s × level/2, Shift ×3) | start (0, 4.8, 8), yaw 0, pitch -0.28 rad, fov 60°, near 0.1, far 5000 |
| dev URL | `?inspectTime=<s>` sets the start time; `?inspectPaused=1` pauses (dev builds only) | **use these for reference captures** |

### 3.4 Hidden constants worth exposing as tunables in the port
`uSkyReflectionBlur 0.22`, `uEnvironmentIntensity 1`, `uWaterIOR 1.31`, `uRefractionStrength 0.25`, `sigma`, `deep`, `transmissionTint`, foam τ 1.48 s, foam source gains .99/1.10, threshold gain 1.6, breaking gain 0.6, haze distance 280/220, spec cap 22 and gain 4.5, roughness floor 0.018 (0.12 for the specular), farFade 200→1200 m, chop 1.5.

---

## 4. Performance-relevant facts (reference, High quality)

| item | size / count |
|---|---|
| FFT per cascade | initial RGBA32F 256² (1 MB) + freq R32F (256 KB) + 2 ping RGBA32F + 2 snapshots RGBA32F (4 MB) + output 2×RGBA16F + mips (~1.4 MB) ≈ 6.7 MB × 3 ≈ **20 MB** |
| FFT passes per snapshot | 3 × (1 evolve + 4 H + 4 V) = **27 draws**, once per 1/rate s (≈ every frame at 60 Hz High) |
| per-frame sim passes | 3 blend (MRT) + mip regeneration of 6 textures + 1 foam + 1 buoyancy (2×1) |
| foam | 2 × RGBA16F 1024² = 16 MB |
| refraction pre-pass | RGBA16F + Depth32 at the full drawing buffer (1080p × 1.75 DPR → 3360×1890 ≈ 76 MB); renders the seabed (524 k tris) + whale |
| water mesh | 393 k vertices / 786 k triangles, 1 draw; vertex shader = 3 cascades × (bicubic 4 taps or 1 tap) chained |
| water fragment | ~3 field (bicubic up to 12 taps) + 3 normal taps, PMREM 2-4 taps, 2-3 env taps (haze, visibility, diffuse), 3 foam taps + 1 history, refraction 4-5 depth/colour taps, 10+ `dFdx` |
| main pass draws | sky sphere (32×16), seabed display, whale meshes, sea, breath (instanced 38 quads) |
| env | 2 cube RTs up to 1024² × 6 × RGBA16F + mips (≈ 2 × 67 MB) + 2 PMREM 768×1024 RGBA16F; rebuilt only on a lighting change (1 face per frame) |
| sky photo | 2 × 4096×2048 RGBA8 (64 MB) |

Port targets (spec): compute FFT, evaluate per frame, f32 storage; aim at ≤ the reference cost per tier and record GPU ms in TUNING.md.

---

## 5. VISUAL FEATURE CHECKLIST (for reviewers)

Canonical cameras (world coordinates, Y up, metres; yaw 0 looks toward -Z; forward = (sin yaw·cos pitch, sin pitch, -cos yaw·cos pitch)):

| cam | position | yaw | pitch | purpose |
|---|---|---|---|---|
| **C1 default** | (0, 4.8, 8) | 0 | -0.28 | the app's start view (rocks and shallows bottom, open sea, horizon) |
| **C2 sun** | (0, 4.8, 8) | -0.925 (toward the sun azimuth) | -0.12 | glitter path, crest glow, Fresnel |
| **C3 anti-sun** | (0, 4.8, 8) | 2.217 | -0.12 | sky reflection, foam on lit faces, haze colour |
| **C4 high** | (0, 60, 8) | 0 | -0.75 | cascade tiling, foam streak pattern, LOD |
| **C5 grazing** | (0, 1.2, 8) | -0.3 | -0.02 | crest silhouettes, horizon, haze, swell |
| **C6 close-down** | (20, 3, -20) | 0 | -1.2 | fine cascade, normal detail, bicubic smoothness, no tiling |
| **C7 underwater** | (0, -3, 8) | 0 | 0.3 | parity of the (lack of) underwater handling |

Capture each preset (golden / azure / storm) at `inspectTime = 30`, paused, plus a 10 s clip in motion for the motion items.

| ID | Feature | How to see it | Failure looks like |
|---|---|---|---|
| **F1** | Large swell, λ ≈ 45 m, Tp ≈ 5.5 s, Hs 2.2 / 2.6 / 4.1 m by preset | C5, C1, storm vs golden | flat or tiny waves; wrong scale (crests 10 m or 200 m apart); all presets look alike |
| **F2** | Sharp crests, round troughs (negated height + chop 1.5·amp) | C5 silhouettes | sinusoidal or inverted (sharp troughs) waves; self-intersecting loops beyond moderate folding |
| **F3** | Direction: waves travel toward +X/+Z (wind (0.54, 0.84)), roughly toward and to the right of the C1 view; ~19 % opposing energy (slight standing chop) | C1 or C4 in motion | waves moving away from the camera or sideways; perfectly unidirectional |
| **F4** | Three-scale detail: swell + 1-12 m chop + 0.1-1 m ripples; no visible 1024/96/9 m repetition | C4, C6 | visible tiling grid; "plastic" smooth surface (fine cascade missing); noise without swell |
| **F5** | Cascade chaining (small waves ride the large ones' horizontal motion) | C6 in motion | small ripples sliding against the swell |
| **F6** | Distance-filtered normals: the surface goes smoothly glossy with distance, without shimmering, moiré or sparkle aliasing | C1/C5 in motion at 100-400 m | crawling noise, moiré, strobing at mid distance |
| **F7** | Sun glitter path: a broad GGX streak toward the sun, sparkly near, wide far; much weaker in storm | C2 golden/azure | one tight hot spot; no path; spec visible in storm at full strength |
| **F8** | Roughness from the slope variance (mip shortening plus screen-space dN): the glitter widens with distance | C2 | glitter as sharp at 300 m as at 10 m (aliasing) |
| **F9** | Fresnel: dark transparent water looking down, bright sky reflection at grazing angles (F0 0.018, split-sum) | C1 bottom vs top of the water | reflections equally strong everywhere; mirror at nadir |
| **F10** | Blurry env reflection (PMREM ρ ≥ 0.24) with clouds and a warm horizon reflected softly | C3 | mirror-sharp clouds; or uniform grey |
| **F11** | Water body colour: deep (.016,.061,.100) × sky light; teal-blue | C3 azure | green, black or saturated cyan |
| **F12** | **Refraction of the seabed**: turquoise shallows, rocks seen through the water with wobble from the normals; absorption darkens with depth (σ = .296/.105/.095) | C1 (rocks at the bottom of the frame) | no seabed; seabed without absorption (too clear); no distortion; distortion tearing at the screen edges or haloing around the rocks |
| **F13** | Contact rim: soft fade plus a thin white foam rim where the water meets rocks | C1 rocks | hard intersection line; no rim |
| **F14** | Crest translucency: teal-green glow in the upper part of the crests, strongest against the sun | C2 golden, C5 | none; or the whole surface glowing |
| **F15** | **Whitecaps from the Jacobian** on the downwind faces of crests, more with wave height (storm ≫ golden) | C1/C4 storm | foam everywhere uniformly; foam in the troughs; no foam in storm; foam identical across presets |
| **F16** | **Persistent foam streaks**: foam decays with τ ≈ 1.5 s and is left behind breaking crests; lace texture stretched along the wind (9 × 17 m tile) | C4 storm, in motion | foam blinking on and off with the crest; no trailing; the pattern drifting or swimming with the camera; visible 400 m window edge |
| **F17** | Foam lace texture: bubbly holes, not a flat white; sparse static surface foam (G channel) | C6 storm | flat white blobs; noisy hash pattern |
| **F18** | Foam shading: grey/white `.3+.7·N·L`, no specular on foam; not tinted orange at sunset | C2 golden | foam takes on the sun colour; specular sparkle on the foam |
| **F19** | Gaussian distance haze to the horizon env colour (≈ 63 % at 280 m, storm 220 m), so the horizon blends seamlessly into the sky | C5 | hard horizon line; visible grid edge; fog too thin or too exponential |
| **F20** | Displacement fade 200-1200 m: no geometric aliasing or popping far away | C5 in motion | far vertices jittering; popping rings |
| **F21** | Mesh stability: no swimming or vertex sliding when the camera moves | any cam, strafing | the surface pattern crawls with the camera (bad snapping) |
| **F22** | Sky and sun: photo sky warped to the elevation; sun disc 0.31° radius; sunset tint at 10° | C2 golden | wrong sun position vs the glitter; the reflected sun is not where the disc is |
| **F23** | Tonemap parity: ACES-fit + gamma 2.2, exposure 1 | all | washed out (sRGB OETF instead of 2.2), clipped, or the wrong exposure |
| **F24** | Preset transitions are smooth (sliders lerp, ~0.3 s env blend) | switch presets live | pops in waves, light or foam |
| **F25** | Temporal smoothness: no stepping (the reference interpolates 20-60 Hz snapshots) | C6 in motion at 144 Hz | visible stepping of the fine ripples |
| **F26** | Underwater view (C7): same shader on the back faces; no crash or NaN | C7 | black or NaN pixels |
| **F27** | Far from the origin (spec): C1 offset by (100 km, 0, 100 km) looks identical | C1 + origin | jitter, quantised waves |

---

## 6. WebGL → WebGPU/TSL (three r186) differences to plan for

| Topic | WebGL reference | WebGPU/TSL approach |
|---|---|---|
| FFT | fragment ping-pong, RGBA32F RTs, `texelFetch`, `gl_FragCoord` | compute (`Fn().compute(n)`), `textureStore` to storage textures or storage buffers; one dispatch per stage, or a shared-memory 256-point FFT per row (`workgroupArray`, 256 threads) to cut 9 passes to 2 per cascade. Bit reversal via `reverseBits` (WGSL built-in). |
| Float filtering | output is **RGBA16F** so linear/mip filtering works | `rgba32float` is **not filterable** without the optional `float32-filterable` feature. Keep the sampled outputs `rgba16float` (storage-capable in core WebGPU), or use 32F with manual bilinear. Displacement precision of 16F at Hs 4 m: about 0.002 m, fine. |
| Mipmaps of sim outputs | `generateMipmaps` on the MRT render target each frame | storage textures do not auto-mip. Write a compute 2×2 downsample chain (8 levels for 256²) per output, or check `StorageTexture` + `generateMipmaps` support in r186 before relying on it. The **normal mips must be box-averaged unnormalised normals** (the shortening is the roughness signal). |
| MRT blend | one fragment pass, 2 targets | one compute kernel writing 2 storage textures |
| Explicit-LOD sampling in the vertex stage | `textureLod` in VS | `texture(tex, uv).level(lod)` in TSL → `textureSampleLevel`; fine in vertex. |
| Derivatives | `dFdx/dFdy` inside `if(uSpectral)`, ripple loops, foam | WGSL `dpdx`/`textureSample` require **uniform control flow**. Compute all derivatives at the top level before any data-dependent branch; use explicit LOD everywhere (the reference already does, except `texture2D` on the foam pattern and history). |
| Readback | `readPixels` (sync) for validation, `readRenderTargetPixelsAsync` for debug | `renderer.getArrayBufferAsync(storageBufferAttribute)` / `readRenderTargetPixelsAsync`. One frame late, **never awaited on the render path** (spec rule 8). Drop the sync `validate()` probes. |
| Half-float / RGBA8 fallbacks | `EXT_color_buffer_float` checks, foam RGBA8 + dither path | WebGPU core has rgba16float render and storage: drop all fallbacks (foam always 16F, `uFoamScale = 1`, no quantisation dither). |
| Screen-space refraction | `gl_FragCoord.xy/uViewport`, depth texture, `linearEyeDepth(z)=n f/(f - z(f-n))` | use the TSL `screenUV` (handles the WebGPU top-left Y origin) and `viewportDepthTexture` / render-graph `scene.depth`; linearise with `perspectiveDepthToViewZ` (respects reversed-depth/log-depth settings). With MSAA the depth texture must come from a resolved / non-MSAA pre-pass. |
| Sampler / texture limits | ~16 textures in the water FS (6 sim, 2 env cubes, 2 PMREM, foam ×2, capillary ×2, scene colour and depth, creature) | WebGPU default `maxSampledTexturesPerShaderStage` = 16, `maxSamplersPerShaderStage` = 16. Drop the env A/B cross-fade (single env + PMREM, or `pmremTexture`), drop the capillaries, share samplers. Pack height+disp and normal+foam as in the reference (2 textures per cascade) or use texture arrays (3 layers) to reduce bindings. |
| PMREM | custom cubeUV sampling copied from three (prefiltered-environment.js) | use `pmremTexture(envMap, R, roughness)` in TSL (same cubeUV layout and roughness→mip mapping as `nagiRoughnessToMip`). |
| Cube capture of the sky | `WebGLCubeRenderTarget` + CubeCamera, one face per frame | `CubeRenderTarget` + `CubeCamera` in WebGPU, or the engine's SkyLight env. The demo only. |
| uint hash | GLSL `uvec2`, `0x…u` multiply wrap | TSL `uint` ops wrap mod 2³² in WGSL: same result. Only needed if the procedural foam fallback / wake noise is ported. |
| Uniform arrays | `uWavePhase[7]`, `uRipplePhase[12]` | `uniformArray` (only for the fallbacks, so not needed). |
| Tonemap / colour space | per-material tonemap, Linear output | the water outputs linear HDR × preExposure; the demo does tonemap as a post pass with the **custom ACES-fit + pow(1/2.2)** node and `outputColorSpace = LinearSRGBColorSpace` (or equivalent) to match. |
| Precision / time | CPU rebases phase every 120 s | pass `t - base` or per-mode phases; floating-origin `wrapped(1024/96/9)` offsets added to `renderPos` before computing UVs (`(renderPos.xz + wrapped)/domain`). |
| DoubleSide | `side: DoubleSide` | same; `frontFacing` is available in TSL if an underwater path is added. |
| Big index buffers | 786 k-triangle indexed mesh | fine in WebGPU (uint32 indices). The port replaces it with a camera-centred LOD anyway. |

---

## 7. Wave behaviour to verify in motion

| Behaviour | Expected (from code) | How to check |
|---|---|---|
| Dominant wavelength | 45-47 m (peak bin λ 45.2 m) | C5: about 2 crests per 90 m; at C4 (60 m altitude) ~4-5 swell crests across the near view |
| Dominant period | T ≈ 5.4-5.5 s | watch one point bob in C5: about 11 rises per minute |
| Phase speed of the swell | c = g/ω ≈ 8.6 m/s; group speed ≈ 4.3 m/s (wave groups move half as fast as the crests) | C4 in motion: crests overtake the groups |
| Medium chop (96 m cascade) | λ 1-12 m, periods 0.9-2.8 s, c 1.3-4.3 m/s | C6 |
| Fine ripples (9 m cascade) | λ 0.1-1.1 m, T 0.25-0.85 s (pure gravity dispersion, no capillary term) | C6 close |
| Direction | crests advance along (0.54, 0.84) in xz; 19 % energy floor in opposing directions + 6.5 % conj mixing (some standing / short-crested look) | C4 |
| Crest shape | crest-to-trough asymmetry grows with the Wave slider (chop 1.5 × amp): storm crests pinch and fold (Jacobian < 0 → whitecaps); golden crests are rounded with occasional caps | C5 storm vs golden |
| Short-crestedness | the directional spread `cos^{2s}(θ/2)`, with s ≈ 15 at the peak dropping to 0.5 at high k, gives fairly long crests in the swell and isotropic small chop | C4 |
| Loop | everything repeats exactly after 8192 s (quantised ω) | not visually checkable; test |
| Foam timing | foam appears as a crest steepens on its lee face, persists ~1.5 s (63 % decay), and trails behind the crest; storm should show large white patches (screenshot docs/screenshots/desktop-paused.jpg) | C4 storm motion |
| Foam ↔ wind slider | the wind slider changes the amplitude by only ±5-30 %, so foam changes a little; the Wave slider changes it a lot | sliders |
| Time smoothing | no visible 30/60 Hz stepping | slow-motion capture |

---

## 8. Reference media notes
* `docs/screenshots/desktop-sunset.jpg` (golden, Light quality): warm overcast sky, low sun on the left with a broad glitter path, muted brown-grey sea, greenish crest glow, turquoise shallows with dark rocks in the foreground, strong horizon haze, no whitecaps.
* `docs/screenshots/desktop-paused.jpg` (azure, High): deep navy-blue sea, **extensive white lacy foam on the wind-facing crests and streaks**, soft cloud reflections, turquoise shallows with rocks and white contact rims, soft hazy horizon.
* `references/Ocean reference.mp4` (40 MB, the target look) was **not extracted** (no ffmpeg or python on this machine). A reviewer should watch it directly.
