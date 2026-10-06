# Ocean

A Three.js and WebGL 2 ocean simulation with spectral waves, foam, refraction, environment lighting, and a swimming whale.

## Screenshots

![Sunset scene with the settings panel open](./docs/screenshots/desktop-sunset.jpg)

![Paused blue-hour scene with visible whitecaps](./docs/screenshots/desktop-paused.jpg)

![Mobile layout with the settings panel open](./docs/screenshots/mobile-settings.jpg)

The mobile screenshot uses a simulated 390 × 844 browser viewport, not a physical device.

## Setup

Requires Node.js `^24.15.0` or `>=26.0.0`.

```bash
npm ci
npm run dev
```

Open the displayed local URL in a WebGL 2 compatible browser.

## Controls

- Drag: look around. Moving right or up turns the view right or up
- Scroll: camera speed only, from 1 to 5. The default is 2. The current setting is shown in the header and in the settings panel. Movement stays on the keys
- `W` `A` `S` `D` (or `Z` `Q` `S` `D` on an AZERTY keyboard): fly along the look direction, including above and below the surface. Hold `Shift` to move faster. Horizontal movement stays within 150 m of the start
- `Q` and `E` move straight down and up. On an AZERTY keyboard those keys are labeled `A` and `E`
- `Space`: play or pause
- `H`: show or hide the settings panel
- `P`: show or hide the frame rate counter
- **Settings**: choose the scene and adjust waves, wind, sun, and rendering quality
- **Shadow between the waves**: show or hide the whale
- **EN / JA**: switch between English and Japanese. The choice is remembered in this browser

## Verification

```bash
npm test
npm run build
```

## Credits and license

Based on [`noxellab/nagi-ocean-sim` at `16c552b`](https://github.com/noxellab/nagi-ocean-sim/commit/16c552b1eda48249b23d633617d67127fe7755c1).

- Code and foam image: [MIT License](./LICENSE)
- Sky images and radiance data: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- Whale model: [“Blue Whale - Textured” by Bohdan Lvov, CC BY 4.0](./public/BLUE_WHALE_ATTRIBUTION.txt)
- Third-party notices: [THIRD_PARTY_NOTICES.txt](./THIRD_PARTY_NOTICES.txt)
