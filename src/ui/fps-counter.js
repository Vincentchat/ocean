const STORAGE_KEY = "ocean-fps-visible";
const SAMPLE_MS = 500;

function readVisible() {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function writeVisible(visible) {
  try {
    localStorage.setItem(STORAGE_KEY, visible ? "1" : "0");
  } catch {
    /* private browsing or a blocked storage API */
  }
}

export function createFpsCounter({ parent, label = "FPS" } = {}) {
  const el = parent.ownerDocument.createElement("div");
  el.className = "fps-counter";
  el.setAttribute("role", "status");
  el.setAttribute("aria-live", "off");
  let visible = readVisible(),
    frames = 0,
    windowStart = 0,
    lastFps = null,
    lastMs = null;
  el.hidden = !visible;
  parent.appendChild(el);

  function paint() {
    el.textContent =
      lastFps == null
        ? label
        : `${Math.round(lastFps)} ${label} · ${lastMs.toFixed(1)} ms`;
  }

  function tick(now) {
    if (!visible || !Number.isFinite(now)) return;
    if (!windowStart) {
      windowStart = now;
      return;
    }
    frames++;
    const elapsed = now - windowStart;
    if (elapsed < SAMPLE_MS) return;
    lastFps = (frames * 1000) / elapsed;
    lastMs = elapsed / frames;
    frames = 0;
    windowStart = now;
    paint();
  }

  function setVisible(value) {
    visible = Boolean(value);
    el.hidden = !visible;
    writeVisible(visible);
    if (!visible) {
      frames = 0;
      windowStart = 0;
    }
  }

  function toggle() {
    setVisible(!visible);
    return visible;
  }

  function setLabel(next) {
    label = next;
    paint();
  }

  paint();
  return { tick, toggle, setVisible, setLabel, element: el };
}
