const MOVEMENT = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
  "KeyQ",
  "KeyE",
  "ShiftLeft",
  "ShiftRight",
]);

function isTypingTarget(target) {
  return Boolean(
    target?.matches?.("input,textarea,select,[contenteditable='true']"),
  );
}

// `code` is the physical key. AZERTY Z/Q/S/D report KeyW/KeyA/KeyS/KeyD.
// KeyQ is Q on QWERTY and A on AZERTY. KeyE is E on both.
export function createKeyboard({ target = globalThis, onShortcut } = {}) {
  const held = new Set();
  const doc = target.document ?? globalThis.document;

  function clear() {
    held.clear();
  }

  function onKeyDown(event) {
    if (isTypingTarget(event.target)) return;
    if (MOVEMENT.has(event.code)) {
      held.add(event.code);
      event.preventDefault();
    }
    if (event.repeat) return;
    if (event.code === "Space") {
      if (event.target?.matches?.("button,a")) return;
      event.preventDefault();
      onShortcut?.("pause");
    } else if (event.code === "KeyH") onShortcut?.("hud");
    else if (event.code === "Escape") onShortcut?.("escape");
    else if (event.code === "KeyP") onShortcut?.("fps");
  }

  function onKeyUp(event) {
    held.delete(event.code);
  }

  function onVisibility() {
    if (doc?.hidden) clear();
  }

  target.addEventListener("keydown", onKeyDown);
  target.addEventListener("keyup", onKeyUp);
  target.addEventListener("blur", clear);
  doc?.addEventListener("visibilitychange", onVisibility);

  return {
    isDown(code) {
      return held.has(code);
    },
    dispose() {
      target.removeEventListener("keydown", onKeyDown);
      target.removeEventListener("keyup", onKeyUp);
      target.removeEventListener("blur", clear);
      doc?.removeEventListener("visibilitychange", onVisibility);
      clear();
    },
  };
}
