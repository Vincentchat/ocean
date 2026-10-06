import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { createFpsCounter } from "../src/ui/fps-counter.js";

function parent() {
  return new JSDOM("<!doctype html><body></body>", {
    url: "http://example.test/",
  }).window.document.body;
}

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => (values.has(key) ? values.get(key) : null),
    setItem: (key, value) => values.set(key, String(value)),
  };
}

test("averages frame time across a 500 ms window", () => {
  const previous = globalThis.localStorage;
  globalThis.localStorage = memoryStorage();
  try {
    const counter = createFpsCounter({ parent: parent(), label: "FPS" });
    counter.setVisible(true);
    const start = 1000;
    counter.tick(start);
    for (let i = 1; i <= 30; i++) counter.tick(start + (500 / 30) * i);
    assert.equal(counter.element.textContent, "60 FPS · 16.7 ms");
    assert.equal(counter.element.hidden, false);
  } finally {
    globalThis.localStorage = previous;
  }
});

test("toggles visibility and remembers it", () => {
  const previous = globalThis.localStorage;
  globalThis.localStorage = memoryStorage();
  try {
    const first = createFpsCounter({ parent: parent(), label: "FPS" });
    assert.equal(first.element.hidden, true);
    assert.equal(first.toggle(), true);
    assert.equal(first.element.hidden, false);
    const second = createFpsCounter({ parent: parent(), label: "FPS" });
    assert.equal(second.element.hidden, false);
    second.setLabel("コマ");
    assert.equal(second.element.textContent, "コマ");
    assert.equal(second.toggle(), false);
    assert.equal(second.element.hidden, true);
    const third = createFpsCounter({ parent: parent(), label: "FPS" });
    assert.equal(third.element.hidden, true);
  } finally {
    globalThis.localStorage = previous;
  }
});
