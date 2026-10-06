import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { createI18n } from "../src/i18n/i18n.js";
import { englishLocale } from "../src/i18n/locales/en.js";
import { japaneseLocale } from "../src/i18n/locales/ja.js";

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => (values.has(key) ? values.get(key) : null),
    setItem: (key, value) => values.set(key, String(value)),
  };
}

function page() {
  return new JSDOM(
    `<!doctype html><html lang="ja"><head>
      <meta name="description" data-i18n-content="metaDescription" content="old">
    </head><body>
      <h1 data-i18n="headline">old</h1>
      <button id="hide" data-i18n-title="settingsShowTitle" data-i18n-aria-label="settingsShow"></button>
    </body></html>`,
    { url: "http://example.test/" },
  ).window.document;
}

test("catalogs cover the same keys and missing keys fall back to English", () => {
  assert.deepEqual(
    Object.keys(englishLocale).sort(),
    Object.keys(japaneseLocale).sort(),
  );
  const i18n = createI18n({
    locales: {
      en: { hello: "Hello", onlyEn: "English only" },
      ja: { hello: "こんにちは" },
    },
    fallback: "en",
    storageKey: "ocean-locale-fallback",
    storage: memoryStorage(),
  });
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("hello"), "Hello");
  assert.equal(i18n.t("missing"), "missing");
  assert.equal(i18n.setLocale("ja"), true);
  assert.equal(i18n.t("hello"), "こんにちは");
  assert.equal(i18n.t("onlyEn"), "English only");
  assert.equal(i18n.setLocale("fr"), false);
  assert.equal(i18n.getLocale(), "ja");
  assert.equal(i18n.t("qualityStatus", { label: "High" }), "qualityStatus");
});

test("translations fill text, titles, labels and the document language", () => {
  const storage = memoryStorage();
  const i18n = createI18n({
    locales: { en: englishLocale, ja: japaneseLocale },
    fallback: "en",
    storageKey: "ocean-locale",
    storage,
  });
  const document = page();
  const seen = [];
  i18n.onLocaleChange((locale) => seen.push(locale));
  i18n.applyTranslations(document);
  assert.equal(document.documentElement.lang, "en");
  assert.equal(document.querySelector("h1").textContent, englishLocale.headline);
  assert.equal(
    document.querySelector("meta").getAttribute("content"),
    englishLocale.metaDescription,
  );
  const hide = document.getElementById("hide");
  assert.equal(hide.title, englishLocale.settingsShowTitle);
  assert.equal(hide.getAttribute("aria-label"), englishLocale.settingsShow);
  assert.equal(i18n.setLocale("ja"), true);
  i18n.applyTranslations(document);
  assert.deepEqual(seen, ["ja"]);
  assert.equal(document.documentElement.lang, "ja");
  assert.equal(document.querySelector("h1").textContent, japaneseLocale.headline);
  assert.equal(hide.title, japaneseLocale.settingsShowTitle);
  assert.equal(storage.getItem("ocean-locale"), "ja");
});

test("a stored locale is restored and an unknown value falls back", () => {
  const restored = createI18n({
    locales: { en: englishLocale, ja: japaneseLocale },
    fallback: "en",
    storageKey: "ocean-locale",
    storage: memoryStorage({ "ocean-locale": "ja" }),
  });
  assert.equal(restored.getLocale(), "ja");
  assert.equal(restored.t("settings"), "設定");
  const fallback = createI18n({
    locales: { en: englishLocale, ja: japaneseLocale },
    fallback: "en",
    storageKey: "ocean-locale",
    storage: memoryStorage({ "ocean-locale": "fr" }),
  });
  assert.equal(fallback.getLocale(), "en");
});

test("interpolation substitutes named placeholders", () => {
  const i18n = createI18n({
    locales: { en: englishLocale, ja: japaneseLocale },
    fallback: "en",
    storage: memoryStorage(),
  });
  assert.equal(
    i18n.t("qualityStatus", { label: "High" }),
    "Quality: High (with time interpolation)",
  );
});
