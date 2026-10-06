function readStorage(storage, key) {
  try {
    return storage?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

function writeStorage(storage, key, value) {
  try {
    storage?.setItem(key, value);
  } catch {
    /* private browsing or a blocked storage API */
  }
}

function interpolate(text, vars) {
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (match, name) =>
    Object.hasOwn(vars, name) ? String(vars[name]) : match,
  );
}

export function createI18n({
  locales,
  fallback = "en",
  storageKey = "ocean-locale",
  storage = globalThis.localStorage,
} = {}) {
  if (!locales?.[fallback]) throw Error("Missing fallback locale");
  const listeners = new Set();
  const stored = readStorage(storage, storageKey);
  let locale = stored && Object.hasOwn(locales, stored) ? stored : fallback;

  function t(key, vars) {
    const table = locales[locale] || {};
    const text = Object.hasOwn(table, key)
      ? table[key]
      : Object.hasOwn(locales[fallback], key)
        ? locales[fallback][key]
        : key;
    return interpolate(text, vars);
  }

  function getLocale() {
    return locale;
  }

  function setLocale(next) {
    if (!Object.hasOwn(locales, next) || next === locale) return false;
    locale = next;
    writeStorage(storage, storageKey, locale);
    for (const listener of listeners) listener(locale);
    return true;
  }

  function onLocaleChange(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function applyTranslations(root = document) {
    const doc = root.nodeType === 9 ? root : root.ownerDocument;
    if (doc?.documentElement) doc.documentElement.lang = locale;
    for (const el of root.querySelectorAll("[data-i18n]"))
      el.textContent = t(el.getAttribute("data-i18n"));
    for (const el of root.querySelectorAll("[data-i18n-title]"))
      el.title = t(el.getAttribute("data-i18n-title"));
    for (const el of root.querySelectorAll("[data-i18n-aria-label]"))
      el.setAttribute(
        "aria-label",
        t(el.getAttribute("data-i18n-aria-label")),
      );
    for (const el of root.querySelectorAll("[data-i18n-content]"))
      el.setAttribute("content", t(el.getAttribute("data-i18n-content")));
  }

  return { t, getLocale, setLocale, onLocaleChange, applyTranslations };
}
