// Bal-Setu Service Worker: Cache First for Zero-Cost Offline Execution
const CACHE_NAME = "balsetu-pwa-v2";
const ASSETS_TO_CACHE = [
  "/pwa/",
  "/pwa/index.html",
  "/pwa/styles.css",
  "/pwa/app.js",
  "/pwa/manifest.json"
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[Bal-Setu SW] Pre-caching offline parent loop shell...");
      return cache.addAll(ASSETS_TO_CACHE).catch(err => {
        console.warn("[Bal-Setu SW] Non-fatal caching warning:", err);
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((k) => {
          if (k !== CACHE_NAME) return caches.delete(k);
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  e.respondWith(
    caches.match(e.request).then((cached) => {
      return cached || fetch(e.request).catch(() => {
        return caches.match("/pwa/index.html");
      });
    })
  );
});
