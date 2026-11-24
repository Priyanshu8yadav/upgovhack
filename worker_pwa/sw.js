// Bal-Setu Worker PWA Service Worker (sw.js)
// Enables 100% Offline Screening in Rural Connectivity Dead Zones

const CACHE_NAME = 'balsetu-worker-v1';
const ASSETS_TO_CACHE = [
  '/worker/',
  '/worker/index.html',
  '/worker/styles.css',
  '/worker/app.js',
  '/worker/manifest.json',
  '/calibcard-assets/calibcard_front.png',
  '/calibcard-assets/calibcard_specs.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Bal-Setu SW] Pre-caching offline worker shell');
      return cache.addAll(ASSETS_TO_CACHE).catch((err) => {
        console.warn('[Bal-Setu SW] Asset caching note:', err);
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[Bal-Setu SW] Removing old cache:', key);
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Offline-first strategy for static assets, network-first with offline fallback for APIs
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request).catch(() => {
        return new Response(
          JSON.stringify({
            offline: true,
            status: "cached_offline_queue",
            message: "कार्यवाही ऑफ़लाइन सहेजी गई। नेटवर्क मिलने पर e-Kavach में सिंक होगी।"
          }),
          { headers: { 'Content-Type': 'application/json' } }
        );
      })
    );
  } else {
    event.respondWith(
      caches.match(event.request).then((response) => {
        return response || fetch(event.request);
      })
    );
  }
});
