// Bal-Setu Worker PWA Service Worker (sw.js)
// Enables 100% Offline Screening in Rural Connectivity Dead Zones
// Precaches ONNX WASM Runtime, Neural Models, and Native Audio Prompts

const CACHE_NAME = 'balsetu-worker-v6';
const ASSETS_TO_CACHE = [
  '/worker/',
  '/worker/index.html',
  '/worker/styles.css',
  '/worker/app.js',
  '/worker/manifest.json',
  '/worker/libs/ort.min.js',
  '/worker/libs/ort-wasm.wasm',
  '/worker/models/skin_pallor_mobilenetv3.onnx',
  '/worker/models/field_lcd_caries_detector.onnx',
  '/worker/models/speech_acoustic_prosody.onnx',
  '/worker/audio/step1_hi.mp3',
  '/worker/audio/step1_bho.mp3',
  '/worker/audio/step2_hi.mp3',
  '/worker/audio/step2_bho.mp3',
  '/worker/audio/step3_hi.mp3',
  '/worker/audio/step3_bho.mp3',
  '/worker/audio/step4_hi.mp3',
  '/worker/audio/step4_bho.mp3',
  '/worker/audio/step5_hi.mp3',
  '/worker/audio/step5_bho.mp3',
  '/worker/audio/quality_ok_hi.mp3',
  '/worker/audio/quality_ok_bho.mp3',
  '/worker/audio/quality_blur_hi.mp3',
  '/worker/audio/quality_blur_bho.mp3',
  '/worker/audio/quality_light_hi.mp3',
  '/worker/audio/quality_light_bho.mp3',
  '/worker/audio/ocr_success_hi.mp3',
  '/worker/audio/ocr_success_bho.mp3',
  '/calibcard-assets/calibcard_front.png',
  '/calibcard-assets/calibcard_specs.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Bal-Setu SW] Pre-caching offline worker shell, ONNX models, and voice assets...');
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
          { headers: { 'Content-Type': 'application/json' }, status: 503 }
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
