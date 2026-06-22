// Increment this version number every time you make changes
const CACHE_NAME = 'cardiocura-v5';

// Only cache the app shell — NOT images
// Images will always load fresh from server
const urlsToCache = [
    '/',
    '/static/css/mstyle.css',
    '/static/css/style.css',
    '/static/js/main.js',
];

// INSTALL
self.addEventListener('install', function(event) {
    console.log('[SW] Installing v5...');
    // Force this SW to activate immediately
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME).then(function(cache) {
            return cache.addAll(urlsToCache).catch(function(err) {
                console.warn('[SW] Cache failed:', err);
            });
        })
    );
});

// ACTIVATE — delete ALL old caches
self.addEventListener('activate', function(event) {
    console.log('[SW] Activating v5...');
    event.waitUntil(
        caches.keys().then(function(cacheNames) {
            return Promise.all(
                cacheNames.map(function(cacheName) {
                    // Delete every old cache
                    if (cacheName !== CACHE_NAME) {
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(function() {
            // Take control of all open tabs immediately
            return self.clients.claim();
        })
    );
});

// FETCH — Network first for images, cache first for CSS/JS
self.addEventListener('fetch', function(event) {
    // Skip non-GET requests
    if (event.request.method !== 'GET') return;

    const url = event.request.url;

    // Always fetch images fresh from network — never cache them
    if (url.match(/\.(jpg|jpeg|png|gif|webp|svg)$/i)) {
        event.respondWith(
            fetch(event.request).catch(function() {
                // If network fails, try cache as last resort
                return caches.match(event.request);
            })
        );
        return;
    }

    // For CSS and JS — cache first, then network
    event.respondWith(
        caches.match(event.request).then(function(response) {
            if (response) {
                return response;
            }
            return fetch(event.request);
        }).catch(function() {
            return caches.match(event.request);
        })
    );
});