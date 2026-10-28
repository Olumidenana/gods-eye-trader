// Service worker: network-first with cache-busting on navigation + auto-cleanup.
const CACHE = 'ictflux-gdeye-v38-live';

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(['/manifest.json']).catch(() => {}))
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('message', e => {
  if (!e.data) return;
  if (e.data === 'SKIP_WAITING' || e.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  if (e.data === 'CLEAR_CACHE' || e.data.type === 'CLEAR_CACHE') {
    e.waitUntil(
      caches.keys().then(keys => Promise.all(keys.map(k => caches.delete(k))))
    );
  }
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  if (url.pathname.startsWith('/api/') || url.pathname === '/version.json') return;
  if (url.hostname === 'accounts.google.com') return;
  const isNav = e.request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html');
  const req = isNav ? new Request(e.request.url, { cache: 'no-store' }) : e.request;
  e.respondWith(
    fetch(req).then(res => {
      if (res && res.ok && url.origin === self.location.origin) {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }
      return res;
    }).catch(() => caches.match(e.request))
  );
});
