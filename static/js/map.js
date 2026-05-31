/* BunkerVision map — MapLibre GL JS
   Vessel state colours:
     alongside → green  (#3fb950)
     stopped   → amber  (#d29922)
     underway  → blue   (#58a6ff)
     other     → grey   (#8b949e)
*/

const STATE_COLOUR = {
  alongside: '#3fb950',
  stopped:   '#d29922',
  underway:  '#58a6ff',
  other:     '#8b949e',
};

const STATE_RADIUS = {
  alongside: 9,
  stopped:   7,
  underway:  5,
  other:     3.5,
};

// Vessel positions: mmsi → {mmsi, imo, name, lat, lon, sog, state, is_bunker}
const vessels = {};
let map, popup;

function initMap() {
  // Build style inline — avoids any external style JSON dependency
  const tileStyle = MAPBOX_TOKEN
    ? {
        version: 8,
        sources: { mapbox: {
          type: 'raster',
          tiles: [`https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/{z}/{x}/{y}?access_token=${MAPBOX_TOKEN}`],
          tileSize: 512,
        }},
        layers: [{ id: 'bg', type: 'raster', source: 'mapbox' }],
      }
    : {
        version: 8,
        sources: { carto: {
          type: 'raster',
          tiles: [
            'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          ],
          tileSize: 256,
          attribution: '© CARTO · © OpenStreetMap contributors',
        }},
        layers: [{ id: 'bg', type: 'raster', source: 'carto' }],
        glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
      };

  map = new maplibregl.Map({
    container: 'map',
    style: tileStyle,
    center: [MAP_CENTRE.lon, MAP_CENTRE.lat],
    zoom:   MAP_CENTRE.zoom,
    attributionControl: false,
    dragRotate: false,          // flat north-up map — avoids accidental rotation on touch
    pitchWithRotate: false,
    touchPitch: false,          // two-finger drag pans/zooms instead of tilting on mobile
    cooperativeGestures: false,
  });
  // Single-finger touch pans the map without rotating it.
  if (map.touchZoomRotate) map.touchZoomRotate.disableRotation();

  // Settle layout once the canvas has a real size (e.g. shown inside a tab).
  map.on('load', () => map.resize());

  map.addControl(new maplibregl.NavigationControl(), 'top-right');
  map.addControl(new maplibregl.AttributionControl({ compact: true }));

  popup = new maplibregl.Popup({ closeButton: true, closeOnClick: false, maxWidth: '260px' });

  map.on('load', () => {
    // OpenSeaMap nautical overlay
    map.addSource('openseamap', {
      type: 'raster',
      tiles: ['https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenSeaMap',
    });
    map.addLayer({ id: 'openseamap', type: 'raster', source: 'openseamap', paint: { 'raster-opacity': 0.6 } });

    // Vessel source (will be updated via updateMap)
    map.addSource('vessels', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    });

    // Non-bunker vessels (dimmer, smaller)
    map.addLayer({
      id: 'vessels-other',
      type: 'circle',
      source: 'vessels',
      filter: ['==', ['get', 'is_bunker'], false],
      paint: {
        'circle-radius': 3.5,
        'circle-color': '#8b949e',
        'circle-opacity': 0.6,
        'circle-stroke-width': 0,
      },
    });

    // Bunker vessels — halo
    map.addLayer({
      id: 'vessels-bunker-halo',
      type: 'circle',
      source: 'vessels',
      filter: ['==', ['get', 'is_bunker'], true],
      paint: {
        'circle-radius': [
          'match', ['get', 'state'],
          'alongside', 14, 'stopped', 11, 8,
        ],
        'circle-color': [
          'match', ['get', 'state'],
          'alongside', '#3fb950', 'stopped', '#d29922', '#58a6ff',
        ],
        'circle-opacity': 0.18,
        'circle-stroke-width': 0,
      },
    });

    // Bunker vessels — core dot
    map.addLayer({
      id: 'vessels-bunker',
      type: 'circle',
      source: 'vessels',
      filter: ['==', ['get', 'is_bunker'], true],
      paint: {
        'circle-radius': [
          'match', ['get', 'state'],
          'alongside', 9, 'stopped', 7, 5,
        ],
        'circle-color': [
          'match', ['get', 'state'],
          'alongside', '#3fb950', 'stopped', '#d29922', '#58a6ff',
        ],
        'circle-opacity': 0.95,
        'circle-stroke-width': 1.5,
        'circle-stroke-color': 'rgba(0,0,0,0.4)',
      },
    });

    // Labels for bunker vessels
    map.addLayer({
      id: 'vessels-bunker-label',
      type: 'symbol',
      source: 'vessels',
      filter: ['==', ['get', 'is_bunker'], true],
      layout: {
        'text-field': ['get', 'name'],
        'text-size': 10,
        'text-offset': [0, 1.4],
        'text-anchor': 'top',
        'text-allow-overlap': false,
        'text-optional': true,
      },
      paint: {
        'text-color': '#c9d1d9',
        'text-halo-color': 'rgba(13,17,23,0.8)',
        'text-halo-width': 1.5,
      },
    });

    // Click handler
    map.on('click', 'vessels-bunker', onVesselClick);
    map.on('click', 'vessels-other', onVesselClick);
    map.on('mouseenter', 'vessels-bunker', () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', 'vessels-bunker', () => { map.getCanvas().style.cursor = ''; });

    // Initial data load
    fetchAndUpdatePositions();
  });
}

function onVesselClick(e) {
  if (!e.features || !e.features.length) return;
  const p = e.features[0].properties;
  const sogKt = parseFloat(p.sog || 0).toFixed(1);
  const stateLabel = { alongside: 'Alongside (bunkering)', stopped: 'Stopped', underway: 'Underway', other: 'Other' }[p.state] || p.state;
  const stateCol = STATE_COLOUR[p.state] || '#8b949e';

  const draughtLine = p.draught && p.draught > 0
    ? `<div class="vessel-popup-row">Draught <span>${parseFloat(p.draught).toFixed(1)} m</span></div>`
    : '';
  popup.setLngLat(e.lngLat).setHTML(`
    <div class="vessel-popup-name">${p.name || 'Unknown vessel'}</div>
    <div class="vessel-popup-row">IMO <span>${p.imo || '—'}</span></div>
    <div class="vessel-popup-row">MMSI <span>${p.mmsi}</span></div>
    <div class="vessel-popup-row">SOG <span>${sogKt} kt</span></div>
    ${draughtLine}
    <div class="vessel-popup-row">Status <span style="color:${stateCol}">${stateLabel}</span></div>
  `).addTo(map);
}

function updateMap(positions) {
  const features = positions.map(p => ({
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [p.lon, p.lat] },
    properties: {
      mmsi:      p.mmsi,
      imo:       p.imo || '',
      name:      p.name || '',
      sog:       p.sog || 0,
      state:     p.state || 'other',
      is_bunker: !!p.is_bunker,
    },
  }));

  const src = map.getSource('vessels');
  if (src) src.setData({ type: 'FeatureCollection', features });

  // Update live badge
  const badge = document.getElementById('live-badge');
  if (badge) {
    const bunkerCount = positions.filter(p => p.is_bunker).length;
    badge.textContent = `${bunkerCount} bunker / ${positions.length} live`;
    badge.className = 'badge ' + (bunkerCount > 0 ? 'bg-success' : 'bg-secondary');
    badge.style.fontSize = '0.7rem';
  }
}

async function fetchAndUpdatePositions() {
  try {
    const port = (typeof CURRENT_PORT !== 'undefined') ? CURRENT_PORT : 'singapore';
    const positions = await fetch(`/api/positions?port=${port}`).then(r => r.json());
    // On each full refresh, clear stale vessels from other ports
    Object.keys(vessels).forEach(k => delete vessels[k]);
    positions.forEach(p => { vessels[p.mmsi] = p; });
    if (map.isStyleLoaded()) updateMap(Object.values(vessels));
  } catch(e) {
    console.warn('Position fetch failed', e);
  }
}

// ── SSE for real-time updates ──────────────────────────────────────────────────
function startSSE() {
  const es = new EventSource('/stream');
  let sseActive = false;

  es.onmessage = function(e) {
    try {
      const data = JSON.parse(e.data);
      if (data.heartbeat) {
        const badge = document.getElementById('live-badge');
        if (badge && badge.textContent === 'AIS connecting…') {
          badge.textContent = 'SSE ok — waiting for AIS…';
          badge.className = 'badge bg-warning text-dark';
          badge.style.fontSize = '0.7rem';
        }
        return;
      }
      sseActive = true;

      const badge = document.getElementById('live-badge');
      if (badge && !badge.classList.contains('bg-success')) {
        badge.className = 'badge bg-success';
        badge.style.fontSize = '0.7rem';
      }

      // Only show vessels in the current port's bbox
      const bb = PORT_CFG && PORT_CFG.bbox;
      if (bb) {
        const lat = data.lat, lon = data.lon;
        if (lat < bb[0][0] || lat > bb[1][0] || lon < bb[0][1] || lon > bb[1][1]) return;
      }

      vessels[data.mmsi] = data;

      // Throttle map redraws to every 2 seconds
      scheduleMapUpdate();
    } catch(err) {}
  };

  es.onerror = function() {
    const badge = document.getElementById('live-badge');
    if (badge) { badge.textContent = 'AIS disconnected'; badge.className = 'badge bg-danger'; badge.style.fontSize = '0.7rem'; }
    setTimeout(() => startSSE(), 5000);
    es.close();
  };
}

let _mapUpdateTimer = null;
function scheduleMapUpdate() {
  if (_mapUpdateTimer) return;
  _mapUpdateTimer = setTimeout(() => {
    _mapUpdateTimer = null;
    if (map.isStyleLoaded()) updateMap(Object.values(vessels));
  }, 2000);
}

// Full position refresh every 60s (catches any SSE gaps)
setInterval(fetchAndUpdatePositions, 60_000);

// ── Boot ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  startSSE();
});
