import re

html_file = 'index.html'
with open(html_file, 'r') as f:
    html = f.read()

# Bust cache
html = re.sub(r'href="style.css[^"]*"', 'href="style.css?v=3"', html)

# Nested Wrappers
map_wrapper = """<div id="map-outer-wrapper" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; pointer-events: none; will-change: transform; z-index: 10;">
                    <div id="map-wrapper" style="position: absolute; width: 300%; height: 300%; top: -100%; left: -100%; pointer-events: auto; transform-origin: center center;">
                        <div id="map" style="width: 100%; height: 100%;"></div>
                    </div>
                </div>"""
html = re.sub(r'<div id="map-wrapper">.*?</div>\s*</div>', map_wrapper, html, flags=re.DOTALL)

with open(html_file, 'w') as f:
    f.write(html)


js_file = 'app.js'
with open(js_file, 'r') as f:
    js = f.read()

# Strip obsolete globals
js = re.sub(r'let mapTx\s*=[^;]*;\s*let mapTy\s*=[^;]*;\s*let lastOriginX.*?;\s*let lastOriginY.*?;\s*let lastAngle.*?;\n?', '', js, flags=re.DOTALL)
js = re.sub(r'let previousDragX = null;\n?', '', js)

# Inject Stateless Math directly over whatever is currently in applyAngleToTrack mapWrapper block
stateless_math = """        const mapWrapper = document.getElementById('map-wrapper');
        if (mapWrapper) {
            const pt = map.latLngToContainerPoint(siteLocation);
            const cx = mapWrapper.offsetWidth / 2;
            const cy = mapWrapper.offsetHeight / 2;
            
            const vx = pt.x - cx;
            const vy = pt.y - cy;
            
            const rad = (-currentAngle) * (Math.PI / 180);
            const rx = vx * Math.cos(rad) - vy * Math.sin(rad);
            const ry = vx * Math.sin(rad) + vy * Math.cos(rad);
            
            const tx = vx - rx;
            const ty = vy - ry;
            
            mapWrapper.style.transform = `translate(${tx}px, ${ty}px) rotate(${-currentAngle}deg)`;
        }"""
js = re.sub(r'const mapWrapper = document\.getElementById\(\'map-wrapper\'\);.*?mapWrapper\.style\.transformOrigin.*?;\n\s*mapWrapper\.style\.transform.*?;(?:\n\s*lastAngle = currentAngle;)?\n\s*\}', stateless_math, js, flags=re.DOTALL)


# Fix custom panning script block dynamically to use outer wrapper tracking
custom_pan = """// --- Custom Rotation-Aware Map Panning ---
const mapPanelDOM = document.querySelector('.map-panel');
let isPanningMap = false;
let mapOuterTx = 0;
let mapOuterTy = 0;

mapPanelDOM.addEventListener('mousedown', (e) => {
    if (e.target.closest('.leaflet-control') || e.target.closest('#map-info-hud') || e.target.closest('#site-name-hud') || e.target.closest('#compass-overlay')) return;
    if (e.button !== 0) return; 
    isPanningMap = true;
});

window.addEventListener('mousemove', (e) => {
    if (!isPanningMap || !map) return;
    mapOuterTx += e.movementX;
    mapOuterTy += e.movementY;
    const outer = document.getElementById('map-outer-wrapper');
    if(outer) outer.style.transform = `translate(${mapOuterTx}px, ${mapOuterTy}px)`;
});

window.addEventListener('mouseup', () => { isPanningMap = false; });

let lastTouchX = null; let lastTouchY = null;
mapPanelDOM.addEventListener('touchstart', (e) => {
    if (e.target.closest('.leaflet-control') || e.target.closest('#map-info-hud') || e.target.closest('#site-name-hud') || e.target.closest('#compass-overlay')) return;
    if (e.touches.length === 1) {
        isPanningMap = true;
        lastTouchX = e.touches[0].clientX;
        lastTouchY = e.touches[0].clientY;
    }
}, {passive: false});

mapPanelDOM.addEventListener('touchmove', (e) => {
    if (!isPanningMap || !map || e.touches.length !== 1) return;
    e.preventDefault(); 
    mapOuterTx += (e.touches[0].clientX - lastTouchX);
    mapOuterTy += (e.touches[0].clientY - lastTouchY);
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
    const outer = document.getElementById('map-outer-wrapper');
    if(outer) outer.style.transform = `translate(${mapOuterTx}px, ${mapOuterTy}px)`;
}, {passive: false});

mapPanelDOM.addEventListener('touchend', () => { isPanningMap = false; lastTouchX = null; lastTouchY = null; });
"""
js = re.sub(r'// --- Custom Rotation-Aware Map Panning ---.*', custom_pan, js, flags=re.DOTALL)


# Fix the hardcoded azimuths if config is empty directly in updateSectorMapPolygons
azimuth_fix = """function updateSectorMapPolygons() {
    if (!map) return;
    
    sectorPolygons.forEach(p => map.removeLayer(p));
    sectorPolygons = [];
    
    let config = radioConfig[activeConfigStr] || [];
    
    if (config.length === 0) {
       config = [
           { name: 'Secteur 1', azimuth: 50 },
           { name: 'Secteur 2', azimuth: 130 },
           { name: 'Secteur 3', azimuth: 220 }
       ];
    }
    
    if (hudAzimuts) {"""
js = re.sub(r'function updateSectorMapPolygons\(\) \{.*?\n\s*if \(hudAzimuts\) \{', azimuth_fix, js, flags=re.DOTALL)

with open(js_file, 'w') as f:
    f.write(js)

