// DOM Elements
const photoUpload = document.getElementById('photo-upload');
const fileCount = document.getElementById('file-count');
const generateBtn = document.getElementById('generate-btn');
const latInput = document.getElementById('lat-input');
const lngInput = document.getElementById('lng-input');
const setupScreen = document.getElementById('setup-screen');
const viewerScreen = document.getElementById('viewer-screen');
const panoramaContainer = document.getElementById('panorama-container');
const panoramaTrack = document.getElementById('panorama-track');
const angleBadge = document.getElementById('angle-badge');
const restartBtn = document.getElementById('restart-btn');

// State
let photos = [];
let currentAngle = 0; // 0 to 359 degrees
let siteLocation = [48.8584, 2.2945];

// Map Objects
let map = null;
let siteMarker = null;
let viewCone = null;

// The field of view of a single photo
const FOV_DEGREES = 30;

// --- UPLOAD LOGIC ---

photoUpload.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
        alert("Please select valid image files.");
        return;
    }
    
    // Sort files by name using natural numeric sort so '30.jpg' comes before '120.jpg'
    imageFiles.sort((a, b) => a.name.localeCompare(b.name, undefined, {numeric: true, sensitivity: 'base'}));

    photos = imageFiles.map(file => ({
        file: file,
        url: URL.createObjectURL(file),
        name: file.name
    }));

    fileCount.textContent = `${photos.length} image(s) selected`;
    
    if (photos.length > 0) {
        generateBtn.disabled = false;
        if (photos.length !== 12) {
            fileCount.textContent += " (Warning: App expects 12 photos for exactly 360°)";
        }
    } else {
        generateBtn.disabled = true;
    }
});

generateBtn.addEventListener('click', () => {
    const rawLat = parseFloat(latInput.value);
    const rawLng = parseFloat(lngInput.value);
    
    if (isNaN(rawLat) || isNaN(rawLng)) {
        alert("Please enter valid numerical coordinate for Latitude and Longitude.");
        return;
    }
    
    if (photos.length === 0) {
        alert("Please upload photos first.");
        return;
    }

    siteLocation = [rawLat, rawLng];
    
    // Transition to viewer
    setupScreen.classList.remove('active');
    viewerScreen.classList.add('active');

    // Give DOM time to apply active class and set flexbox dimensions
    setTimeout(() => {
        initViewer();
    }, 100);
});

restartBtn.addEventListener('click', () => {
    viewerScreen.classList.remove('active');
    setTimeout(() => {
        setupScreen.classList.add('active');
        panoramaTrack.innerHTML = '';
    }, 500);
});

// --- VIEWER LOGIC ---

function initViewer() {
    currentAngle = 0;
    
    // Populate the track with 3 identical sets of the photos for infinite scrolling
    // Set 0 (left buffer), Set 1 (center main), Set 2 (right buffer)
    panoramaTrack.innerHTML = '';
    for (let i = 0; i < 3; i++) {
        photos.forEach((photo, idx) => {
            const img = document.createElement('img');
            img.src = photo.url;
            img.draggable = false;
            panoramaTrack.appendChild(img);
        });
    }

    // Wait for first image to load to measure width properly
    const firstImg = panoramaTrack.querySelector('img');
    if (firstImg) {
        if (firstImg.complete) {
            applyAngleToTrack();
        } else {
            firstImg.onload = () => applyAngleToTrack();
        }
    }
    
    // Init or update map
    if (!map) {
        initMap();
    } else {
        map.invalidateSize();
        map.setView(siteLocation, 19);
        siteMarker.setLatLng(siteLocation);
        updateViewCone();
    }
}

// 360 Continuous Drag interaction
let isDragging = false;
let startX = 0;
let startAngle = 0;

function getSingleSetWidth() {
    if (panoramaTrack.children.length === 0) return 0;
    // One set represents 360 degrees (e.g. 12 images)
    return panoramaTrack.children[0].clientWidth * photos.length;
}

panoramaContainer.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startAngle = currentAngle;
});

panoramaContainer.addEventListener('touchstart', (e) => {
    isDragging = true;
    startX = e.touches[0].clientX;
    startAngle = currentAngle;
}, {passive: true});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    handleDrag(e.clientX);
});

window.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    handleDrag(e.touches[0].clientX);
}, {passive: true});

window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('touchend', () => isDragging = false);
window.addEventListener('resize', applyAngleToTrack);

function handleDrag(currentX) {
    const deltaX = currentX - startX;
    
    const setWidth = getSingleSetWidth();
    if (setWidth === 0) return;
    
    // 360 degrees = setWidth pixels
    // Drag left (negative deltaX) -> angle increases
    const angleShift = -(deltaX / setWidth) * 360;
    let newAngle = startAngle + angleShift;
    
    // Keep angle positive and wrap around 360
    newAngle = ((newAngle % 360) + 360) % 360;
    
    currentAngle = newAngle;
    applyAngleToTrack();
}

function applyAngleToTrack() {
    const setWidth = getSingleSetWidth();
    if (setWidth === 0) return;
    
    // Base offset: start at the second set (index 1) which represents 0 degrees
    const baseOffset = -setWidth;
    
    // Calculate pixel shift for currentAngle
    const angleOffset = -(currentAngle / 360) * setWidth;
    
    const totalOffset = baseOffset + angleOffset;
    
    // Center the rendering horizontally inside the container viewport
    const containerWidth = panoramaContainer.clientWidth;
    const imgWidth = panoramaTrack.children[0].clientWidth;
    
    // Shift by half container width + half image width so that the 0° point (center of first image of set 1) is screen center
    const finalTx = totalOffset + (containerWidth / 2) - (imgWidth / 2);
    
    panoramaTrack.style.transform = `translateX(${finalTx}px)`;
    
    // Update UI badge
    const displayAngle = Math.round(currentAngle);
    angleBadge.textContent = 'Azimuth: ' + displayAngle + '°';
    
    // Sync Map continuously
    if (map) {
        updateViewCone();
        
        // Rotate the map wrapper to simulate heading up
        const mapWrapper = document.getElementById('map-wrapper');
        if (mapWrapper) {
            mapWrapper.style.transform = `rotate(${-currentAngle}deg)`;
        }
        
        // Rotate the compass arrow to indicate where north is
        const compass = document.querySelector('.compass-arrow');
        if (compass) {
            compass.style.transform = `rotate(${-currentAngle}deg)`;
        }
    }
}

// --- MAP LOGIC ---

function initMap() {
    map = L.map('map', {
        zoomControl: false
    });
    
    // Use ResizeObserver for bulletproof map sizing when layout changes
    const mapPanel = document.querySelector('.map-panel');
    if (mapPanel) {
        new ResizeObserver(() => {
            if (map) map.invalidateSize();
        }).observe(mapPanel);
    }
    
    // Force immediate invalidation
    map.invalidateSize();
    map.setView(siteLocation, 19);
    
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Add Esri Satellite Imagery
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP',
        maxZoom: 19
    }).addTo(map);

    // Center marker
    const iconHtml = `<div style="background-color: #2f81f7; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>`;
    const customIcon = L.divIcon({
        className: 'custom-site-marker',
        html: iconHtml,
        iconSize: [16, 16],
        iconAnchor: [8, 8]
    });

    siteMarker = L.marker(siteLocation, {icon: customIcon}).addTo(map);
    
    updateViewCone();
}

function updateViewCone() {
    if (viewCone) {
        map.removeLayer(viewCone);
    }
    
    const viewCenterRaw = currentAngle;
    
    const halfFov = FOV_DEGREES / 2;
    const rightAngle = viewCenterRaw + halfFov;
    const leftAngle = viewCenterRaw - halfFov;
    
    const distanceMeters = 60;
    
    const centerPoint = siteLocation;
    const leftPoint = destinationPoint(centerPoint[0], centerPoint[1], leftAngle, distanceMeters);
    const rightPoint = destinationPoint(centerPoint[0], centerPoint[1], rightAngle, distanceMeters);
    
    viewCone = L.polygon([
        centerPoint,
        leftPoint,
        rightPoint
    ], {
        color: '#2f81f7',
        fillColor: '#2f81f7',
        fillOpacity: 0.35,
        weight: 1,
        className: 'sight-cone',
        interactive: false
    }).addTo(map);
}

// Haversine formula based helper
function destinationPoint(lat, lng, bearingDeg, distanceM) {
    const R = 6371e3;
    const brng = bearingDeg * Math.PI / 180;
    const lat1 = lat * Math.PI / 180;
    const lon1 = lng * Math.PI / 180;
    
    var lat2 = Math.asin( Math.sin(lat1)*Math.cos(distanceM/R) +
                          Math.cos(lat1)*Math.sin(distanceM/R)*Math.cos(brng) );
    var lon2 = lon1 + Math.atan2(Math.sin(brng)*Math.sin(distanceM/R)*Math.cos(lat1),
                                 Math.cos(distanceM/R)-Math.sin(lat1)*Math.sin(lat2));
                                 
    return [lat2 * 180 / Math.PI, lon2 * 180 / Math.PI];
}
