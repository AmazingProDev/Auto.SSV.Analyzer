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

photoUpload.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    
    // Check if an Excel file is uploaded
    const xlsxFile = files.find(file => file.name.toLowerCase().endsWith('.xlsx'));
    if (xlsxFile) {
        await handleExcelUpload(xlsxFile);
        return;
    }
    
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

async function handleExcelUpload(file) {
    fileCount.textContent = `Analyzing Excel file...`;
    generateBtn.disabled = true;
    
    try {
        const zip = await JSZip.loadAsync(file);
        const parser = new DOMParser();
        
        // 1. Read shared strings
        const sstXmlObj = zip.file('xl/sharedStrings.xml');
        let stringArray = [];
        if (sstXmlObj) {
            const sstText = await sstXmlObj.async('string');
            const sstDoc = parser.parseFromString(sstText, "text/xml");
            const siNodes = sstDoc.getElementsByTagName("si");
            Array.from(siNodes).forEach((si) => {
                const ts = si.getElementsByTagName("t");
                let fullText = "";
                for(let t of ts) fullText += t.textContent;
                stringArray.push(fullText.trim());
            });
        }
        
        // 2. Discover Target Angle Strings
        const targetAnglesStr = [];
        for (let i = 0; i < 360; i += 30) {
            targetAnglesStr.push(`${i} DEGRÉS`);
        }
        
        let targetStringIndexes = {};
        stringArray.forEach((s, idx) => {
            const upperS = s.toUpperCase().replace(/\s+/g, ' '); 
            if (targetAnglesStr.includes(upperS)) {
                const angle = parseInt(upperS.split(' ')[0]);
                targetStringIndexes[idx.toString()] = angle;
            }
        });

        // 3. Find target cells in sheet1
        const sheetXmlObj = zip.file('xl/worksheets/sheet1.xml');
        if (!sheetXmlObj) throw new Error("Could not find sheet1.xml in .xlsx");
        const sheetText = await sheetXmlObj.async('string');
        const sheetDoc = parser.parseFromString(sheetText, "text/xml");
        
        let foundAngleCells = [];
        const rows = sheetDoc.getElementsByTagName("row");
        Array.from(rows).forEach(row => {
            const rowNum = parseInt(row.getAttribute("r"));
            const cells = row.getElementsByTagName("c");
            Array.from(cells).forEach(c => {
                if (c.getAttribute("t") === "s") {
                    const v = c.getElementsByTagName("v")[0];
                    if (v && targetStringIndexes[v.textContent] !== undefined) {
                        const cRef = c.getAttribute("r");
                        const colStr = cRef.replace(/[0-9]/g, '');
                        foundAngleCells.push({
                            row: rowNum,
                            colStr: colStr,
                            angle: targetStringIndexes[v.textContent]
                        });
                    }
                }
            });
        });
        
        let angleRows = {};
        foundAngleCells.forEach(fc => {
            angleRows[fc.row] = angleRows[fc.row] || [];
            angleRows[fc.row].push(fc);
        });

        // 4. Drawing Rels Mapping
        const relsXmlObj = zip.file('xl/drawings/_rels/drawing1.xml.rels');
        let drawingRels = {};
        if (relsXmlObj) {
            const relsText = await relsXmlObj.async('string');
            const relsDoc = parser.parseFromString(relsText, "text/xml");
            const rels = relsDoc.getElementsByTagName("Relationship");
            Array.from(rels).forEach(r => {
                drawingRels[r.getAttribute("Id")] = r.getAttribute("Target");
            });
        }
        
        // 5. Drawing Anchor matching
        const drawXmlObj = zip.file('xl/drawings/drawing1.xml');
        if (!drawXmlObj) throw new Error("Could not find drawing1.xml");
        const drawText = await drawXmlObj.async('string');
        const drawDoc = parser.parseFromString(drawText, "text/xml");
        const anchors1 = drawDoc.getElementsByTagName("xdr:twoCellAnchor");
        const anchors2 = drawDoc.getElementsByTagName("twoCellAnchor");
        const anchors = anchors1.length > 0 ? anchors1 : anchors2;
        
        Array.from(anchors).forEach(anchor => {
            const from = anchor.getElementsByTagName("xdr:from")[0] || anchor.getElementsByTagName("from")[0];
            if (from) {
                const rowNode = from.getElementsByTagName("xdr:row")[0] || from.getElementsByTagName("row")[0];
                const colNode = from.getElementsByTagName("xdr:col")[0] || from.getElementsByTagName("col")[0];
                
                if (rowNode && colNode) {
                    const imgRow = parseInt(rowNode.textContent);
                    const imgColRaw = parseInt(colNode.textContent);
                    
                    if (angleRows[imgRow]) {
                        const blip1 = anchor.getElementsByTagName("a:blip")[0];
                        const blip2 = anchor.getElementsByTagName("blip")[0];
                        const blip = blip1 || blip2;
                        if (blip) {
                            const rId = blip.getAttribute("r:embed");
                            if (rId && drawingRels[rId]) {
                                if (!angleRows[imgRow].images) angleRows[imgRow].images = [];
                                angleRows[imgRow].images.push({ col: imgColRaw, target: drawingRels[rId] });
                            }
                        }
                    }
                }
            }
        });
        
        // 6. Match labels to images by column ordering
        let extractedImages = [];
        Object.keys(angleRows).forEach(rowStr => {
            const rowData = angleRows[rowStr];
            const images = rowData.images || [];
            
            rowData.sort((a, b) => {
                if(a.colStr.length !== b.colStr.length) return a.colStr.length - b.colStr.length;
                return a.colStr.localeCompare(b.colStr);
            });
            images.sort((a, b) => a.col - b.col);
            
            for (let i = 0; i < Math.min(rowData.length, images.length); i++) {
                extractedImages.push({
                    angle: rowData[i].angle,
                    target: images[i].target 
                });
            }
        });
        
        if (extractedImages.length < 12) {
            throw new Error(`Only found ${extractedImages.length}/12 panoramic photos. Expected 12 images correctly anchored under 'DEGRÉS' cells.`);
        }
        
        // 7. Load Binaries and Build Photos Array
        photos = [];
        for (let i = 0; i < extractedImages.length; i++) {
            const item = extractedImages[i];
            let mediaPath = item.target.startsWith('../') ? item.target.substring(3) : item.target;
            mediaPath = 'xl/' + mediaPath; 
            
            const mediaFile = zip.file(mediaPath);
            if (mediaFile) {
                const ext = mediaPath.split('.').pop().toLowerCase();
                let mime = 'image/jpeg';
                if (ext === 'png') mime = 'image/png';
                
                const blob = await mediaFile.async('blob');
                const typedBlob = new Blob([blob], { type: mime });
                
                photos.push({
                    file: typedBlob,
                    url: URL.createObjectURL(typedBlob),
                    name: `Angle_${item.angle}°`,
                    angle: item.angle
                });
            }
        }
        
        photos.sort((a, b) => a.angle - b.angle);
        fileCount.textContent = `Successfully extracted ${photos.length} exact panoramic images from Exce!`;
        generateBtn.disabled = false;
        
    } catch (err) {
        console.error(err);
        fileCount.textContent = "Error parsing Excel: " + err.message;
        generateBtn.disabled = true;
    }
}

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
