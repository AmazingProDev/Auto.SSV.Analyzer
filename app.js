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

// Radio Config State
let radioConfig = { avant: [], apres: [] }; // { name: 'Secteur 1', azimuth: 90, url: '...', target: '...' }
let activeConfigStr = 'avant';
const configSelect = document.getElementById('config-select');
const sectorHud = document.getElementById('sector-hud');
const sectorHudTitle = document.getElementById('sector-hud-title');
const sectorHudImg = document.getElementById('sector-hud-img');
let sectorPolygons = []; // map layers

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
    fileCount.textContent = `Analyzing Excel file for Panoramas and Sectors...`;
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
        
        // 2. Discover Target Angle Strings array
        const targetAnglesStr = [];
        for (let i = 0; i < 360; i += 30) {
            targetAnglesStr.push(`${i} DEGRÉS`);
        }
        
        // 3. Parse Sheet1 quickly into 2D map
        const sheetXmlObj = zip.file('xl/worksheets/sheet1.xml');
        if (!sheetXmlObj) throw new Error("Could not find sheet1.xml in .xlsx");
        const sheetText = await sheetXmlObj.async('string');
        const sheetDoc = parser.parseFromString(sheetText, "text/xml");
        
        let cellsData = {}; 
        const rows = sheetDoc.getElementsByTagName("row");
        Array.from(rows).forEach(row => {
            const rowNum = parseInt(row.getAttribute("r"));
            cellsData[rowNum] = {};
            const cells = row.getElementsByTagName("c");
            Array.from(cells).forEach(c => {
                const cRef = c.getAttribute("r");
                const colStr = cRef.replace(/[0-9]/g, '');
                const v = c.getElementsByTagName("v")[0];
                if (v) {
                    let val = v.textContent;
                    let isStr = c.getAttribute("t") === "s";
                    if (isStr) val = stringArray[parseInt(val)];
                    cellsData[rowNum][colStr] = { value: val, isStr: isStr };
                }
            });
        });
        
        // 4. Find Panoramic degree cells
        let foundAngleCells = [];
        Object.keys(cellsData).forEach(rStr => {
            const r = parseInt(rStr);
            Object.keys(cellsData[r]).forEach(cStr => {
                const cell = cellsData[r][cStr];
                if (cell.isStr && typeof cell.value === 'string') {
                   const upperS = cell.value.toUpperCase().replace(/\s+/g, ' ');
                   if (targetAnglesStr.includes(upperS)) {
                       const angle = parseInt(upperS.split(' ')[0]);
                       foundAngleCells.push({row: r, colStr: cStr, angle: angle});
                   }
                }
            });
        });
        
        let angleRows = {};
        foundAngleCells.forEach(fc => {
            angleRows[fc.row] = angleRows[fc.row] || [];
            angleRows[fc.row].push(fc);
        });

        // 5. Extract CGPS and Sector Azimuths
        let coordRows = [];
        Object.keys(cellsData).forEach(rStr => {
            const r = parseInt(rStr);
            const rowObj = cellsData[r];
            Object.keys(rowObj).forEach(cStr => {
                if (rowObj[cStr].isStr && typeof rowObj[cStr].value === 'string' && rowObj[cStr].value.includes('Coordonnées GPS Lat:')) {
                    coordRows.push(r);
                }
            });
        });
        coordRows.sort((a,b)=>a-b);
        
        let cgpsLat = null, cgpsLng = null;
        let avantConfig = [], apresConfig = [];
        
        function extractConfigAzimuths(startRow, configArr) {
            let azimutColStr = 'E'; // fallback
            // Search for Azimuth header explicitly in the rows roughly before Sector listings
            for(let r=startRow; r<startRow+15; r++) {
               if(!cellsData[r]) continue;
               Object.keys(cellsData[r]).forEach(cStr => {
                   if (cellsData[r][cStr].isStr && typeof cellsData[r][cStr].value === 'string' && cellsData[r][cStr].value.includes('Azimut')) {
                       azimutColStr = cStr;
                   }
               });
            }
            
            for(let r=startRow+1; r<startRow+20; r++) {
                const rowObj = cellsData[r];
                if(!rowObj) continue;
                Object.keys(rowObj).forEach(cStr => {
                    const val = rowObj[cStr].value;
                    if(rowObj[cStr].isStr && typeof val === 'string' && val.startsWith('Secteur ')) {
                        const az = rowObj[azimutColStr] ? parseFloat(rowObj[azimutColStr].value) : null;
                        if(az !== null && !isNaN(az)) {
                            configArr.push({ name: val, azimuth: az });
                        }
                    }
                });
            }
        }
        
        if (coordRows.length > 0) {
            const r = coordRows[0];
            cgpsLat = cellsData[r]['E'] ? parseFloat(cellsData[r]['E'].value) : null;
            cgpsLng = cellsData[r]['H'] ? parseFloat(cellsData[r]['H'].value) : null;
            extractConfigAzimuths(r, avantConfig);
        }
        if (coordRows.length > 1) {
            extractConfigAzimuths(coordRows[1], apresConfig);
        }
        
        // 6. Finding Photos Azimut anchor rows
        let photoMainRow = null;
        Object.keys(cellsData).forEach(rStr => {
            const r = parseInt(rStr);
            Object.values(cellsData[r]).forEach(cell => {
                if (cell.isStr && typeof cell.value === 'string' && cell.value.toUpperCase().includes('PHOTOS AZIMUT')) {
                    photoMainRow = r;
                }
            });
        });
        
        let avantPhotoAnchorRow = null;
        let apresPhotoAnchorRow = null;
        if (photoMainRow) {
            let foundAvantHeader = false;
            let foundApresHeader = false;
            for(let r=photoMainRow+1; r<photoMainRow+100; r++) {
                if(!cellsData[r]) continue;
                let textConcat = Object.values(cellsData[r]).filter(c=>c.isStr && typeof c.value === 'string').map(c=>c.value.toUpperCase()).join(' ');
                
                if (textConcat.includes('AVANT OPTIMISATION')) { foundAvantHeader = true; continue; }
                if (textConcat.includes('APRES OPTIMISATION')) { foundApresHeader = true; continue; }
                
                if (textConcat.includes('SECTEUR 1')) {
                    if (foundAvantHeader && !avantPhotoAnchorRow) avantPhotoAnchorRow = r;
                    else if (foundApresHeader && !apresPhotoAnchorRow) apresPhotoAnchorRow = r;
                }
                
                if (avantPhotoAnchorRow && apresPhotoAnchorRow) break;
            }
        }
        
        function populateAngleRowsForSectors(anchorRow) {
           if (!angleRows[anchorRow]) angleRows[anchorRow] = [];
           Object.keys(cellsData[anchorRow]).forEach(cStr => {
                const cell = cellsData[anchorRow][cStr];
                if (cell.isStr && typeof cell.value === 'string' && cell.value.startsWith('Secteur')) {
                     angleRows[anchorRow].push({ colStr: cStr, angle: cell.value });
                }
           });
        }
        if (avantPhotoAnchorRow) populateAngleRowsForSectors(avantPhotoAnchorRow);
        if (apresPhotoAnchorRow) populateAngleRowsForSectors(apresPhotoAnchorRow);

        // 7. Drawing Rels Mapping
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
        
        // 8. Drawing Anchor matching
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
        
        // 9. Match labels to images by column ordering
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
                const labelAngle = rowData[i].angle;
                if (typeof labelAngle === 'number') {
                    extractedImages.push({ angle: labelAngle, target: images[i].target });
                } else {
                    const rowN = parseInt(rowStr);
                    if (rowN === avantPhotoAnchorRow) {
                         let cfg = avantConfig.find(c => c.name === labelAngle);
                         if (cfg) cfg.target = images[i].target;
                    } else if (rowN === apresPhotoAnchorRow) {
                         let cfg = apresConfig.find(c => c.name === labelAngle);
                         if (cfg) cfg.target = images[i].target;
                    }
                }
            }
        });
        
        if (extractedImages.length < 12) {
            throw new Error(`Only found ${extractedImages.length}/12 panoramic photos. Expected 12 images correctly anchored under 'DEGRÉS' cells.`);
        }
        
        // 10. Load Binaries
        async function loadBinaries(itemsArray) {
            for (let i = 0; i < itemsArray.length; i++) {
                const item = itemsArray[i];
                if (!item.target) continue;
                let mediaPath = item.target.startsWith('../') ? item.target.substring(3) : item.target;
                mediaPath = 'xl/' + mediaPath; 
                
                const mediaFile = zip.file(mediaPath);
                if (mediaFile) {
                    const ext = mediaPath.split('.').pop().toLowerCase();
                    let mime = 'image/jpeg';
                    if (ext === 'png') mime = 'image/png';
                    
                    const blob = await mediaFile.async('blob');
                    const typedBlob = new Blob([blob], { type: mime });
                    item.url = URL.createObjectURL(typedBlob);
                }
            }
        }
        
        await loadBinaries(extractedImages);
        await loadBinaries(avantConfig);
        await loadBinaries(apresConfig);
        
        // Finalize state
        photos = extractedImages.map(item => ({
            file: null, url: item.url, name: `Angle_${item.angle}°`, angle: item.angle
        }));
        photos.sort((a, b) => a.angle - b.angle);
        
        radioConfig.avant = avantConfig.filter(c => c.url);
        radioConfig.apres = apresConfig.filter(c => c.url);
        
        if (cgpsLat !== null && cgpsLng !== null) {
            latInput.value = cgpsLat;
            lngInput.value = cgpsLng;
        }
        
        if (radioConfig.avant.length > 0 || radioConfig.apres.length > 0) {
            configSelect.style.display = 'block';
            activeConfigStr = radioConfig.avant.length > 0 ? 'avant' : 'apres';
            configSelect.value = activeConfigStr;
        } else {
            configSelect.style.display = 'none';
        }
        
        fileCount.textContent = `Successfully extracted ${photos.length} panoramas, Coordinates & Sectors!`;
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
            const wrapper = document.createElement('div');
            wrapper.className = 'photo-wrapper';
            
            const img = document.createElement('img');
            img.src = photo.url;
            img.draggable = false;
            
            // Debug label to show the user the strict ordering
            const label = document.createElement('div');
            label.className = 'photo-label';
            label.textContent = photo.name; 
            
            wrapper.appendChild(img);
            wrapper.appendChild(label);
            panoramaTrack.appendChild(wrapper);
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

function getTrackGeometry(targetAngle) {
    if (panoramaTrack.children.length === 0 || photos.length === 0) return { offset: 0, setWidth: 0 };
    
    const count = photos.length;
    let leftEdges = [0];
    for (let i = 0; i < count; i++) {
        leftEdges.push(leftEdges[i] + panoramaTrack.children[i].clientWidth);
    }
    const setWidth = leftEdges[count];
    
    let centers = [];
    for (let i = 0; i < count; i++) {
        centers.push(leftEdges[i] + panoramaTrack.children[i].clientWidth / 2);
    }
    centers.push(setWidth + panoramaTrack.children[0].clientWidth / 2);
    
    // Calculate fractional index (0 to count)
    const targetIndex = (targetAngle / 360) * count;
    const intIndex = Math.floor(targetIndex);
    const frac = targetIndex - intIndex;
    
    // Smoothly interpolate center point between adjacent photos
    const center1 = centers[intIndex];
    const center2 = centers[intIndex + 1];
    const pCenter = center1 + frac * (center2 - center1);
    
    return { offset: pCenter, setWidth: setWidth };
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
    
    const { setWidth } = getTrackGeometry(0);
    if (setWidth === 0) return;
    
    const angleShift = -(deltaX / setWidth) * 360;
    let newAngle = startAngle + angleShift;
    
    newAngle = ((newAngle % 360) + 360) % 360;
    
    currentAngle = newAngle;
    applyAngleToTrack();
}

function applyAngleToTrack() {
    const { offset, setWidth } = getTrackGeometry(currentAngle);
    if (setWidth === 0) return;
    
    // Target the center duplicate set (Set index 1 acts as baseline)
    const targetCenter = setWidth + offset;
    const containerWidth = panoramaContainer.clientWidth;
    
    // Center logic: We want targetCenter to be in the middle of container width
    const finalTx = (containerWidth / 2) - targetCenter;
    
    panoramaTrack.style.transform = `translateX(${finalTx}px)`;
    
    const displayAngle = Math.round(currentAngle);
    angleBadge.textContent = 'Azimuth: ' + displayAngle + '°';
    
    if (map) {
        map.setView(siteLocation, map.getZoom(), {animate: false}); // Mathematical enforcement of pivot center
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
    
    // HUD Logic
    const activeCfg = radioConfig[activeConfigStr] || [];
    let foundSector = null;
    
    for(let sector of activeCfg) {
        if (sector.azimuth === null || !sector.url) continue;
        const diff = Math.abs((((sector.azimuth - currentAngle) % 360) + 360) % 360);
        const dist = Math.min(diff, 360 - diff);
        if (dist <= 15) { // Show HUD when within 15 degrees
            foundSector = sector;
            break;
        }
    }
    
    if (foundSector) {
        sectorHudTitle.textContent = `${foundSector.name} (${foundSector.azimuth}°)`;
        if (sectorHudImg.src !== foundSector.url) sectorHudImg.src = foundSector.url;
        sectorHud.classList.add('visible');
    } else {
        sectorHud.classList.remove('visible');
    }
}

// Config Toggle
configSelect.addEventListener('change', (e) => {
    activeConfigStr = e.target.value;
    if (map) updateSectorMapPolygons();
    applyAngleToTrack();
});

function updateSectorMapPolygons() {
    if (!map) return;
    
    sectorPolygons.forEach(p => map.removeLayer(p));
    sectorPolygons = [];
    
    const config = radioConfig[activeConfigStr] || [];
    const distanceMeters = 80; 
    const fov = 60; // standard 60 deg antenna beamwidth visualization
    const centerPoint = siteLocation;
    
    config.forEach(sector => {
        if (sector.azimuth === null) return;
        const leftA = sector.azimuth - fov/2;
        const rightA = sector.azimuth + fov/2;
        
        let points = [centerPoint];
        for(let a=leftA; a<=rightA; a += 5) {
            points.push(destinationPoint(centerPoint[0], centerPoint[1], a, distanceMeters));
        }
        points.push(destinationPoint(centerPoint[0], centerPoint[1], rightA, distanceMeters));
        
        const poly = L.polygon(points, {
            color: '#ff3333',
            fillColor: '#ff3333',
            fillOpacity: 0.15,
            weight: 2,
            interactive: false
        }).addTo(map);
        sectorPolygons.push(poly);
    });
}

// --- MAP LOGIC ---

function initMap() {
    map = L.map('map', {
        zoomControl: false,
        dragging: false,            // Lock map panning to firmly keep the site dot perfectly anchored
        scrollWheelZoom: 'center'   // Force scroll zooms to anchor exactly on the center marker 
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
    updateSectorMapPolygons();
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
