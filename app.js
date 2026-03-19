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
const hudCgps = document.getElementById('hud-cgps');
const hudDirection = document.getElementById('hud-direction');
const hudAzimuts = document.getElementById('hud-azimuts');
const displaySiteName = document.getElementById('display-site-name');
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
            const rowObj = cellsData[r];
            
            let latCol = null, lngCol = null;
            const cols = Object.keys(rowObj).sort(); // Sort alphabetically A, B, C...
            for (let i = 0; i < cols.length; i++) {
                const cStr = cols[i];
                const cell = rowObj[cStr];
                if (cell.isStr && typeof cell.value === 'string') {
                    if (cell.value.includes('Lat:') && i + 1 < cols.length) latCol = cols[i + 1];
                    if (cell.value.includes('Long:') && i + 1 < cols.length) lngCol = cols[i + 1];
                }
            }
            
            cgpsLat = latCol ? parseFloat(rowObj[latCol].value) : null;
            cgpsLng = lngCol ? parseFloat(rowObj[lngCol].value) : null;
            
            // Moroccan Site Auto-Correction (Lat MUST be North/Positive, Lng MUST be West/Negative)
            if (cgpsLat !== null) cgpsLat = Math.abs(cgpsLat);
            if (cgpsLng !== null) cgpsLng = -Math.abs(cgpsLng);
            
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
        
        if (displaySiteName) {
            displaySiteName.textContent = file.name.replace('.xlsx', '');
        }
        
        if (hudAzimuts) {
            // we'll populate this properly inside the select change listener or defaults
        }
        
        if (cgpsLat !== null && cgpsLng !== null) {
            latInput.value = cgpsLat;
            lngInput.value = cgpsLng;
            if (hudCgps) hudCgps.textContent = `${cgpsLat.toFixed(6)}, ${cgpsLng.toFixed(6)}`;
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
    if (hudDirection) hudDirection.textContent = `${displayAngle}°`;
    
    if (map) {
        updateViewCone();
        
        // Rotate the map safely around the CGPS marker's accurate screen location
                const mapWrapper = document.getElementById('map-wrapper');
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
    
    let config = radioConfig[activeConfigStr] || [];
    
    if (config.length === 0) {
       config = [
           { name: 'Secteur 1', azimuth: 50 },
           { name: 'Secteur 2', azimuth: 130 },
           { name: 'Secteur 3', azimuth: 220 }
       ];
    }
    
    if (hudAzimuts) {
        const azList = config.map(c => c.azimuth).filter(a => a !== null);
        hudAzimuts.textContent = azList.length > 0 ? (azList.join('°/') + '°') : '--';
    }
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

// --- KMZ EXPORT (with embedded images) ---

document.getElementById('export-kml-btn').addEventListener('click', exportKMZ);

async function exportKMZ() {
    const btn = document.getElementById('export-kml-btn');
    btn.textContent = '⏳';
    btn.disabled = true;
    
    try {
        const siteName = displaySiteName ? displaySiteName.textContent : 'Unknown Site';
        const lat = siteLocation[0];
        const lng = siteLocation[1];
        
        let config = radioConfig[activeConfigStr] || [];
        if (config.length === 0) {
            config = [
                { name: 'Secteur 1', azimuth: 50 },
                { name: 'Secteur 2', azimuth: 130 },
                { name: 'Secteur 3', azimuth: 220 }
            ];
        }
        
        const distanceMeters = 80;
        const fov = 60;
        const kmzZip = new JSZip();
        const imgFolder = kmzZip.folder('images');
        
        // --- Helper: fetch blob URL as ArrayBuffer ---
        async function fetchBlobAsArrayBuffer(blobUrl) {
            const res = await fetch(blobUrl);
            return await res.arrayBuffer();
        }
        
        // --- 1. Package 12 panoramic photos ---
        let photoPlacemarks = '';
        for (let i = 0; i < photos.length; i++) {
            const photo = photos[i];
            if (!photo.url) continue;
            
            const angle = photo.angle !== undefined ? photo.angle : i * 30;
            const filename = `pano_${angle}.jpg`;
            
            try {
                const buf = await fetchBlobAsArrayBuffer(photo.url);
                imgFolder.file(filename, buf);
            } catch (e) {
                console.warn(`Could not package photo ${filename}:`, e);
                continue;
            }
            
            // Place a small marker at a short distance from site along the bearing
            const markerPt = destinationPoint(lat, lng, angle, 30);
            
            photoPlacemarks += `
    <Placemark>
      <name>${angle}°</name>
      <description><![CDATA[
        <h3>Panorama ${angle}°</h3>
        <img src="images/${filename}" width="400"/>
      ]]></description>
      <Style>
        <IconStyle>
          <scale>0.6</scale>
          <Icon><href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href></Icon>
        </IconStyle>
        <BalloonStyle><bgColor>ff1a1a2e</bgColor><textColor>ffffffff</textColor></BalloonStyle>
      </Style>
      <Point><coordinates>${markerPt[1]},${markerPt[0]},0</coordinates></Point>
    </Placemark>`;
        }
        
        // --- 2. Package 3 sector antenna photos + polygons ---
        let sectorPlacemarks = '';
        const sectorColors = ['ff0000ff', 'ff00ff00', 'ffff0000'];
        
        for (let i = 0; i < config.length; i++) {
            const sector = config[i];
            if (sector.azimuth === null) continue;
            
            const color = sectorColors[i % sectorColors.length];
            const leftA = sector.azimuth - fov / 2;
            const rightA = sector.azimuth + fov / 2;
            
            // Build polygon coordinates
            let coords = `${lng},${lat},0\n`;
            for (let a = leftA; a <= rightA; a += 5) {
                const pt = destinationPoint(lat, lng, a, distanceMeters);
                coords += `          ${pt[1]},${pt[0]},0\n`;
            }
            const lastPt = destinationPoint(lat, lng, rightA, distanceMeters);
            coords += `          ${lastPt[1]},${lastPt[0]},0\n`;
            coords += `          ${lng},${lat},0`;
            
            // Try to package sector antenna image
            let sectorImgTag = '';
            if (sector.url) {
                const sectorFilename = `sector_${i + 1}.jpg`;
                try {
                    const buf = await fetchBlobAsArrayBuffer(sector.url);
                    imgFolder.file(sectorFilename, buf);
                    sectorImgTag = `<br/><img src="images/${sectorFilename}" width="350"/>`;
                } catch (e) {
                    console.warn(`Could not package sector image ${i + 1}:`, e);
                }
            }
            
            sectorPlacemarks += `
    <Placemark>
      <name>${sector.name} (${sector.azimuth}°)</name>
      <description><![CDATA[
        <h3>${sector.name}</h3>
        <p>Azimuth: ${sector.azimuth}° | Beamwidth: ${fov}°</p>
        ${sectorImgTag}
      ]]></description>
      <Style>
        <LineStyle><color>${color}</color><width>2</width></LineStyle>
        <PolyStyle><color>40${color.substring(2)}</color></PolyStyle>
        <BalloonStyle><bgColor>ff1a1a2e</bgColor><textColor>ffffffff</textColor></BalloonStyle>
      </Style>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
          ${coords}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>`;
        }
        
        // --- 3. Assemble KML ---
        const azListStr = config.map(c => `${c.name}: ${c.azimuth}°`).join(' | ');
        
        const kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>${siteName} - Site Audit</name>
    <description><![CDATA[
      <h2>360° Telecom Site Audit</h2>
      <p>CGPS: ${lat.toFixed(6)}, ${lng.toFixed(6)}</p>
      <p>${azListStr}</p>
      <p>Config: ${activeConfigStr === 'avant' ? 'Avant Optimisation' : 'Après Optimisation'}</p>
    ]]></description>
    
    <Style id="siteIcon">
      <IconStyle>
        <color>ffff7700</color>
        <scale>1.3</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>
      </IconStyle>
      <LabelStyle><color>ffffffff</color><scale>1.1</scale></LabelStyle>
    </Style>
    
    <Folder>
      <name>Site</name>
      <Placemark>
        <name>${siteName}</name>
        <description><![CDATA[
          <p><b>CGPS:</b> ${lat.toFixed(6)}, ${lng.toFixed(6)}</p>
          <p><b>Azimuths:</b> ${config.map(c => c.azimuth + '°').join(' / ')}</p>
        ]]></description>
        <styleUrl>#siteIcon</styleUrl>
        <Point><coordinates>${lng},${lat},0</coordinates></Point>
      </Placemark>
    </Folder>
    
    <Folder>
      <name>Sectors (${activeConfigStr === 'avant' ? 'Avant' : 'Après'})</name>
      ${sectorPlacemarks}
    </Folder>
    
    <Folder>
      <name>360° Panoramic Photos</name>
      ${photoPlacemarks}
    </Folder>
  </Document>
</kml>`;
        
        // --- 4. Build KMZ ---
        kmzZip.file('doc.kml', kml);
        
        const kmzBlob = await kmzZip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
        const url = URL.createObjectURL(kmzBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${siteName.replace(/[^a-zA-Z0-9_-]/g, '_')}_audit.kmz`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
    } catch (err) {
        console.error('KMZ export failed:', err);
        alert('Export failed: ' + err.message);
    } finally {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> KML`;
        btn.disabled = false;
    }
}

// --- MAP LOGIC ---

function initMap() {
    map = L.map('map', {
        zoomControl: false,
        dragging: false,             // Locked to force custom rotational vector math
        scrollWheelZoom: 'center',   // Force scroll zooms to anchor exactly on origin
        touchZoom: 'center'          // Force mobile pinch-zoom to anchor on origin
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
    
    // Bind moving to continuously lock the flawless stateless tracking algorithm!
    map.on('move', applyAngleToTrack);
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

// --- Custom Rotation-Aware Map Panning ---
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
