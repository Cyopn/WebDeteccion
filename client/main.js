const fileInput = document.getElementById('fileInput');
const btnDetectImage = document.getElementById('btnDetectImage');
const btnDetectVideo = document.getElementById('btnDetectVideo');
const classSelectImage = document.getElementById('classSelect_image');
const classSelectVideo = document.getElementById('classSelect_video');
const classSelectCamera = document.getElementById('classSelect_camera');
const resultImage = document.getElementById('resultImage');
const resultVideo = document.getElementById('resultVideo');
const jsonOutImage = document.getElementById('jsonOut_image');
const jsonOutVideo = document.getElementById('jsonOut_video');
const jsonOutCamera = document.getElementById('jsonOut_camera');
const videoFileInput = document.getElementById('videoFile');
const cameraUrlInput = document.getElementById('cameraUrl');
const frameStepInput = document.getElementById('frameStep');
const maxFramesInput = document.getElementById('maxFrames');
const visualizeVideoInput = document.getElementById('visualizeVideo');
const timelineContainer = document.getElementById('timeline_container');
const timelineBar = document.getElementById('timeline_bar');
const timelineCursor = document.getElementById('timeline_cursor');
const timelineInfo = document.getElementById('timeline_info');
const confInputCamera = document.getElementById('confInput_camera');
const confInputImage = document.getElementById('conf_image');
const confInputVideo = document.getElementById('conf_video');
const confInputCameraNew = document.getElementById('conf_camera');
const btnStartLive = document.getElementById('btnStartLive');
const btnStopLive = document.getElementById('btnStopLive');
const liveView = document.getElementById('liveView');
const btnToggleRawLogs = document.getElementById('btnToggleRawLogs');
let showRawCameraLogs = false;
const tabButtons = Array.from(document.querySelectorAll('.tabbtn'));
let _logPollId = null;
const _lastStatus = { image: null, video: null, camera: null };

const COCO_CLASSES = [
    'sin etiqueta', 'persona', 'bicicleta', 'automovil', 'motocicleta', 'avion',
    'autobus', 'tren', 'camion', 'barco', 'semaforo', 'boca de incendio', 'street sign',
    'señal de alto', 'parquimetro', 'banco', 'pájaro', 'gato', 'perro', 'caballo',
    'oveja', 'vaca', 'elefante', 'oso', 'zebra', 'girafa', 'sombrero', 'mochila',
    'paraguas', 'zapato', 'lentes', 'bolso de mano', 'tie', 'maleta', 'disco volador',
    'esquís', 'patineta de nieve', 'pelota de deportes', 'cometa', 'bat de béisbol',
    'guante de béisbol', 'patineta', 'tabla de surf', 'raqueta de tennis', 'botella',
    'lámina', 'copa de vino', 'taza', 'tenedor', 'cuchillo', 'cuchara', 'bol',
    'platano', 'manzana', 'sandwich', 'naranja', 'brocoli', 'zanahoria', 'hot dog',
    'pizza', 'dona', 'pastel', 'silla', 'sofá', 'planta en maceta', 'cama', 'espejo',
    'comedor', 'ventana', 'escritorio', 'baño', 'puerta', 'televisión', 'laptop',
    'mouse', 'remoto', 'teclado', 'celular', 'microondas', 'horno', 'tostador',
    'sink', 'refrigerador', 'licuadora', 'libro', 'reloj', 'florero', 'tijeras',
    'oso de peluche', 'secador de pelo', 'cepillo de dientes', 'cepillo de pelo'
];

function getClassIndex(className) {
    if (className === 'all') return -1;
    const index = COCO_CLASSES.indexOf(className);
    return index >= 0 ? index : -1;
}

function getStatusElement(tab) {
    if (!tab) return null;
    const id = `status_${tab}`;
    return document.getElementById(id);
}

function renderDetectionsToElement(elId, detections) {
    const el = document.getElementById(elId);
    if (!el) return;
    if (!detections || !detections.length) {
        el.textContent = '(sin detecciones)';
        return;
    }
    const lines = detections.map((d, i) => {
        const label = d.label || d.class || d.class_name || d.class_id || 'obj';
        const conf = (typeof d.confidence !== 'undefined') ? d.confidence : (d.score || d.conf || 0);
        const bbox = (d.x !== undefined && d.y !== undefined) ? ` [x:${Math.round(d.x)},y:${Math.round(d.y)},w:${Math.round(d.w || d.width || 0)},h:${Math.round(d.h || d.height || 0)}]` : '';
        return `${i + 1}. ${label} (${(conf * 100).toFixed(1)}%)${bbox}`;
    });
    el.textContent = lines.join('\n');
}

function renderVideoSampleToElement(elId, sample, fps) {
    const el = document.getElementById(elId);
    if (!el) return;
    if (!sample || !sample.length) { el.textContent = '(sin muestras)'; return; }
    el.innerHTML = '';
    el.style.cursor = 'default';
    const personMap = {};
    for (let s of sample) {
        if (!s.detections || !s.detections.length) continue;
        const persons = s.detections.filter(d => {
            const label = String(d.label || d.class || '').toLowerCase();
            return label === 'persona' || label === 'person';
        });

        if (persons.length === 0) continue;

        const frame = s.frame || s.index || s.f || 0;
        const timestamp = s.timestamp || (fps ? frame / fps : 0);

        for (let p of persons) {
            const personId = (p.person_id !== undefined && p.person_id !== null) ? p.person_id : 'desconocido';
            const personColor = p.person_color || [100, 100, 100];
            const conf = (typeof p.confidence !== 'undefined') ? p.confidence : (p.score || p.conf || 0);
            if (!personMap[personId]) {
                personMap[personId] = { color: personColor, segments: [] };
            }
            const person = personMap[personId];
            const lastSeg = person.segments[person.segments.length - 1];
            if (lastSeg && timestamp - lastSeg.end <= 1.0) {
                lastSeg.end = timestamp;
                lastSeg.frames += 1;
                lastSeg.totalConf += conf;
            } else {
                person.segments.push({
                    start: timestamp,
                    end: timestamp,
                    firstTimestamp: timestamp,
                    frames: 1,
                    totalConf: conf
                });
            }
        }
    }

    let personCount = 0;
    Object.entries(personMap).forEach(([pid, info]) => {
        personCount++;
        const bgr = info.color;
        const rgb = [bgr[2], bgr[1], bgr[0]];
        const colorStr = `rgb(${rgb.join(',')})`;

        const totalDuration = info.segments.reduce((sum, seg) => sum + (seg.end - seg.start), 0);
        const totalFrames = info.segments.reduce((sum, seg) => sum + seg.frames, 0);
        const totalConf = info.segments.reduce((sum, seg) => sum + seg.totalConf, 0);
        const avgConf = totalConf / totalFrames;

        const timeRanges = info.segments.map(seg =>
            seg.start === seg.end ? `${seg.start.toFixed(1)}s` : `${seg.start.toFixed(1)}-${seg.end.toFixed(1)}s`
        ).join(', ');

        const detectionDiv = document.createElement('div');
        detectionDiv.style.padding = '12px';
        detectionDiv.style.margin = '6px 0';
        detectionDiv.style.cursor = 'pointer';
        detectionDiv.style.borderRadius = '6px';
        detectionDiv.style.background = '#1a2530';
        detectionDiv.style.borderLeft = `6px solid ${colorStr}`;
        detectionDiv.style.transition = 'all 0.2s';

        const personIdHtml = `<div style="
            background: ${colorStr};
            color: white;
            padding: 6px 12px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 14px;
            display: inline-block;
            margin-bottom: 6px;
        ">#${pid}</div>`;

        const statsHtml = `<div style="color: #aaa; font-size: 13px; margin-top: 4px;">
            <strong>${totalFrames}</strong> detecciones en <strong>${totalDuration.toFixed(1)}s</strong>
            — Confianza promedio: <strong>${(avgConf * 100).toFixed(1)}%</strong>
        </div>`;

        const rangesHtml = `<div style="color: #888; font-size: 12px; margin-top: 4px;">
            Rangos: ${timeRanges}
        </div>`;

        detectionDiv.innerHTML = personIdHtml + statsHtml + rangesHtml;

        detectionDiv.onmouseenter = () => {
            detectionDiv.style.background = '#2a3540';
        };
        detectionDiv.onmouseleave = () => {
            detectionDiv.style.background = '#1a2530';
        };

        detectionDiv.onclick = () => {
            const video = document.getElementById('resultVideo');
            if (video && info.segments.length > 0) {
                video.currentTime = info.segments[0].firstTimestamp;
                if (video.paused) video.play();
                detectionDiv.style.background = '#0066cc';
                setTimeout(() => {
                    detectionDiv.style.background = '#1a2530';
                }, 500);
            }
        };

        el.appendChild(detectionDiv);
    });

    if (personCount === 0) {
        el.textContent = '(sin detecciones)';
    }
}


function renderTimeline(metadata) {
    if (!timelineContainer || !timelineBar || !metadata) return;

    const existingHighlights = timelineBar.querySelectorAll('.timeline-highlight');
    existingHighlights.forEach(h => h.remove());

    const duration = metadata.duration || 1;
    const detections = metadata.detections || [];

    if (detections.length === 0) {
        if (timelineInfo) timelineInfo.textContent = 'No se detectaron objetos en el video.';
        return;
    }

    const personMap = {};

    const ensurePerson = (pid, colorBgr) => {
        if (!personMap[pid]) {
            personMap[pid] = {
                color: colorBgr || [0, 200, 255],
                segments: [],
            };
        }
        return personMap[pid];
    };

    for (let det of detections) {
        const ts = det.timestamp || (det.frame / (metadata.fps || 25));
        const dets = det.detections || [];
        const persons = dets.filter(d => {
            const label = String(d.label || d.class || '').toLowerCase();
            return label === 'persona' || label === 'person';
        });

        if (persons.length === 0) {
            const p = ensurePerson('desconocido', [120, 120, 120]);
            pushSegment(p, ts, det.count || persons.length || 1);
            continue;
        }

        for (let p of persons) {
            const pid = (p.person_id !== undefined && p.person_id !== null) ? `ID#${p.person_id}` : 'desconocido';
            const colorBgr = Array.isArray(p.person_color) && p.person_color.length === 3 ? p.person_color : [0, 200, 255];
            const person = ensurePerson(pid, colorBgr);
            pushSegment(person, ts, 1);
        }
    }

    function pushSegment(person, ts, count) {
        const segments = person.segments;
        const last = segments[segments.length - 1];
        if (last && ts - last.end <= 1.0) {
            last.end = ts;
            last.count += count;
        } else {
            segments.push({ start: ts, end: ts, count: count });
        }
    }

    let totalSegments = 0;
    Object.entries(personMap).forEach(([pid, info]) => {
        const colorBgr = info.color || [0, 200, 255];
        const colorCss = `rgb(${colorBgr[2]}, ${colorBgr[1]}, ${colorBgr[0]})`;
        info.segments.forEach(seg => {
            totalSegments += 1;
            const startPercent = (seg.start / duration) * 100;
            const widthPercent = ((seg.end - seg.start + 0.5) / duration) * 100;

            const highlight = document.createElement('div');
            highlight.className = 'timeline-highlight';
            highlight.style.left = `${startPercent}%`;
            highlight.style.width = `${Math.max(widthPercent, 0.5)}%`;
            highlight.style.background = colorCss;
            highlight.style.opacity = '0.75';
            highlight.title = `${pid} — ${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s (${seg.count} detecciones)`;
            timelineBar.appendChild(highlight);
        });
    });

    if (timelineInfo) {
        const personsCount = Object.keys(personMap).length;
        timelineInfo.textContent = `Total: ${metadata.total_detections} detecciones, ${personsCount} objetos, ${totalSegments} segmentos`;
    }

    timelineContainer.classList.remove('hidden');
}

function syncTimelineCursor(video) {
    if (!video || !timelineCursor || !video.duration) return;
    const percent = (video.currentTime / video.duration) * 100;
    timelineCursor.style.left = `${percent}%`;
}

function setupTimelineClick(video, metadata) {
    if (!timelineBar || !video || !metadata) return;
    timelineBar.addEventListener('click', (e) => {
        const rect = timelineBar.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percent = clickX / rect.width;
        const duration = metadata.duration || video.duration || 1;
        const targetTime = percent * duration;
        video.currentTime = targetTime;
    });
}

function setStatus(text, tab) {
    const el = getStatusElement(tab);
    const key = tab || 'global';
    if (_lastStatus.hasOwnProperty(key) && _lastStatus[key] === text) return;
    _lastStatus[key] = text;
    if (el) {
        const lower = String(text || '').toLowerCase();
        el.textContent = text;
        el.classList.remove('error', 'ok');
        if (lower.includes('error') || lower.includes('no existe') || lower.includes('no encontrado') || lower.includes('no se')) {
            el.classList.add('error');
        } else if (lower.includes('listo') || lower.includes('iniciado') || lower.includes('ok')) {
            el.classList.add('ok');
        }
    } else {
        console.log('[status]', key, text);
    }
}

function friendlyCameraMessageFromLine(line) {
    const L = String(line || '').toLowerCase();
    if (L.includes('connection refused')) return 'Conexión rechazada (verifica IP/puerto)';
    if (L.includes('connection to') && L.includes('failed')) return 'Conexión fallida a la cámara';
    if (L.includes('no se pudo abrir')) return 'No se pudo abrir la fuente de video';
    if (L.includes('overread')) return 'Stream corrupto o invalido (overread)';
    if (L.includes('mjpeg') && L.includes('overread')) return 'Stream MJPEG corrupto';
    if (L.includes('tcp @') && L.includes('failed')) return 'Fallo en conexión TCP';
    return line;
}

const API_BASE = (function () {
    if (typeof window.API_BASE === 'string' && window.API_BASE.length) return window.API_BASE.replace(/\/$/, '');
    const qp = new URLSearchParams(window.location.search).get('api');
    if (qp) return qp.replace(/\/$/, '');
    return 'http://127.0.0.1:5501';
})();

function activateTab(tabId) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tabbtn').forEach(b => b.classList.remove('active'));
    const panel = document.getElementById(tabId);
    const btn = document.querySelector(`.tabbtn[data-tab="${tabId}"]`);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');

    const mapping = {
        'tab-image': 'result-panel-image',
        'tab-video': 'result-panel-video',
        'tab-camera': 'result-panel-camera'
    };
    document.querySelectorAll('.result-panel').forEach(p => p.classList.remove('active'));
    const rid = mapping[tabId];
    if (rid) {
        const rp = document.getElementById(rid);
        if (rp) rp.classList.add('active');
    }
    const key = tabIdToKey(tabId);
    if (key) clearLogsForKey(key);
}

function tabIdToKey(tabId) {
    if (!tabId) return null;
    if (tabId === 'tab-image') return 'image';
    if (tabId === 'tab-video') return 'video';
    if (tabId === 'tab-camera') return 'camera';
    return null;
}

function clearLogsForKey(key) {
    if (!key) return;
    const statusEl = getStatusElement(key);
    if (statusEl) {
        statusEl.textContent = '';
        statusEl.classList.remove('error', 'ok');
    }
    if (_lastStatus.hasOwnProperty(key)) _lastStatus[key] = null;
    if (key === 'image' && jsonOutImage) jsonOutImage.textContent = '';
    if (key === 'video' && jsonOutVideo) jsonOutVideo.textContent = '';
    if (key === 'camera' && jsonOutCamera) jsonOutCamera.textContent = '';
}

function clearAllLogs() {
    ['image', 'video', 'camera'].forEach(k => clearLogsForKey(k));
}

tabButtons.forEach(b => b.addEventListener('click', (e) => {
    const t = e.currentTarget.getAttribute('data-tab');
    activateTab(t);
}));

if (btnDetectImage) {
    btnDetectImage.addEventListener('click', async () => {
        const file = fileInput && fileInput.files ? fileInput.files[0] : null;
        const conf = confInputImage ? confInputImage.value : '0.4';
        const classIndex = classSelectImage ? getClassIndex(classSelectImage.value) : -1;
        if (!file) { setStatus('Selecciona primero una imagen.', 'image'); return; }
        setStatus('Enviando imagen...', 'image');
        try {
            const form = new FormData(); form.append('image', file);
            let url = `${API_BASE}/detect/image?visualize=true&conf=${encodeURIComponent(conf)}`;
            if (classIndex >= 0) url += `&class_id=${classIndex}`;
            const res = await fetch(url, { method: 'POST', body: form });
            if (!res.ok) { const txt = await res.text(); setStatus(`Error: ${res.status} - ${txt}`, 'image'); return; }
            const contentType = res.headers.get('content-type') || '';
            if (contentType.includes('image')) {
                const blob = await res.blob();
                if (resultImage) { resultImage.style.display = 'block'; resultImage.src = URL.createObjectURL(blob); }
                if (resultVideo) resultVideo.style.display = 'none';
                if (jsonOutImage) jsonOutImage.textContent = 'Imagen retornada con cajas (visualize=true)';
                setStatus('Listo', 'image');
                try { pollLogsOnce(); } catch (e) { }
                try {
                    const form2 = new FormData(); form2.append('image', file);
                    let url2 = `${API_BASE}/detect/image?visualize=false&conf=${encodeURIComponent(conf)}`;
                    if (classIndex >= 0) url2 += `&class_id=${classIndex}`;
                    const r2 = await fetch(url2, { method: 'POST', body: form2 });
                    if (r2.ok) {
                        const j2 = await r2.json();
                        if (j2 && j2.detections) renderDetectionsToElement('log_image', j2.detections);
                        if (j2) {
                            const cnt = j2.count || (j2.detections && j2.detections.length) || 0;
                            const t = j2.elapsed_seconds ? `${j2.elapsed_seconds.toFixed(2)}s` : '';
                            setStatus(`Detección de imagen OK (count=${cnt} time=${t})`, 'image');
                        }
                    }
                } catch (e) { }
            } else {
                const j = await res.json();
                const timeStr = j.elapsed_seconds ? ` (${j.elapsed_seconds.toFixed(2)}s)` : '';
                if (jsonOutImage) jsonOutImage.textContent = JSON.stringify(j, null, 2);
                const cnt = j.count || (j.detections && j.detections.length) || 0;
                setStatus(`Detección de imagen OK (count=${cnt} time=${timeStr.trim()})`, 'image');
                try { pollLogsOnce(); } catch (e) { }
                try { if (j && j.detections) renderDetectionsToElement('log_image', j.detections); } catch (e) { }
            }
        } catch (err) { setStatus('Error: ' + (err.message || err), 'image'); }
    });
}

if (btnDetectVideo) {
    btnDetectVideo.addEventListener('click', async () => {
        const vidFile = videoFileInput && videoFileInput.files ? videoFileInput.files[0] : null;
        const frameStep = frameStepInput ? frameStepInput.value : '1';
        const conf = confInputVideo ? confInputVideo.value : '0.4';
        const maxFrames = maxFramesInput ? maxFramesInput.value : '0';
        const showRectangles = visualizeVideoInput ? visualizeVideoInput.checked : false;
        const classIndex = classSelectVideo ? getClassIndex(classSelectVideo.value) : -1;
        if (!vidFile) { setStatus('Selecciona primero un archivo de video.', 'video'); return; }

        if (timelineContainer) timelineContainer.classList.add('hidden');

        setStatus('Enviando video...', 'video');
        try {
            const form = new FormData();
            form.append('video', vidFile);
            form.append('frame_step', frameStep);
            form.append('max_frames', maxFrames);
            form.append('conf', conf);
            if (classIndex >= 0) form.append('class_id', classIndex);

            form.append('timeline', 'true');

            if (showRectangles) {
                form.append('visualize', 'true');
                form.append('transcode', '1');
            }

            const res = await fetch(`${API_BASE}/detect/video`, { method: 'POST', body: form });
            if (!res.ok) { const txt = await res.text(); setStatus(`Error: ${res.status} - ${txt}`, 'video'); return; }

            const contentType = res.headers.get('content-type') || '';
            const contentDisp = res.headers.get('content-disposition') || '';
            const isLikelyVideo = contentType.includes('video') || contentType.includes('application/octet-stream') || /filename=.*\.(mp4|avi|mov|mkv)/i.test(contentDisp);

            if (contentType.includes('application/json')) {
                const j = await res.json();
                console.log('JSON response received:', {
                    has_timeline: !!j.timeline,
                    has_video_data: !!j.video_data,
                    video_data_size: j.video_data ? j.video_data.length : 0,
                    video_mime: j.video_mime,
                    total_detections: j.timeline?.total_detections,
                    frames: j.frames_processed
                });

                if (j.timeline) {
                    const metadata = j.timeline;
                    console.log('Timeline metadata:', metadata);

                    if (j.video_data) {
                        try {
                            console.log('Decoding base64 video...');
                            const videoBlob = await (async () => {
                                const byteCharacters = atob(j.video_data);
                                const byteNumbers = new Array(byteCharacters.length);
                                for (let i = 0; i < byteCharacters.length; i++) {
                                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                                }
                                const byteArray = new Uint8Array(byteNumbers);
                                return new Blob([byteArray], { type: j.video_mime || 'video/mp4' });
                            })();

                            console.log('Blob created:', { size: videoBlob.size, type: videoBlob.type });

                            if (resultVideo) {
                                resultVideo.style.display = 'block';
                                const url = URL.createObjectURL(videoBlob);
                                console.log('Object URL created:', url);
                                resultVideo.src = url;
                                resultVideo.load();
                                console.log('Video element load() called');
                                resultVideo.onloadedmetadata = () => {
                                    console.log('Video loadedmetadata event fired');
                                    try { resultVideo.play().catch(() => { }); } catch (e) { }

                                    renderTimeline(metadata);
                                    setupTimelineClick(resultVideo, metadata);
                                    resultVideo.addEventListener('timeupdate', () => syncTimelineCursor(resultVideo));
                                };
                                if (resultImage) resultImage.style.display = 'none';
                            }

                            if (jsonOutVideo) {
                                jsonOutVideo.textContent = `Video con ${metadata.total_detections} detecciones`;
                            }

                            if (metadata.detections) {
                                renderVideoSampleToElement('log_video', metadata.detections, metadata.fps);
                            }

                            const timeStr = j.elapsed_seconds ? ` (${j.elapsed_seconds.toFixed(2)}s)` : '';
                            const total = metadata.total_detections || j.total_detections || 0;
                            setStatus(`Detección de video OK (total=${total} time=${j.elapsed_seconds ? j.elapsed_seconds.toFixed(2) + 's' : ''})`, 'video');
                            try { pollLogsOnce(); } catch (e) { }
                        } catch (e) {
                            console.error('Error decodificando video base64:', e);
                            console.error('Stack:', e.stack);
                            setStatus('Error decodificando video: ' + e.message, 'video');
                        }
                    } else {
                        console.warn('Respuesta timeline sin video_data:', j);
                        console.warn('Campos disponibles:', Object.keys(j));
                        if (jsonOutVideo) jsonOutVideo.textContent = JSON.stringify(j, null, 2);
                        setStatus('Metadata de detecciones recibida (video no disponible)', 'video');

                        if (metadata.detections) {
                            renderVideoSampleToElement('log_video', metadata.detections, metadata.fps);
                        }
                    }
                } else {
                    if (jsonOutVideo) jsonOutVideo.textContent = JSON.stringify(j, null, 2);
                    const sampleCount = (j.sample && j.sample.reduce((s, f) => s + (f.count || 0), 0)) || j.total_detections || 0;
                    setStatus(`Detección de video OK (total=${sampleCount} time=${j.elapsed_seconds ? j.elapsed_seconds.toFixed(2) + 's' : ''})`, 'video');
                    try { pollLogsOnce(); } catch (e) { }
                    try { if (j && j.sample) renderVideoSampleToElement('log_video', j.sample, j.fps); } catch (e) { }
                }
            } else if (isLikelyVideo) {
                const blob = await res.blob();

                if (resultVideo) {
                    resultVideo.style.display = 'block';
                    const url = URL.createObjectURL(blob);
                    resultVideo.src = url;
                    resultVideo.load();
                    resultVideo.onloadedmetadata = () => {
                        try { resultVideo.play().catch(() => { }); } catch (e) { }
                    };

                    setTimeout(() => {
                        const downloadContainer = document.getElementById('video_download');
                        if (resultVideo && (resultVideo.readyState < 2 || resultVideo.error)) {
                            try {
                                if (downloadContainer) {
                                    downloadContainer.innerHTML = '';
                                    const a = document.createElement('a');
                                    a.href = url;
                                    a.download = 'detections_video';
                                    a.textContent = 'Descargar video procesado (si el navegador no reproduce)';
                                    a.className = 'tabbtn';
                                    downloadContainer.appendChild(a);
                                }
                                setStatus('El navegador no puede reproducir este formato; puedes descargarlo.', 'video');
                            } catch (e) { }
                        } else {
                            if (downloadContainer) downloadContainer.innerHTML = '';
                        }
                    }, 1500);
                    if (resultImage) resultImage.style.display = 'none';
                }

                if (jsonOutVideo) jsonOutVideo.textContent = 'Video procesado descargado (rectangulos mostrados)';
                setStatus('Listo', 'video');
                try { pollLogsOnce(); } catch (e) { }

                try {
                    const form2 = new FormData();
                    form2.append('video', vidFile);
                    form2.append('model', model);
                    form2.append('frame_step', frameStep);
                    form2.append('conf', conf);
                    const fetchMaxFrames = (parseInt(maxFrames, 10) > 0) ? maxFrames : '50';
                    form2.append('max_frames', fetchMaxFrames);
                    const r2 = await fetch(`${API_BASE}/detect/video?visualize=false`, { method: 'POST', body: form2 });
                    if (r2.ok) {
                        const j2 = await r2.json();
                        if (j2 && j2.sample) renderVideoSampleToElement('log_video', j2.sample, j2.fps);
                    }
                } catch (e) { }
            }
        } catch (err) { setStatus('Error: ' + (err.message || err), 'video'); }
    });
}

function startLiveView() {
    const camUrl = cameraUrlInput && cameraUrlInput.value ? cameraUrlInput.value.trim() : '';
    const frameStep = frameStepInput ? frameStepInput.value : '1';
    const conf = confInputCameraNew ? parseFloat(confInputCameraNew.value) : 0.4;
    const classIndex = classSelectCamera ? getClassIndex(classSelectCamera.value) : -1;
    if (!camUrl) { setStatus('Proporciona la URL de la cámara para vista en vivo.', 'camera'); return; }
    try {
        const parsed = new URL(camUrl);
        const scheme = parsed.protocol.replace(':', '').toLowerCase();
        if (!['http', 'https', 'rtsp'].includes(scheme)) {
            setStatus(`URL de cámara inválida (esquema ${parsed.protocol})`, 'camera');
            return;
        }
        if (!parsed.hostname) {
            setStatus('URL de cámara inválida (host ausente)', 'camera');
            return;
        }
    } catch (err) {
        setStatus('URL de cámara inválida', 'camera');
        return;
    }
    let streamUrl = `${API_BASE}/stream/video?camera_url=${encodeURIComponent(camUrl)}&frame_step=${encodeURIComponent(frameStep)}&conf=${encodeURIComponent(conf)}`;
    if (classIndex >= 0) streamUrl += `&class_id=${classIndex}`;
    if (liveView) { liveView.src = streamUrl; liveView.style.display = 'block'; }
    if (resultImage) resultImage.style.display = 'none';
    if (resultVideo) resultVideo.style.display = 'none';
    if (jsonOutCamera) jsonOutCamera.textContent = '';
    if (btnStartLive) btnStartLive.disabled = true;
    if (btnStopLive) btnStopLive.disabled = false;
    setStatus('Vista en vivo iniciado (MJPEG)', 'camera');
}

function stopLiveView() {
    if (liveView) { try { liveView.src = ''; } catch (e) { } liveView.style.display = 'none'; }
    if (btnStartLive) btnStartLive.disabled = false;
    if (btnStopLive) btnStopLive.disabled = true;
    if (jsonOutCamera) jsonOutCamera.textContent = 'Vista en vivo detenido';
    setStatus('Vista en vivo detenido', 'camera');
}

if (btnStartLive) btnStartLive.addEventListener('click', startLiveView);
if (btnStopLive) btnStopLive.addEventListener('click', stopLiveView);


async function fetchLogs(tail = 200) {
    try {
        const res = await fetch(`${API_BASE}/logs`);
        if (!res.ok) return [];
        const j = await res.json();
        return j.lines || [];
    } catch (e) {
        return [];
    }
}

function analyzeLogs(lines) {
    let lastCamera = null;
    let lastImage = null;
    let lastVideo = null;
    let globalError = null;
    const cameraLines = [];
    let lastCriticalCameraLine = null;
    const imageLines = [];
    const videoLines = [];

    const httpReqRe = /"(GET|POST) ([^ ]+) HTTP\/[0-9.]+"\s+(\d{3})/i;
    const responseRe = /^RESPONSE\s+path=([^\s]+)\s+status=(\d+)(.*)$/i;

    for (let i = lines.length - 1; i >= 0; --i) {
        const L = String(lines[i] || '');
        const lower = L.toLowerCase();

        const rm = L.match(responseRe);
        if (rm) {
            const path = rm[1];
            const code = parseInt(rm[2], 10);
            const rest = rm[3] || '';
            if (path.includes('/detect/image')) {
                if (code >= 200 && code < 300) lastImage = `Detección de imagen OK (${rest.trim()})`;
                else lastImage = `Error en detección de imagen (HTTP ${code}) ${rest.trim()}`;
                imageLines.push(L);
            } else if (path.includes('/detect/video')) {
                if (code >= 200 && code < 300) lastVideo = `Detección de video OK (${rest.trim()})`;
                else lastVideo = `Error en detección de video (HTTP ${code}) ${rest.trim()}`;
                videoLines.push(L);
            } else {
                globalError = globalError || (`Server response: ${path} ${code}`);
            }
            continue;
        }

        const m = L.match(httpReqRe);
        if (m) {
            const method = m[1];
            const path = m[2];
            const code = parseInt(m[3], 10);
            if (path.includes('/stream/video')) {
                if (code === 200) {
                    lastCamera = `Stream OK (HTTP ${code})`;
                } else {
                    const camMatch = path.match(/camera_url=([^&]+)/i);
                    if (camMatch) {
                        try {
                            const raw = decodeURIComponent(camMatch[1]);
                            const low = raw.toLowerCase();
                            if (!(low.startsWith('http://') || low.startsWith('https://') || low.startsWith('rtsp://'))) {
                                lastCamera = `URL de cámara inválida: ${raw}`;
                            } else {
                                lastCamera = `No se recibe video en camera_url (HTTP ${code})`;
                            }
                        } catch (e) {
                            lastCamera = `No se recibe video en camera_url (HTTP ${code})`;
                        }
                    } else {
                        lastCamera = `No se recibe video en camera_url (HTTP ${code})`;
                    }
                }
                cameraLines.push(L);
                if (!lastCriticalCameraLine && code >= 500) lastCriticalCameraLine = L;
            }
            if (path.includes('/detect/image')) {
                if (code >= 200 && code < 300) lastImage = `Detección de imagen OK (HTTP ${code})`;
                else lastImage = `Error en detección de imagen (HTTP ${code})`;
                imageLines.push(L);
            }
            if (path.includes('/detect/video')) {
                if (code >= 200 && code < 300) lastVideo = `Detección de video OK (HTTP ${code})`;
                else lastVideo = `Error en detección de video (HTTP ${code})`;
                videoLines.push(L);
            }
            if (!globalError && code >= 500) globalError = `Server error (HTTP ${code})`;
        }

        if (!lastCamera && (lower.includes('connection refused') || lower.includes('connection to') || lower.includes('overread') || lower.includes('mjpeg') || lower.includes('tcp @') || lower.includes('ffmpeg') || lower.includes('videoio') || lower.includes('no se pudo abrir'))) {
            lastCamera = friendlyCameraMessageFromLine(L);
            cameraLines.push(L);
            if (!lastCriticalCameraLine) lastCriticalCameraLine = L;
        }

        if (!lastImage && lower.includes('/detect/image')) { lastImage = L; imageLines.push(L); }
        if (!lastVideo && lower.includes('/detect/video')) { lastVideo = L; videoLines.push(L); }

        if (!globalError && (lower.includes('error') || lower.includes('failed') || lower.includes('refused') || lower.includes('500'))) {
            globalError = L;
        }

        if (lastCamera && lastImage && lastVideo && globalError) break;
    }

    return { lastCamera, lastImage, lastVideo, globalError, cameraLines, imageLines, videoLines, lastCriticalCameraLine };
}

async function pollLogsOnce() {
    const lines = await fetchLogs(300);
    if (!lines || !lines.length) return;
    const { lastCamera, lastImage, lastVideo, globalError, cameraLines, imageLines, videoLines, lastCriticalCameraLine } = analyzeLogs(lines);
    if (lastCamera) setStatus(lastCamera, 'camera');
    if (lastImage) setStatus(lastImage, 'image');
    if (lastVideo) setStatus(lastVideo, 'video');
    if (jsonOutCamera) {
        if (showRawCameraLogs) {
            jsonOutCamera.textContent = (cameraLines && cameraLines.length) ? cameraLines.slice(-50).join('\n') : '';
            jsonOutCamera.classList.remove('hidden');
        } else {
            jsonOutCamera.classList.add('hidden');
        }
    }
    if (jsonOutImage) jsonOutImage.textContent = (imageLines && imageLines.length) ? imageLines.slice(-50).join('\n') : jsonOutImage.textContent;
    if (jsonOutVideo) jsonOutVideo.textContent = (videoLines && videoLines.length) ? videoLines.slice(-50).join('\n') : jsonOutVideo.textContent;

    if (globalError) {
        if (globalError && !lastCamera) setStatus(globalError, 'camera');
        if (globalError && !lastVideo) setStatus(globalError, 'video');
        if (globalError && !lastImage) setStatus(globalError, 'image');
    }
    const camEx = document.getElementById('camera_excerpt');
    if (camEx) {
        if (lastCriticalCameraLine) {
            const friendly = friendlyCameraMessageFromLine(lastCriticalCameraLine);
            camEx.textContent = String(friendly || lastCriticalCameraLine);
            camEx.classList.add('log-excerpt');
        } else {
            camEx.textContent = '';
            camEx.classList.remove('log-excerpt');
        }
    }
    try {
        const logCamEl = document.getElementById('log_camera');
        if (logCamEl) {
            if (cameraLines && cameraLines.length) {
                logCamEl.textContent = cameraLines.slice(-30).join('\n');
            } else if (lastCamera) {
                logCamEl.textContent = String(lastCamera);
            } else {
                logCamEl.textContent = '(sin datos)';
            }
        }
    } catch (e) { }
}

function startLogPolling(intervalMs = 2000) {
    if (_logPollId) return;
    pollLogsOnce();
    _logPollId = setInterval(pollLogsOnce, intervalMs);
}

function stopLogPolling() {
    if (_logPollId) { clearInterval(_logPollId); _logPollId = null; }
}

window.addEventListener('load', () => {
    clearAllLogs();
});

if (btnToggleRawLogs) {
    btnToggleRawLogs.addEventListener('click', (e) => {
        showRawCameraLogs = !showRawCameraLogs;
        if (showRawCameraLogs) {
            btnToggleRawLogs.textContent = 'Ocultar logs crudos';
            if (jsonOutCamera) jsonOutCamera.classList.remove('hidden');
        } else {
            btnToggleRawLogs.textContent = 'Mostrar logs crudos';
            if (jsonOutCamera) jsonOutCamera.classList.add('hidden');
        }
        if (showRawCameraLogs) pollLogsOnce();
    });
}

window.addEventListener('beforeunload', () => { stopLogPolling(); });
