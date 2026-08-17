/* ============================================================
   FenoFresa · Motor TFJS + COCO-SSD + Bounding Boxes
   Todo corre local en el teléfono (GPU/WebGL).
   ============================================================ */

import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// ---------- CONSTANTES ----------
const DB_NAME = 'fenofresa';
const STORE = 'registros';
let _db = null;

// Mapeo de clases COCO → estadios Fresa (demo)
// ¡CAMBIA ESTO CUANDO ENTRENES TU PROPIO MODELO!
const CLASS_MAP = {
  'apple': 'rojo',
  'orange': 'envero',    // naranja → envero (parecido al rosado)
  'banana': 'blanco',    // amarillo pálido → blanco
  'broccoli': 'verde',   // brócoli → verde
  'carrot': 'verde',     // zanahoria (tallo) → verde
  'flower': 'flor',      // flor genérica
  'potted plant': 'verde',
  'vase': 'verde'
};

// Para conteo manual
const ESTADIOS = ['verde', 'blanco', 'envero', 'rojo', 'flores', 'botones'];

// ---------- UTILERÍAS ----------
const $ = id => document.getElementById(id);
function uuid() { return crypto.randomUUID?.() || `id-${Date.now()}-${Math.random().toString(16).slice(2)}`; }
function pct(a, b) { return b ? Math.round((a / b) * 1000) / 10 : 0; }
function stamp() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

// ---------- INDEXEDDB ----------
function openDB() {
  return new Promise((res, rej) => {
    if (_db) return res(_db);
    const req = indexedDB.open(DB_NAME, 2);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' });
        store.createIndex('fecha', 'fecha_iso', { unique: false });
      }
    };
    req.onsuccess = () => { _db = req.result; res(_db); };
    req.onerror = () => rej(req.error);
  });
}
async function dbAdd(rec) {
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(rec);
    tx.oncomplete = res; tx.onerror = () => rej(tx.error);
  });
}
async function dbAll() {
  const db = await openDB();
  return new Promise((res, rej) => {
    const out = [];
    const tx = db.transaction(STORE, 'readonly');
    tx.objectStore(STORE).openCursor().onsuccess = e => {
      const c = e.target.result;
      if (c) { out.push(c.value); c.continue(); }
      else res(out.sort((a,b) => a.fecha_iso < b.fecha_iso ? 1 : -1));
    };
    tx.onerror = () => rej(tx.error);
  });
}
async function dbDelete(id) {
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(id);
    tx.oncomplete = res; tx.onerror = () => rej(tx.error);
  });
}
async function dbClear() {
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).clear();
    tx.oncomplete = res; tx.onerror = () => rej(tx.error);
  });
}

// ---------- MODELO TFJS (COCO-SSD) ----------
let detector = null;
let modelReady = false;

async function loadModel() {
  try {
    $('model-status').textContent = '⏳ Descargando modelo COCO-SSD (15MB)…';
    await tf.ready();
    detector = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    modelReady = true;
    $('model-status').textContent = '✅ Modelo cargado. Listo para detectar.';
    $('model-status').style.background = '#D4E8D0';
  } catch (e) {
    console.error(e);
    $('model-status').textContent = '❌ Error cargando modelo. Revisa internet.';
    $('model-status').style.background = '#FCE4E4';
  }
}

// ---------- ANÁLISIS CON BBOX ----------
let currentCanvas = null;
let currentDetections = [];  // array de { clase, x, y, w, h, confianza }
let showBoxes = true;

async function analyzeImage(file) {
  $('analyze-status').textContent = '⏳ Procesando imagen…';
  if (!modelReady) {
    $('analyze-status').textContent = '⏳ Esperando que cargue el modelo…';
    await loadModel();
    if (!modelReady) return;
  }

  // 1. Cargar imagen a canvas
  const cv = await loadToCanvas(file);
  currentCanvas = cv;
  const w = cv.width, h = cv.height;

  // 2. Inferencia
  const imgTensor = tf.browser.fromPixels(cv);
  const predictions = await detector.detect(imgTensor);
  tf.dispose(imgTensor);

  // 3. Mapear a nuestros estadios y normalizar coordenadas
  const detections = [];
  predictions.forEach(p => {
    const clase = CLASS_MAP[p.class] || null;
    if (!clase) return;
    const [x, y, bw, bh] = p.bbox; // [x, y, width, height] en píxeles
    detections.push({
      clase,
      x: x / w,
      y: y / h,
      w: bw / w,
      h: bh / h,
      confianza: Math.round(p.score * 100)
    });
  });

  // 4. Contar por clase
  const counts = { verde: 0, blanco: 0, envero: 0, rojo: 0, flores: 0 };
  detections.forEach(d => { if (counts[d.clase] !== undefined) counts[d.clase]++; });

  // 5. Cobertura (área de cajas)
  const cov = { verde: 0, blanco: 0, envero: 0, rojo: 0, flores: 0 };
  detections.forEach(d => {
    const area = d.w * d.h * 100;
    cov[d.clase] = (cov[d.clase] || 0) + area;
  });
  const total = Object.values(cov).reduce((a,b) => a+b, 0) || 1;
  Object.keys(cov).forEach(k => cov[k] = +((cov[k] / total) * 100).toFixed(1));

  // 6. Guardar global y dibujar
  currentDetections = detections;
  drawPreview();
  updateUI(counts, cov);
  $('results').classList.remove('hidden');
  $('analyze-status').textContent = `✅ ${detections.length} objetos detectados. Revisa y guarda.`;
}

// Cargar imagen a canvas (redimensiona para rendimiento)
async function loadToCanvas(file) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
  catch { 
    bmp = await new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img); img.onerror = rej;
      img.src = URL.createObjectURL(file);
    });
  }
  const maxDim = 800;
  const s = Math.min(1, maxDim / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  cv.getContext('2d').drawImage(bmp, 0, 0, w, h);
  return cv;
}

// ---------- DIBUJAR PREVIEW CON CAJAS ----------
function drawPreview() {
  const cv = $('preview');
  if (!currentCanvas) return;
  cv.width = currentCanvas.width; cv.height = currentCanvas.height;
  const ctx = cv.getContext('2d');
  ctx.drawImage(currentCanvas, 0, 0);

  if (!showBoxes) {
    $('preview-wrap').classList.remove('hidden');
    return;
  }

  const colores = { verde: '#5C8C3F', blanco: '#D8D2B0', envero: '#E48AA3', rojo: '#C4303C', flor: '#E9C46A' };
  const w = cv.width, h = cv.height;

  currentDetections.forEach(d => {
    const x = d.x * w, y = d.y * h, bw = d.w * w, bh = d.h * h;
    const color = colores[d.clase] || '#fff';

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowColor = 'rgba(0,0,0,0.4)';
    ctx.shadowBlur = 6;
    ctx.strokeRect(x, y, bw, bh);
    ctx.shadowBlur = 0;

    const label = `${d.clase} ${d.confianza}%`;
    ctx.font = 'bold 13px Inter, sans-serif';
    const m = ctx.measureText(label);
    const tw = m.width + 10, th = 26;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - th, tw, th);

    ctx.fillStyle = '#000';
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText(label, x + 5, y - 7);
  });

  $('preview-wrap').classList.remove('hidden');
}

// ---------- UI: ACTUALIZAR CONTEOS ----------
function updateUI(counts, cov) {
  ['verde', 'blanco', 'envero', 'rojo', 'flores'].forEach(k => {
    setCount(k, counts[k] || 0);
  });
  const cvbox = $('coverage');
  cvbox.innerHTML = '';
  [['rojo','Rojo'], ['envero','Envero'], ['blanco','Blanco'], ['verde','Verde'], ['flores','Flores']]
    .forEach(([k,l]) => {
      const s = document.createElement('span');
      s.innerHTML = `${l} <b>${cov[k]||0}%</b>`;
      cvbox.appendChild(s);
    });
  updateFrutos();
}

function getCount(key) { return Math.max(0, parseInt($(`c-${key}`).value || '0', 10)); }
function setCount(key, val) {
  const el = $(`c-${key}`);
  if (el) { el.value = Math.max(0, val); }
  updateFrutos();
}
function updateFrutos() {
  const total = ['verde','blanco','envero','rojo'].reduce((a,k) => a + getCount(k), 0);
  $('frutos-total').textContent = total;
  const meter = $('stage-meter');
  meter.innerHTML = '';
  const map = { verde: '#5C8C3F', blanco: '#D8D2B0', envero: '#E48AA3', rojo: '#C4303C' };
  let any = false;
  Object.keys(map).forEach(k => {
    const c = getCount(k);
    if (!c) return;
    any = true;
    const p = pct(c, total);
    const d = document.createElement('div');
    d.className = 'seg';
    d.style.width = p + '%';
    d.style.background = map[k];
    d.style.color = k === 'blanco' ? '#3a3720' : '#fff';
    d.textContent = p >= 9 ? p + '%' : '';
    meter.appendChild(d);
  });
  if (!any) meter.innerHTML = '<div class="seg empty-seg">Sin frutos</div>';
}

// ---------- GUARDAR REGISTRO (con BBox) ----------
async function saveRecord() {
  const rec = {
    id: uuid(),
    fecha_iso: new Date().toISOString(),
    fecha: new Date().toLocaleString('es-MX'),
    parcela: $('f-parcela').value.trim(),
    variedad: $('f-variedad').value.trim(),
    verde: getCount('verde'), blanco: getCount('blanco'),
    envero: getCount('envero'), rojo: getCount('rojo'),
    flores: getCount('flores'), botones: getCount('botones'),
    notas: $('f-notas').value.trim(),
    metodo: 'TFJS COCO-SSD',
    // ¡ANOTACIONES VALIDADAS PARA ENTRENAMIENTO!
    anotaciones: currentDetections.map(d => ({
      clase: d.clase,
      x: Math.round(d.x * 1000) / 1000,
      y: Math.round(d.y * 1000) / 1000,
      w: Math.round(d.w * 1000) / 1000,
      h: Math.round(d.h * 1000) / 1000,
      confianza: d.confianza
    }))
  };
  rec.frutos = rec.verde + rec.blanco + rec.envero + rec.rojo;
  await dbAdd(rec);
  $('analyze-status').textContent = '✅ Guardado con bounding boxes.';
  refreshTable();
}

// ---------- EXPORTAR YOLO (formato .txt) ----------
async function exportYOLO() {
  const rows = await dbAll();
  if (!rows.length) return alert('No hay registros.');
  // Mapeo de clases a IDs (orden alfabético o el que definas)
  const classList = ['verde', 'blanco', 'envero', 'rojo', 'flor'];
  let output = '';
  rows.forEach((r, idx) => {
    if (!r.anotaciones || !r.anotaciones.length) return;
    output += `# Imagen ${idx+1}: ${r.fecha} (${r.parcela || 'sin parcela'})\n`;
    r.anotaciones.forEach(a => {
      const classId = classList.indexOf(a.clase);
      if (classId === -1) return;
      // YOLO: class_id x_center y_center width height (normalizados 0-1)
      const cx = a.x + a.w/2;
      const cy = a.y + a.h/2;
      output += `${classId} ${cx.toFixed(4)} ${cy.toFixed(4)} ${a.w.toFixed(4)} ${a.h.toFixed(4)}\n`;
    });
    output += '\n';
  });
  const blob = new Blob([output], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `yolo_annotations_${stamp()}.txt`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// ---------- TABLA ----------
async function refreshTable() {
  const rows = await dbAll();
  $('rec-count').textContent = rows.length;
  const tb = $('tbody'); tb.innerHTML = '';
  const empty = $('empty-rec');
  const wrap = $('table-wrap');
  if (!rows.length) { empty.classList.remove('hidden'); wrap.classList.add('hidden'); return; }
  empty.classList.add('hidden'); wrap.classList.remove('hidden');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="dim">${r.fecha}</td>
      <td>${r.parcela || '—'}</td>
      <td class="strong">${r.frutos}</td>
      <td class="mono">${r.verde}</td>
      <td class="mono">${r.blanco}</td>
      <td class="mono">${r.envero}</td>
      <td class="mono">${r.rojo}</td>
      <td class="mono">${r.flores}</td>
      <td class="mono">${r.botones}</td>
      <td>${r.anotaciones ? r.anotaciones.length : 0}</td>
      <td><button class="del" data-id="${r.id}">×</button></td>`;
    tb.appendChild(tr);
  });
  tb.querySelectorAll('.del').forEach(b => b.addEventListener('click', async () => {
    await dbDelete(b.dataset.id); refreshTable();
  }));
}

// ---------- OTROS EXPORT/IMPORT ----------
function download(content, ext, mime) {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `fenofresa_${stamp()}.${ext}`;
  document.body.appendChild(a); a.click(); a.remove();
}
async function exportCSV() {
  const rows = await dbAll(); if (!rows.length) return;
  const cols = ['fecha','parcela','variedad','frutos','verde','blanco','envero','rojo','flores','botones','metodo','notas','bbox_count'];
  const esc = v => `"${String(v??'').replace(/"/g,'""')}"`;
  const body = rows.map(r => [r.fecha, r.parcela, r.variedad, r.frutos, r.verde, r.blanco, r.envero, r.rojo, r.flores, r.botones, r.metodo, r.notas, r.anotaciones?.length || 0].map(esc).join(','));
  download('\uFEFF' + [cols.join(','), ...body].join('\n'), 'csv', 'text/csv;charset=utf-8');
}
async function exportJSON() {
  const rows = await dbAll(); if (!rows.length) return;
  download(JSON.stringify({ herramienta: 'FenoFresa TFJS', generado: new Date().toISOString(), registros: rows }, null, 2), 'json', 'application/json');
}
function importJSON(file) {
  const r = new FileReader();
  r.onload = async () => {
    try {
      const data = JSON.parse(r.result);
      const list = data.registros || data;
      for (const x of list) {
        x.id = x.id || uuid();
        x.frutos = x.frutos ?? (x.verde + x.blanco + x.envero + x.rojo);
        await dbAdd(x);
      }
      refreshTable();
      $('analyze-status').textContent = '✅ Importado.';
    } catch { $('analyze-status').textContent = '❌ JSON inválido.'; }
  };
  r.readAsText(file);
}

// ---------- PWA / INSTALACIÓN ----------
let deferredPrompt = null;
window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault(); deferredPrompt = e; $('btn-install').classList.remove('hidden');
});
function netStatus() {
  const dot = $('net-dot');
  dot.classList.toggle('on', navigator.onLine);
  dot.title = navigator.onLine ? 'En línea' : 'Sin conexión';
}
window.addEventListener('online', netStatus);
window.addEventListener('offline', netStatus);

// ---------- BINDINGS ----------
function bind() {
  $('file-input').addEventListener('change', e => { const f = e.target.files[0]; if (f) analyzeImage(f); });
  $('btn-analyze').addEventListener('click', () => $('file-input').click());
  $('btn-camera').addEventListener('click', () => $('camera-input').click());
  $('camera-input').addEventListener('change', e => { const f = e.target.files[0]; if (f) analyzeImage(f); });
  $('btn-overlay').addEventListener('click', () => {
    showBoxes = !showBoxes;
    $('btn-overlay').textContent = showBoxes ? 'Ocultar cajas' : 'Mostrar cajas';
    drawPreview();
  });
  ['verde','blanco','envero','rojo','flores','botones'].forEach(k => {
    $(`c-${k}`).addEventListener('input', updateFrutos);
    $(`s-${k}-up`).addEventListener('click', () => setCount(k, getCount(k)+1));
    $(`s-${k}-dn`).addEventListener('click', () => setCount(k, getCount(k)-1));
  });
  $('btn-save').addEventListener('click', saveRecord);
  $('btn-csv').addEventListener('click', exportCSV);
  $('btn-json').addEventListener('click', exportJSON);
  $('btn-yolo').addEventListener('click', exportYOLO);
  $('btn-import').addEventListener('click', () => $('import-input').click());
  $('import-input').addEventListener('change', e => { if (e.target.files[0]) importJSON(e.target.files[0]); e.target.value = ''; });
  $('btn-clear').addEventListener('click', async () => {
    if (confirm('¿Vaciar toda la base?')) { await dbClear(); refreshTable(); }
  });
  $('btn-install').addEventListener('click', async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt(); await deferredPrompt.userChoice;
    deferredPrompt = null; $('btn-install').classList.add('hidden');
  });
}

// ---------- INICIO ----------
async function init() {
  bind();
  netStatus();
  updateFrutos();
  await refreshTable();
  // Cargar modelo al inicio
  loadModel();
}
document.addEventListener('DOMContentLoaded', init);