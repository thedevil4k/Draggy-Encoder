import { chromium, _electron as electron } from 'playwright';
import path from 'node:path';

// Uso:
//   node e2e.mjs http://127.0.0.1:4173/                     -> navegador contra una URL
//   node e2e.mjs electron:D:/Draggy-Encoder/desktop          -> app Electron en desarrollo
//   node e2e.mjs "electron:D:/.../Draggy Encoder.exe"        -> app empaquetada
const target = process.argv[2];
if (!target) throw new Error('Falta la URL o electron:<carpeta o .exe>');

const isElectron = target.startsWith('electron:');
const appTarget = isElectron ? path.resolve(target.slice('electron:'.length)) : null;
const packagedExe = isElectron && appTarget.toLowerCase().endsWith('.exe') ? appTarget : null;

let browser = null;
let electronApp = null;
let page;

if (isElectron) {
  electronApp = await electron.launch(
    packagedExe
      ? { executablePath: packagedExe, args: [] }
      : {
          executablePath: path.join(appTarget, 'node_modules', 'electron', 'dist', 'electron.exe'),
          args: [appTarget],
          cwd: appTarget,
        },
  );
  page = await electronApp.firstWindow();
  await page.waitForLoadState('domcontentloaded');
} else {
  browser = await chromium.launch({
    headless: false,
    args: ['--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'],
  });
  page = await browser.newPage();
}

const errors = [];
page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(`console: ${message.text()}`);
});

if (!isElectron) await page.goto(target, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);

console.log('motor      :', isElectron ? (packagedExe ? 'electron empaquetado' : 'electron') : 'chromium');
console.log('gpu-badge  :', (await page.locator('#gpu-badge').textContent())?.trim());
console.log('cadena     :', (await page.locator('.workflow').innerText()).replace(/\s*\n\s*/g, ' | '));

// La ventana original: top bar de 5 botones, Import 150x150, pasos AI 200x150,
// Compress 90x150, panel de ajustes y fila de acciones.
const layout = await page.evaluate(() => {
  const size = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return 'missing';
    const rect = el.getBoundingClientRect();
    return `${Math.round(rect.width)}x${Math.round(rect.height)}`;
  };
  const texts = (selector) => [...document.querySelectorAll(selector)].map((el) => el.textContent.trim());
  return {
    topbar: texts('.topbar .topbtn').length,
    import: size('#node-import'),
    upscale: size('#node-upscale'),
    sharpen: size('#node-sharpen'),
    compress: size('#node-compress'),
    group: texts('.group legend'),
    fields: texts('.grid > label'),
    actions: texts('.actions button'),
    tasks: texts('.tasks-actions button'),
  };
});
console.log('layout     :', JSON.stringify(layout));

// Clip sintético generado dentro de la propia página (nada descargado, nada en disco).
const clipBytes = await page.evaluate(async () => {
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 180;
  const ctx = canvas.getContext('2d');
  const stream = canvas.captureStream(30);
  const recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
  const chunks = [];
  recorder.ondataavailable = (event) => chunks.push(event.data);
  const stopped = new Promise((resolve) => {
    recorder.onstop = resolve;
  });
  recorder.start();
  for (let i = 0; i < 30; i++) {
    ctx.fillStyle = `hsl(${i * 12}, 70%, 50%)`;
    ctx.fillRect(0, 0, 320, 180);
    ctx.fillStyle = '#000';
    ctx.font = 'bold 48px sans-serif';
    ctx.fillText(String(i), 130, 115);
    await new Promise((resolve) => setTimeout(resolve, 33));
  }
  recorder.stop();
  await stopped;

  const blob = new Blob(chunks, { type: 'video/webm' });
  const file = new File([blob], 'clip-sintetico.webm', { type: 'video/webm' });
  const transfer = new DataTransfer();
  transfer.items.add(file);
  const input = document.querySelector('#file');
  input.files = transfer.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
  return blob.size;
});
console.log('clip       :', clipBytes, 'bytes webm generados en memoria');

await page.waitForFunction(
  () => document.querySelector('#import-info')?.textContent?.includes('·'),
  null,
  { timeout: 30000 },
);
console.log('import     :', (await page.locator('#import-info').textContent())?.trim());
console.log(
  'nodos      :',
  (await page.locator('#upscale-status').textContent())?.trim(),
  '·',
  (await page.locator('#sharpen-status').textContent())?.trim(),
);

await page.waitForSelector('#btn-compress:not([disabled])', { timeout: 30000 }).catch(async () => {
  console.log('status     :', (await page.locator('#status').textContent())?.trim());
  throw new Error('El botón Compress sigue deshabilitado');
});

const before = Date.now();
await page.locator('#btn-compress').click();
await page.waitForSelector('.task-card a[download]', { timeout: 180000 });
const seconds = ((Date.now() - before) / 1000).toFixed(1);

console.log('estado     :', (await page.locator('#status').textContent())?.trim());
console.log('tarea      :', (await page.locator('.task-card').innerText()).replace(/\s*\n\s*/g, ' | '));
console.log('duración   :', seconds, 's');

const href = await page.locator('.task-card a[download]').first().getAttribute('href');
const size = await page.evaluate(async (objectUrl) => {
  const response = await fetch(objectUrl);
  const buffer = await response.arrayBuffer();
  const head = new Uint8Array(buffer.slice(0, 12));
  return { bytes: buffer.byteLength, ftyp: String.fromCharCode(...head.slice(4, 8)) };
}, href);
console.log('mp4        :', size.bytes, 'bytes · marca de contenedor:', size.ftyp);

console.log('errores    :', errors.length ? errors.slice(0, 4).join(' | ') : 'ninguno');

if (electronApp) await electronApp.close();
else await browser.close();
