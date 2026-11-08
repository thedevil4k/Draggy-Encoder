const { app, BrowserWindow, Menu, protocol, net } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const { pathToFileURL } = require('node:url');

// En desarrollo se sirve el dist de ../web; empaquetado, la copia en resources/dist.
const distDir = app.isPackaged
  ? path.join(process.resourcesPath, 'dist')
  : process.env.DRAGGY_DIST || path.join(__dirname, '..', 'web', 'dist');

// Nada de datos en C:: el perfil vive junto al ejecutable (o en desktop/.data en desarrollo).
const dataDir = app.isPackaged
  ? path.join(path.dirname(app.getPath('exe')), 'data')
  : path.join(__dirname, '.data');

// También el temporal: Chromium escribe cachés/restos en %TEMP%, y aquí viven en D:.
const tempDir = path.join(dataDir, 'temp');
try {
  fs.mkdirSync(tempDir, { recursive: true });
} catch {
  // Si no se puede crear, Electron seguirá con el temporal del sistema.
}

for (const key of ['userData', 'sessionData', 'crashDumps', 'logs', 'temp']) {
  try {
    app.setPath(key, key === 'logs' ? path.join(dataDir, 'logs') : key === 'temp' ? tempDir : dataDir);
  } catch {
    // Si alguna ruta no existe todavía, Electron la crea al primer uso.
  }
}

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'app',
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true },
  },
]);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.webp': 'image/webp',
};

function resolveRequest(requestUrl) {
  const { pathname } = new URL(requestUrl);
  const clean = decodeURIComponent(pathname === '/' ? '/index.html' : pathname);
  const filePath = path.join(distDir, path.normalize(clean).replace(/^([/\\])+/, ''));
  if (!filePath.startsWith(distDir)) return null;
  return { filePath, hasExtension: path.extname(filePath) !== '' };
}

async function serve(request) {
  const hit = resolveRequest(request.url);
  if (!hit) return new Response('Forbidden', { status: 403 });

  const send = async (filePath) => {
    try {
      const response = await net.fetch(pathToFileURL(filePath).toString());
      if (!response.ok) return null;
      const headers = new Headers(response.headers);
      headers.set('content-type', MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream');
      return new Response(response.body, { status: 200, headers });
    } catch {
      return null;
    }
  };

  return (await send(hit.filePath)) || (hit.hasExtension ? new Response('Not found', { status: 404 }) : (await send(path.join(distDir, 'index.html'))) || new Response('Not found', { status: 404 }));
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 720,
    backgroundColor: '#0f0f1a',
    title: 'Draggy Encoder',
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });

  win.once('ready-to-show', () => win.show());
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('app://')) event.preventDefault();
  });

  win.loadURL('app://draggy/index.html');
  return win;
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    const [win] = BrowserWindow.getAllWindows();
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(() => {
    Menu.setApplicationMenu(null);
    protocol.handle('app', serve);
    createWindow();
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on('window-all-closed', () => app.quit());
}
