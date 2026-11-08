# Draggy Encoder · versión web (WebGPU)

Versión navegador del pipeline de Draggy Encoder. El vídeo **no se sube a ningún
sitio**: se decodifica en la pestaña, se procesa en tu GPU con WebGPU y se
exporta como MP4 en memoria.

Publicado en:
<https://site-c3c66ce1d2b1482ab0c19feea55fcfb8.freebuff.page>

## Qué hace hoy

- Flujo de trabajo ordenable (el concepto de la app de escritorio): añadir,
  reordenar y quitar pasos. Hoy hay tres tipos:
  - **Reescalar** ×2 / ×3 / ×4 — bicúbica Catmull-Rom en un shader WGSL.
  - **Realzar detalle** — máscara de desenfoque con cantidad ajustable.
  - **Comprimir / exportar** — H.264 con bitrate objetivo.
- Lectura de MP4/WebM/MKV mediante `mediabunny` (decode con WebCodecs).
- Exportación MP4 (H.264) con `mediabunny` + `CanvasSource`.
- Progreso con fps y tiempo restante, cancelable con `Esc`.

## Qué **no** hace todavía

- **No hay modelos de IA.** Los pasos actuales son shaders, no redes
  neuronales: no hay Real-ESRGAN, ni RIFE, ni restauración de caras. Un `.onnx`
  de upscale real pesa decenas o cientos de MB y no se carga en esta versión.
- **No conserva el audio**: el MP4 sale sin pista de audio.
- Solo salida H.264/MP4; no hay NVENC ni control de CRF, porque eso es ffmpeg.
- Límite de 7680 px en la dimensión mayor del resultado (x4 sobre 4K se sale).

## Desarrollo

```bash
cd web
npm install          # usa la caché de .npmrc (fuera de C:)
npm run dev          # http://localhost:5173
npm run build        # dist/
```

Requisitos: Node 20+ y un navegador con WebGPU (Chrome/Edge actuales; Firefox
en Windows 141+; Safari 26+).

## Docker

```bash
cd web
docker compose up -d --build   # http://localhost:8080
```

El contenedor es solo `nginx` + estáticos: **no necesita GPU**, porque el
cómputo ocurre en el navegador del visitante.

### Mantener Docker fuera de C:

Docker Desktop guarda imágenes y volúmenes en C: por defecto. En Windows:

- Docker Desktop → Settings → Resources → Advanced → *Disk image location* →
  una ruta en `D:`.
- O, con WSL2, en `%UserProfile%\.wslconfig`:
  ```ini
  [wsl2]
  swapFile=D:\\docker\\swap.vhdx
  ```

Comprueba con `docker system df` y `docker info --format '{{.DockerRootDir}}'`
que la raíz está en `D:`.

## Prueba end-to-end

Genera un clip sintético dentro de la página, lo procesa en la GPU y valida que
el MP4 descargado tiene marca de contenedor `ftyp`:

```bash
export PLAYWRIGHT_BROWSERS_PATH=D:/draggy-web-build/pw-browsers
node e2e.mjs https://site-c3c66ce1d2b1482ab0c19feea55fcfb8.freebuff.page
```

Necesita `playwright` + `npx playwright install chromium`, con los navegadores
en `D:` (esa variable de entorno). El test abre una ventana real porque WebGPU
no existe en el Chromium headless de Playwright.
