# Draggy Encoder — auditoría de rendimiento (solo exploración)

**Fecha:** 2026-09-19 · **Equipo:** RTX 3060 12 GB (driver 610.74, CC 8.6) · Windows 11 26200
**Runtime:** Python 3.14.6 · onnxruntime-gpu 1.30.0 · torch 2.14.0+cu130
**Caso medido (el del usuario):** `SHIN CHAN EN LA ISLA DEL TESORO.mkv` — 720x480 H.264 **10 bits** (yuv420p10le),
25 fps, salida 1080x720 (`hevc_nvenc` 1600k), modelo `realesr-animevideov3` (SRVGG, nativo **x4**), petición de upscale **x2**.
**Nada de este informe se ha implementado: no se ha tocado ni una línea de la app.**

---

## 1. Resumen ejecutivo

El render va a **5 fps (≈200 ms/frame)**. El desglose medido es:

| Etapa (por frame, 720x480 → x4 → 1080x720) | Medido | % del frame |
|---|---|---|
| Inferencia ONNX Runtime CUDA **fp32** (lo que hace la app hoy) | **71.4 ms** | 37 % |
| **Post-proceso en CPU** del tensor x4 fp32 (clip×255, uint8, BGR, contiguo) | **68.1 ms** | 35 % |
| **Resize PIL** 2880x1920 → 1080x720 (`cv2` no está instalado → `_resize_frame` cae a PIL) | **33.9 ms** | 18 % |
| Decode + pipe + `tobytes` + encode NVENC + progreso | ~12 ms | 6 % |
| Pre-proceso CPU (BGR→RGB→NCHW fp32) | ~5 ms | 3 % |
| **Total** | **~190 ms → 5.2 fps** | |

Eso reproduce exactamente los 5 fps observados. **Más de la mitad del frame no es inferencia: es trabajo de CPU que
podría hacerse en la GPU.**

Rutas equivalentes medidas hoy con el mismo modelo y el mismo tamaño de frame:

| Ruta | ms/frame | fps | vs. hoy |
|---|---|---|---|
| **App hoy** — engine ONNX fp32 + post CPU + resize PIL | **~180** | 5.5 | 1.0x |
| `PyTorchInferenceEngine` de la app (autocast fp16, uint8 en GPU) | ~88 | 11.4 | 1.7x |
| Prototipo A: fp32 + pre/post **dentro del grafo ONNX** (uint8 entra, uint8 sale) | 66 | 15.1 | 2.7x |
| Prototipo B: A + resize a 1080x720 **en la GPU, dentro del grafo** | 66 | 15.2 | 2.8x |
| Prototipo C: **fp16** + B | **42** | 23.7 | **4.3x** |
| Prototipo D: torch **fp16 + channels_last** + resize GPU + uint8 D2H | **32** | **30.9** | **5.6x** |

Conclusión corta: **hay 4-6x de margen sin cambiar de modelo**, casi todo quitando trabajo de CPU del camino crítico y
usando fp16. Video2X (≈10 fps) queda por debajo de cualquiera de estos prototipos.

Lo que **no** hace falta tocar: decode, encode, batching, TensorRT, NPU, hwaccel (ver §6, todo descartado con datos).

---

## 2. Cómo se ha medido

Scripts de solo-lectura dejados en `tmp-perf/` (no escriben fuera de esa carpeta; el modelo fp16 y los prototipos que
generan también viven ahí):

| Script | Qué mide |
|---|---|
| `tmp-perf/probe_ai.py` | ORT: línea base de la app, opciones de provider, CUDA graph |
| `tmp-perf/probe_accel2.py` | fp16, torch (con `cuda.synchronize`), motores reales de la app, post-proceso a tamaño real |
| `tmp-perf/probe_accel3.py` | **prototipos A-D** (pre/post dentro del grafo, fp16, torch ideal) |
| `tmp-perf/probe_batch.py` | batching (batch 1/2/4/8) |
| `tmp-perf/probe_ffmpeg.py` | decode / encode / bucle sin IA |

Resultados crudos: `probe_results.json`, `probe_results2.json`, `probe_results3.json`.
Reproducir: `python tmp-perf\probe_accel3.py` (usa los runtimes privados de `bin/runtimes/ai_modules`).

---

## 3. Dónde se va el tiempo (código señalado)

1. **Inferencia fp32, sin opciones de sesión** — `src/ai/inference.py:232-235`: `ort.InferenceSession(path, providers=...)`
   sin `SessionOptions`. Se pierde `cudnn_conv_algo_search="HEURISTIC"` (el default EXHAUSTIVE hace lento el primer
   frame de cada shape), `arena_extend_strategy`, `enable_cuda_graph`, `gpu_mem_limit`.
   Medido: 71.4 ms fp32 → 65.8 ms con `enable_cuda_graph` + IO binding (−8 %).
2. **Post-proceso en CPU sobre un tensor de 66 MB** — `src/ai/inference.py:468-474`:
   `squeeze → transpose → clip(x*255) → astype(uint8) → [:, :, ::-1] → ascontiguousarray`.
   Con el modelo x4 y un frame de 720x480 el tensor de salida es **1x3x1920x2880 fp32 = 66 MB**: se transfiere
   entero por PCIe y se procesa en numpy. **68.1 ms medidos** (≈1/3 del frame).
   El motor PyTorch (`inference.py:620-624`) ya lo hace bien: convierte a uint8 **en la GPU** y mueve 16 MB.
3. **Resize con PIL** — `src/processing/pipeline.py:1189-1229` (rama PIL en 1210-1216).
   `cv2` **no está en `requirements.txt` ni en `requirements-build.txt`** (sí `pillow`) y no existe en el intérprete
   con el que corre la app: verificado, `import cv2` → `ImportError`. Así que en la práctica *todo* render usa
   PIL **BILINEAR** para pasar de 2880x1920 a 1080x720: **33.9 ms** y peor calidad que `INTER_AREA` en reducción.
4. **Bucle estrictamente serie** — `src/processing/pipeline.py:1231-1305`: `read(frame)` → pre → inferencia →
   post → resize → `tobytes` → `stdin.write`. Mientras se hace el post-proceso y el resize (~100 ms) **la GPU está
   parada**; el perfil de ORT lo confirma: en fp32 los kernels suman **38.97 ms** de 71 ms (55 % de ocupación).
   En fp16 con el tail en el grafo: 42.8 ms de pared ≈ **91 % de ocupación**.
5. **Desperdicio estructural: se pide x2 pero se ejecuta la red x4** — `src/ui/main_window.py:597` (`_target_scale`)
   solo cambia lo que la engine *reporta* para dimensionar el encoder; la red sigue produciendo 2880x1920 y luego
   se reduce a 1080x720 (factor 2.67). Eso es lo que hace gigantes el post-proceso y el resize.
6. **"auto" eligió el motor más lento para este caso** — `bin/.session/engines.json` guarda el benchmark cacheado
   (512x288, `src/ai/benchmark.py:40`): onnx 72.9 ms vs torch 78.7 ms → `auto` se queda con **onnxruntime**.
   Medido hoy **a la resolución real del render**: engine ONNX 146.2 ms vs engine PyTorch 87.8 ms
   (**el de torch es 1.66x más rápido**). El benchmark se hizo a 1/2.34 de los píxeles y con post-proceso incluido,
   así que compara peras con manzanas y decide con datos viejos.
7. **Interpolación después del upscale** (cadena del usuario: Upscale → Interp → …): RIFE trabaja sobre frames
   **2880x1920** en vez de sobre 720x480. `_split_ai_chain` (`pipeline.py:1651`) ya permite poner el nodo de
   interpolación antes del upscale **sin tocar código**; merece la pena medirlo, porque puede costar más que el
   propio upscale.

---

## 4. Plan propuesto (robustez primero, luego rendimiento)

### P0 — Telemetría por etapa (riesgo 0, imprescindible)
Añadir al `render_*.json` el desglose por frame: `decode_ms / pre_ms / infer_ms / post_ms / resize_ms / write_ms`
(medianas cada N frames) y usarlo también para la ETA de la UI.
**Criterio de aceptación:** el log de un render de 20 s permite ver las 6 etapas sin profiler externo.
Sin esto, cualquier optimización posterior es a ciegas y las ETAs seguirán mal.

### P1 — Sacar el trabajo de CPU del camino crítico (bajo riesgo · ~2.7x)
* Convertir/trasponer/normalizar **dentro del grafo ONNX** (cabeza: `Cast→Transpose→Mul(1/255)`; cola:
  `Mul(255)→Clip→Cast(uint8)→[Resize]→Transpose NHWC`) y usar **IO binding**. Medido en el prototipo A/B: **66 ms**,
  con la GPU al ~90 % y el CPU casi sin tocar (queda `tobytes`, 2.3 MB ≈ 1 ms).
  * El grafo derivado se genera **una vez por render** en `tmp` a partir del `.onnx` instalado: no hay que
    reentrenar ni reconvertir nada, y si algo falla se usa el modelo original (fallback trivial).
  * La cabeza debe incluir el **swap BGR→RGB** (en mi prototipo lo omití: por eso el `max diff` salió 255; se
    resuelve con un `Gather` de índices [2,1,0] sobre el eje de canales).
  * Si el `Resize` va dentro del grafo con `sizes` fijos, el modelo queda atado a ese tamaño de salida y a batch=1
    (por eso el test de batch>1 falló). Para no atarse: calcular `sizes` con `Shape/Concat/Slice` o regenerar el
    grafo por render (más simple y suficiente).
* **Alternativa más barata aún:** que el motor ONNX haga lo que ya hace el de torch (uint8 en GPU antes del
  `.cpu()`). Es un cambio pequeño en `inference.py` y no toca grafos.
* Si se prefiere no tocar inferencia: **empaquetar `opencv-python-headless`** (o usar el escalador de ffmpeg /
  `torch.nn.functional.interpolate`) quita los 34 ms de PIL, pero no los 68 ms del post-proceso.

### P2 — fp16 verificado (riesgo medio · hasta 4.3x acumulado)
* `convert_float_to_float16(..., keep_io_types=False)` sobre el modelo ya envuelto con cabeza/cola:
  **46.0 ms** crudo y **42.3 ms** el camino completo con resize en GPU (prototipo C).
* **Atención (trampa real encontrada):** con `keep_io_types=True` el grafo convertido **se rompe** en este modelo
  (`/Add` con error de tipo/shape; el motor de la app pasó a **4448 ms/frame**, 0.22 fps). Hay que verificar
  numéricamente contra fp32 (mi prototipo quedó en `max|diff|` de 1 nivel de 0-255) y **no** publicar un fp16 sin
  comprobación por frame o, mejor, **exportar el ONNX ya en fp16 desde torch** (`model.half()`), que es el camino
  reproducible.
* Añadir en la sesión: `cudnn_conv_algo_search="HEURISTIC"`, `arena_extend_strategy="kSameAsRequested"`,
  `enable_cuda_graph` + `io_binding` (medidos: −8 % en fp32; sin efecto negativo).

### P3 — Usar el mejor motor por defecto (riesgo bajo · ~5.6x acumulado)
* Con el motor PyTorch: `channels_last` + fp16 + resize en GPU + uint8 D2H = **32.3 ms** (prototipo D).
  `channels_last` solo es un `to(memory_format=...)` en `load()`/`process_frame` y da 30.1 vs 42.8 ms fp16.
* `cudnn.benchmark` **no** aporta nada aquí (67.2 vs 66.3 ms fp32): no hace falta tocarlo.
* Rehacer el benchmark del backend **a la resolución real del render** (o al menos escalar por píxeles) y con el
  mismo post-proceso en ambos motores; hoy `auto` decide con 512x288 y datos posiblemente medidos bajo carga.
* `auto` debería además invalidar la caché si cambia la resolución del job, no solo runtime/GPU/Python.

### P4 — Solape decode / inferencia / encode (riesgo medio · hasta ~25 fps)
El suelo real hoy es el kernel de la GPU: **39 ms** (medido en el perfil de ORT). Con el bucle en serie no se baja de
~45 ms. Un esquema de 3 hilos con colas acotadas (decoder → `infer` → `post+resize+write`) o `run_async` de ORT con
doble buffer permite acercarse a ese suelo y, de paso, mantener la UI fluida.
**Criterio de aceptación:** ocupación de GPU > 85 % sostenida y 0 frames perdidos/desordenados en 10 000 frames,
con abort en <1 s (los `ProcessSupervisor` y los gates de integridad actuales deben seguir pasando).

### P5 — Palancas de calidad/velocidad a explorar después
* **Modelo nativo x2** (o el truco de pre-escalar la entrada: 480x320 → x4 = 1920x1280 → 1080x720) en vez de
  ejecutar la red x4 y reducir 2.67x. Ahorra cómputo y memoria de salida, a cambio de algo de detalle.
* **Tiling** para frames grandes (ver §5.2).
* **Orden de la cadena** con interpolación (ver §3.7).

---

## 5. Cosas de robustez detectadas (esto es lo que puede "romper" al acelerar)

1. **Un fallo de engine a mitad de render degrada en silencio.** `pipeline.py:1169-1187` captura cualquier excepción
   de un engine, lo marca `_failed` y **deja pasar los frames sin procesar** el resto del vídeo; el render termina en
   "éxito". Un OOM de CUDA en el frame 5000 = el resto del vídeo sin upscale sin que el usuario lo note con claridad.
   *Sugerencia:* aviso visible en la UI + opción "estricto" (abortar) o reintento automático con tiling.
2. **No hay tiling en ninguna parte** (ni en `inference.py` ni en `pipeline.py`): la memoria escala con el frame
   completo. Con un modelo x4 (p. ej. `RealESRGAN_x4plus`, RRDBNet) a 1080p de entrada el pico de activaciones es de
   varios GB; a 4K es un OOM garantizado en 12 GB y en 8 GB ni se intenta. Video2X trocea. Es *el* fallo duro que
   aparece justo cuando alguien sube de resolución — y encima cae en el punto 1.
3. **fp16 no es un drop-in** en este grafo (ver P2): documentar y verificar antes de activarlo por defecto.
4. **TensorRT está anunciado pero no es usable.** `ort.get_available_providers()` devuelve
   `TensorrtExecutionProvider`, pero su DLL no carga: `onnxruntime_providers_tensorrt.dll` depende de
   **`nvinfer_10.dll`, que no está** → ORT hace *fallback* silencioso a CUDA (medido: 73 ms, lo mismo que CUDA).
   El benchmark de la app guarda TRT como provider "disponible" y la UI lo trata como GPU.
   *Sugerencia:* o se instalan las DLL de TensorRT 10 como dependencia opcional real, o se filtra TRT de
   `get_available_providers()` para no prometer lo que no hay.
5. **Fallback silencioso a CPU.** Si CUDA se cae, `_try_cpu_fallback` (`inference.py:325-361`) solo deja una línea
   en el log. En CPU este modelo va a **0.67 fps** (dato del propio benchmark cacheado de `realesr-general-x4v3`).
   Debe verse en la UI con opción de abortar.
6. **ETAs y benchmark engañosos**: 512x288, 5 frames, con post-proceso incluido, cacheado indefinidamente
   (`benchmark.py:40-42`, `engines.json`). En este caso predice 72.9 ms cuando el frame real cuesta ~146 ms.
7. **`cv2` no es dependencia** y el código lo prefiere (`pipeline.py:1197-1207`): en la práctica siempre se usa PIL,
   con reducción BILINEAR (aliasing al reducir 2.67x). O se añade opencv, o se elige el interpolador correcto en PIL
   (`Image.Resampling.BOX`/`LANCZOS`).
8. **Este equipo no tiene NPU.** No aparece ningún dispositivo `ComputeAccelerator`; el único acelerador es la
   RTX 3060. Un NPU Intel/AMD requeriría otra build de ORT (OpenVINO / Vitis AI) y no compensa frente a CUDA aquí.
9. **NVDEC no sirve para esta fuente**: el vídeo es H.264 **10 bits** y `-hwaccel cuda` falla
   ("Hardware is lacking required capabilities") en GA106. Da igual: el decode software cuesta **2.7 ms/frame**
   (373 fps medidos en el bucle sin IA). Para futuras fuentes 10-bit HEVC/AV1 sí convendría añadir hwaccel.

---

## 6. Descartado (medido, para no perder tiempo)

| Idea | Veredicto |
|---|---|
| Batching (2/4/8 frames por `run`) | **No**. En fp16 la GPU ya está ~91 % ocupada (42.8 ms de pared vs 39 ms de kernels). Solo ayudaría en fp32 (55 %), que es justo lo que queremos abandonar. Además exige un grafo batch-aware. |
| TensorRT EP | No instalable sin las DLL de TensorRT 10 y con coste de build/cache alto; la ganancia sobre fp16+CUDA graph es incierta. Contradice "robustez primero". |
| `cudnn.benchmark=True` en torch | Ruido estadístico (66.3 → 67.2 ms). |
| `-hwaccel cuda` | No soporta este 10-bit; y el decode ya cuesta 2.7 ms/frame. |
| Optimizar el encoder | `hevc_nvenc` a 1080x720 = **2.5 ms/frame** (397 fps). Nada que rascar. |
| CPU | 0.67 fps (benchmark cacheado real de `realesr-general-x4v3`). Solo último recurso. |
| NPU | No existe en este equipo. |

---

## 7. Siguiente paso propuesto

1. P0 (telemetría) → una sesión de trabajo, riesgo nulo, y da el presupuesto real de cada etapa.
2. P1 (post-proceso y resize en GPU) → el mayor salto por unidad de riesgo: **~2.7x** sin cambiar de modelo ni de
   precisión, con fallback trivial al modelo original.
3. P2/P3 (fp16 + `channels_last` + benchmark a resolución real) → **hasta 5.6x**, con verificación numérica obligatoria.
4. P4 (solape de etapas) solo después de que P1-P3 estén verdes y con el suelo de 39 ms como objetivo.

Nada de lo anterior requiere tocar la cadena de IA, los modelos instalados ni el flujo de FFmpeg: son cambios
localizados en `src/ai/inference.py` (sesión y conversión), `src/ai/benchmark.py` (medición real) y
`src/processing/pipeline.py` (resize y bucle), con los prototipos ya validados en `tmp-perf/`.
