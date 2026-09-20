# Draggy Encoder — auditoría de rendimiento

**Fecha:** 2026-09-19 (exploración) · **Implementado y validado:** 2026-09-20
**Equipo:** RTX 3060 12 GB (driver 610.74, CC 8.6) · Windows 11 26200
**Runtime:** Python 3.14.6 · onnxruntime-gpu 1.30.0 · torch 2.14.0+cu130
**Caso medido (el del usuario):** `SHIN CHAN EN LA ISLA DEL TESORO.mkv` — 720x480 H.264 **10 bits** (yuv420p10le),
25 fps, salida 1080x720 (`hevc_nvenc` 1600k), modelo `realesr-animevideov3` (SRVGG, nativo **x4**), petición de upscale **x2**.

---

## 0. Estado: P0 + P1 + P2 implementados y medidos

Lo que sigue (§1-§7) es la exploración original. **Ya está implementado** lo
aprobado: telemetría por etapa (P0), post-proceso en GPU detrás de preferencia
apagada por defecto con autocomprobación y fallback (P1), y selector de
precisión fp16/fp32/del modelo (P2). Medido con el **pipeline real**
(`compress_video_with_ai`, motor ONNX, 150 frames de tu MKV a 720x480 → 720p,
`hevc_nvenc`):

| Configuración | fps reales | mejora | 150 frames tardan |
|---|---|---|---|
| A · clásico (lo de hoy) | **5,37** | 1,00x | 28,5 s |
| B · GPU post-processing (fp32) | **14,37** | **2,68x** | 12,1 s |
| C · GPU post-processing + FP16 | **21,74** | **4,05x** | 8,5 s |

Presupuesto por etapa que ahora escribe el render log (P0), mediana por frame:

| Etapa | A (hoy) | B (GPU) | C (GPU+fp16) |
|---|---|---|---|
| decode | 1,00 ms | 1,01 ms | 1,01 ms |
| ai_pre (CPU) | 4,84 ms | 0,00 ms | 0,00 ms |
| ai_infer (GPU) | 72,15 ms | 67,53 ms | 44,05 ms |
| ai_post (CPU) | **71,30 ms** | **0,01 ms** | **0,01 ms** |
| resize (PIL) | **34,95 ms** | — | — |
| write (pipe) | 0,90 ms | 0,88 ms | 0,87 ms |
| **frame_total** | **186,3 ms** | **69,6 ms** | **46,0 ms** |

La imagen cambia menos de lo que parece: A vs C sobre el mismo frame da
``max|diff| = 52`` y ``mean|diff| = 1,66`` sobre 255 (la diferencia es el filtro
del reescalado, no la IA: la comprobación "gruesa" del self-check da 0,96/255).

**Cómo se activa:** *Preferences → Behavior → AI performance* →
`GPU post-processing` (apagado por defecto) e `Inference precision`
(Model default / FP16 / FP32). Si algo no cuadra, la app sigue por el camino
clásico y lo avisa en el panel de tareas.

**Trampas encontradas al implementarlo** (todas verificadas en esta máquina):

1. **`IOBinding` con entrada en CPU entrega CEROS** en ONNX Runtime 1.30 + CUDA
   con este grafo: `bind_cpu_input`, `bind_input(buffer_ptr=…)` y
   `bind_ortvalue_input` con un OrtValue reutilizado devuelven una imagen plana
   idéntica para cualquier entrada (el dato nunca llega). `session.run` (numpy)
   y un OrtValue nuevo por frame sí funcionan. Consecuencia: **el camino rápido
   usa `session.run` y NO usa CUDA Graph** (que exige IO binding). Las
   mediciones de los prototipos de §1 se hicieron con ese binding roto: los
   *tiempos* siguen valiendo (se ejecutan los mismos kernels: 66/42 ms de
   prototipo vs 67,5/44,1 ms medidos en el pipeline real), pero la validación
   numérica de entonces no valía, y la recomendación de "CUDA Graph −8 %" queda
   **descartada** para este build.
2. **La cabeza del grafo no puede reescribirse a sí misma.** Al reemplazar los
   consumidores de la entrada original hay que recorrer solo los nodos
   originales: si se recorre también el `Cast` de la propia cabeza se crea un
   ciclo (Cast ← dg_norm ← Mul ← … ← Cast) y ONNX Runtime **no da error**:
   devuelve una imagen plana. Lo detectó la autocomprobación.
3. **numpy no tiene SIMD para float16.** El post-proceso clásico sobre un
   tensor fp16 pasaba de 68 a **180 ms** por frame (el camino clásico completo
   subía a 405 ms). Arreglado subiendo a float32 antes de operar; con eso, FP16
   sin GPU post queda en ~163 ms (neutro) y por eso la UI avisa de que la
   ganancia de FP16 se aprovecha **con** GPU post activado.
4. **`convert_float_to_float16(keep_io_types=True)` rompe este modelo** (el
   `Add` residual con `Resize`): 4448 ms/frame. Con `keep_io_types=False` y la
   cabeza/cola en fp16 funciona (44 ms).
5. **Fallo de Trim encontrado y arreglado:** los campos Start/End de la UI son
   TEXTO (`"00:01:30"`) y el pipeline de IA les hacía `float()`. Con ese
   formato lanzaba `ValueError`, se caía al camino "sin límite" y **el render
   seguía hasta el final del vídeo ignorando el End** (con tu MKV, 90 minutos
   de GPU en vez de 6 segundos). Ahora se parsea con `_parse_time_to_seconds`
   y, si el valor no se entiende, se avisa en la UI. Además, la validación
   final del render ya no se salta cuando hay Trim: comprueba contra la
   duración real del trozo pedido.

**Tests:** `tests/test_gpu_post_timing.py` y `tests/test_pipeline_robustness.py`
nuevos; suite completa en verde.

**Ficheros tocados:** `src/ai/gpu_post.py` (nuevo), `src/utils/timing.py` (nuevo),
`src/ai/inference.py`, `src/processing/pipeline.py`,
`src/ui/preferences_dialog.py`, `tests/test_gpu_post_timing.py`,
`tests/test_pipeline_robustness.py`. Nada de FFmpeg, modelos instalados ni
diseño de UI.

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

## 6.5 Fallos visibles, baremo de modelos y un bug de calidad encontrado (2026-09-20)

### a) El usuario SIEMPRE se entera de un fallo de IA
Antes: si un motor fallaba a mitad de render (típicamente **VRAM insuficiente**),
`_apply_steps` capturaba el error, lo apuntaba en el log, desactivaba esa engine
y **el render terminaba en "éxito" con el resto del vídeo sin procesar**. Ahora:

* Aviso inmediato en el panel de tareas y en la barra de estado, clasificado:
  memoria de GPU / CUDA / runtime ausente / modelo incompatible, cada uno con su
  **consejo accionable** ("cierra otros programas de GPU, baja la resolución de
  export o usa un modelo más ligero").
* La fase de esa engine se marca **failed** en el panel (queda visible al final).
* **Comprobación de VRAM antes de renderizar** (`nvidia-smi`): si quedan menos de
  1,5 GB libres se avisa ANTES de empezar, no en el frame 4000.
* Al terminar: aviso con el número de pasos desactivados, y en el render log
  (`ai_failures`, `gpu_post_failures`, `vram`).
* Si el camino GPU se abandona a mitad (tamaño inesperado, fallo del grafo), la
  engine **vuelve al camino clásico en ese mismo frame** en lugar de quedarse
  desactivada, y se avisa igualmente.

### b) Baremo real: calidad y velocidad medidas

Hasta ahora el catálogo mostraba `quality_score: 74` y `popularity_score: 50`,
números escritos a mano que no significan nada para quien sólo quiere saber
"¿cuál va rápido?". `src/ai/model_rating.py` mide las dos cosas en el equipo del
usuario, con su modelo, sin descargas:

* **calidad**: 5 imágenes de referencia sintéticas (líneas finas, franjas,
  degradados con ruido, textura y carta de color), degradadas por bloques y
  reconstruidas por el modelo. Se miden **fidelidad** (SSIM/PSNR), **detalle**
  (energía de bordes frente a un reescalado sin IA) y **artefactos** (halos en
  zonas planas + deriva de color), y se combinan en un índice 0-100 con pesos
  documentados en el tooltip. Se guarda en `bin/.session/model_ratings.json`.
* **velocidad**: el benchmark que ya existía (ms/frame a 512x288), mostrado con
  etiqueta en palabras ("muy rápida", "media"...).
* La fila de cada motor muestra `★ calidad · ⚡ velocidad` y, en el tooltip, de
  dónde sale cada número y con qué se midió.

Medido en este equipo (referencia x4 a 512x288; RTX 3060):

| modelo | calidad | velocidad | detalle | halos |
|---|---|---|---|---|
| realesr-general-wdn-x4v3 | 55,9 (muy alta) | 16,9 fps | 2,79x | 6,4% |
| RealESRGAN_x4plus | 53,5 (muy alta) | 1,7 fps | 2,47x | 5,0% |
| AnimeVideo v3 · onnx | 48,5 (buena) | 14,4 fps | 1,98x | 3,6% |
| General v4v3 · onnx | 43,9 (buena) | 10,1 fps | 2,03x | 7,7% |
| AnimeVideo v3 · torch | 40,8 (buena) | 21,8 fps | 1,83x | 4,2% |
| RealESRNet_x4plus | 32,2 (correcta) | 1,7 fps | 1,39x | 3,0% |
| *(sin IA)* nearest | 31,9 (correcta) | — | 1,12x | 4,2% |
| RealESRGAN_x4plus_anime_6B | 23,5 (básica) | 4,8 fps | 1,77x | 12,5% |
| *(sin IA)* bilineal | 19,8 (básica) | — | 1,00x | — |

Aviso honesto: es un baremo **repetible**, no una verdad absoluta. El juego de
referencia es sintético (favorece a los modelos que no inventan detalle) y el
especialista en anime puntúa bajo por halos reales en bordes duros sintéticos;
sobre anime de verdad puede verse mejor de lo que dice la tabla.

### c) El baremo encontró un bug de calidad de verdad (arreglado)

`_RRDB.forward` aplicaba un `LeakyReLU` **después** de la suma residual; el
original de Real-ESRGAN no lleva activación ahí (`out * 0.2 + x`). Con 23
bloques, cada uno apagaba la señal negativa (x0.2) y la imagen salía oscura y
sin contraste. Además el upsampling estaba invertido (`conv_up` → `interpolate`
en lugar de `interpolate` → `conv_up`). Medido con `RealESRGAN_x4plus`:

| | antes | después |
|---|---|---|
| media de la salida (original: 221) | 126 | **225** |
| SSIM | 0,537 | **0,848** |
| detalle vs bilineal | 0,52x (**más borroso**) | **2,74x** |
| índice del baremo | 19,5 (básica) | **53,5 (muy alta)** |

Afectaba a **toda la familia RRDBNet** (x4plus, anime_6B, RealESRNet, y
cualquier x2plus/x4plus de usuario). `load_state_dict(strict=True)` no lo
detecta: las formas coinciden. Como el `.onnx` exportado también arrastraba el
error, el conversor ahora escribe una **marca de versión**
(`ONNX_EXPORT_VERSION = 2`) junto al `.onnx` y re-exporta los antiguos (si la
re-exportación falla, conserva el viejo).

### d) Los tres formatos, soportados y verificados (2026-09-20)

`onnx`, `safetensors` y PyTorch (`.pth`/`.pt`) cargan por la puerta real de la
app (`create_inference_engine`), verificado uno a uno:

| formato | backend | resultado |
|---|---|---|
| `.onnx` | ONNX Runtime | directo, 0,7 s |
| `.pth` | PyTorch | 2,5 s (y se puede convertir a ONNX) |
| `.safetensors` | PyTorch | 7,3 s la primera vez (convierte), 3,6 s después |
| `.safetensors` | ONNX Runtime | conversión automática a ONNX + CUDA |

El blocker era que **`safetensors` no se instalaba**: ahora va con PyTorch
(0,4 MB, se sirve desde PyPI y el instalador ya añadía PyPI como índice extra),
se valida en el staging como los demás paquetes y se desinstala con PyTorch
(no con ONNX Runtime). Con eso quedan operativos OmniSR x2/x3/x4, RIFE y FILM
en cuanto al *formato*.

Ojo: en el caso de **interpolación** el formato no era el único blocker. No hay
ninguna clase de motor que implemente `interpolate_frame`, así que RIFE/FILM
siguen sin poder usarse (el pipeline busca ese método y `architectures.py` no
tiene constructor para `rife`/`film`). Es trabajo aparte.

### e) Baremo corregido para comparar escalas distintas

La primera versión degradaba la referencia según la escala **nativa** de cada
modelo: a un x4 le pedía reconstruir desde 1/16 de los píxeles y a un x2 desde
1/4, así que el x4 parecía peor sólo por tener una tarea más difícil. Ahora
**todos se miden con la misma tarea** (degradación ×2 y la salida llevada al
tamaño final, exactamente como hace el render real cuando exportas por debajo de
la escala nativa). Tabla final, misma tarea para todos:

| modelo | formato | calidad | fps | detalle | halos |
|---|---|---|---|---|---|
| **OmniSR X2** | torch | **61,2** (muy alta) | 3,9 | 5,49x | 5,6% |
| OmniSR X4 | torch | 54,5 (muy alta) | 3,7 | 7,06x | 5,0% |
| **wdn-x4v3** | torch | **53,8** (muy alta) | **17,3** | 1,41x | 1,0% |
| RealESRGAN x4plus | torch | 46,8 (buena) | 1,7 | 1,25x | 1,2% |
| General v4v3 | onnx | 44,7 (buena) | 11,5 | 1,25x | 3,3% |
| AnimeVideo v3 | torch | 42,8 (buena) | **22,6** | 1,09x | 1,2% |
| AnimeVideo v3 | onnx | 42,7 (buena) | 16,0 | 1,09x | 1,2% |
| OmniSR X3 | torch | 40,0 (buena) | 3,8 | 20,6x | **45,9%** |
| anime_6B | torch | 16,6 (básica) | 4,7 | 1,07x | 17,0% |
| *(sin IA)* nearest | — | 31,9 (correcta) | — | 1,12x | 4,2% |
| *(sin IA)* bilineal | — | 19,8 (básica) | — | 1,00x | — |

Lectura rápida: **wdn-x4v3 da casi la mejor nota a un 77% de la velocidad del
más rápido** (el mejor equilibrio), **OmniSR X2 es el de mejor nota pero 5,8×
más lento**, y x4plus no compensa (13× más lento que AnimeVideo v3 para +4
puntos). **OmniSR X3 sale corrupto** (SSIM 0,39, halos 46%): o el checkpoint de
ese repo es raro o la implementación no cubre ese caso; X2 y X4 están bien.

### f) Otros fallos encontrados por el camino

* **`realesrgan-x4plus` del catálogo trae un `.onnx` con entrada fija 64x64**
  (`bin/models/upscale/realesrgan-x4plus/onnx/model.onnx`): cualquier render a
  otra resolución falla con "Got: 288 Expected: 64". Ahora la engine lo **avisa
  al cargar** (y queda en el render log como `model_note`). Su `.pth` de 64 MB
  sí funciona (46,8/100 a 1,7 fps).
* Los dos `.pth` llamados "x2plus" que hay en disco (`RealESRGAN_x2plus.pth`,
  `AI-Forever_x2plus.pth`) son **de 12 canales de entrada**: no valen para este
  pipeline (la app los rechaza con un mensaje claro). El único x2 nativo
  utilizable de esta máquina es **OmniSR X2**.
* **(resuelto)** `safetensors` no estaba instalado en el store: ver (d).

---

## 6.6 Empaquetado (PyInstaller / Inno Setup / AppImage / Flatpak): comprobado

Se ejecutó **el mismo camino que el CI**, no una suposición:

| comprobación | resultado |
|---|---|
| `python -m compileall -q src main.py packaging` (paso del CI Linux) | OK |
| `python main.py --selftest` offscreen (CI Linux) | `SELFTEST PASS`, con las 3 líneas que exige el CI |
| `capture_screenshots.py` (CI Linux) | genera los 2 PNG |
| **Build real con `draggy_encoder.spec`** (PyInstaller 6.22) | `Build complete`, bundle de 233 MB |
| ¿Están los módulos nuevos dentro del bundle? | `src.ai.gpu_post`, `src.ai.model_rating`, `src.utils.timing` **en `PYZ-00.toc`** |
| `DraggyEncoder.exe --selftest` (CI Windows, app congelada) | las 3 líneas offscreen presentes |
| `draggy_encoder.iss` | copia `dist\DraggyEncoder\*` recursivo: **no hay lista de ficheros que actualizar** |

Única diferencia con el CI en el build local: el CI añade antes del selftest el
Python privado (`embed-python/` en Windows, `runtime-python/` en Linux) y yo no,
así que el selftest local falló sólo en ese punto ("No se encontró el Python
privado") — esperado y ajeno a estos cambios.

**Conclusiones de empaquetado**

1. **No hace falta tocar el spec.** `safetensors` ya está en su lista
   `excludes` (igual que `torch`/`onnxruntime`): no se empaqueta, se instala en
   el store privado en tiempo de ejecución. El wheel es `cp310-abi3`, así que el
   guard `_wheel_tag_matches` lo acepta para el Python 3.13 del bundle
   (abi3 = compatible hacia delante) y no marca el store como incompatible.
2. **No hace falta tocar los workflows ni el `.iss`.** Todo lo nuevo es Python
   puro importado estáticamente (también los imports dentro de funciones, que
   PyInstaller detecta) más `numpy`/`PIL`, que ya eran dependencias del bundle.
3. **Rutas de escritura**: `gpu_post_cache/` y `model_ratings.json` van a
   `get_session_dir()`, o sea `bin/.session` → redirigido por los tres
   lanzadores de Linux (`AppRun`, wrapper del RPM y wrapper de Flatpak) a
   `$XDG_DATA_HOME/draggy-encoder` vía `DRAGGY_DATA_DIR`. Verificado: el
   selftest con `DRAGGY_DATA_DIR` apunta a esa carpeta. Nada escribe en `/opt`
   ni en el montaje de sólo lectura del AppImage.
4. **AppImage sí, Flatpak con matices.** El AppImage corre fuera de sandbox (la
   GPU se ve directa) y su wrapper ya redirige los datos: es el camino bueno, y
   ya se genera. El manifiesto de Flatpak ya trae `--share=network`
   (imprescindible: la app descarga FFmpeg y los runtimes por pip),
   `--filesystem=home` (datos + tus vídeos) y `--device=dri`. Lo que **no**
   declara es la extensión `org.freedesktop.Platform.GL.nvidia-*`, de la que
   depende CUDA dentro del sandbox: hay que probarlo en una máquina NVIDIA real
   antes de publicar Flatpak, porque si CUDA no ve la GPU el motor cae a CPU
   (0,67 fps medidos) y la app lo dirá en el panel de tareas. Recomendación:
   AppImage/tar.gz como objetivo Linux; Flatpak después y con esa prueba hecha.
5. **`.gitignore`**: se añadió `tmp-perf/` para que las sondas, los modelos
   derivados, los vídeos de validación y las capturas (≈15 MB) no suban a
   GitHub. `perf-audit.md` se deja a propósito fuera de esa lista (si prefieres
   tratarlo como `problems.md`/`features.md`, es una línea más).
6. **Detalle de Linux detectado (no de estos cambios)**: en
   `bin/models/` conviven `upscale/` (minúscula, la convención actual de
   `get_engine_dir`, que saca la categoría del `engine_id`) y un `Upscale/` con
   mayúscula y restos de una instalación antigua. En Windows da igual (rutas
   insensibles a mayúsculas); en Linux son dos carpetas distintas, así que ese
   resto no se encontraría ni se limpiaría. Es basura local: se puede borrar.
7. **Sugerencia opcional para `build-installer.yml`**: el paso "Selftest the
   built EXE" invoca el `.exe` (que es `console=False`) y comprueba
   `$LASTEXITCODE` justo después. En mi ejecución local PowerShell **no esperó**
   al proceso (devolvió al instante, sin código de salida y sin log todavía). En
   el CI parece funcionar, pero blindarlo es barato:
   `$p = Start-Process -FilePath .\dist\DraggyEncoder\DraggyEncoder.exe -ArgumentList '--selftest' -Wait -PassThru; if ($p.ExitCode -ne 0) { exit 1 }`.

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
