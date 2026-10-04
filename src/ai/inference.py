"""Motores de inferencia de IA para procesamiento frame-by-frame.

Proporciona una abstracción sobre los motores de inferencia (ONNX Runtime,
PyTorch) que permite cargar un modelo, procesar fotogramas individuales en
formato RGB uint8 y liberar recursos.

Diseñado para usarse dentro de un QThread (CompressionWorker) durante el
procesamiento frame-by-frame: FFmpeg decodifica → este motor upscalea →
FFmpeg codifica.

Optimizaciones de rendimiento:
  - Pinned memory (numpy contiguous) para transferencia CPU↔GPU
  - Pre/post-procesamiento eficiente con numpy
  - Reutilización de buffers para evitar allocations repetidas
  - ONNX Runtime con CUDAExecutionProvider (GPU) con fallback a CPU
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from typing import Optional

import numpy as np

# Los runtimes de IA se resuelven exclusivamente desde el store privado.
# La función se importa aquí solo para que los llamadores que necesitan
# torch/onnxruntime tengan una frontera explícita; no se escanean los
# site-packages del sistema ni se añaden sus DLLs al PATH.
try:
    from src.runtimes.manager import (
        import_local_runtime,
        inject_runtime_paths,
        ensure_runtime_dlls_on_path,
    )
    inject_runtime_paths()
    ensure_runtime_dlls_on_path()
except Exception:
    # La UI puede arrancar sin IA instalada; los engines mostrarán un error
    # accionable al intentar cargarse.
    import_local_runtime = None

import src.preferences as preferences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carga segura de checkpoints (.pth / .pt)
# ---------------------------------------------------------------------------

# Preferencia que habilita la deserialización insegura de checkpoints. Por
# defecto OFF: ``torch.load`` con ``weights_only=False`` ejecuta pickle sobre
# un fichero de terceros, lo que permite ejecutar código arbitrario embebido
# en el checkpoint. Solo se degrada a ese camino si el usuario lo activa
# explícitamente en Preferences > Behavior > Checkpoint security.
_INSECURE_CHECKPOINT_PREF = "allow_insecure_checkpoint_load"


def _is_weights_only_restriction(exc) -> bool:
    """True si ``exc`` es el error específico de ``weights_only=True``.

    Cuando un checkpoint contiene objetos no-tensor (metadata, wrappers
    antiguos, optimizer state...), torch lanza un error que sugiere volver
    a cargar con ``weights_only=False``. Sólo en ese caso tiene sentido
    plantear la degradación: un fichero corrupto o ilegible debe seguir
    fallando, sin ofrecer el camino inseguro.
    """
    low = str(exc).lower()
    return (
        "weights only load failed" in low
        or "weights_only" in low
        or "unsupported global" in low
    )


def load_checkpoint_state_dict(path: str):
    """Carga el state_dict de un checkpoint ``.pth``/``.pt`` de forma segura.

    Intenta SIEMPRE ``torch.load(..., weights_only=True)`` primero (no
    ejecuta código embebido en el fichero). Sólo si el checkpoint requiere
    deserialización insegura Y el usuario ha activado explícitamente la
    preferencia ``allow_insecure_checkpoint_load``, se reintenta con
    ``weights_only=False`` (con un warning en el log). En cualquier otro
    caso se lanza ``RuntimeError`` con instrucciones accionables.

    ``.safetensors`` no pasa por aquí: su formato no usa pickle y se carga
    con ``safetensors.torch.load_file`` en los llamadores.
    """
    torch = import_local_runtime("torch")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not _is_weights_only_restriction(exc):
            raise
        if not preferences.get(_INSECURE_CHECKPOINT_PREF, False):
            raise RuntimeError(
                "This checkpoint cannot be loaded safely: it contains objects "
                "that require Python unpickling, which can execute arbitrary "
                "code embedded in the file. Refusing to load it with "
                "weights_only=False.\n\n"
                "To load it anyway, enable 'Allow insecure checkpoint loading' "
                "in Preferences (Behavior tab), or re-save the model as "
                ".safetensors.\n\n"
                f"File: {path}\nUnderlying error: {exc}"
            ) from exc
        logger.warning(
            "Loading checkpoint with weights_only=False (user opted in): %s",
            path,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _pth_channel_count(pth_path: str) -> Optional[int]:
    """Peek at the first Conv2d input channels of a .pth checkpoint.

    Returns the channel count, or None if it can't be determined.
    This is a lightweight check — it loads the state_dict but doesn't
    build the model. Used by the workflow widget to skip incompatible
    models (e.g. 12-channel RRDBNet) that would always fail to load.
    """
    try:
        torch = import_local_runtime("torch")
        sd = load_checkpoint_state_dict(pth_path)
        if not isinstance(sd, dict):
            return None
        # Unwrap common checkpoint wrappers (params_ema, params, state_dict)
        for wrapper_key in ("params_ema", "params", "state_dict"):
            if wrapper_key in sd and isinstance(sd[wrapper_key], dict):
                sd = sd[wrapper_key]
                break
        for key in ("conv_first.weight", "body.0.weight", "input.weight"):
            if key in sd:
                return int(sd[key].shape[1])
        return 3  # default assumption
    except Exception:
        return None


class AIInferenceEngine:
    """Clase base abstracta para motores de inferencia.

    Subclases deben implementar: load(), process_frame(), unload().
    """

    def __init__(self, model_path: str, engine_id: str):
        self.model_path = model_path
        self.engine_id = engine_id
        self._loaded = False
        self._input_name = ""
        self._output_name = ""
        self._input_shape = None
        self._output_shape = None
        # Escala objetivo solicitada por la UI (el usuario puede querer
        # x2 aunque el peso sea x4). None → usar la escala nativa del
        # modelo detectada en load().
        self._target_scale = None
        # Telemetría (P0): acumulado por frame, para que el pipeline pida el
        # total una sola vez al terminar el render.
        self._timing_totals = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}
        self._timing_calls = 0
        # Post-proceso en GPU (P1): None = no configurado o inactivo.
        self._gpu_post = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------ telemetría
    def _add_timing(self, pre_ms: float, infer_ms: float, post_ms: float) -> None:
        """Suma los ms de las tres fases de un frame. Nunca lanza."""
        try:
            self._timing_totals["pre_ms"] += float(pre_ms)
            self._timing_totals["infer_ms"] += float(infer_ms)
            self._timing_totals["post_ms"] += float(post_ms)
            self._timing_calls += 1
        except Exception:
            pass

    def get_cumulative_timing(self) -> dict:
        """Milisegundos acumulados por fase + nº de frames procesados (P0)."""
        return {
            "frames": self._timing_calls,
            "pre_ms": round(self._timing_totals.get("pre_ms", 0.0), 3),
            "infer_ms": round(self._timing_totals.get("infer_ms", 0.0), 3),
            "post_ms": round(self._timing_totals.get("post_ms", 0.0), 3),
        }

    def reset_cumulative_timing(self) -> None:
        """Descarta lo acumulado (los frames de la autocomprobación no cuentan)."""
        self._timing_totals = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}
        self._timing_calls = 0

    # --------------------------------------------------------- post GPU (P1)
    def configure_gpu_post(self, width=None, height=None, resize_fn=None,
                           input_size=None) -> dict:
        """Activa el pre/post-proceso en GPU si el usuario lo pidió (P1).

        Devuelve ``{"active": bool, "reason": str, ...}``. La implementación
        por defecto no ofrece este camino; las subclases que sí lo soportan lo
        sobrescriben. Nunca lanza: ante cualquier problema se devuelve
        ``active=False`` y el pipeline sigue con el camino de siempre.
        """
        return {"active": False, "reason": "not-supported"}

    def gpu_post_active(self) -> bool:
        """True si esta engine está usando el camino de post-proceso en GPU."""
        try:
            return bool(self._gpu_post and self._gpu_post.get("active"))
        except Exception:
            return False

    def disable_gpu_post(self) -> None:
        """Vuelve al camino clásico. Las subclases lo sobrescriben."""
        self._gpu_post = None

    def describe_execution(self) -> dict:
        """Cómo está corriendo realmente la engine (para el log del render)."""
        return {
            "engine_id": self.engine_id,
            "backend": type(self).__name__,
            "gpu_post": self.gpu_post_active(),
        }

    def load(self) -> bool:
        raise NotImplementedError

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un fotograma RGB uint8 (H, W, 3) y devuelve el resultado
        también en RGB uint8 (H', W', 3)."""
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def get_scale_factor(self) -> int:
        """Devuelve el factor de escala del modelo (2, 4, etc.)."""
        return 2


class ONNXRUNTIMEInferenceEngine(AIInferenceEngine):
    """Motor de inferencia con ONNX Runtime y soporte CUDA.

    Usa CPUExecutionProvider por defecto para evitar fallos de CUDA/cuDNN.
    En modo 'Auto' intenta CUDA y, si no funciona, recae en CPU.
    """

    def __init__(self, model_path: str, engine_id: str):
        super().__init__(model_path, engine_id)
        self._session = None
        self._providers = None
        self._exec_mode = "Auto"
        self._is_first_frame = True
        self._cpu_fallback_done = False
        # Precisión (P2) y ruta real del .onnx de la sesión. Si el usuario
        # pide fp16/fp32 y el archivo es de otra precisión, la sesión se crea
        # a partir de una copia convertida (en la caché de sesión, nunca
        # tocando el modelo instalado); si la conversión no es viable se usa
        # el modelo tal cual.
        self._precision_pref = "model"
        self._precision = "fp32"
        self._native_dtype = "fp32"
        self._session_model_path = model_path
        self._input_dtype = np.float32
        self._gpu_post_conversion_note = ""
        # Aviso si el .onnx trae un tamaño de entrada fijo (export mal hecho).
        self._fixed_input_note = ""
        # Escala nativa del modelo, medida una vez (ver get_scale_factor).
        self._native_scale = None
        # Motivo por el que el camino GPU se abandonó a mitad de render (si
        # pasó). El pipeline lo lee al terminar para avisar en la UI.
        self._gpu_post_failure = None

    def _resolve_precision(self, ort) -> None:
        """Resuelve la precisión efectiva antes de crear la sesión (P2)."""
        try:
            from src.ai import gpu_post
            self._precision_pref = gpu_post.get_precision_preference()
            self._native_dtype = gpu_post.model_float_dtype(self.model_path)
        except Exception as exc:
            logger.debug("Precision preference unavailable: %s", exc)
            self._precision_pref, self._native_dtype = "model", "fp32"

        target = (self._precision_pref if self._precision_pref in ("fp16", "fp32")
                  else self._native_dtype)
        self._precision = self._native_dtype
        self._session_model_path = self.model_path
        self._gpu_post_conversion_note = ""
        if target == self._native_dtype:
            return
        try:
            from src.ai import gpu_post
            path, info = gpu_post.build_precision_model(self.model_path, target)
        except Exception as exc:
            path, info = None, {"reason": str(exc)}
        if path:
            self._session_model_path = path
            self._precision = target
            self._gpu_post_conversion_note = (
                f"converted {self._native_dtype}->{target}"
            )
            logger.info("ONNX precision: %s -> %s (%s)", self._native_dtype,
                        target, os.path.basename(path))
            if target == "fp16":
                try:
                    from src.ai import gpu_post as _gpu_post
                    if not _gpu_post.gpu_post_enabled():
                        # Honestidad medida: fp16 acelera la inferencia, pero
                        # el post-proceso clásico en numpy sobre un tensor fp16
                        # se come la ganancia (numpy no tiene SIMD para fp16).
                        logger.warning(
                            "FP16 is selected without GPU post-processing: the "
                            "numpy post-process offsets most of the gain on the "
                            "ONNX backend. Enable 'GPU post-processing' in "
                            "Preferences for the full speed-up."
                        )
                except Exception:
                    pass
        else:
            # Rechazo explícito y visible: el usuario pidió una precisión que
            # no se pudo preparar, así que corremos la nativa y lo decimos.
            self._gpu_post_conversion_note = (
                f"precision {target} unavailable ({info.get('reason', '?')})"
            )
            logger.warning(
                "Requested precision %s could not be prepared for %s (%s); "
                "using the model as installed (%s).",
                target, os.path.basename(self.model_path),
                info.get("reason", "unknown"), self._native_dtype,
            )

    def load(self) -> bool:
        """Carga el modelo ONNX. Devuelve True si tiene éxito."""
        if not os.path.exists(self.model_path):
            logger.error("ONNX model not found: %s", self.model_path)
            return False

        try:
            ort = import_local_runtime("onnxruntime")
        except ImportError:
            logger.error(
                "ONNX Runtime is not installed. Open Manage Dependencies → "
                "Runtimes & Binaries to install it, or use Python 3.11–3.14 "
                "for automatic download."
            )
            return False

        # Detectar proveedores disponibles y aplicar preferencia del usuario.
        # 'Auto' intenta CUDA con fallback a CPU; 'CPU' fuerza CPU;
        # 'CUDA' fuerza CUDA y falla si no funciona.
        # Por defecto usamos Auto: el warmup en load() detectará un CUDA
        # roto y caerá a CPU automáticamente. El usuario puede forzar CPU
        # o CUDA desde Preferences > AI Execution Provider.
        self._exec_mode = str(preferences.get("execution_provider", "Auto")).strip() or "Auto"
        available = ort.get_available_providers()
        if self._exec_mode == "CPU":
            self._providers = ["CPUExecutionProvider"]
        elif self._exec_mode == "CUDA":
            self._providers = ["CUDAExecutionProvider"]
        else:  # Auto (default)
            if "CUDAExecutionProvider" in available:
                self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._providers = ["CPUExecutionProvider"]

        self._resolve_precision(ort)

        try:
            self._session = ort.InferenceSession(
                self._session_model_path,
                providers=self._providers,
            )
        except Exception as exc:
            # En modo Auto, si la sesión CUDA no se puede crear, intentamos
            # crearla con CPU antes de rendirnos. Esto cubre sistemas en
            # los que get_available_providers reporta CUDA pero el driver
            # o cuDNN no permiten inicializar la sesión.
            if (
                self._exec_mode == "Auto"
                and "CUDAExecutionProvider" in self._providers
            ):
                logger.warning(
                    "Failed to create CUDA session, falling back to CPU: %s", exc
                )
                self._providers = ["CPUExecutionProvider"]
                try:
                    self._session = ort.InferenceSession(
                        self._session_model_path,
                        providers=self._providers,
                    )
                except Exception as cpu_exc:
                    logger.error("Failed to load ONNX model on CPU: %s", cpu_exc)
                    return False
            else:
                logger.error("Failed to load ONNX model: %s", exc)
                return False

        self._is_first_frame = True
        self._cpu_fallback_done = False
        # Obtener nombres de entrada/salida
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if inputs:
            self._input_name = inputs[0].name
            self._input_shape = inputs[0].shape
        if outputs:
            self._output_name = outputs[0].name
            self._output_shape = outputs[0].shape

        # El pipeline siempre alimenta frames BGR de 3 canales. Si el modelo
        # ONNX tiene un canal fijo distinto (p.ej. 12) rechazamos la carga
        # ANTES de empezar el render: un crash a mitad de video es opaco
        # ("Got: 3 Expected: 12") y confunde al usuario.
        if self._input_shape and len(self._input_shape) >= 2:
            in_ch = self._input_shape[1]
            if isinstance(in_ch, int) and in_ch != 3:
                logger.error(
                    "ONNX model %s expects %d input channels; the frame "
                    "pipeline always feeds 3-channel BGR. Refusing to load.",
                    os.path.basename(self.model_path), in_ch,
                )
                return False
        # Altura/anchura fijas distintas de la del vídeo = .onnx mal exportado
        # (visto en el catálogo: un modelo con 64x64 fijos). Se avisa AL CARGAR
        # con la salida accionable, en vez de dejar que reviente en cada frame
        # del render (allí el motor se desactiva y el vídeo sale sin upscale).
        try:
            fixed_h = self._input_shape[2] if len(self._input_shape) > 2 else None
            fixed_w = self._input_shape[3] if len(self._input_shape) > 3 else None
            if isinstance(fixed_h, int) and isinstance(fixed_w, int) and fixed_h > 0 and fixed_w > 0:
                self._fixed_input_note = (
                    f"model has a FIXED {fixed_w}x{fixed_h} input; it can only "
                    f"process frames of that exact size"
                )
                logger.warning(
                    "ONNX model %s declares a fixed %dx%d input. Renders at any "
                    "other resolution will fail; re-convert the weights from the "
                    ".pth in Manage Dependencies to get a dynamic-shape export.",
                    os.path.basename(self.model_path), fixed_w, fixed_h,
                )
        except Exception:
            pass

        # En modo Auto con CUDA, hacemos una inferencia de prueba para
        # forzar el fallo de cuDNN (si existe) ANTES de que el pipeline
        # comience. Si falla, recreamos la sesión con CPU. Así el usuario
        # no ve el error en mitad del procesamiento real.
        # El dtype de alimentación se fija ANTES del warmup: con un modelo
        # fp16, alimentar float32 hace fallar la sesión
        # ("Unexpected input data type. Actual: tensor(float)").
        self._input_dtype = np.float16 if self._precision == "fp16" else np.float32
        if self._exec_mode == "Auto" and "CUDAExecutionProvider" in self._providers:
            try:
                self._dummy_inference()
            except Exception as exc:
                if not self._try_cpu_fallback(ort, "warmup"):
                    return False

        self._loaded = True
        # Reafirmado tras el fallback a CPU (por si cambió la precisión).
        self._input_dtype = np.float16 if self._precision == "fp16" else np.float32
        # Log which providers are ACTUALLY in use (ORT may silently drop one).
        actual = self._session.get_providers()
        logger.info(
            "ONNX model loaded: %s (requested: %s, active: %s, precision: %s%s)",
            os.path.basename(self.model_path), self._providers, actual,
            self._precision,
            f", {self._gpu_post_conversion_note}" if self._gpu_post_conversion_note else "",
        )
        if "CUDAExecutionProvider" not in actual and "CUDAExecutionProvider" in self._providers:
            logger.warning(
                "CUDA was requested but is NOT active. CUDA provider was dropped. "
                "Check that CUDA DLLs (cublas, cudnn, cufft) are on PATH."
            )
        return True

    def get_active_providers(self) -> list:
        """Devuelve los providers REALMENTE activos en la sesión.

        ORT puede descartar CUDA silenciosamente (DLLs ausentes, driver,
        cuDNN), por lo que ``_providers`` no basta: hay que mirar la sesión.
        """
        try:
            if self._session is not None:
                return list(self._session.get_providers())
        except Exception:
            pass
        return []

    def _try_cpu_fallback(self, ort, source=""):
        """Recreate the session with CPU provider and run a dummy inference.

        Returns True if the CPU session works, False otherwise.
        """
        try:
            logger.info("Falling back to CPU (source=%s)", source)
            self._providers = ["CPUExecutionProvider"]
            self._cpu_fallback_done = True
            self._session = ort.InferenceSession(
                self._session_model_path,
                providers=self._providers,
            )
            # Refresh input/output metadata after recreating session
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if inputs:
                self._input_name = inputs[0].name
                self._input_shape = inputs[0].shape
            if outputs:
                self._output_name = outputs[0].name
                self._output_shape = outputs[0].shape
            if self._input_shape and len(self._input_shape) >= 2:
                in_ch = self._input_shape[1]
                if isinstance(in_ch, int) and in_ch != 3:
                    logger.error(
                        "ONNX model %s expects %d input channels (CPU session); "
                        "refusing to load.",
                        os.path.basename(self.model_path), in_ch,
                    )
                    return False
            self._dummy_inference()
            logger.info("CPU fallback successful")
            return True
        except Exception as cpu_exc:
            logger.error("CPU fallback failed: %s", cpu_exc)
            return False

    def _dummy_inference(self):
        """Ejecuta una inferencia de prueba para validar que la sesión funciona.

        Usa un tensor dummy pequeño compatible con el shape de entrada del modelo.
        """
        if not self._input_shape or len(self._input_shape) < 4:
            # Asumimos entrada NCHW de imagen; 1x3x64x64 es seguro para la mayoría.
            dummy_shape = (1, 3, 64, 64)
        else:
            # Reemplazar dimensiones dinámicas (None o strings) por valores fijos.
            dummy_shape = []
            for dim in self._input_shape:
                try:
                    dim_int = int(dim)
                    dummy_shape.append(dim_int)
                except (TypeError, ValueError):
                    dummy_shape.append(64)
            if len(dummy_shape) < 4:
                dummy_shape = (1, 3, 64, 64)
            else:
                dummy_shape = tuple(dummy_shape[:4])

        dummy = np.ascontiguousarray(
            np.zeros(dummy_shape, dtype=getattr(self, "_input_dtype", np.float32))
        )
        return self._session.run(
            [self._output_name],
            {self._input_name: dummy},
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un fotograma BGR uint8 (H, W, 3) y devuelve BGR uint8.

        El pipeline de DRAGGY ENCODER lee BGR24 de FFmpeg (``-pix_fmt
        bgr24``). Esta engine convierte internamente a RGB antes de
        alimentar el modelo, y vuelve a BGR antes de devolverlo, para
        que la salida sea consistente con la entrada y con la engine
        PyTorch (que ya hacía la conversión).

        Pre (BGR→RGB):  invierte canales y aplica transpose HWC→CHW,
        float32 [0,1], añade batch dim y contigüidad para pinned memory.
        Post (RGB→BGR): squeeze, transpose CHW→HWC, clamp y vuelve a
        invertir canales antes del ``ascontiguousarray`` final.

        Antes de este fix, la engine no convertía canales y el output
        quedaba con R↔B invertidos cuando se usaba con el pipe BGR24.
        """
        if not self._loaded or self._session is None:
            raise RuntimeError("Model not loaded")

        # Asegurar formato correcto: (H, W, 3) uint8
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR frame (H, W, 3), got shape {frame.shape}")

        if self.gpu_post_active():
            return self._process_frame_gpu_post(frame)

        h, w = frame.shape[:2]

        t_pre = time.perf_counter()
        # BGR uint8 → RGB float32 [0,1] NCHW contiguous.
        # ``[:, :, ::-1]`` invierte el eje de canales (BGR ↔ RGB).
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        input_data = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # NCHW
        input_data = np.ascontiguousarray(input_data)
        if getattr(self, "_input_dtype", np.float32) != np.float32:
            input_data = input_data.astype(self._input_dtype)
        t_run = time.perf_counter()

        # Inferencia. En 'Auto' con CUDA, si cuDNN o el driver no están
        # disponibles (aunque get_available_providers reporte CUDA),
        # recreamos la sesión con CPU y reintentamos. El fallback se hace
        # una sola vez: una vez demotado a CPU, _providers ya no contiene
        # CUDAExecutionProvider.
        try:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: input_data},
            )
        except Exception as exc:
            # ONNX Runtime lanza excepciones propias (por ejemplo
            # onnxruntime.capi.onnxruntime_pybind11_state.Fail) que no son
            # RuntimeError estándar. Capturamos Exception para asegurar que
            # cualquier fallo de CUDA se traduce en fallback a CPU.
            if (
                self._exec_mode == "Auto"
                and "CUDAExecutionProvider" in self._providers
                and not self._cpu_fallback_done
            ):
                logger.warning(
                    "[%s] CUDA inference failed, falling back to CPU: %s",
                    type(exc).__name__, exc,
                )
                ort = import_local_runtime("onnxruntime")
                if self._try_cpu_fallback(ort, "runtime"):
                    outputs = self._session.run(
                        [self._output_name],
                        {self._input_name: input_data},
                    )
                else:
                    raise
            else:
                logger.error(
                    "ONNX inference failed (%s): %s",
                    type(exc).__name__, exc,
                )
                raise

        t_post = time.perf_counter()
        self._is_first_frame = False

        # NCHW → CHW → HWC → uint8 → BGR
        output = outputs[0]
        # numpy no tiene SIMD para float16: multiplicar/recortar un tensor
        # fp16 en numpy cuesta ~3x más que en float32 (medido: el post-proceso
        # clásico pasaba de 68 a 180 ms/frame con un modelo fp16). Se sube a
        # float32 antes de operar, que es una conversión vectorizada barata.
        if output.dtype == np.float16:
            output = output.astype(np.float32)
        output = np.squeeze(output, axis=0)  # CHW
        output = np.transpose(output, (1, 2, 0))  # HWC
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # RGB → BGR (invierte canales) y fija contigüidad para el pipe.
        output = np.ascontiguousarray(output[:, :, ::-1])
        self._add_timing((t_run - t_pre) * 1000.0,
                         (t_post - t_run) * 1000.0,
                         (time.perf_counter() - t_post) * 1000.0)
        return output

    # ------------------------------------------------- post-proceso GPU (P1)
    def _process_frame_gpu_post(self, frame: np.ndarray) -> np.ndarray:
        """Camino con cabeza/cola dentro del grafo (P1).

        La entrada va por ``session.run`` (numpy) y no por IO binding: en este
        build de ONNX Runtime (1.30 + CUDA) cualquier binding de ENTRADA por
        CPU (``bind_cpu_input``/``bind_input``/``bind_ortvalue_input`` con un
        OrtValue reutilizado) entrega ceros al grafo y la salida sale plana
        —verificado con el grafo envuelto—. ``session.run`` sí entrega el dato
        y además se queda con la única copia que hace falta: 1 MB de entrada y
        2.3 MB de salida uint8 ya al tamaño del encoder.
        """
        state = self._gpu_post or {}
        session = state.get("session")
        if session is None:
            # Estado incompleto: no es un fallo del render, es "vuelve al
            # camino de siempre" (nunca se desactiva la engine por esto).
            self.disable_gpu_post()
            return self.process_frame(frame)

        expected = state.get("input_size") or (0, 0)
        if expected[0] and tuple(frame.shape[:2]) != tuple(expected):
            # El grafo derivado está atado a un tamaño de entrada y a uno de
            # salida. Si llega otro tamaño (p. ej. otro nodo de la cadena
            # cambió la resolución) se vuelve al camino clásico: mejor perder
            # velocidad que dejar el frame sin procesar.
            self._gpu_post_failure = (
                f"input size changed ({frame.shape[1]}x{frame.shape[0]} vs "
                f"{expected[1]}x{expected[0]})"
            )
            logger.warning("gpu_post: %s; volviendo al camino clásico",
                           self._gpu_post_failure)
            self.disable_gpu_post()
            return self.process_frame(frame)

        t_pre = time.perf_counter()
        input_data = frame[None, ...]
        t_run = time.perf_counter()
        try:
            outputs = session.run([state["output_name"]],
                                  {state["input_name"]: input_data})
        except Exception as exc:
            # Cualquier fallo del grafo envuelto (memoria, kernel, dtype) cae
            # al camino clásico en vez de propagarse: el frame se procesa
            # igual, sólo que más despacio.
            self._gpu_post_failure = f"{type(exc).__name__}: {exc}"
            logger.error("gpu_post: falló el grafo envuelto (%s); volviendo al "
                         "camino clásico para el resto del render",
                         self._gpu_post_failure)
            self.disable_gpu_post()
            return self.process_frame(frame)
        t_post = time.perf_counter()
        output = outputs[0]
        if output.ndim == 4:
            output = output[0]
        self._add_timing((t_run - t_pre) * 1000.0,
                         (t_post - t_run) * 1000.0,
                         (time.perf_counter() - t_post) * 1000.0)
        self._is_first_frame = False
        return output

    def _build_gpu_post_session(self, model_path):
        """Sesión CUDA del grafo envuelto (opciones prudentes, sin CUDA Graph)."""
        ort = import_local_runtime("onnxruntime")
        options = {
            "cudnn_conv_algo_search": "HEURISTIC",
            "arena_extend_strategy": "kSameAsRequested",
        }
        providers = [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
        return ort.InferenceSession(model_path, providers=providers)

    def configure_gpu_post(self, width=None, height=None, resize_fn=None,
                           input_size=None) -> dict:
        """Construye y valida el grafo con cabeza/cola (P1).

        Devuelve ``{"active": bool, "reason": str, ...}``. Sólo se activa si:
          - el usuario lo pidió en Preferences (apagado por defecto),
          - hay CUDA activo,
          - el grafo derivado se construye, se ejecuta y su salida coincide
            con el camino actual dentro de tolerancia.

        Cualquier fallo devuelve ``active=False`` con el motivo, y el render
        continúa con el camino de siempre.
        """
        from src.ai import gpu_post

        if not gpu_post.gpu_post_enabled():
            return {"active": False, "reason": "preference-off"}
        if not self._loaded or self._session is None:
            return {"active": False, "reason": "not-loaded"}
        if "CUDAExecutionProvider" not in (self.get_active_providers() or []):
            return {"active": False, "reason": "no-cuda"}

        in_h, in_w = 0, 0
        if input_size:
            in_h, in_w = int(input_size[1]), int(input_size[0])
        if in_h <= 0 or in_w <= 0:
            return {"active": False, "reason": "input-size-unknown"}

        source = getattr(self, "_session_model_path", self.model_path)
        target = None
        if width and height:
            target = (int(width), int(height))
        path, info = gpu_post.build_wrapped_model(
            source, out_size=target, precision=self._precision,
        )
        if not path:
            return {"active": False, "reason": info.get("reason", "build-failed"),
                    "detail": info}

        try:
            session = self._build_gpu_post_session(path)
            state = {
                "session": session,
                "input_name": session.get_inputs()[0].name,
                "output_name": session.get_outputs()[0].name,
                "input_size": (in_h, in_w),
                "out_size": target,
            }
            # Prueba en frío: si el grafo no puede ejecutarse con este tamaño,
            # mejor enterarse aquí (antes del render) que en el frame 1.
            session.run([state["output_name"]],
                        {state["input_name"]: np.zeros((1, in_h, in_w, 3),
                                                       dtype=np.uint8)})
        except Exception as exc:
            return {"active": False, "reason": "session-failed",
                    "detail": f"{type(exc).__name__}: {exc}"}

        frames = gpu_post.sample_frames(in_h, in_w)

        def fast(fr):
            return self._run_with_state(state, fr)

        def reference(fr):
            out = self.process_frame(fr)
            if target and resize_fn:
                if (out.shape[1], out.shape[0]) != target:
                    out = resize_fn(out, target[0], target[1])
            return out

        ok, details = gpu_post.run_self_check(fast, reference, frames=frames)
        if not ok:
            logger.warning(
                "gpu_post: autocomprobación fallida (%s); se sigue con el "
                "camino de siempre", details.get("reason"),
            )
            return {"active": False, "reason": "check-failed", "detail": details}

        state["active"] = True
        state["precision"] = self._precision
        state["check"] = details
        self._gpu_post = state
        # Los frames de la autocomprobación no son frames del render.
        self.reset_cumulative_timing()
        logger.info(
            "gpu_post: ACTIVO (%s, salida %s, coarse_diff %.2f, channel_diff "
            "%.2f, max_diff %.1f)", self._precision, target or "sin resize",
            details.get("worst_coarse", 0.0), details.get("worst_channel", 0.0),
            details.get("worst_max", 0.0),
        )
        return {
            "active": True,
            "reason": "ok",
            "precision": self._precision,
            "out_size": target,
            "max_diff": round(float(details.get("worst_max", 0.0)), 2),
            "mean_diff": round(float(details.get("worst_mean", 0.0)), 3),
            "coarse_diff": round(float(details.get("worst_coarse", 0.0)), 2),
            "channel_diff": round(float(details.get("worst_channel", 0.0)), 2),
            "converted": bool(info.get("converted")),
            "cached": bool(info.get("cached")),
        }

    def _run_with_state(self, state, frame):
        """Ejecuta un frame con un estado de post-proceso concreto."""
        outputs = state["session"].run(
            [state["output_name"]], {state["input_name"]: frame[None, ...]},
        )
        output = outputs[0]
        return output[0] if output.ndim == 4 else output

    def disable_gpu_post(self) -> None:
        """Vuelve al camino clásico (lo usa el pipeline si algo se tuerce)."""
        self._gpu_post = None

    def describe_execution(self) -> dict:
        info = {
            "engine_id": self.engine_id,
            "backend": "onnxruntime",
            "provider": ",".join(self.get_active_providers() or []),
            "precision": self._precision,
            "precision_preference": self._precision_pref,
            "native_dtype": self._native_dtype,
            "gpu_post": self.gpu_post_active(),
        }
        if self._gpu_post_conversion_note:
            info["precision_note"] = self._gpu_post_conversion_note
        if getattr(self, "_fixed_input_note", ""):
            info["model_note"] = self._fixed_input_note
        if self.gpu_post_active():
            info["gpu_post_detail"] = {
                "out_size": list(self._gpu_post.get("out_size") or []),
                "max_diff": round(float(
                    (self._gpu_post.get("check") or {}).get("worst_max", 0.0)), 2),
                "coarse_diff": round(float(
                    (self._gpu_post.get("check") or {}).get("worst_coarse", 0.0)), 2),
                "channel_diff": round(float(
                    (self._gpu_post.get("check") or {}).get("worst_channel", 0.0)), 2),
            }
        return info

    def _detect_native_scale(self) -> int:
        """Factor de escala REAL del .onnx, leído por dentro.

        Las formas declaradas sólo sirven si H y W están fijos. En los SR
        exportados son dinámicas (``['batch', 3, 'height', 'width']`` en
        entrada y salida) y entonces la escala se lee del grafo
        (``Resize``/``Upsample``/``ConvTranspose``). Antes esto devolvía 2
        a ciegas para todo modelo dinámico, así que un x4 real se
        dimensionaba como si fuera x2: la mitad de píxeles de los que el
        modelo produce de verdad.
        """
        scale = 0
        if self._input_shape and self._output_shape:
            try:
                in_h = int(self._input_shape[2])
                out_h = int(self._output_shape[2])
                if in_h > 0 and out_h > 0:
                    scale = max(1, out_h // in_h)
            except (TypeError, ValueError, IndexError):
                scale = 0
        if scale <= 1:
            try:
                from src.ai.architectures import detect_onnx_scale
                scale = int(detect_onnx_scale(self.model_path))
            except Exception as exc:
                logger.debug("native scale from graph failed: %s", exc)
                scale = 0
        if scale <= 1:
            # Último recurso: la pista del nombre del fichero.
            try:
                from src.ai.architectures import parse_scale_from_name
                scale = int(parse_scale_from_name(os.path.basename(self.model_path)))
            except Exception:
                scale = 0
        if scale > 1:
            logger.info("ONNX native scale: x%d (%s)", scale,
                        os.path.basename(self.model_path))
        return max(1, int(scale))

    def get_scale_factor(self) -> int:
        """Factor de escala del modelo (2, 4, etc.)."""
        if self._native_scale is None:
            self._native_scale = self._detect_native_scale()
        native = self._native_scale
        # La escala solicitada por la UI manda si es menor que la nativa.
        if self._target_scale and 1 < self._target_scale < native:
            return int(self._target_scale)
        return native

    def unload(self) -> None:
        self._session = None
        self._loaded = False


class PyTorchInferenceEngine(AIInferenceEngine):
    """Motor de inferencia con PyTorch y soporte CUDA.

    Para modelos Real-ESRGAN (.pth) y Real-CUGAN (.pth) usando
    arquitecturas auto-detectadas desde los state_dict.
    """

    def __init__(self, model_path: str, engine_id: str):
        super().__init__(model_path, engine_id)
        self._model = None
        self._device = None
        self._scale = 4
        # Precisión (P2): "model" mantiene el comportamiento de siempre
        # (autocast fp16 en GPU), "fp16" lo fuerza y "fp32" lo desactiva.
        self._precision_pref = "model"
        self._autocast = True
        self._gpu_resize = None
        self._last_phases = (0.0, 0.0, 0.0)

    def _resolve_precision(self) -> None:
        """Lee la precisión elegida por el usuario (P2)."""
        try:
            from src.ai import gpu_post
            self._precision_pref = gpu_post.get_precision_preference()
        except Exception:
            self._precision_pref = "model"
        # "model" conserva el comportamiento histórico (autocast fp16 en GPU
        # cuando el modelo está en fp32): no cambiamos nada por defecto.
        self._autocast = self._precision_pref != "fp32"

    def load(self) -> bool:
        try:
            torch = import_local_runtime("torch")
        except ImportError:
            logger.error(
                "PyTorch is not installed. Open Manage Dependencies → "
                "Runtimes & Binaries to install it, or use Python 3.11–3.14 "
                "for automatic download."
            )
            return False

        if not os.path.exists(self.model_path):
            logger.error("Model not found: %s", self.model_path)
            return False

        # Prefer CUDA when available, but gracefully fall back to CPU if
        # the CUDA context cannot be created (broken drivers, missing DLLs,
        # out-of-memory at load time, etc.).
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._resolve_precision()

        try:
            from src.ai.architectures import build_model_from_state_dict
            if self.model_path.lower().endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(self.model_path, device="cpu")
            else:
                state_dict = load_checkpoint_state_dict(self.model_path)
            if not isinstance(state_dict, dict):
                logger.error("Unexpected checkpoint format: %s", type(state_dict))
                return False
            model = build_model_from_state_dict(state_dict)

            # El pipeline alimenta frames BGR de 3 canales; rechazar modelos
            # con otra entrada fija (p.ej. RRDBNet de 12 canales) antes de
            # gastar GPU y de reventar a mitad de render.
            try:
                first_conv = next(
                    m for m in model.modules()
                    if isinstance(m, torch.nn.Conv2d)
                )
                in_ch = first_conv.in_channels
            except Exception:
                in_ch = None
            if in_ch is not None and in_ch != 3:
                logger.error(
                    "Model %s expects %d input channels; the frame pipeline "
                    "always feeds 3-channel BGR. Refusing to load.",
                    os.path.basename(self.model_path), in_ch,
                )
                return False

            try:
                model = model.to(self._device)
            except Exception as cuda_exc:
                logger.warning(
                    "Failed to move model to %s, falling back to CPU: %s",
                    self._device, cuda_exc,
                )
                self._device = torch.device("cpu")
                model = model.to(self._device)

            model.eval()
            self._model = model

            # Detect scale factor from model output
            try:
                dummy = torch.randn(1, 3, 16, 16, device=self._device)
                with torch.no_grad():
                    out = model(dummy)
                self._scale = out.shape[2] // 16
            except Exception:
                self._scale = 4

            self._loaded = True
            logger.info("PyTorch model loaded: %s (device=%s, scale=%d)",
                        self.model_path, self._device, self._scale)
            return True
        except Exception as exc:
            logger.error("Failed to load PyTorch model: %s", exc, exc_info=True)
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un fotograma BGR uint8 (H, W, 3) y devuelve BGR uint8
        con el upscale aplicado.

        Con el post-proceso en GPU activo (P1) la salida llega ya al tamaño
        del encoder: el reescalado se hace en la tarjeta, evitando el viaje de
        ida y vuelta a PIL (medido: 34 ms por frame en 2880x1920 -> 1080x720).
        """
        t_pre = time.perf_counter()
        output = self._forward(frame, resize=self._gpu_resize if self.gpu_post_active() else None)
        self._add_timing(*self._last_phases)
        return output

    def _forward(self, frame: np.ndarray, resize=None) -> np.ndarray:
        """Inferencia completa; ``resize`` = (w, h) para reescalar en la GPU."""
        torch = import_local_runtime("torch")

        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()
        # BGR → RGB, HWC → CHW, float32 [0,1]
        rgb = frame[:, :, ::-1].copy()
        input_data = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.ascontiguousarray(input_data)

        input_tensor = torch.from_numpy(input_data).to(self._device)
        t1 = time.perf_counter()

        autocast = (self._autocast and torch.cuda.is_available()
                    and self._device.type == "cuda")
        with torch.no_grad():
            if autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    output_tensor = self._model(input_tensor)
            else:
                output_tensor = self._model(input_tensor)
        t2 = time.perf_counter()

        # CHW → HWC, clamp, scale and convert on device (fp16 numpy ops are
        # extremely slow), then RGB → BGR, uint8
        output = output_tensor.squeeze(0).clamp(0, 1) * 255.0
        if resize:
            width, height = int(resize[0]), int(resize[1])
            if (output.shape[2], output.shape[1]) != (width, height):
                output = torch.nn.functional.interpolate(
                    output.unsqueeze(0), size=(height, width),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
        output = output.to(torch.uint8)
        output = output.permute(1, 2, 0).cpu().numpy()
        output = np.ascontiguousarray(output[:, :, ::-1])  # RGB → BGR
        self._last_phases = ((t1 - t0) * 1000.0, (t2 - t1) * 1000.0,
                             (time.perf_counter() - t2) * 1000.0)
        return output

    def configure_gpu_post(self, width=None, height=None, resize_fn=None,
                           input_size=None) -> dict:
        """Reescalado en la GPU dentro del propio motor PyTorch (P1).

        Aquí no hay grafo que envolver: la conversión a uint8 ya se hacía en
        la tarjeta, así que lo único que se mueve es el reescalado final. Se
        valida igual que en ONNX (comparación contra el camino actual) antes
        de darlo por bueno.
        """
        from src.ai import gpu_post

        if not gpu_post.gpu_post_enabled():
            return {"active": False, "reason": "preference-off"}
        if not self._loaded or self._model is None:
            return {"active": False, "reason": "not-loaded"}
        if self._device is None or self._device.type != "cuda":
            return {"active": False, "reason": "no-cuda"}
        if not width or not height:
            return {"active": False, "reason": "no-target-size"}
        target = (int(width), int(height))

        in_h, in_w = (int(input_size[1]), int(input_size[0])) if input_size else (0, 0)
        if in_h <= 0 or in_w <= 0:
            in_h, in_w = 96, 128
        frames = gpu_post.sample_frames(in_h, in_w)

        def fast(fr):
            return self._forward(fr, resize=target)

        def reference(fr):
            out = self._forward(fr, resize=None)
            if (out.shape[1], out.shape[0]) != target:
                if resize_fn:
                    out = resize_fn(out, target[0], target[1])
            return out

        ok, details = gpu_post.run_self_check(fast, reference, frames=frames)
        if not ok:
            logger.warning("gpu_post (torch): autocomprobación fallida (%s)",
                           details.get("reason"))
            return {"active": False, "reason": "check-failed", "detail": details}
        self._gpu_resize = target
        self._gpu_post = {"active": True, "out_size": target, "check": details}
        self.reset_cumulative_timing()
        logger.info("gpu_post (torch): ACTIVO (salida %s, max_diff %.1f, "
                    "mean_diff %.2f)", target, details.get("worst_max", 0.0),
                    details.get("worst_mean", 0.0))
        return {
            "active": True, "reason": "ok", "precision": self._precision_pref,
            "out_size": target, "cuda_graph": False,
            "max_diff": round(float(details.get("worst_max", 0.0)), 2),
            "mean_diff": round(float(details.get("worst_mean", 0.0)), 3),
        }

    def disable_gpu_post(self) -> None:
        self._gpu_post = None
        self._gpu_resize = None

    def describe_execution(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "backend": "torch",
            "device": str(self._device),
            "precision": self._precision_pref,
            "autocast": bool(self._autocast),
            "gpu_post": self.gpu_post_active(),
            "gpu_post_detail": (
                {"out_size": list(self._gpu_resize or [])}
                if self.gpu_post_active() else {}
            ),
        }

    def get_scale_factor(self) -> int:
        # Si el usuario pidió una escala menor que la nativa del peso
        # (p.ej. x2 con un modelo x4), reportar la solicitada: la
        # pipeline dimensiona el encoder con este valor.
        if self._target_scale and 1 < self._target_scale < self._scale:
            return int(self._target_scale)
        return self._scale

    def unload(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
        self._device = None
        self._loaded = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_inference_engine(
    engine_id: str,
    model_path: str,
) -> AIInferenceEngine:
    """Crea el motor de inferencia apropiado según el engine_id.

    Para modelos .pth, intenta auto-convertir a ONNX para aprovechar
    la GPU via ONNX Runtime con CUDA. Si ONNX no funciona, cae a
    PyTorch, que también intentará usar CUDA si está disponible.

    Args:
        engine_id: "onnxruntime" o "torch".
        model_path: Ruta al archivo del modelo.

    Returns:
        Instancia de AIInferenceEngine lista para usar.

    Raises:
        RuntimeError: si ningún motor pudo cargarse.
    """
    last_error = None

    # If we already have an .onnx file, go straight to ONNX Runtime.
    if model_path and model_path.endswith(".onnx"):
        try:
            engine = ONNXRUNTIMEInferenceEngine(model_path, engine_id)
            if engine.load():
                return engine
        except Exception as exc:
            last_error = f"ONNX load failed: {exc}"
        raise RuntimeError(
            f"Could not load ONNX Runtime engine for {model_path}. {last_error or ''}"
        )

    # For .pth models, try ONNX conversion first (GPU via CUDA)
    if model_path and model_path.endswith((".pth", ".pt", ".safetensors")):
        try:
            from src.ai.architectures import is_torch_available
            if is_torch_available():
                onnx_path = convert_pth_to_onnx(model_path)
            else:
                onnx_path = None
                last_error = "PyTorch not installed — cannot convert .pth to .onnx"
                logger.warning(last_error)
        except Exception as exc:
            onnx_path = None
            last_error = f"ONNX conversion failed: {exc}"
            logger.warning(last_error)
        if onnx_path:
            logger.info("Using ONNX Runtime (CUDA) for %s", os.path.basename(model_path))
            try:
                engine = ONNXRUNTIMEInferenceEngine(onnx_path, engine_id)
                if engine.load():
                    return engine
            except Exception as exc:
                last_error = f"ONNX load failed: {exc}"
                logger.warning(last_error)
            logger.warning("ONNX load failed, falling back to PyTorch (CUDA/CPU)")

    if engine_id == "onnxruntime":
        try:
            engine = ONNXRUNTIMEInferenceEngine(model_path, engine_id)
            if engine.load():
                return engine
        except Exception as exc:
            last_error = f"ONNX load failed: {exc}"
        raise RuntimeError(
            f"Could not load ONNX Runtime engine for {model_path}. {last_error or ''}"
        )
    elif engine_id == "torch":
        torch_exc = None
        try:
            engine = PyTorchInferenceEngine(model_path, engine_id)
            if engine.load():
                return engine
            torch_exc = "PyTorch load returned False (check log for the underlying exception)"
        except Exception as exc:
            torch_exc = f"PyTorch load failed: {exc}"
        details = " ".join(str(x) for x in [last_error, torch_exc] if x)
        raise RuntimeError(
            f"Could not load PyTorch engine for {model_path}. {details}"
        )
    else:
        # Ids de catálogo (p.ej. "upscale-realesr-animevideov3") que
        # llegan aquí sin un ONNX convertible: intentamos PyTorch como
        # última vía antes de rendirnos.
        torch_exc = None
        if model_path and model_path.endswith((".pth", ".pt", ".safetensors")):
            try:
                engine = PyTorchInferenceEngine(model_path, engine_id)
                if engine.load():
                    return engine
                torch_exc = "PyTorch load returned False (check the log for the underlying exception)"
            except Exception as exc:
                torch_exc = f"PyTorch load failed: {exc}"
        details = " ".join(str(x) for x in [last_error, torch_exc] if x)
        raise RuntimeError(
            f"Could not load inference engine for {engine_id}. {details}"
        )


def find_onnx_model(model_dir: str) -> Optional[str]:
    """Busca un archivo .onnx en el directorio del modelo.

    Si el modelo fue convertido, el archivo está en model_dir/onnx/.
    Si no, busca en el directorio raíz.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return None

    # Buscar en subdirectorio onnx/ (OnnxConvertWorker lo crea allí)
    onnx_subdir = os.path.join(model_dir, "onnx")
    if os.path.isdir(onnx_subdir):
        for fname in os.listdir(onnx_subdir):
            if fname.endswith(".onnx"):
                return os.path.join(onnx_subdir, fname)

    # Buscar en el directorio raíz
    for fname in os.listdir(model_dir):
        if fname.endswith(".onnx"):
            return os.path.join(model_dir, fname)

    return None


def _onnx_input_channels_ok(onnx_path: str) -> Optional[bool]:
    """Probe barata de una sesión ORT.

    True  -> el canal de entrada fijo del modelo es 3 (o dinámico).
    False -> canal fijo != 3 (un .onnx huérfano convertido desde el archivo
             equivocado, p.ej. un RRDBNet de 12 canales).
    None  -> no se pudo verificar (onnxruntime ausente, sesión rota...). El
             llamador NO debe borrar el archivo: un .onnx previo puede ser
             perfectamente válido aunque ORT no pueda abrirlo en este entorno.
    """
    try:
        ort = import_local_runtime("onnxruntime")

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        if len(shape) >= 2 and isinstance(shape[1], int):
            return shape[1] == 3
        return True
    except Exception:
        return None


def convert_pth_to_onnx(pth_path: str) -> Optional[str]:
    """Convert a .pth model to .onnx for GPU inference via ONNX Runtime.

    Saves the .onnx alongside the .pth file. Returns the .onnx path
    or None on failure. Use :func:`convert_pth_to_onnx_ex` when the UI
    needs to know *why* a conversion failed.
    """
    path, _error = convert_pth_to_onnx_ex(pth_path)
    return path


#: Versión del grafo que produce ``torch.onnx.export``. Se sube cuando un
#: arreglo de arquitectura cambia lo que la red calcula: un ``.onnx`` exportado
#: con una versión anterior se RE-EXPORTA, porque si no seguiría dando una
#: imagen equivocada (oscura, borrosa) sin que nadie se enterase.
#: v2 = arreglo de RRDBNet (LeakyReLU de más tras la suma residual + orden
#:      incorrecto de conv_up/interpolate).
ONNX_EXPORT_VERSION = 2


def _export_marker_path(onnx_path: str) -> str:
    return onnx_path + ".export-version"


def _read_export_marker(onnx_path: str) -> int:
    """Versión con la que se exportó ese ``.onnx`` (0 si no hay marca)."""
    try:
        with open(_export_marker_path(onnx_path), "r", encoding="utf-8") as handle:
            return int(str(handle.read()).strip() or 0)
    except Exception:
        return 0


def _write_export_marker(onnx_path: str) -> None:
    try:
        with open(_export_marker_path(onnx_path), "w", encoding="utf-8") as handle:
            handle.write(str(ONNX_EXPORT_VERSION))
    except Exception as exc:
        logger.debug("could not write the ONNX export marker: %s", exc)


def convert_pth_to_onnx_ex(pth_path: str) -> tuple[Optional[str], str]:
    """Like :func:`convert_pth_to_onnx` but reports the failure reason.

    Returns ``(onnx_path, "")`` on success or ``(None, reason)`` where
    ``reason`` is a human-readable, user-facing explanation (missing
    weights, unsupported architecture, non-3-channel input, export
    failure, ...).
    """
    if not os.path.exists(pth_path):
        return None, f"Model weight file not found:\n{pth_path}"

    onnx_path = os.path.splitext(pth_path)[0] + ".onnx"
    stale_backup = None

    def _restore_stale():
        """Devuelve el .onnx viejo a su sitio si la re-exportación falló."""
        if not stale_backup:
            return
        try:
            if not os.path.exists(onnx_path) and os.path.exists(stale_backup):
                os.replace(stale_backup, onnx_path)
                logger.warning(
                    "Re-export failed; keeping the previous (outdated) ONNX "
                    "file at %s", onnx_path,
                )
        except Exception:
            logger.debug("could not restore the previous ONNX file", exc_info=True)

    if os.path.exists(onnx_path):
        # Reutilizar el .onnx solo si es compatible con el pipeline: un
        # .onnx huérfano (convertido desde el archivo equivocado, p.ej. con
        # 12 canales de entrada) reventaría el render. Si el probe no puede
        # verificar (None: onnxruntime ausente o sesión rota) se MANTIENE el
        # archivo: fue creado por una conversión previa y borrarlo sin poder
        # reconvertir dejaría al usuario sin ONNX. Solo se borra cuando se
        # prueba POSITIVAMENTE que el canal fijo no es 3.
        probe = _onnx_input_channels_ok(onnx_path)
        marker = _read_export_marker(onnx_path)
        if probe is not False and marker >= ONNX_EXPORT_VERSION:
            return onnx_path, ""
        if probe is False:
            try:
                os.remove(onnx_path)
            except OSError as exc:
                return None, f"Could not replace the existing ONNX file: {exc}"
        else:
            # Grafo exportado con una versión anterior de la app (o sin marca):
            # puede calcular mal. Se aparta y se re-exporta; si la
            # re-exportación falla se restaura, porque un ONNX viejo es mejor
            # que quedarse sin ninguno.
            try:
                stale_backup = onnx_path + ".stale"
                os.replace(onnx_path, stale_backup)
                logger.info(
                    "ONNX export version %s < %s: re-exporting %s",
                    marker, ONNX_EXPORT_VERSION, os.path.basename(onnx_path),
                )
            except OSError as exc:
                logger.warning("Could not move the outdated ONNX aside (%s); "
                               "keeping it as is", exc)
                return onnx_path, ""

    try:
        torch = import_local_runtime("torch")
        from src.ai.architectures import build_model_from_state_dict, detect_model_arch_and_scale

        # Redirect stdout/stderr to avoid cp1252 UnicodeEncodeError on Windows
        # during torch.onnx.export logging. We use devnull to suppress, then
        # read back any warnings from the logger.
        import io
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        _devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _devnull
        sys.stderr = _devnull

        error_msg = ""
        model = None
        try:
            logger.info("Converting %s to ONNX...", pth_path)
            if pth_path.lower().endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(pth_path, device="cpu")
            else:
                state_dict = load_checkpoint_state_dict(pth_path)
            if not isinstance(state_dict, dict):
                error_msg = "The weight file does not contain a state dict."
            else:
                # Unwrap common checkpoint wrappers (params_ema, params, state_dict)
                if "body.0.rdb1.conv1.weight" not in state_dict:
                    for wrapper_key in ("params_ema", "params", "state_dict"):
                        if wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
                            state_dict = state_dict[wrapper_key]
                            break

                # Detect architecture first so we fail fast with a helpful
                # message instead of a cryptic exporter error.
                detected = detect_model_arch_and_scale(state_dict)
                if not detected:
                    error_msg = (
                        "Unknown model architecture — only Real-ESRGAN "
                        "(RRDB / SRVGG), Real-CUGAN and OmniSR weights can "
                        "be converted to ONNX. The render will use PyTorch "
                        "instead."
                    )
                else:
                    try:
                        model = build_model_from_state_dict(state_dict)
                        model.eval()
                        error_msg = ""
                    except Exception as exc:
                        error_msg = f"Could not build the model architecture: {exc}"
            if error_msg:
                pass
            elif model is not None:
                # Detect num_in_ch from the appropriate key for each architecture
                if "conv_first.weight" in state_dict:
                    num_in_ch = state_dict["conv_first.weight"].shape[1]
                elif "body.0.weight" in state_dict:
                    num_in_ch = state_dict["body.0.weight"].shape[1]
                elif "input.weight" in state_dict:
                    num_in_ch = state_dict["input.weight"].shape[1]
                else:
                    num_in_ch = 3

                # El pipeline alimenta frames BGR de 3 canales: un modelo con
                # otra entrada fija (p.ej. RRDBNet de 12 canales) convertiría a
                # un ONNX que revienta en el primer frame real ("Expected: 12").
                if num_in_ch != 3:
                    error_msg = (
                        f"The model expects {num_in_ch} input channels, but "
                        "the render pipeline always feeds 3-channel images. "
                        "The render will use PyTorch instead."
                    )
                else:
                    dummy = torch.randn(1, num_in_ch, 64, 64)
                    try:
                        try:
                            import onnx  # noqa: F401
                        except ImportError:
                            raise RuntimeError(
                                "The Python package 'onnx' is missing. PyTorch "
                                "2.6+ needs it to run torch.onnx.export. Reinstall "
                                "PyTorch from 'Runtimes & Binaries' (it now "
                                "installs onnx automatically) and try again."
                            )
                        torch.onnx.export(
                            model, dummy, onnx_path,
                            opset_version=18,
                            input_names=["input"],
                            output_names=["output"],
                            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                                          "output": {0: "batch", 2: "height", 3: "width"}},
                            dynamo=False,
                        )
                    except Exception as exc:
                        # Si el export con ejes dinámicos falla por el operador
                        # (p.ej. F.interpolate en algunos RRDBNet), reintentar
                        # con tamaño fijo: menos flexible pero compatible con
                        # cualquier build de torch/onnx.
                        retry = None
                        try:
                            torch.onnx.export(
                                model, dummy, onnx_path,
                                opset_version=18,
                                input_names=["input"],
                                output_names=["output"],
                                dynamo=False,
                            )
                            retry = True
                        except Exception as retry_exc:
                            retry = f"{retry_exc}"
                        if retry is True:
                            error_msg = ""
                        elif "onnx" in str(exc).lower() and "not installed" in str(exc).lower():
                            error_msg = (
                                "torch.onnx.export needs the 'onnx' package, which "
                                "is not installed. Reinstall PyTorch from 'Runtimes "
                                "& Binaries' (it now installs onnx automatically). "
                                "The render will use PyTorch instead."
                            )
                        else:
                            error_msg = (
                                f"torch.onnx.export failed for this model: {exc} "
                                f"(fixed-size retry: {retry}). "
                                "The render will use PyTorch instead."
                            )
        finally:
            _devnull.close()
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr

        if error_msg:
            logger.warning("ONNX conversion skipped for %s: %s", pth_path, error_msg)
            try:
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
            except Exception:
                pass
            _restore_stale()
            return None, error_msg

        _write_export_marker(onnx_path)
        if stale_backup and os.path.exists(stale_backup):
            try:
                os.remove(stale_backup)
            except Exception:
                pass
        logger.info("ONNX conversion saved: %s", onnx_path)
        return onnx_path, ""
    except Exception as exc:
        logger.error("ONNX conversion failed for %s: %s", pth_path, exc)
        # Clean up partial file
        try:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
        except Exception:
            pass
        _restore_stale()
        return None, f"Unexpected error during conversion: {exc}"


# Extensiones de pesos PyTorch que el pipeline puede cargar directamente.
_TORCH_WEIGHT_EXTS = (".safetensors", ".bin", ".pth", ".pt")


def list_weight_files(model_dir: str, recursive: bool = True) -> list[str]:
    """List all weight files (.pth, .pt, .safetensors, .bin) in a directory.

    Returns paths sorted by depth (shallow first) then alphabetically.
    Used by the fallback logic to try alternative weight files when
    the first choice has incompatible channel count.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return []
    results: list[tuple[int, str]] = []
    if recursive:
        try:
            for root, dirs, files in os.walk(model_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                depth = root[len(model_dir):].count(os.sep)
                for fname in files:
                    if fname.lower().endswith(_TORCH_WEIGHT_EXTS):
                        results.append((depth, os.path.join(root, fname)))
        except OSError:
            pass
    else:
        try:
            results = [
                (0, os.path.join(model_dir, fname))
                for fname in os.listdir(model_dir)
                if fname.lower().endswith(_TORCH_WEIGHT_EXTS)
            ]
        except OSError:
            pass
    results.sort(key=lambda c: (c[0], c[1].lower()))
    return [path for _, path in results]


def pick_weight_file(model_dir: str, engine_id: str,
                     recursive: bool = True) -> Optional[str]:
    """Elige el archivo de pesos que MEJOR coincide con ``engine_id``.

    Los repos de modelos contienen varios pesos (p.ej. el repo de
    Real-ESRGAN trae 9 .pth: 6 RRDBNet + 3 SRVGG). Coger "el primero en
    orden de listado" era un azar: el conversor ONNX podía convertir el
    modelo equivocado (un RRDBNet de 67 MB en vez del SRVGG de 2,4 MB
    del AnimeVideo v3) o un peso con 12 canales de entrada que reventaba
    el render. Se puntúa cada archivo por coincidencia con los tokens
    del engine_id y se elige el mejor; en empate gana el más superficial
    (los pesos del catálogo viven en la raíz).

    ``recursive=False`` limita la búsqueda a la raíz de ``model_dir``
    (comportamiento histórico de ``get_model_path``).
    """
    if not model_dir or not os.path.isdir(model_dir):
        return None
    skip_parts = ("upscale", "interpolate", "denoise", "colorize", "facerestore")
    tokens = [
        t for t in re.split(r"[^a-z0-9]+", (engine_id or "").lower())
        if t and t not in skip_parts
    ]
    joined = "-".join(tokens)

    candidates: list[tuple[int, str]] = []  # (depth, abspath)
    if recursive:
        try:
            for root, dirs, files in os.walk(model_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                depth = root[len(model_dir):].count(os.sep)
                for fname in files:
                    if fname.lower().endswith(_TORCH_WEIGHT_EXTS):
                        candidates.append((depth, os.path.join(root, fname)))
        except OSError:
            pass
    else:
        try:
            candidates = [
                (0, os.path.join(model_dir, fname))
                for fname in os.listdir(model_dir)
                if fname.lower().endswith(_TORCH_WEIGHT_EXTS)
            ]
        except OSError:
            pass
    if not candidates:
        return None

    def _score(fname: str) -> int:
        n = re.sub(r"[^a-z0-9]+", "-", fname.lower()).strip("-")
        score = sum(1 for t in tokens if t in n)
        if joined and joined in n:
            score += 1
        return score

    best_path: Optional[str] = None
    best_key = (-1, 0)  # (score, -depth): a igual score gana la raíz
    for depth, path in candidates:
        score = _score(os.path.basename(path))
        key = (score, -depth)
        if key > best_key:
            best_key, best_path = key, path
    if best_path is None or best_key[0] <= 0:
        # Sin coincidencia con los tokens: orden estable por (profundidad,
        # ruta) para no depender del orden aleatorio del walk.
        best_path = min(candidates, key=lambda c: (c[0], c[1].lower()))[1]
    return best_path


def get_model_path(engine_id_catalog: str, engine_id_inference: str = "onnxruntime") -> Optional[str]:
    """Obtiene la ruta al modelo para un engine del catálogo.

    Args:
        engine_id_catalog: ID del engine en el catálogo (ej: "upscale-realesrgan-x4plus").
        engine_id_inference: Motor de inferencia ("onnxruntime", "torch").

    Returns:
        Ruta al archivo del modelo, o None si no está disponible.
    """
    try:
        from src.ai.engines_state import get_state
        from src.ai.registry import get_engine_dir

        state = get_state(engine_id_catalog) or {}
        if not state.get("installed"):
            return None

        model_dir = state.get("install_path") or get_engine_dir(engine_id_catalog)

        if engine_id_inference == "onnxruntime":
            # Si está convertido a ONNX, usar el archivo ONNX
            if state.get("onnx_converted") and state.get("onnx_path"):
                return state["onnx_path"]
            # Si no está convertido, buscar .onnx en el directorio
            onnx_path = find_onnx_model(model_dir)
            if onnx_path:
                return onnx_path
            # No hay ONNX disponible
            return None
        elif engine_id_inference == "torch":
            # Buscar archivos de peso de PyTorch puntuando por coincidencia
            # con el engine_id (misma lógica que pick_weight_file, limitada a
            # la raíz para no cambiar el comportamiento histórico).
            return pick_weight_file(model_dir, engine_id_catalog, recursive=False)

    except Exception as exc:
        logger.error("Failed to get model path: %s", exc)
        return None
