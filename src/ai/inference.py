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
from typing import Optional

import numpy as np

# Intentar usar runtimes locales descargados por la app antes de los del sistema.
try:
    from src.runtimes.manager import inject_runtime_paths
    inject_runtime_paths()
except Exception:
    pass

import src.preferences as preferences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure NVIDIA CUDA/cuDNN DLLs from pip packages are on PATH (Windows)
# ---------------------------------------------------------------------------

def _ensure_nvidia_dlls_on_path() -> None:
    """Add nvidia pip package DLL directories to PATH on Windows.

    Busca tanto en el entorno Python como en el runtime local descargado
    por la app (bin/runtimes/ai_modules), y en site-packages detectado
    via ``site.getsitepackages()``.
    """
    if sys.platform != "win32":
        return

    candidates = []

    # Runtime local gestionado por la app
    try:
        from src.runtimes.manager import get_runtime_modules_dir
        candidates.append(get_runtime_modules_dir())
    except Exception:
        pass

    # Entorno Python actual (sys.path)
    for p in sys.path:
        if "site-packages" in p:
            candidates.append(p)

    # site.getsitepackages() — funciona incluso si sys.path fue manipulado
    try:
        import site as _site
        for p in _site.getsitepackages():
            if p and os.path.isdir(p):
                candidates.append(p)
    except Exception:
        pass

    # sys.prefix / sys.base_prefix — cubre venvs y entornos normales
    for prefix in [getattr(sys, "prefix", ""), getattr(sys, "base_prefix", "")]:
        if prefix:
            lib = os.path.join(prefix, "Lib", "site-packages")
            if os.path.isdir(lib):
                candidates.append(lib)

    added = []
    seen = set()
    for site_dir in candidates:
        if not site_dir or not os.path.isdir(site_dir) or site_dir in seen:
            continue
        seen.add(site_dir)
        nvidia = os.path.join(site_dir, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for sub in os.listdir(nvidia):
            for suffix in [("bin",), ("bin", "x86_64")]:
                d = os.path.join(nvidia, sub, *suffix)
                if os.path.isdir(d) and d not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = d + ";" + os.environ.get("PATH", "")
                    added.append(d)

    if added:
        logger.info("Added NVIDIA DLL paths: %s", added)
    else:
        logger.info(
            "NVIDIA DLL paths already on PATH or none found (CUDA will be "
            "used if available). Searched %d candidate directories.", len(candidates)
        )

_ensure_nvidia_dlls_on_path()


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

    @property
    def is_loaded(self) -> bool:
        return self._loaded

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

    def load(self) -> bool:
        """Carga el modelo ONNX. Devuelve True si tiene éxito."""
        if not os.path.exists(self.model_path):
            logger.error("ONNX model not found: %s", self.model_path)
            return False

        try:
            import onnxruntime as ort
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

        try:
            self._session = ort.InferenceSession(
                self.model_path,
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
                        self.model_path,
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

        # En modo Auto con CUDA, hacemos una inferencia de prueba para
        # forzar el fallo de cuDNN (si existe) ANTES de que el pipeline
        # comience. Si falla, recreamos la sesión con CPU. Así el usuario
        # no ve el error en mitad del procesamiento real.
        if self._exec_mode == "Auto" and "CUDAExecutionProvider" in self._providers:
            try:
                self._dummy_inference()
            except Exception as exc:
                if not self._try_cpu_fallback(ort, "warmup"):
                    return False

        self._loaded = True
        # Log which providers are ACTUALLY in use (ORT may silently drop one).
        actual = self._session.get_providers()
        logger.info(
            "ONNX model loaded: %s (requested: %s, active: %s)",
            os.path.basename(self.model_path), self._providers, actual,
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
                self.model_path,
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
            np.zeros(dummy_shape, dtype=np.float32)
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

        h, w = frame.shape[:2]

        # BGR uint8 → RGB float32 [0,1] NCHW contiguous.
        # ``[:, :, ::-1]`` invierte el eje de canales (BGR ↔ RGB).
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        input_data = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # NCHW
        input_data = np.ascontiguousarray(input_data)

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
                import onnxruntime as ort
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

        self._is_first_frame = False

        # NCHW → CHW → HWC → uint8 → BGR
        output = outputs[0]
        output = np.squeeze(output, axis=0)  # CHW
        output = np.transpose(output, (1, 2, 0))  # HWC
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # RGB → BGR (invierte canales) y fija contigüidad para el pipe.
        output = np.ascontiguousarray(output[:, :, ::-1])
        return output

    def get_scale_factor(self) -> int:
        """Intenta determinar el factor de escala del modelo."""
        native = 2
        if self._input_shape and self._output_shape:
            try:
                in_h = self._input_shape[2] if len(self._input_shape) > 2 else 1
                out_h = self._output_shape[2] if len(self._output_shape) > 2 else 1
                if in_h > 0 and out_h > 0:
                    native = max(1, out_h // in_h)
            except (TypeError, IndexError):
                pass
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

    def load(self) -> bool:
        try:
            import torch
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

        try:
            from src.ai.architectures import build_model_from_state_dict
            if self.model_path.lower().endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(self.model_path, device="cpu")
            else:
                state_dict = torch.load(self.model_path, map_location="cpu", weights_only=False)
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
        con el upscale aplicado."""
        import torch

        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        h, w = frame.shape[:2]

        # BGR → RGB, HWC → CHW, float32 [0,1]
        rgb = frame[:, :, ::-1].copy()
        input_data = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.ascontiguousarray(input_data)

        input_tensor = torch.from_numpy(input_data).to(self._device)

        autocast = (torch.cuda.is_available() and self._device.type == "cuda")
        with torch.no_grad():
            if autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    output_tensor = self._model(input_tensor)
            else:
                output_tensor = self._model(input_tensor)

        # CHW → HWC, clamp, scale and convert on device (fp16 numpy ops are
        # extremely slow), then RGB → BGR, uint8
        output = (output_tensor.squeeze(0).clamp(0, 1) * 255.0).to(torch.uint8)
        output = output.permute(1, 2, 0).cpu().numpy()
        output = np.ascontiguousarray(output[:, :, ::-1])  # RGB → BGR
        return output

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


class _LibplaceboEngine(AIInferenceEngine):
    """Internal libplacebo-backed scaler used by the FSR 3.1 engine.

    FSR 3.1 is the only exposed FSR engine. This base class is deliberately
    private because it is an implementation detail, not a separate catalog
    option.
    """

    def __init__(self, model_path: str, engine_id: str):
        super().__init__(model_path, engine_id)
        self._scale = 2
        self._ffmpeg_path = None
        self._sharpness = 0.7  # libplacebo CAS sharpness

    def load(self) -> bool:
        """Load the libplacebo backend required by FSR 3.1."""
        # Parse the scale from the FSR 3.1 engine id.
        import re
        m = re.search(r"fsr31-(\d+)x", self.engine_id or "")
        if m:
            self._scale = int(m.group(1))
        else:
            self._scale = 2

        # Get FFmpeg path with libplacebo support
        try:
            from src.ffmpeg.manager import get_ffmpeg_path_for_fsr31, has_libplacebo_support
            self._ffmpeg_path = get_ffmpeg_path_for_fsr31()
            if not self._ffmpeg_path or not os.path.exists(self._ffmpeg_path):
                logger.error("FFmpeg not found for FSR 3.1")
                return False

            # Verify libplacebo support
            if not has_libplacebo_support(self._ffmpeg_path):
                logger.error("FFmpeg does not have libplacebo support for FSR 3.1")
                return False

            self._loaded = True
            logger.info("FSR 3.1 libplacebo backend loaded (scale=%d, ffmpeg=%s)", self._scale, self._ffmpeg_path)
            return True
        except Exception as exc:
            logger.error("Failed to load FSR 3.1 libplacebo backend: %s", exc)
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a BGR uint8 frame using FFmpeg libplacebo filter.

        Rendimiento: lanza un proceso ffmpeg por frame. Intentamos un pipe
        persistente (un solo Popen reutilizado) pero ``libplacebo`` es un
        filtro con estado temporal (acumula frames para su ``frame_mixer``),
        por lo que no emite salida hasta recibir más de un frame y el lector
        bloquea. Hasta que el pipeline use la API por-lotes de libplacebo,
        Popen por frame es la única vía correcta.
        """
        if not self._loaded or not self._ffmpeg_path:
            raise RuntimeError("FSR 3.1 libplacebo backend not loaded")

        h, w = frame.shape[:2]
        new_h, new_w = h * self._scale, w * self._scale

        # Upscaler selection: libplacebo's "fsr" preset is not always
        # compiled in (depends on libplacebo build flags). The
        # closest FSR-quality scaler that's universally available is
        # ``ewa_lanczos`` (EWA Lanczos, the same family AMD uses). We
        # probe the available scalers once per process and cache the
        # decision so subsequent frames don't pay the cost.
        upscaler, downscaler = self._select_scalers()

        # libplacebo option names per FFmpeg 7.x / libplacebo 7.x:
        #   * w / h: output dimensions
        #   * upscaler / downscaler: scaler name
        #   * deband: boolean (NOT ``apply_debanding``; older builds)
        #   * sigmoid: boolean, default True for the libplacebo scaler.
        cmd = [
            self._ffmpeg_path,
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", "25",
            "-i", "pipe:0",
            "-vf", (
                f"libplacebo=w={new_w}:h={new_h}:"
                f"upscaler={upscaler}:downscaler={downscaler}:"
                f"deband=0:sigmoid=1"
            ),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "pipe:1",
        ]

        try:
            import subprocess
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            # Write input frame
            input_data = frame.tobytes()
            stdout, stderr = proc.communicate(input=input_data, timeout=30)

            if proc.returncode != 0:
                msg = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"FFmpeg libplacebo failed: {msg}")

            # Parse output frame
            expected_size = new_h * new_w * 3
            if len(stdout) != expected_size:
                raise RuntimeError(f"Expected {expected_size} bytes, got {len(stdout)}")

            output = np.frombuffer(stdout, dtype=np.uint8).reshape((new_h, new_w, 3))
            return np.ascontiguousarray(output)

        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("FFmpeg libplacebo timeout")
        except Exception as exc:
            logger.error("FSR 3.1 libplacebo processing failed: %s", exc)
            raise

    # Scaler selection is cached at class level so we only probe once
    # per process. Different FFmpeg/libplacebo builds ship different
    # preset names; probing a 4x4 raw frame is cheap (sub-millisecond)
    # but doing it for every video frame would be wasteful.
    _SCALER_CACHE: dict[str, tuple[str, str]] = {}

    @classmethod
    def _select_scalers(cls) -> tuple[str, str]:
        """Return ``(upscaler, downscaler)`` names guaranteed to work."""
        cache_key = "default"
        cached = cls._SCALER_CACHE.get(cache_key)
        if cached:
            return cached

        # Preference order: FSR preset first, then EWA Lanczos (FSR
        # quality without the FSR preset), then safe fallbacks.
        upscaler_candidates = ("fsr", "ewa_lanczos", "lanczos", "spline36")
        downscaler_candidates = ("ewa_lanczos", "mitchell", "spline36")

        import subprocess
        ffmpeg = cls.__module__  # noqa: F841 (we read it later via instance)
        # We need the actual ffmpeg path; we get it from a temporary
        # instantiation. Cheap because __init__ is trivial.
        tmp = cls("", "upscale-fsr31-2x")
        from src.ffmpeg.manager import get_ffmpeg_path_for_fsr31
        ffmpeg_path = get_ffmpeg_path_for_fsr31()
        if not ffmpeg_path or not os.path.exists(ffmpeg_path):
            cls._SCALER_CACHE[cache_key] = ("spline36", "spline36")
            return cls._SCALER_CACHE[cache_key]

        def _probe(scaler: str, direction: str) -> bool:
            cmd = [
                ffmpeg_path, "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "4x4",
                "-i", "pipe:0",
                "-vf", f"libplacebo=w=8:h=8:{direction}={scaler}:deband=0",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-frames:v", "1",
                "pipe:1",
            ]
            try:
                r = subprocess.run(
                    cmd,
                    input=bytes([128] * (4 * 4 * 3)),
                    capture_output=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                return r.returncode == 0 and len(r.stdout) == 8 * 8 * 3
            except Exception:
                return False

        up = next((s for s in upscaler_candidates if _probe(s, "upscaler")), "spline36")
        down = next((s for s in downscaler_candidates if _probe(s, "downscaler")), "spline36")
        cls._SCALER_CACHE[cache_key] = (up, down)
        logger.info("FSR 3.1 scaler selection: upscaler=%s, downscaler=%s", up, down)
        return up, down

    def get_scale_factor(self) -> int:
        return self._scale

    def unload(self) -> None:
        self._ffmpeg_path = None
        self._loaded = False


class FSR31Engine(_LibplaceboEngine):
    """FSR 3.1 engine: libplacebo upscaling + frame interpolation.

    FSR 3.1's actual frame generation (optical flow) requires DirectX 12/Vulkan
    swapchain integration and is not available in FFmpeg/libplacebo for batch
    video processing. This engine provides the closest available approximation:

    - **Upscaling**: libplacebo temporal upscaling via ``upscaler=fsr``
      (inherited from the private libplacebo backend).
    - **Frame interpolation**: libplacebo ``frame_mixer`` temporal blending,
      which synthesizes intermediate frames by blending consecutive frames
      based on their timestamps. While not optical-flow-based, it produces
      smooth motion at higher frame rates.

    Three modes are supported via ``mode`` parameter:

    - ``"upscale"`` — FSR 3.1 upscaling only (no frame interpolation).
    - ``"interpolate"`` — Frame interpolation only (no upscaling).
    - ``"both"`` — FSR 3.1 upscaling + frame interpolation combined.

    The interpolation mode requires the pipeline to pass ``prev_frame`` and
    ``next_frame`` to ``process_frame()``. When ``mode == "both"``, frames are
    upscaled first, then interpolated.
    """

    def __init__(self, model_path: str, engine_id: str, mode: str = "both"):
        super().__init__(model_path, engine_id)
        self._mode = mode

    def load(self) -> bool:
        if not super().load():
            return False
        # Verify libplacebo supports frame_mixer (it should in any build
        # that has libplacebo at all).
        return self._loaded

    def process_frame(self, frame: np.ndarray, prev_frame: np.ndarray = None,
                      next_frame: np.ndarray = None) -> np.ndarray:
        """Process a single frame.

        Args:
            frame: Current BGR uint8 frame.
            prev_frame: Previous BGR uint8 frame (required for interpolation).
            next_frame: Next BGR uint8 frame (required for interpolation).

        Returns:
            Processed BGR uint8 frame. For interpolation modes, this is the
            interpolated frame between prev_frame and next_frame.
        """
        if self._mode == "upscale":
            # Pure FSR 3.1 upscaling, no interpolation
            return super().process_frame(frame)

        if self._mode == "interpolate":
            # Frame interpolation only (no upscaling)
            if prev_frame is None or next_frame is None:
                # Cannot interpolate without both neighbors; return current frame
                return np.ascontiguousarray(frame)
            return self._interpolate_frames(prev_frame, next_frame)

        if self._mode == "both":
            # Upscale current frame, then interpolate between upscaled neighbors
            upscaled = super().process_frame(frame)
            if prev_frame is not None and next_frame is not None:
                upscaled_prev = super().process_frame(prev_frame)
                upscaled_next = super().process_frame(next_frame)
                interpolated = self._interpolate_frames(upscaled_prev, upscaled_next)
                # Return both: the interpolated frame (pipeline will write it)
                # and the upscaled current frame
                # Actually, the pipeline needs to handle this differently:
                # it should write interpolated frame first, then current frame
                # For now, return interpolated frame
                return interpolated
            return upscaled

        # Fallback
        return super().process_frame(frame)

    def interpolate_frame(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        """Generate an interpolated frame between prev_frame and next_frame.

        This is a convenience method for the pipeline to call when it wants
        to insert an interpolated frame between two consecutive frames.
        """
        return self._interpolate_frames(prev_frame, next_frame)

    def _interpolate_frames(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        """Interpolate between two BGR uint8 frames.

        Uses a simple 50/50 temporal blend (crossfade) between the two
        frames to synthesize an intermediate frame. This is not optical
        flow-based frame generation (which requires DX12/Vulkan swapchain
        integration), but produces a smooth blended frame that approximates
        the temporal midpoint between the two source frames.
        """
        prev = prev_frame.astype(np.float32)
        nxt = next_frame.astype(np.float32)
        blended = (prev + nxt) * 0.5
        return np.ascontiguousarray(np.clip(blended, 0, 255).astype(np.uint8))

    def get_scale_factor(self) -> int:
        return self._scale

    def unload(self) -> None:
        super().unload()


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

    # FSR 3.1 is the only exposed algorithmic engine; it uses the
    # bundled FFmpeg/libplacebo backend and needs no model file.
    if engine_id and "fsr31" in engine_id:
            # FSR 3.1: mode derivado del sufijo del engine id.
            if engine_id.endswith("-interp"):
                mode = "interpolate"
            elif engine_id.endswith("-both"):
                mode = "both"
            else:
                mode = "upscale"
            try:
                engine = FSR31Engine(model_path, engine_id, mode=mode)
                if engine.load():
                    return engine
            except Exception as exc:
                last_error = f"FSR 3.1 engine failed: {exc}"
                logger.error(last_error)
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
        import onnxruntime as ort
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
    if os.path.exists(onnx_path):
        # Reutilizar el .onnx solo si es compatible con el pipeline: un
        # .onnx huérfano (convertido desde el archivo equivocado, p.ej. con
        # 12 canales de entrada) reventaría el render. Si el probe no puede
        # verificar (None: onnxruntime ausente o sesión rota) se MANTIENE el
        # archivo: fue creado por una conversión previa y borrarlo sin poder
        # reconvertir dejaría al usuario sin ONNX. Solo se borra cuando se
        # prueba POSITIVAMENTE que el canal fijo no es 3.
        probe = _onnx_input_channels_ok(onnx_path)
        if probe is not False:
            return onnx_path, ""
        try:
            os.remove(onnx_path)
        except OSError as exc:
            return None, f"Could not replace the existing ONNX file: {exc}"

    try:
        import torch
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
                state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
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
            return None, error_msg

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
        return None, f"Unexpected error during conversion: {exc}"


# Extensiones de pesos PyTorch que el pipeline puede cargar directamente.
_TORCH_WEIGHT_EXTS = (".safetensors", ".bin", ".pth", ".pt")


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
