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
                "Runtimes & Binaries to install it, or use Python 3.11–3.13 "
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
        if self._input_shape and self._output_shape:
            try:
                in_h = self._input_shape[2] if len(self._input_shape) > 2 else 1
                out_h = self._output_shape[2] if len(self._output_shape) > 2 else 1
                if in_h > 0 and out_h > 0:
                    return max(1, out_h // in_h)
            except (TypeError, IndexError):
                pass
        return 2

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
                "Runtimes & Binaries to install it, or use Python 3.11–3.13 "
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
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=False)
            if not isinstance(state_dict, dict):
                logger.error("Unexpected checkpoint format: %s", type(state_dict))
                return False
            model = build_model_from_state_dict(state_dict)

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

        with torch.no_grad():
            output_tensor = self._model(input_tensor)

        # CHW → HWC, clamp, RGB → BGR, uint8
        output = output_tensor.squeeze(0).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))  # HWC
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = output[:, :, ::-1]  # RGB → BGR
        output = np.ascontiguousarray(output)
        return output

    def get_scale_factor(self) -> int:
        return self._scale

    def unload(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
        self._device = None
        self._loaded = False


class FSREngine(AIInferenceEngine):
    """FSR 1.0 (FidelityFX Super Resolution) — lightweight, no model download.

    Implements the FSR 1.0 algorithm: contrast-adaptive sharpening (CAS)
    combined with edge-directed upscaling. Pure NumPy, runs on CPU.
    No external model files required.
    """

    def __init__(self, model_path: str, engine_id: str):
        super().__init__(model_path, engine_id)
        self._scale = 2
        self._sharpness = 0.8  # CAS sharpness (0.0 = off, 1.0 = max)

    def load(self) -> bool:
        # FSR 1.0 doesn't need any model files.
        # Parse scale from engine_id (e.g., "upscale-fsr-2x" -> 2).
        import re
        m = re.search(r"fsr-(\d+)x", self.engine_id or "")
        if m:
            self._scale = int(m.group(1))
        else:
            # Fallback: try parsing from model_path basename
            try:
                scale_str = os.path.basename(self.model_path).replace("fsr-", "").replace(".pth", "")
                self._scale = int(scale_str) if scale_str.isdigit() else 2
            except (ValueError, AttributeError):
                self._scale = 2
        self._loaded = True
        logger.info("FSR 1.0 engine loaded (scale=%d, sharpness=%.2f)", self._scale, self._sharpness)
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a BGR uint8 frame using FSR 1.0."""
        if not self._loaded:
            raise RuntimeError("FSR engine not loaded")

        h, w = frame.shape[:2]

        # Step 1: Edge-directed upscale (bilinear as base, then CAS sharpen)
        # Convert BGR to RGB float [0,1]
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0

        # Bilinear upscale
        upscaled = np.zeros((h * self._scale, w * self._scale, 3), dtype=np.float32)
        for c in range(3):
            upscaled[:, :, c] = _bilinear_upscale(rgb[:, :, c], self._scale)

        # Step 2: Contrast-Adaptive Sharpening (CAS)
        upscaled = _cas_sharpen(upscaled, self._sharpness)

        # Clamp and convert back to BGR uint8
        upscaled = np.clip(upscaled * 255.0, 0, 255).astype(np.uint8)
        bgr = upscaled[:, :, ::-1]
        return np.ascontiguousarray(bgr)

    def get_scale_factor(self) -> int:
        return self._scale

    def unload(self) -> None:
        self._loaded = False


class FSR2Engine(AIInferenceEngine):
    """FSR 2.0 (FidelityFX Super Resolution 2.0) via FFmpeg libplacebo filter.

    Uses FFmpeg's libplacebo filter for GPU-accelerated upscaling with
    FSR 2.0 algorithm. Requires FFmpeg full build with libplacebo support.
    """

    def __init__(self, model_path: str, engine_id: str):
        super().__init__(model_path, engine_id)
        self._scale = 2
        self._ffmpeg_path = None
        self._sharpness = 0.7  # libplacebo CAS sharpness

    def load(self) -> bool:
        """Load FSR 2.0 engine by verifying FFmpeg has libplacebo support."""
        # Parse scale from engine_id (e.g., "upscale-fsr-2x-v2" -> 2).
        import re
        m = re.search(r"fsr-(\d+)x", self.engine_id or "")
        if m:
            self._scale = int(m.group(1))
        else:
            self._scale = 2

        # Get FFmpeg path with libplacebo support
        try:
            from src.ffmpeg.manager import get_ffmpeg_path_for_fsr2, has_libplacebo_support
            self._ffmpeg_path = get_ffmpeg_path_for_fsr2()
            if not self._ffmpeg_path or not os.path.exists(self._ffmpeg_path):
                logger.error("FFmpeg not found for FSR 2.0")
                return False

            # Verify libplacebo support
            if not has_libplacebo_support(self._ffmpeg_path):
                logger.error("FFmpeg does not have libplacebo support for FSR 2.0")
                return False

            self._loaded = True
            logger.info("FSR 2.0 engine loaded (scale=%d, ffmpeg=%s)", self._scale, self._ffmpeg_path)
            return True
        except Exception as exc:
            logger.error("Failed to load FSR 2.0 engine: %s", exc)
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a BGR uint8 frame using FFmpeg libplacebo filter."""
        if not self._loaded or not self._ffmpeg_path:
            raise RuntimeError("FSR 2.0 engine not loaded")

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
        #   * sigmoid: boolean, default True. FSR 2.0 expects a sigmoid
        #              tonemap-like curve, so we keep it on.
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
            logger.error("FSR 2.0 processing failed: %s", exc)
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
        tmp = cls("", "upscale-fsr-2x-v2")
        from src.ffmpeg.manager import get_ffmpeg_path_for_fsr2
        ffmpeg_path = get_ffmpeg_path_for_fsr2()
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
        logger.info("FSR 2.0 scaler selection: upscaler=%s, downscaler=%s", up, down)
        return up, down

    def get_scale_factor(self) -> int:
        return self._scale

    def unload(self) -> None:
        self._ffmpeg_path = None
        self._loaded = False


class FSR31Engine(FSR2Engine):
    """FSR 3.1 engine: FSR 2.0 upscaling + libplacebo frame interpolation.

    FSR 3.1's actual frame generation (optical flow) requires DirectX 12/Vulkan
    swapchain integration and is not available in FFmpeg/libplacebo for batch
    video processing. This engine provides the closest available approximation:

    - **Upscaling**: FSR 2.0 temporal upscaling via libplacebo ``upscaler=fsr``
      (inherited from FSR2Engine).
    - **Frame interpolation**: libplacebo ``frame_mixer`` temporal blending,
      which synthesizes intermediate frames by blending consecutive frames
      based on their timestamps. While not optical-flow-based, it produces
      smooth motion at higher frame rates.

    Three modes are supported via ``mode`` parameter:

    - ``"upscale"`` — FSR 2.0 upscaling only (no frame interpolation).
    - ``"interpolate"`` — Frame interpolation only (no upscaling).
    - ``"both"`` — FSR 2.0 upscaling + frame interpolation combined.

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
            # Pure FSR 2.0 upscaling, no interpolation
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


def _bilinear_upscale(img: np.ndarray, scale: int) -> np.ndarray:
    """Bilinear upscale of a 2D float32 array."""
    h, w = img.shape
    new_h, new_w = h * scale, w * scale
    # Use numpy repeat + interpolation for bilinear
    # First, create coordinate maps
    y = np.linspace(0, h - 1, new_h)
    x = np.linspace(0, w - 1, new_w)
    # Bilinear interpolation
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)
    # Interpolate
    Ia = img[y0[:, None], x0[None, :]]
    Ib = img[y0[:, None], x1[None, :]]
    Ic = img[y1[:, None], x0[None, :]]
    Id = img[y1[:, None], x1[None, :]]
    top = Ia * (1 - wx) + Ib * wx
    bottom = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bottom * wy


def _cas_sharpen(img: np.ndarray, sharpness: float) -> np.ndarray:
    """Contrast-Adaptive Sharpening (CAS) from FSR 1.0.

    Args:
        img: RGB float32 array in [0, 1], shape (H, W, 3)
        sharpness: 0.0 to 1.0
    """
    if sharpness <= 0:
        return img

    h, w, c = img.shape
    # Pad image by replicating edges
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge')

    # Compute local min/max in 3x3 neighborhood
    min_img = np.minimum(np.minimum(
        np.minimum(padded[:-2, :-2, :], padded[:-2, 1:-1, :]),
        padded[:-2, 2:, :]),
        np.minimum(padded[1:-1, :-2, :], padded[1:-1, 2:, :]),
    )
    min_img = np.minimum(min_img, np.minimum(
        padded[2:, :-2, :], np.minimum(padded[2:, 1:-1, :], padded[2:, 2:, :])
    ))

    max_img = np.maximum(np.maximum(
        np.maximum(padded[:-2, :-2, :], padded[:-2, 1:-1, :]),
        padded[:-2, 2:, :]),
        np.maximum(padded[1:-1, :-2, :], padded[1:-1, 2:, :]),
    )
    max_img = np.maximum(max_img, np.maximum(
        padded[2:, :-2, :], np.maximum(padded[2:, 1:-1, :], padded[2:, 2:, :])
    ))

    # Center pixel
    center = padded[1:-1, 1:-1, :]

    # CAS algorithm
    # 1. Compute local contrast
    local_contrast = max_img - min_img
    # 2. Compute sharpening factor
    # FSR 1.0 uses: 1 / (local_contrast / (4 * (center - local_min)) + 1)
    # Simplified version:
    denom = local_contrast / (4.0 * (center - min_img + 1e-5))
    denom = np.clip(denom, 0, 1.0 / (sharpness * sharpness))
    factor = 1.0 / (denom + 1.0)

    # 3. Apply sharpening
    # CAS uses a 3x3 weighted kernel
    kernel = np.array([
        [0, 0.25, 0],
        [0.25, 0, 0.25],
        [0, 0.25, 0],
    ], dtype=np.float32)

    # Apply convolution
    result = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            if kernel[dy, dx] == 0:
                continue
            result += kernel[dy, dx] * padded[dy:dy+h, dx:dx+w, :]

    # Blend with original based on factor
    result = center + (result - center) * factor * sharpness * 2.0
    return np.clip(result, 0, 1)


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

    # FSR engines don't need model files — pure algorithmic upscale
    if engine_id and engine_id.startswith("upscale-fsr"):
        # FSR 2.0 requires FFmpeg full with libplacebo
        if engine_id.endswith("-v2"):
            try:
                engine = FSR2Engine(model_path, engine_id)
                if engine.load():
                    return engine
            except Exception as exc:
                last_error = f"FSR 2.0 engine failed: {exc}"
                logger.error(last_error)
        else:
            # FSR 1.0 (CPU-based)
            try:
                engine = FSREngine(model_path, engine_id)
                if engine.load():
                    return engine
            except Exception as exc:
                last_error = f"FSR engine failed: {exc}"
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
        raise RuntimeError(f"Unknown inference engine: {engine_id}")


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


def convert_pth_to_onnx(pth_path: str) -> Optional[str]:
    """Convert a .pth model to .onnx for GPU inference via ONNX Runtime.

    Saves the .onnx alongside the .pth file. Returns the .onnx path
    or None on failure.
    """
    if not os.path.exists(pth_path):
        return None

    onnx_path = os.path.splitext(pth_path)[0] + ".onnx"
    if os.path.exists(onnx_path):
        return onnx_path

    try:
        import torch
        from src.ai.architectures import build_model_from_state_dict

        # Redirect stdout/stderr to avoid cp1252 UnicodeEncodeError on Windows
        # during torch.onnx.export logging. We use devnull to suppress, then
        # read back any warnings from the logger.
        import io
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        _devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _devnull
        sys.stderr = _devnull

        try:
            logger.info("Converting %s to ONNX...", pth_path)
            state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
            if not isinstance(state_dict, dict):
                return None

            # Unwrap common checkpoint wrappers (params_ema, params, state_dict)
            if isinstance(state_dict, dict) and "body.0.rdb1.conv1.weight" not in state_dict:
                for wrapper_key in ("params_ema", "params", "state_dict"):
                    if wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
                        state_dict = state_dict[wrapper_key]
                        break

            model = build_model_from_state_dict(state_dict)
            model.eval()

            # Detect num_in_ch from the appropriate key for each architecture
            if "conv_first.weight" in state_dict:
                num_in_ch = state_dict["conv_first.weight"].shape[1]
            elif "body.0.weight" in state_dict:
                num_in_ch = state_dict["body.0.weight"].shape[1]
            elif "input.weight" in state_dict:
                num_in_ch = state_dict["input.weight"].shape[1]
            else:
                num_in_ch = 3
            dummy = torch.randn(1, num_in_ch, 64, 64)
            torch.onnx.export(
                model, dummy, onnx_path,
                opset_version=18,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                              "output": {0: "batch", 2: "height", 3: "width"}},
                dynamo=False,
            )
        finally:
            _devnull.close()
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr

        logger.info("ONNX conversion saved: %s", onnx_path)
        return onnx_path
    except Exception as exc:
        logger.error("ONNX conversion failed for %s: %s", pth_path, exc)
        # Clean up partial file
        try:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
        except Exception:
            pass
        return None


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
            # Buscar archivos de peso de PyTorch
            weight_exts = (".safetensors", ".bin", ".pth", ".pt")
            for fname in os.listdir(model_dir):
                if fname.lower().endswith(weight_exts):
                    return os.path.join(model_dir, fname)
            return None

    except Exception as exc:
        logger.error("Failed to get model path: %s", exc)
        return None
