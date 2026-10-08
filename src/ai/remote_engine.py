"""Motor de inferencia que corre en el HOST, no en el proceso de la GUI.

``AIInferenceEngine`` es la interfaz que usa el pipeline (``load``,
``process_frame``, ``get_scale_factor``, telemetría...). Esta implementación
cumple la misma interfaz enviando el trabajo al proceso del host: los frames
viajan como blobs crudos (ver ``src.ai.host_protocol``) y vuelven igual.

**Por qué.** Importar torch/onnxruntime dentro del proceso congelado de la GUI
cargaba cientos de MB de extensiones nativas y Windows mataba la aplicación
con una violación de acceso (0xc0000005 en python313.dll). Con este motor, el
proceso que carga torch es el host: si un modelo nativo revienta, se lleva su
proceso y la app sigue viva con un error accionable.

**Qué NO hace (todavía).** El post-proceso en GPU (P1) se declara inactivo a
propósito: hoy vive en ``src/ai/gpu_post.py``, que carga onnxruntime en el
proceso que lo ejecute. Activarlo desde aquí devolvería un motor que la GUI
creería envuelto y no lo está; el pipeline usa entonces el camino estándar,
que es exactamente el fallback que ya tiene para cuando P1 no aplica.
"""

from __future__ import annotations

import logging

from src.ai import host_client

logger = logging.getLogger(__name__)


class RemoteInferenceEngine:
    """Proxy del motor real: la carga y el cómputo ocurren en el host."""

    def __init__(self, engine_id: str, model_path: str):
        self.engine_id = engine_id
        self.model_path = model_path
        self._handle: str | None = None
        self._loaded = False
        self._scale = 1.0
        self._execution: dict | str = {}
        # El pipeline consulta estos dos atributos para degradar un motor que
        # falló a mitad de render (``_apply_steps``); se exponen igual que en
        # los motores locales.
        self._failed = False
        self._failure_detail: dict = {}
        self._target_scale = None
        self._gpu_post = None

    # ------------------------------------------------------------- contrato
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Crea y carga el motor EN EL HOST (misma fábrica que la app)."""
        host = host_client.get_host()
        self._handle = host.engine_create(self.engine_id, self.model_path or "")
        info = host.engine_info(self._handle)
        self._loaded = True
        try:
            self._scale = float(info.get("scale_factor") or 1.0)
        except (TypeError, ValueError):
            self._scale = 1.0
        self._execution = info.get("execution") or {}
        if info.get("engine_id"):
            # La fábrica del host puede haber resuelto otro backend (p. ej. un
            # .pth convertido a ONNX): el pipeline etiqueta con lo real.
            self.engine_id = str(info["engine_id"])
        logger.info(
            "motor remoto listo: %s (%s) escala x%s",
            self.engine_id, self.model_path, self._scale,
        )
        return True

    def process_frame(self, frame, resize=None):
        """Envía un frame al host y devuelve el frame procesado."""
        if not self._loaded or self._handle is None:
            raise RuntimeError("Model not loaded")
        out = host_client.get_host().engine_process(
            self._handle, [frame], resize=resize
        )
        if not out:
            raise RuntimeError("el host no devolvió frame")
        return out[0]

    def unload(self) -> None:
        if self._handle is not None:
            host_client.get_host().engine_unload(self._handle)
        self._handle = None
        self._loaded = False

    # -------------------------------------------------------------- estado
    def get_scale_factor(self) -> float:
        return self._scale

    def describe_execution(self):
        return self._execution

    def get_cumulative_timing(self) -> dict:
        """Telemetría real del host (misma forma que los motores locales)."""
        if self._handle is None:
            return {"frames": 0, "pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}
        try:
            info = host_client.get_host().engine_info(self._handle)
        except host_client.HostError as exc:
            logger.warning("sin telemetría del host: %s", exc)
            return {"frames": 0, "pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}
        timing = info.get("timing") or {}
        return {
            "frames": int(timing.get("frames") or 0),
            "pre_ms": float(timing.get("pre_ms") or 0.0),
            "infer_ms": float(timing.get("infer_ms") or 0.0),
            "post_ms": float(timing.get("post_ms") or 0.0),
        }

    def reset_cumulative_timing(self) -> None:
        """No-op: el host acumula la suya propia (el pipeline solo usa deltas)."""
        return None

    # ------------------------------------------------------- post GPU (P1)
    def configure_gpu_post(self, width=None, height=None, resize_fn=None,
                           input_size=None) -> dict:
        """P1 sigue en el proceso local: aquí se declara INACTIVO, no fingido."""
        reason = (
            "el post-proceso en GPU aún no vive en el host de IA; se usa el "
            "camino estándar (mismo resultado, sin la optimización)"
        )
        self._gpu_post = None
        return {"active": False, "reason": reason}

    @property
    def gpu_post_active(self) -> bool:
        return False

    def disable_gpu_post(self) -> None:
        self._gpu_post = None

    def __repr__(self) -> str:  # pragma: no cover - ayuda de diagnóstico
        return f"<RemoteInferenceEngine {self.engine_id} {self.model_path!r} loaded={self._loaded}>"
