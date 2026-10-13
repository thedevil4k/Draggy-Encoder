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

**El post-proceso en GPU (P1) también vive en el host.** Activar P1 en el
proceso de la GUI no era una opción (es justo el import que se quitó de ahí),
así que ``configure_gpu_post`` cruza el protocolo: el host construye el grafo
envuelto (cabeza/cola dentro de ONNX, resize final en la tarjeta) sobre el
motor que ya tiene cargado, lo autocomprueba y responde si quedó activo. El
``resize_fn`` del pipeline no puede cruzar el protocolo, así que el host usa su
propio reescalado equivalente (mismo filtro: cv2 AREA/LINEAR con PIL y numpy
de respaldo) para la autocomprobación.

**Sin peticiones extra por frame.** La foto del motor (telemetría acumulada,
si P1 sigue activo, motivo del abandono) viaja DENTRO de la respuesta de
``engine.process``: leerla aparte costaba dos round-trips por frame (~0.3 ms
cada uno más el coste de serializar, y con un render de 20k frames eso son
40k mensajes que no aportan nada). El cliente la guarda y aquí se lee gratis.
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
        # Post-proceso en GPU (P1): dict con ``active`` cuando el HOST lo dio
        # por bueno, None mientras no esté activo. ``_gpu_post_failure`` es el
        # motivo por el que se abandonó a mitad de render (si pasó).
        self._gpu_post = None
        self._gpu_post_failure = None
        # Última foto que mandó el host (ver módulo): identidad, escala,
        # telemetría y estado de P1.
        self._state: dict = {}
        self._timing = {"frames": 0, "pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}

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
        self._adopt_state(info)
        logger.info(
            "motor remoto listo: %s (%s) escala x%s",
            self.engine_id, self.model_path, self._scale,
        )
        return True

    def process_frame(self, frame, resize=None):
        """Envía un frame al host y devuelve el frame procesado.

        Después de cada frame se adopta la foto que vino con la respuesta: así
        la telemetría y el estado de P1 están al día sin pedir nada aparte, y
        si el host abandonó el camino GPU a mitad de render (tamaño de entrada
        distinto, fallo del grafo) el pipeline se entera y lo reporta.
        """
        if not self._loaded or self._handle is None:
            raise RuntimeError("Model not loaded")
        out = host_client.get_host().engine_process(
            self._handle, [frame], resize=resize
        )
        self._adopt_state(host_client.get_host().engine_last_state(self._handle))
        if not out:
            raise RuntimeError("el host no devolvió frame")
        return out[0]

    def unload(self) -> None:
        if self._handle is not None:
            try:
                host_client.get_host().engine_unload(self._handle)
            finally:
                self._handle = None
                self._loaded = False

    # -------------------------------------------------------------- estado
    def _adopt_state(self, state) -> None:
        """Fusiona una foto del host en el estado de este proxy. Nunca lanza."""
        if not isinstance(state, dict) or not state:
            return
        try:
            self._state.update(state)
            timing = state.get("timing")
            if isinstance(timing, dict):
                self._timing = {
                    "frames": int(timing.get("frames") or 0),
                    "pre_ms": float(timing.get("pre_ms") or 0.0),
                    "infer_ms": float(timing.get("infer_ms") or 0.0),
                    "post_ms": float(timing.get("post_ms") or 0.0),
                }
            execution = state.get("execution")
            if isinstance(execution, dict):
                self._execution = execution
            if "gpu_post_active" in state:
                if state["gpu_post_active"]:
                    self._gpu_post = {"active": True,
                                      "out_size": (self._gpu_post or {}).get("out_size")}
                elif self._gpu_post is not None:
                    # El host lo apagó (o el motor lo abandonó). No es un fallo
                    # del render, es volver al camino clásico: se anota para
                    # que el log del render lo diga.
                    self._gpu_post = None
            failure = state.get("gpu_post_failure")
            if failure and not self._gpu_post_failure:
                self._gpu_post_failure = str(failure)
                logger.warning("gpu_post desactivado en el host: %s", failure)
        except Exception:  # telemetría best-effort, jamás romper un frame
            logger.debug("no se pudo adoptar el estado del host", exc_info=True)

    def get_scale_factor(self) -> float:
        return self._scale

    def describe_execution(self):
        """Cómo corre el motor de verdad, según el host."""
        if isinstance(self._execution, dict):
            return {**self._execution, "gpu_post": self.gpu_post_active(),
                    "remote": True}
        return {"engine_id": self.engine_id, "backend": str(self._execution),
                "gpu_post": self.gpu_post_active(), "remote": True}

    def get_cumulative_timing(self) -> dict:
        """Telemetría real del host (misma forma que los motores locales).

        Se devuelve la última foto recibida, que el host manda con cada frame:
        pedirla aparte serían dos round-trips extra por frame (el pipeline la
        lee antes y después de cada llamada al motor). Si todavía no hay foto
        (nada procesado y sin ``engine.info`` previo), se pide una vez.
        """
        if self._handle is None:
            return {"frames": 0, "pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}
        if not self._state:
            try:
                self._adopt_state(host_client.get_host().engine_info(self._handle))
            except host_client.HostError as exc:
                logger.warning("sin telemetría del host: %s", exc)
        return dict(self._timing)

    def reset_cumulative_timing(self) -> None:
        """No-op: el host acumula la suya propia (el pipeline solo usa deltas)."""
        return None

    # ------------------------------------------------------- post GPU (P1)
    def configure_gpu_post(self, width=None, height=None, resize_fn=None,
                           input_size=None, in_scale=None) -> dict:
        """Pide al HOST activar P1 en este motor (ver docstring del módulo).

        ``in_scale`` viaja también: es el ajuste de detalle (cuánta resolución
        de la fuente entra al modelo), el único que cambia el coste de la
        inferencia. El host es quien construye el grafo, así que él decide.

        Devuelve el mismo informe que devolvería el motor local, más los
        motivos por los que no se pudo activar. Nunca lanza: si el host no
        está, el render sigue por el camino clásico.
        """
        if self._handle is None:
            return {"active": False, "reason": "not-loaded"}
        try:
            report = host_client.get_host().engine_gpu_post(
                self._handle,
                width=width,
                height=height,
                input_size=list(input_size) if input_size else None,
                in_scale=in_scale,
            )
        except host_client.HostError as exc:
            logger.error("gpu_post rechazado por el host: %s", exc)
            self._gpu_post = None
            return {"active": False, "reason": f"host-error:{type(exc).__name__}"}
        self._adopt_state(report.pop("info", None))
        if report.get("active"):
            self._gpu_post = {
                "active": True,
                "out_size": report.get("out_size"),
                "precision": report.get("precision"),
                "detail": report.get("detail", 1.0),
            }
            logger.info("gpu_post ACTIVO en el host (salida %s, %s, detalle %s)",
                        report.get("out_size"), report.get("precision"),
                        report.get("detail", 1.0))
        else:
            self._gpu_post = None
            logger.info("gpu_post no disponible en el host: %s",
                        report.get("reason"))
        return report

    def gpu_post_active(self) -> bool:
        """True si el HOST confirmó que este motor corre con P1 (misma
        interfaz que los motores locales: ``AIInferenceEngine.gpu_post_active``)."""
        try:
            return bool(self._gpu_post and self._gpu_post.get("active"))
        except Exception:
            return False

    def disable_gpu_post(self) -> None:
        """Vuelve al camino clásico localmente.

        El motor real (y su grafo envuelto) son del host: aquí solo se olvida
        la marca, que es lo que el pipeline consulta. Nada en la aplicación
        llama a esto sobre un motor remoto; el host se desactiva solo cuando un
        frame no encaja con el grafo y lo reporta en la siguiente respuesta.
        """
        self._gpu_post = None

    def __repr__(self) -> str:  # pragma: no cover - ayuda de diagnóstico
        return (f"<RemoteInferenceEngine {self.engine_id} {self.model_path!r} "
                f"loaded={self._loaded} gpu_post={self.gpu_post_active()}>")
