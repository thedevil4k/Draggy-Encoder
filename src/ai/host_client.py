"""Cliente del host de IA: arranca el proceso y le habla por el protocolo.

La GUI usa esta clase para todo lo que necesite torch/onnxruntime. El host es
un proceso APARTE lanzado con el Python privado del instalador, así que las
extensiones nativas que tumbaban DraggyEncoder.exe (0xc0000005 en
python313.dll) ya no se cargan en el proceso de la interfaz: si el host muere,
la aplicación sigue viva y muestra un error accionable.

El host se arranca de forma perezosa (al primer uso) y es único por proceso.
``DRAGGY_AI_HOST=off`` lo desactiva por completo (escape hatch de soporte):
entonces ``host_available()`` devuelve False y los llamadores degradan.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import threading

from src.ai import host_protocol as proto

logger = logging.getLogger(__name__)

#: Tiempos máximos por tipo de petición (segundos).
PING_TIMEOUT = 60.0
OP_TIMEOUT = 900.0
LOAD_TIMEOUT = 1800.0
PROCESS_TIMEOUT = 3600.0

#: Arranque del host: importa python, localiza el store... y responde ping.
START_TIMEOUT = 60.0

_BOOTSTRAP = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from src.ai.host import main\n"
    "sys.exit(main())\n"
)

#: Variables que pueden hacer que el host importe un Python u otro entorno
#: (el host debe ver exactamente el Python privado y el store de la app).
_ENV_STRIP = (
    "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
)


class HostError(RuntimeError):
    """Fallo genérico al comunicarse con el host de IA."""


class HostUnavailable(HostError):
    """No hay host posible: sin Python privado, sin fuentes o desactivado."""


class HostCrashed(HostError):
    """El host murió mientras trabajaba (posible fallo de sus DLL nativas)."""


def host_source_root() -> str | None:
    """Directorio que contiene el paquete ``src`` que ejecutará el host.

    Empaquetado: ``ai_src`` junto al árbol del bundle (lo copia el spec desde
    ``src/``). En fuentes: la raíz del repo, que ya contiene ``src/``.
    """
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
        for candidate in (
            os.path.join(base, "ai_src"),
            os.path.join(os.path.dirname(sys.executable), "ai_src"),
        ):
            if os.path.isdir(os.path.join(candidate, "src")):
                return candidate
        return None
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return root if os.path.isdir(os.path.join(root, "src")) else None


def host_python() -> str | None:
    """Intérprete del host: el Python privado del bundle, o el actual en fuentes."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    try:
        from src.runtimes.manager import get_embed_python_path

        return get_embed_python_path()
    except Exception:
        logger.warning("no se pudo resolver el Python privado para el host")
        return None


def host_disabled() -> bool:
    """True si el usuario/soporte desactivó el host (``DRAGGY_AI_HOST=off``)."""
    return os.environ.get("DRAGGY_AI_HOST", "").strip().lower() in ("off", "0", "false")


def host_available() -> bool:
    """¿Se puede arrancar el host? (no lo arranca, solo comprueba los requisitos)."""
    if host_disabled():
        return False
    return bool(host_source_root() and host_python())


def _host_env() -> dict:
    env = dict(os.environ)
    for key in _ENV_STRIP:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        from src.config import get_data_dir

        # El host debe resolver el MISMO store que la app. En el bundle su
        # ``sys.executable`` es el del embed-python, así que sin esta variable
        # miraría su propio directorio y no vería los runtimes instalados.
        env["DRAGGY_DATA_DIR"] = get_data_dir()
    except Exception:
        pass
    return env


def host_log_path() -> str | None:
    try:
        from src.config import get_logs_dir

        logs = get_logs_dir()
        os.makedirs(logs, exist_ok=True)
        return os.path.join(logs, "ai_host.log")
    except Exception:
        return None


class RuntimeHost:
    """Proceso host + canal de peticiones (una petición en vuelo a la vez)."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._log_handle = None
        self._crashed_reason = ""
        self.ping_info: dict = {}
        #: Última foto que el host mandó de cada motor (telemetría + post GPU).
        #: Viaja DENTRO de la respuesta de procesado: pedirla aparte costaría
        #: dos peticiones más por frame, y en frames grandes eso es fps.
        self._engine_state: dict[str, dict] = {}

    # ------------------------------------------------------------- proceso
    def start(self, timeout: float = START_TIMEOUT) -> bool:
        """Arranca el host y hace el handshake. Lanza ``HostUnavailable``."""
        if host_disabled():
            raise HostUnavailable(
                "el host de IA está desactivado (DRAGGY_AI_HOST=off); borra esa "
                "variable para volver a usarlo"
            )
        python = host_python()
        root = host_source_root()
        if not (python and root):
            raise HostUnavailable(
                "no hay host de IA disponible ("
                f"python={python or 'no encontrado'}, fuentes={root or 'no encontradas'})"
            )
        log_path = host_log_path()
        try:
            self._log_handle = open(log_path, "a", encoding="utf-8") if log_path else None
        except OSError:
            self._log_handle = None
        creationflags = 0
        if os.name == "nt":
            # Sin ventana de consola: el host es un proceso de servicio.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                [python, "-I", "-c", _BOOTSTRAP, root],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle or subprocess.DEVNULL,
                env=_host_env(),
                cwd=root,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise HostUnavailable(f"no se pudo lanzar el host de IA ({python}): {exc}")
        self._crashed_reason = ""
        try:
            info, _ = self.request("ping", timeout=timeout)
        except HostError:
            self.stop()
            raise
        if int(info.get("protocol", 0)) != proto.PROTOCOL_VERSION:
            self.stop()
            raise HostUnavailable(
                "el host de IA tiene otro protocolo "
                f"({info.get('protocol')} != {proto.PROTOCOL_VERSION}); "
                "reinstala la aplicación"
            )
        self.ping_info = info
        logger.info(
            "host de IA listo: pid=%s python=%s store=%s",
            info.get("pid"), info.get("python"), info.get("store"),
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Pide el shutdown y limpia (idempotente)."""
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.poll() is None:
                    with self._lock:
                        proto.write_message(proc.stdin, {"id": 0, "job": "shutdown", "data": {}})
            except Exception:
                pass
            try:
                proc.wait(timeout=timeout)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def ensure_started(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self.stop()
            self.start()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ping(self) -> dict:
        """Arranca el host si hace falta y devuelve su identidad (versión, store...)."""
        self.ensure_started()
        return dict(self.ping_info)

    # ------------------------------------------------------------ peticiones
    def request(self, job: str, data: dict | None = None, blobs=(),
                blob_meta=(), timeout: float | None = OP_TIMEOUT,
                progress_callback=None):
        """Envía un job y devuelve ``(data, blobs)`` de la respuesta.

        Raises:
            HostUnavailable: el host no está en marcha y no se puede arrancar.
            HostCrashed: el proceso murió durante la petición.
            HostError: el job respondió con error (mensaje del host).
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise HostUnavailable(self._dead_message())
            req_id = self._next_id
            self._next_id += 1
            header = {"id": req_id, "job": job, "data": dict(data or {})}
            if blob_meta:
                header["blob_meta"] = [dict(m) for m in blob_meta]
            try:
                proto.write_message(proc.stdin, header, blobs)
            except Exception as exc:
                raise HostCrashed(f"no se pudo escribir al host de IA: {exc}") from exc

            watchdog = None
            if timeout:
                def _on_timeout():
                    # OJO: NO se toma ``self._lock`` aquí (lo tiene la petición
                    # en curso). Matar el proceso cierra sus pipes y hace que
                    # el ``read_message`` de abajo salga con HostCrashed en vez
                    # de quedarse colgado para siempre.
                    self._crashed_reason = (
                        f"el host de IA no respondió en {int(timeout)}s al job "
                        f"'{job}'; se mata y se reinicia"
                    )
                    logger.error(self._crashed_reason)
                    try:
                        proc.kill()
                    except Exception:
                        pass

                watchdog = threading.Timer(timeout, _on_timeout)
                watchdog.daemon = True
                watchdog.start()
            response = None
            out_blobs = []
            try:
                while True:
                    resp, resp_blobs = proto.read_message(proc.stdout)
                    if resp.get("id") not in (req_id, None):
                        raise HostError(
                            f"respuesta de host desincronizada (id {resp.get('id')} != {req_id})"
                        )
                    # Progress message - emit and continue waiting for final
                    if resp.get("progress") is not None and resp.get("final", True) is False:
                        if progress_callback is not None:
                            try:
                                progress_callback(resp.get("progress"), resp.get("message", ""))
                            except Exception:
                                pass
                        continue
                    # Final response
                    response = resp
                    out_blobs = resp_blobs
                    break
            except (EOFError, ValueError, OSError) as exc:
                raise HostCrashed(self._dead_message(extra=str(exc))) from exc
            finally:
                if watchdog is not None:
                    watchdog.cancel()

        if response is None:
            raise HostError("no se recibió respuesta del host")
        if not response.get("ok"):
            detail = response.get("error") or "error desconocido del host"
            raise HostError(str(detail))
        return dict(response.get("data") or {}), out_blobs

    def _dead_message(self, extra: str = "") -> str:
        reason = self._crashed_reason or "el host de IA no está en marcha"
        logs = host_log_path() or "(log del host no disponible)"
        hint = (
            " Si vuelve a ocurrir, adjunta el log del host: es el proceso que "
            "carga torch y el que puede fallar por DLL nativas (VC++ "
            "Redistributable / antivirus)."
        )
        return f"{reason}. Log: {logs}.{(' ' + extra) if extra else ''}{hint}"

    # ----------------------------------------------------------------- ops
    def call_op(self, name: str, timeout: float | None = OP_TIMEOUT, progress_callback=None, **kwargs):
        """Ejecuta una op de la whitelist del host y devuelve su ``data``."""
        self.ensure_started()
        data, _ = self.request(
            "op",
            {"op": name, "kwargs": kwargs},
            timeout=timeout,
            progress_callback=progress_callback,
        )
        return data

    def engine_create(self, engine_id: str, model_path: str) -> str:
        self.ensure_started()
        data, _ = self.request(
            "engine.create", {"engine_id": engine_id, "model_path": model_path}
        )
        return str(data["handle"])

    def engine_load(self, handle: str) -> bool:
        data, _ = self.request("engine.load", {"handle": handle}, timeout=LOAD_TIMEOUT)
        return bool(data.get("loaded"))

    def engine_process(self, handle: str, frames, resize=None):
        """Procesa frames (numpy uint8 HWC) y devuelve los frames de salida."""
        metas, blobs = [], []
        for frame in frames:
            meta, raw = proto.encode_array(frame)
            metas.append(meta)
            blobs.append(raw)
        data, out_blobs = self.request(
            "engine.process",
            {"handle": handle, "resize": list(resize) if resize else None},
            blobs=blobs,
            blob_meta=metas,
            timeout=PROCESS_TIMEOUT,
        )
        state = data.get("info")
        if isinstance(state, dict):
            self._engine_state[handle] = state
        out = [
            proto.decode_array(meta, raw)
            for meta, raw in zip(data.get("outputs") or [], out_blobs)
        ]
        return out

    def engine_info(self, handle: str) -> dict:
        data, _ = self.request("engine.info", {"handle": handle})
        return data

    def engine_last_state(self, handle: str) -> dict | None:
        """Última foto del motor que mandó el host (gratis: vino con el frame)."""
        state = self._engine_state.get(handle)
        return dict(state) if state else None

    def engine_gpu_post(self, handle: str, **kwargs) -> dict:
        """Pide al host activar el post-proceso en GPU de ESE motor (P1)."""
        data, _ = self.request(
            "engine.gpu_post", {"handle": handle, **kwargs}, timeout=LOAD_TIMEOUT
        )
        state = data.get("info")
        if isinstance(state, dict):
            self._engine_state[handle] = state
        return data

    def engine_unload(self, handle: str) -> None:
        try:
            self.request("engine.unload", {"handle": handle}, timeout=60.0)
        finally:
            self._engine_state.pop(handle, None)


_HOST: RuntimeHost | None = None
_HOST_LOCK = threading.Lock()


def get_host() -> RuntimeHost:
    """Host compartido del proceso (arrancado de forma perezosa)."""
    global _HOST
    with _HOST_LOCK:
        if _HOST is None:
            _HOST = RuntimeHost()
        return _HOST


def stop_host() -> None:
    """Apaga el host si está vivo (fin de la aplicación)."""
    global _HOST
    with _HOST_LOCK:
        host, _HOST = _HOST, None
    if host is not None:
        host.stop()


atexit.register(stop_host)
