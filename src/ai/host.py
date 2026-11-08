"""Host de IA: el proceso que SÍ carga torch/onnxruntime.

Se lanza (ver ``src.ai.host_client``) con el Python privado del instalador —
``embed-python/python.exe`` en Windows, ``runtime-python/bin/python3`` en
Linux— y habla el protocolo de ``src.ai.host_protocol`` por stdin/stdout. La
GUI nunca importa aquí los runtimes nativos: se lo pide a este proceso.

Reglas del proceso:

* ``stdout`` es SOLO protocolo. Los logs van a ``stderr`` (el cliente los
  vuelca a ``bin/logs/ai_host.log``); cualquier ``print`` accidental rompería
  el framing y el cliente lo detectaría como protocolo corrupto.
* Los módulos pesados (``src.ai.inference``, ``src.ai.architectures``,
  ``torch``...) se importan DENTRO de cada job, no al arrancar: así el host
  arranca y responde ``ping`` incluso sin runtimes instalados (el CI lo usa
  con el store vacío).
* Cada job se ejecuta en un try/except y devuelve ``{"ok": false, "error"}``
  con el traceback: un fallo del modelo no debe matar el host ni dejar al
  cliente esperando una respuesta que nunca llega.

Ejecutable como ``python -m src.ai.host`` (o vía ``src.ai.host.main()`` con
``sys.path`` apuntando al árbol de fuentes empaquetado).
"""

from __future__ import annotations

import logging
import os
import sys
import traceback

from src.ai import host_protocol as proto

logger = logging.getLogger("draggy.ai_host")

#: Motores vivos en este proceso, por id de sesión.
_ENGINES: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Ops: el trabajo pesado que la GUI delega aquí
# ---------------------------------------------------------------------------

def _op_model_scale(data: dict) -> dict:
    """Escala nativa de un fichero de pesos (lee el grafo o los tensores)."""
    from src.ai.architectures import get_model_file_scale, parse_scale_from_name

    path = str(data.get("path") or "")
    hint = int(parse_scale_from_name(os.path.basename(path)))
    proven = int(get_model_file_scale(path))
    return {
        "scale": int(proven) if proven > 1 else 1,
        "proven": bool(proven > 1),
        "hint": hint,
    }


def _op_detect_arch_and_scale(data: dict) -> dict:
    from src.ai.architectures import detect_arch_and_scale

    arch, scale = detect_arch_and_scale(str(data.get("path") or ""))
    return {"arch": arch, "scale": scale}


def _op_convert_pth_to_onnx(data: dict, progress_cb=None) -> dict:
    from src.ai.inference import convert_pth_to_onnx_ex

    def _cb(pct, msg=""):
        if progress_cb is not None:
            try:
                progress_cb(int(pct), str(msg))
            except Exception:
                pass

    path, message = convert_pth_to_onnx_ex(str(data.get("path") or ""), progress_callback=_cb)
    return {"onnx_path": path, "message": message}


def _op_pth_channel_count(data: dict) -> dict:
    """Canales de entrada de un checkpoint (lo lee torch, aquí dentro)."""
    from src.ai.inference import _pth_channel_count

    return {"channels": _pth_channel_count(str(data.get("path") or ""))}


def _op_benchmark(data: dict) -> dict:
    """``benchmark_backend``/``benchmark_quality`` sin tocar la GUI."""
    from src.ai import benchmark as bench

    name = str(data.get("name") or "benchmark_backend")
    kwargs = dict(data.get("kwargs") or {})
    if "resolution" in kwargs and kwargs["resolution"] is not None:
        kwargs["resolution"] = tuple(kwargs["resolution"])
    fn = getattr(bench, name, None)
    if fn is None or name not in ("benchmark_backend", "benchmark_quality"):
        raise ValueError(f"benchmark desconocido: {name}")
    return {"result": fn(**kwargs)}


def _op_runtime_caps(_data: dict) -> dict:
    """Versiones y capacidades reales, leídas por quien puede importarlas."""
    from src.runtimes import manager as rm

    caps: dict = {"python": sys.version.split()[0]}
    torch_status = rm.torch_status()
    caps["torch"] = {
        "installed": bool(torch_status.get("installed")),
        "version": torch_status.get("version"),
        "gpu": bool(torch_status.get("gpu")),
    }
    onnx_status = rm.onnx_status()
    caps["onnxruntime"] = {
        "installed": bool(onnx_status.get("installed")),
        "version": onnx_status.get("version"),
        "gpu": bool(onnx_status.get("gpu")),
    }
    return caps


#: Whitelist: la GUI no puede ejecutar código arbitrario en este proceso, solo
#: estas operaciones (cada una es una función de la app, no un import libre).
_OPS = {
    "model_scale": _op_model_scale,
    "detect_arch_and_scale": _op_detect_arch_and_scale,
    "convert_pth_to_onnx": _op_convert_pth_to_onnx,
    "pth_channel_count": _op_pth_channel_count,
    "benchmark": _op_benchmark,
    "runtime_caps": _op_runtime_caps,
}


# ---------------------------------------------------------------------------
# Motores
# ---------------------------------------------------------------------------

class _MockEngine:
    """Motor de mentira que reenvía los frames sin tocarlos.

    Existe para poder ejercitar TODO el camino (arranque del host, framing,
    registro de motores, ida y vuelta de blobs) en el CI, donde no hay torch
    instalado. Si esto funciona en el bundle, el cableado funciona.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.engine_id = "mock"
        self.is_loaded = False
        self._gpu_post = None
        self._gpu_post_failure = None

    def load(self) -> bool:
        self.is_loaded = True
        return True

    def unload(self) -> None:
        self.is_loaded = False

    def describe_execution(self) -> str:
        return "mock (loopback sin runtime nativo)"

    def configure_gpu_post(self, width=None, height=None, resize_fn=None,
                           input_size=None, in_scale=None) -> dict:
        """P1 de mentira: activo siempre, para probar el cableado en el CI.

        No construye ningún grafo (aquí no hay onnxruntime); solo declara que
        el camino GPU está activo con el tamaño de salida y el detalle
        pedidos, que es lo que el proxy del cliente tiene que propagar sin
        inventarse nada.
        """
        self._gpu_post = {"active": True,
                          "out_size": (width, height) if width and height else None}
        self._gpu_post_failure = None
        self._gpu_post["detail"] = (float(in_scale) if in_scale is not None
                                    else 1.0)
        return {"active": True, "reason": "ok",
                "out_size": self._gpu_post["out_size"],
                "detail": self._gpu_post["detail"], "precision": "mock"}

    def gpu_post_active(self) -> bool:
        return bool(self._gpu_post)

    def disable_gpu_post(self) -> None:
        self._gpu_post = None

    def get_cumulative_timing(self) -> dict:
        return {"frames": 0, "pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}

    def get_scale_factor(self) -> int:
        return 1


def _job_engine_create(data: dict) -> dict:
    """Crea el motor con la MISMA fábrica que usaba la app en su proceso.

    Así el comportamiento (auto-conversión de .pth a ONNX, elección de
    backend, fallback a PyTorch) es idéntico al de siempre: lo único que
    cambia es DÓNDE corre. Si el motor revienta por una DLL nativa, se lleva
    este proceso y la GUI sigue viva.
    """
    engine_id = str(data.get("engine_id") or data.get("backend") or "")
    model_path = str(data.get("model_path") or "")
    if engine_id == "mock":
        engine = _MockEngine(model_path)
    else:
        from src.ai.inference import create_inference_engine

        engine = create_inference_engine(engine_id, model_path)
    handle = f"eng-{len(_ENGINES) + 1}"
    _ENGINES[handle] = engine
    return {
        "handle": handle,
        "engine_id": getattr(engine, "engine_id", engine_id) or engine_id,
        "loaded": bool(getattr(engine, "is_loaded", False)),
    }


def _job_engine_load(data: dict) -> dict:
    engine = _ENGINES[str(data.get("handle"))]
    return {"loaded": bool(engine.load())}


def _process_frame(engine, frame, resize):
    """``process_frame`` con escala objetivo opcional.

    El motor ONNX no acepta ``resize`` (solo el de PyTorch); se le llama sin
    el argumento y procesa con su escala nativa, igual que hace la app.
    """
    if resize:
        try:
            return engine.process_frame(frame, resize=resize)
        except TypeError as exc:
            if "resize" not in str(exc):
                raise
    return engine.process_frame(frame)


def _job_engine_process(data: dict, blobs: list) -> tuple[dict, list]:
    handle = str(data.get("handle"))
    engine = _ENGINES.get(handle)
    if engine is None:
        raise KeyError(f"motor desconocido en el host: {handle}")
    metas = [dict(m) for m in (data.get("blob_meta") or [])]
    if isinstance(engine, _MockEngine):
        # Sin numpy: solo se comprueba el cableado (blobs intactos).
        return {"outputs": metas, "info": _engine_state(engine)}, list(blobs)

    resize = data.get("resize")
    out_meta, out_blobs = [], []
    for meta, blob in zip(metas, blobs):
        frame = proto.decode_array(meta, blob)
        out = _process_frame(engine, frame, tuple(resize) if resize else None)
        if out is None:
            raise RuntimeError("el motor devolvió None")
        encoded_meta, encoded_blob = proto.encode_array(out)
        out_meta.append(encoded_meta)
        out_blobs.append(encoded_blob)
    return {"outputs": out_meta, "info": _engine_state(engine)}, out_blobs


def _gpu_post_state(engine) -> dict:
    """Estado real del post-proceso GPU de un motor de este proceso.

    El pipeline lo lee dos veces por frame (antes y después de la llamada),
    así que viaja DENTRO de la respuesta de ``engine.process``: si hubiera que
    pedirlo aparte serían dos peticiones extra por frame (y con 20 ms de
    round-trip en frames grandes, eso se nota en el fps).
    """
    state: dict = {}
    active = getattr(engine, "gpu_post_active", None)
    try:
        state["gpu_post_active"] = bool(active()) if callable(active) else bool(active)
    except Exception:
        state["gpu_post_active"] = False
    failure = getattr(engine, "_gpu_post_failure", None)
    if failure:
        state["gpu_post_failure"] = str(failure)
    return state


def _engine_state(engine) -> dict:
    """Foto del motor (identidad, escala, telemetría, post GPU) para el cliente."""
    state: dict = {"engine_id": getattr(engine, "engine_id", "?")}
    for attr, key in (("get_scale_factor", "scale_factor"),
                      ("describe_execution", "execution"),
                      ("get_cumulative_timing", "timing")):
        fn = getattr(engine, attr, None)
        if callable(fn):
            try:
                state[key] = fn()
            except Exception as exc:  # la info es best-effort, no crítica
                state[key] = f"({type(exc).__name__}: {exc})"
    state.update(_gpu_post_state(engine))
    return state


def _host_resize_frame(im, dst_w, dst_h):
    """Reescalado de frames con el MISMO filtro que el pipeline.

    El post-proceso GPU se autocomprueba contra el camino clásico (motor +
    ``pipeline._resize_frame``), así que el filtro tiene que ser el de allí:
    cv2 INTER_AREA al reducir / INTER_LINEAR al agrandar, con PIL y numpy como
    respaldo. Si se usara otro filtro, la comprobación compararía dos imágenes
    legítimamente distintas y rechazaría el camino GPU sin motivo.

    numpy se importa aquí dentro (no en el módulo) porque es el proceso del
    host el que puede pagarlo; y hace falta aunque ``cv2`` esté: el respaldo
    de PIL y el de numpy lo usan.
    """
    import numpy as np

    dst_w, dst_h = int(dst_w), int(dst_h)
    try:
        import cv2

        interp = (cv2.INTER_AREA
                  if (dst_h < im.shape[0] or dst_w < im.shape[1])
                  else cv2.INTER_LINEAR)
        resized = cv2.resize(im, (dst_w, dst_h), interpolation=interp)
        if resized.ndim == 2:
            resized = resized[..., None].repeat(3, axis=2)
        return resized
    except ImportError:
        pass
    try:
        from PIL import Image as _PILImage

        resized = np.asarray(_PILImage.fromarray(im).resize(
            (dst_w, dst_h), _PILImage.Resampling.BILINEAR))
        if resized.ndim == 2:
            resized = resized[..., None].repeat(3, axis=2)
        return resized
    except ImportError:
        rows = np.linspace(0, im.shape[0] - 1, dst_h, dtype=int)
        cols = np.linspace(0, im.shape[1] - 1, dst_w, dtype=int)
        resized = im[rows[:, None], cols[None, :], :] if im.ndim == 3 else \
            im[rows[:, None], cols[None, :]]
        if resized.ndim == 2:
            resized = resized[..., None].repeat(3, axis=2)
        return resized


def _job_engine_info(data: dict) -> dict:
    engine = _ENGINES.get(str(data.get("handle")))
    if engine is None:
        raise KeyError(f"motor desconocido en el host: {data.get('handle')}")
    return _engine_state(engine)


def _release_gpu_memory() -> None:
    """Devuelve a la GPU la memoria que el motor descargado aún retenía.

    torch guarda en su caché de CUDA los bloques liberados; sin
    ``empty_cache`` el host seguía apareciendo como dueño de GB y la
    siguiente preview se quedaba sin VRAM aunque el motor ya no existiera.
    Solo se toca torch si ya está importado en este proceso (no forzamos
    una carga de ~2 s cuando el motor era ONNX Runtime).
    """
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        logger.debug("No se pudo vaciar la caché CUDA:\n%s", traceback.format_exc())


def _job_engine_unload(data: dict) -> dict:
    handle = str(data.get("handle"))
    engine = _ENGINES.pop(handle, None)
    if engine is not None:
        unload = getattr(engine, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                logger.warning("unload falló:\n%s", traceback.format_exc())
    _release_gpu_memory()
    return {"unloaded": engine is not None}


def _job_engine_gpu_post(data: dict) -> dict:
    """P1: activa el pre/post-proceso en GPU DEL MOTOR QUE VIVE AQUÍ.

    Es el mismo ``configure_gpu_post`` de los motores locales (construye el
    grafo envuelto con cabeza/cola, lo autocomprueba y decide), sólo que
    ocurre en este proceso, que es el que tiene onnxruntime. El ``resize_fn``
    del pipeline no puede cruzar el protocolo, así que se pasa el reescalado
    equivalente de aquí (``_host_resize_frame``): mismo filtro, así la
    autocomprobación compara lo comparable.

    Si el motor no soporta el camino GPU (terceros, mocks, PyTorch sin CUDA)
    se responde ``active: False`` con el motivo; nunca se miente ni se rompe
    el render.
    """
    engine = _ENGINES.get(str(data.get("handle")))
    if engine is None:
        raise KeyError(f"motor desconocido en el host: {data.get('handle')}")
    configure = getattr(engine, "configure_gpu_post", None)
    if not callable(configure):
        return {"active": False, "reason": "not-supported",
                "info": _engine_state(engine)}
    input_size = data.get("input_size")
    in_scale = data.get("in_scale")
    scale = float(in_scale) if in_scale is not None else None
    try:
        result = configure(
            data.get("width"), data.get("height"),
            resize_fn=_host_resize_frame,
            input_size=tuple(input_size) if input_size else None,
            in_scale=scale,
        )
    except TypeError as exc:
        # Motores que aún no conocen el ajuste de detalle: se les llama como
        # antes en vez de declarar P1 roto (el detalle es opcional, no un
        # requisito de la interfaz).
        if "in_scale" not in str(exc):
            logger.error("gpu_post falló en el host:\n%s", traceback.format_exc())
            result = {"active": False,
                      "reason": f"error:{type(exc).__name__}: {exc}"}
        else:
            try:
                result = configure(
                    data.get("width"), data.get("height"),
                    resize_fn=_host_resize_frame,
                    input_size=tuple(input_size) if input_size else None,
                )
            except Exception as inner:
                logger.error("gpu_post falló en el host:\n%s", traceback.format_exc())
                result = {"active": False,
                          "reason": f"error:{type(inner).__name__}: {inner}"}
    except Exception as exc:
        logger.error("gpu_post falló en el host:\n%s", traceback.format_exc())
        result = {"active": False,
                  "reason": f"error:{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        result = {"active": False, "reason": "bad-result"}
    report = {"active": bool(result.get("active")),
              "reason": str(result.get("reason", ""))}
    for key, value in result.items():
        if key not in report:
            report[key] = value
    report["info"] = _engine_state(engine)
    logger.info("gpu_post en el host: active=%s reason=%s",
                report["active"], report["reason"])
    return report


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def _ping() -> dict:
    from src.runtimes import manager as rm

    return {
        "protocol": proto.PROTOCOL_VERSION,
        "python": sys.version.split()[0],
        "frozen": bool(getattr(sys, "frozen", False)),
        "store": rm.get_runtime_modules_dir(),
        "store_exists": os.path.isdir(rm.get_runtime_modules_dir()),
        "pid": os.getpid(),
    }


def _dispatch(job: str, data: dict, blobs: list, progress_cb=None) -> tuple[dict, list]:
    if job == "ping":
        return _ping(), []
    if job == "shutdown":
        return {"bye": True}, []
    if job == "op":
        name = str(data.get("op") or "")
        fn = _OPS.get(name)
        if fn is None:
            raise ValueError(
                f"op no permitida: {name!r} (válidas: {', '.join(sorted(_OPS))})"
            )
        try:
            return fn(dict(data.get("kwargs") or {}), progress_cb), []
        except TypeError:
            return fn(dict(data.get("kwargs") or {})), []
    if job == "engine.create":
        return _job_engine_create(data), []
    if job == "engine.load":
        return _job_engine_load(data), []
    if job == "engine.process":
        return _job_engine_process(data, blobs)
    if job == "engine.info":
        return _job_engine_info(data), []
    if job == "engine.gpu_post":
        return _job_engine_gpu_post(data), []
    if job == "engine.unload":
        return _job_engine_unload(data), []
    raise ValueError(f"job desconocido: {job!r}")


def main() -> int:
    """Bucle de servicio: lee jobs de stdin y responde por stdout."""
    # El host comparte la jerarquía de logs del app solo por stderr: stdout es
    # protocolo y no admite ni un print de librería.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        from src.runtimes.manager import ensure_runtime_dlls_on_path, inject_runtime_paths

        inject_runtime_paths()
        ensure_runtime_dlls_on_path()
    except Exception:
        logger.warning("no se pudieron inyectar los runtimes:\n%s", traceback.format_exc())

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    logger.info(
        "host de IA listo (pid=%d, python=%s, store=%s)",
        os.getpid(), sys.version.split()[0], os.environ.get("DRAGGY_DATA_DIR", "?"),
    )

    while True:
        try:
            header, blobs = proto.read_message(stdin)
        except EOFError:
            logger.info("stdin cerrado; el host termina")
            return 0
        except Exception:
            logger.error("protocolo corrupto:\n%s", traceback.format_exc())
            return 2

        job = str(header.get("job") or "")
        req_id = header.get("id")
        data = dict(header.get("data") or {})
        # ``blob_meta`` describe los blobs que acompañan al mensaje y vive en
        # la cabecera (junto a ``blob_lengths``), no dentro de ``data``: los
        # jobs que reciben frames lo necesitan para reconstruir los arrays.
        data.setdefault("blob_meta", list(header.get("blob_meta") or []))
        def _progress_cb(pct, msg=""):
            try:
                proto.write_message(
                    stdout,
                    {
                        "id": req_id,
                        "job": job,
                        "ok": True,
                        "progress": int(pct),
                        "message": str(msg),
                        "final": False,
                    },
                )
            except Exception:
                logger.debug("no se pudo enviar progreso: %s", traceback.format_exc())

        try:
            result, out_blobs = _dispatch(job, data, blobs, progress_cb=_progress_cb)
            response = {"id": req_id, "job": job, "ok": True, "data": result}
        except Exception as exc:
            response = {
                "id": req_id,
                "job": job,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            out_blobs = []
            logger.error("job %s falló:\n%s", job, traceback.format_exc())
        try:
            proto.write_message(stdout, response, out_blobs)
        except Exception:
            logger.error("no se pudo escribir la respuesta:\n%s", traceback.format_exc())
            return 3
        if job == "shutdown":
            logger.info("shutdown pedido por el cliente")
            return 0


if __name__ == "__main__":
    sys.exit(main())
