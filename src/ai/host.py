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

    def load(self) -> bool:
        self.is_loaded = True
        return True

    def unload(self) -> None:
        self.is_loaded = False

    def describe_execution(self) -> str:
        return "mock (loopback sin runtime nativo)"


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
        return {"outputs": metas}, list(blobs)

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
    return {"outputs": out_meta}, out_blobs


def _job_engine_info(data: dict) -> dict:
    engine = _ENGINES.get(str(data.get("handle")))
    if engine is None:
        raise KeyError(f"motor desconocido en el host: {data.get('handle')}")
    info: dict = {"engine_id": getattr(engine, "engine_id", "?")}
    for attr, key in (("get_scale_factor", "scale_factor"),
                      ("describe_execution", "execution"),
                      ("get_cumulative_timing", "timing")):
        fn = getattr(engine, attr, None)
        if callable(fn):
            try:
                info[key] = fn()
            except Exception as exc:  # la info es best-effort, no crítica
                info[key] = f"({type(exc).__name__}: {exc})"
    return info


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
    return {"unloaded": engine is not None}


def _job_engine_gpu_post(data: dict) -> dict:
    """P1 (post-proceso en GPU) todavía corre en el proceso de la app.

    Se responde explícitamente ``active: False`` para que el pipeline use el
    camino de siempre; lo que NO se hace es activarlo aquí y devolver un motor
    que la GUI crea que está envuelto.
    """
    return {
        "active": False,
        "reason": (
            "el post-proceso en GPU aún no vive en el host de IA; "
            "se usa el camino estándar"
        ),
    }


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
