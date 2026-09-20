"""P1: el pre/post-proceso de los upscalers, dentro del grafo ONNX (GPU).

**El problema medido.** El motor ONNX de la app recibe un frame BGR uint8, lo
convierte a RGB/NCHW/float32 en el CPU, ejecuta el modelo y después limpia el
resultado en el CPU (``clip(x*255)``, ``astype(uint8)``, invertir canales,
``ascontiguousarray``). Con un modelo x4 y un frame de 720x480 eso son
**1x3x2880x1920 float32 = 66 MB**: 66 MB de vuelta por PCIe y ~68 ms de numpy
por frame. El motor PyTorch ya lo hacía mejor (uint8 en la GPU), y ese detalle
explica por qué el mismo modelo rinde 146 ms por un camino y 88 ms por el otro
en una RTX 3060.

**La solución.** Envolver el grafo con una cabeza y una cola que hagan ese
trabajo donde ya está el dato:

    entrada: frame BGR uint8 (H, W, 3)  [lo que el pipeline ya tiene]
      cabeza: Cast -> Gather(canales BGR->RGB) -> Transpose NHWC->NCHW -> x(1/255)
      cuerpo: el modelo original (x2/x4, la precisión que elija el usuario)
      cola  : x255 -> Clip(0,255) -> [Resize al tamaño del encoder] ->
              Gather(canales RGB->BGR) -> Cast(uint8) -> Transpose NCHW->NHWC
    salida : frame BGR uint8 (el tamaño del encoder, ya listo para el pipe)

Nada de esto cambia las cuentas: multiplicar por 255, recortar y reordenar
canales son operaciones exactas; lo único que cambia es dónde se ejecutan (y el
filtro del reescalado, que pasa de PIL-BILINEAR a Resize lineal de la GPU).

**Robustez (requisito de diseño).** Este módulo nunca es la única vía:
  - Se construye en ``tmp``/caché; los modelos instalados no se tocan.
  - Antes de usarlo, ``self_check`` compara su salida contra el camino actual
    (motor + ``_resize_frame``) sobre frames sintéticos con canales
    asimétricos: un grafo mal montado (canales invertidos, escala equivocada,
    dtype incorrecto) da diferencias enormes y se descarta.
  - Si algo falla (onnx ausente, conversión imposible, tolerancia superada),
    se devuelve un motivo y el llamador sigue con el camino de siempre.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile

import numpy as np

logger = logging.getLogger(__name__)

#: Preferencia que habilita el camino GPU. APAGADA por defecto: nadie la usa
#: hasta que el usuario la activa en Preferences.
GPU_POST_PREF = "gpu_post_process"

#: Preferencia de precisión de inferencia: "model" | "fp16" | "fp32".
PRECISION_PREF = "inference_precision"

#: Tolerancias de la autocomprobación (escala 0..255). No buscan igualdad bit a
#: bit (el reescalado de la GPU usa otro filtro que PIL) sino detectar un
#: grafo roto: canales invertidos, escala mal aplicada o dtype incorrecto.
#:
#: La comprobación que de verdad manda es la "gruesa" (``coarse``): la imagen
#: se reduce a bloques de ~32x32 promediando, y ahí el filtro de reescalado
#: deja de importar pero un intercambio de canales o una escala mal aplicada
#: siguen siendo diferencias enormes. Las tolerancias finas son informativas y
#: deliberadamente holgadas: en un damero de 4 px, dos filtros distintos
#: pueden separarse 60 niveles sin que ninguno esté "mal".
COARSE_MAX_TOLERANCE = 32.0
COARSE_MEAN_TOLERANCE = 4.0
CHANNEL_MEAN_TOLERANCE = 8.0
MAX_DIFF_TOLERANCE = 128.0
MEAN_DIFF_TOLERANCE = 16.0
MIN_STD_TOLERANCE = 1.0  # una salida plana (todo un color) es sospechosa
COARSE_BLOCKS = 32


# ---------------------------------------------------------------------------
# Preferencias
# ---------------------------------------------------------------------------

def gpu_post_enabled() -> bool:
    """¿El usuario activó el post-proceso en GPU? (default: no)"""
    try:
        import src.preferences as preferences
        return bool(preferences.get(GPU_POST_PREF, False))
    except Exception:
        return False


def get_precision_preference() -> str:
    """Precisión elegida por el usuario: ``"model"``, ``"fp16"`` o ``"fp32"``.

    Tolerante a mayúsculas y a los textos que pueda guardar la UI
    ("FP16", "Del modelo", ...). Cualquier valor raro cae a ``"model"``,
    que significa "no convertir nada" (el comportamiento de siempre).
    """
    try:
        import src.preferences as preferences
        raw = str(preferences.get(PRECISION_PREF, "model") or "").strip().lower()
    except Exception:
        return "model"
    if raw in ("fp16", "half", "16", "float16"):
        return "fp16"
    if raw in ("fp32", "full", "32", "float32", "single"):
        return "fp32"
    return "model"


# ---------------------------------------------------------------------------
# Introspección / conversión de precisión del modelo ONNX
# ---------------------------------------------------------------------------

def model_float_dtype(onnx_path: str) -> str:
    """``"fp16"`` o ``"fp32"`` según el tipo de la primera entrada del grafo."""
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        elem = model.graph.input[0].type.tensor_type.elem_type
        return "fp16" if elem == onnx.TensorProto.FLOAT16 else "fp32"
    except Exception as exc:
        logger.debug("gpu_post: no se pudo leer el dtype de %s: %s", onnx_path, exc)
        return "fp32"


def convert_model_precision(model, target: str):
    """Devuelve ``model`` convertido a ``target`` ("fp16"/"fp32") o ``None``.

    fp32 -> fp16 usa el conversor oficial de ONNX Runtime. fp16 -> fp32 es
    manual (inicializadores + nodos ``Constant`` + ``Cast``), porque ORT no
    trae el camino inverso; si el grafo es exótico, se devuelve ``None`` y el
    llamador se queda con el modelo tal cual.
    """
    from onnx import TensorProto, numpy_helper

    if target == "fp16":
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16
        except Exception as exc:
            logger.info("gpu_post: conversor fp16 no disponible: %s", exc)
            return None
        try:
            # keep_io_types=False: queremos que el grafo sea fp16 de punta a
            # punta, porque la cabeza/cola (uint8 -> fp16 y fp16 -> uint8) las
            # añadimos nosotros después. Con keep_io_types=True este modelo
            # (SRVGG + residual Add con Resize) se rompe: verificado.
            return convert_float_to_float16(model, keep_io_types=False,
                                            disable_shape_infer=False)
        except Exception as exc:
            logger.info("gpu_post: conversión a fp16 falló: %s", exc)
            return None

    if target == "fp32":
        try:
            graph = model.graph
            for index, init in enumerate(list(graph.initializer)):
                array = numpy_helper.to_array(init)
                if array.dtype == np.float16:
                    graph.initializer[index].CopyFrom(
                        numpy_helper.from_array(array.astype(np.float32), init.name)
                    )
            for node in graph.node:
                if node.op_type == "Constant":
                    for attr in node.attribute:
                        if (attr.type == TensorProto.TENSOR
                                and attr.t.data_type == TensorProto.FLOAT16):
                            array = numpy_helper.to_array(attr.t).astype(np.float32)
                            attr.t.CopyFrom(numpy_helper.from_array(array))
                elif node.op_type == "Cast":
                    for attr in node.attribute:
                        if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                            attr.i = TensorProto.FLOAT
            for value_info in (list(graph.input) + list(graph.output)
                               + list(graph.value_info)):
                if value_info.type.tensor_type.elem_type == TensorProto.FLOAT16:
                    value_info.type.tensor_type.elem_type = TensorProto.FLOAT
            return model
        except Exception as exc:
            logger.info("gpu_post: conversión a fp32 falló: %s", exc)
            return None

    return None


def build_precision_model(src_path: str, target: str) -> tuple[str | None, dict]:
    """Copia del modelo convertida a ``target`` ("fp16"/"fp32"), sin cabeza/cola.

    Se usa cuando el usuario elige una precisión distinta de la del archivo
    aunque el post-proceso en GPU esté apagado: así el selector de precisión
    funciona por sí solo. El archivo original nunca se modifica.
    """
    info: dict = {"precision": target}
    if not src_path or not os.path.isfile(src_path):
        return None, {**info, "reason": "model-not-found"}
    native = model_float_dtype(src_path)
    info["native_dtype"] = native
    if native == target:
        return src_path, {**info, "reason": "already-native", "cached": True}
    dst_path = _derived_model_path(src_path, f"prec_{target}")
    if not dst_path:
        return None, {**info, "reason": "no-cache-path"}
    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
        return dst_path, {**info, "reason": "ok", "cached": True}
    try:
        import onnx
        model = onnx.load(src_path)
        converted = convert_model_precision(model, target)
        if converted is None:
            return None, {**info, "reason": "precision-conversion-failed"}
        onnx.save(converted, dst_path)
        logger.info("gpu_post: %s convertido a %s (%s)", os.path.basename(src_path),
                    target, os.path.basename(dst_path))
        return dst_path, {**info, "reason": "ok", "cached": False}
    except Exception as exc:
        try:
            if os.path.isfile(dst_path):
                os.remove(dst_path)
        except Exception:
            pass
        return None, {**info, "reason": f"convert-failed:{exc}"}


def _cache_dir() -> str | None:
    """Carpeta de caché para los grafos derivados (dentro de los datos del usuario)."""
    try:
        from src.config import get_session_dir, ensure_runtime_dirs
        ensure_runtime_dirs()
        path = os.path.join(get_session_dir(), "gpu_post_cache")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        return None


def _derived_model_path(src_path: str, tag: str) -> str | None:
    """Ruta estable (cacheada) para el grafo derivado, o ``None``."""
    base = _cache_dir()
    try:
        stat = os.stat(src_path)
        key = f"{os.path.abspath(src_path)}|{stat.st_size}|{stat.st_mtime_ns}|{tag}"
    except OSError:
        return None
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    name = f"{os.path.splitext(os.path.basename(src_path))[0]}_{tag}_{digest}.onnx"
    if base:
        return os.path.join(base, name)
    try:
        handle, tmp_path = tempfile.mkstemp(prefix="draggy_gpupost_", suffix=".onnx")
        os.close(handle)
        return tmp_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Construcción del grafo envuelto
# ---------------------------------------------------------------------------

def build_wrapped_model(src_path: str, out_size=None,
                        precision: str = "model") -> tuple[str | None, dict]:
    """Construye (o reutiliza de caché) el modelo con cabeza/cola en GPU.

    Args:
        src_path: ``.onnx`` original (no se modifica).
        out_size: ``(width, height)`` del encoder. ``None`` = no reescalar
            (para engines intermedios de una cadena: solo se hace el
            pre/post-proceso, que es lo caro).
        precision: ``"model"`` (tal cual), ``"fp16"`` o ``"fp32"``.

    Returns:
        ``(ruta_del_modelo_derivado | None, info)``. ``info`` siempre trae
        ``reason`` cuando algo no se pudo hacer.
    """
    info: dict = {"precision": precision, "out_size": out_size}
    if not src_path or not os.path.isfile(src_path):
        return None, {**info, "reason": "model-not-found"}

    native = model_float_dtype(src_path)
    target = precision if precision in ("fp16", "fp32") else native
    info["native_dtype"] = native
    info["dtype"] = target
    info["converted"] = (target != native)

    dst_path = _derived_model_path(src_path, f"{target}_{out_size or 'raw'}")
    if not dst_path:
        return None, {**info, "reason": "no-cache-path"}
    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
        info["cached"] = True
        return dst_path, info

    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except Exception as exc:
        return None, {**info, "reason": f"onnx-unavailable:{exc}"}

    try:
        model = onnx.load(src_path)
        if target != native:
            converted = convert_model_precision(model, target)
            if converted is None:
                return None, {**info, "reason": "precision-conversion-failed"}
            model = converted
    except Exception as exc:
        return None, {**info, "reason": f"load-failed:{exc}"}

    try:
        graph = model.graph
        src_in = graph.input[0].name
        src_out = graph.output[0].name
        dt = np.float16 if target == "fp16" else np.float32
        elem = TensorProto.FLOAT16 if target == "fp16" else TensorProto.FLOAT

        graph.initializer.extend([
            numpy_helper.from_array(np.array(1.0 / 255.0, dtype=dt), "dg_inv255"),
            numpy_helper.from_array(np.array(255.0, dtype=dt), "dg_255"),
            numpy_helper.from_array(np.array(0.0, dtype=dt), "dg_min"),
            numpy_helper.from_array(np.array(255.0, dtype=dt), "dg_max"),
            numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "dg_bgr2rgb"),
        ])
        if out_size:
            width, height = int(out_size[0]), int(out_size[1])
            graph.initializer.append(numpy_helper.from_array(
                np.array([1, 3, height, width], dtype=np.int64), "dg_out_sizes"))

        head = [
            helper.make_node("Cast", [src_in], ["dg_f"], to=elem),
            helper.make_node("Gather", ["dg_f", "dg_bgr2rgb"], ["dg_rgb"],
                             axis=3),
            helper.make_node("Transpose", ["dg_rgb"], ["dg_nchw"],
                             perm=[0, 3, 1, 2]),
            helper.make_node("Mul", ["dg_nchw", "dg_inv255"], ["dg_norm"]),
        ]
        # Los nodos ORIGINALES pasan a consumir la salida de la cabeza. Se
        # toma la lista ANTES de insertar la cabeza: reescribir también la
        # entrada del Cast de la propia cabeza crearía un ciclo
        # (Cast <- dg_norm <- Mul <- ... <- Cast) y ONNX Runtime devolvía una
        # imagen plana sin dar error. Verificado en la prueba de humo.
        original_nodes = list(graph.node)
        for node in original_nodes:
            for index, name in enumerate(node.input):
                if name == src_in:
                    node.input[index] = "dg_norm"

        tail = [
            helper.make_node("Mul", [src_out, "dg_255"], ["dg_scaled"]),
            helper.make_node("Clip", ["dg_scaled", "dg_min", "dg_max"], ["dg_clip"]),
        ]
        last = "dg_clip"
        if out_size:
            tail.append(helper.make_node(
                "Resize", [last, "", "", "dg_out_sizes"], ["dg_resized"],
                mode="linear", coordinate_transformation_mode="half_pixel",
                nearest_mode="round_prefer_floor"))
            last = "dg_resized"
        tail.extend([
            # RGB -> BGR antes del cast: Gather sobre float está garantizado en
            # CUDA, y aquí el tensor ya es pequeño (tamaño del encoder).
            helper.make_node("Gather", [last, "dg_bgr2rgb"], ["dg_bgr"], axis=1),
            helper.make_node("Cast", ["dg_bgr"], ["dg_u8"], to=TensorProto.UINT8),
            helper.make_node("Transpose", ["dg_u8"], ["dg_out"], perm=[0, 2, 3, 1]),
        ])

        # La cabeza va DELANTE (topológicamente válida) y la cola DETRÁS.
        for index, node in enumerate(head):
            graph.node.insert(index, node)
        graph.node.extend(tail)

        del graph.value_info[:]
        del graph.input[:]
        del graph.output[:]
        graph.input.extend([helper.make_tensor_value_info(
            src_in, TensorProto.UINT8, ["n", "h", "w", 3])])
        graph.output.extend([helper.make_tensor_value_info(
            "dg_out", TensorProto.UINT8, ["n", "oh", "ow", 3])])
        model.producer_name = "draggy-encoder-gpu-post"
        onnx.save(model, dst_path)
        info["cached"] = False
        logger.info("gpu_post: grafo derivado listo (%s, %s, resize=%s)",
                    os.path.basename(dst_path), target, out_size or "no")
        return dst_path, info
    except Exception as exc:
        logger.info("gpu_post: no se pudo envolver %s: %s", src_path, exc)
        try:
            if os.path.isfile(dst_path):
                os.remove(dst_path)
        except Exception:
            pass
        return None, {**info, "reason": f"wrap-failed:{exc}"}


# ---------------------------------------------------------------------------
# Autocomprobación
# ---------------------------------------------------------------------------

def sample_frames(height: int = 96, width: int = 128) -> list[np.ndarray]:
    """Frames sintéticos que delatan un grafo mal montado.

    El primero lleva canales muy asimétricos (B=20, G=110, R=230): si la
    cabeza olvidara el BGR->RGB la diferencia sería de ~200 niveles, no de 3.
    El segundo es un damero (detalle fino, sensible al filtro de reescalado)
    y el tercero un degradado suave (detecta un Resize mal orientado).

    Se generan al tamaño real del frame del render cuando se conoce (el
    llamador lo pasa): así la comprobación usa la misma forma que los frames
    reales y no fuerza una segunda captura de CUDA Graph.
    """
    height = max(8, int(height))
    width = max(8, int(width))
    asymmetric = np.zeros((height, width, 3), dtype=np.uint8)
    asymmetric[..., 0] = 20
    asymmetric[..., 1] = 110
    asymmetric[..., 2] = 230
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    ramp_x = np.linspace(0, 255, width, dtype=np.uint8)
    ramp_y = np.linspace(0, 255, height, dtype=np.uint8)
    gradient[..., 0] = ramp_x[None, :]
    gradient[..., 1] = ramp_y[:, None]
    gradient[..., 2] = (ramp_x[None, :] // 2 + ramp_y[:, None] // 2)
    checker = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    # Damero de 8 px (no de 1-4 px): sigue teniendo detalle fino pero no está
    # en el límite de Nyquist, donde dos filtros de reescalado legítimos
    # difieren muchísimo y la comprobación se volvería un falso positivo.
    tile = (((yy // 8) + (xx // 8)) % 2).astype(np.uint8) * 255
    checker[..., 0] = tile
    checker[..., 1] = 255 - tile
    checker[..., 2] = 128
    return [asymmetric, gradient, checker]


def _coarse(frame: np.ndarray, blocks: int = COARSE_BLOCKS) -> np.ndarray:
    """Reduce el frame a ~``blocks``x``blocks`` promediando (insensible al filtro)."""
    height, width = frame.shape[:2]
    block_h = max(1, height // max(1, blocks))
    block_w = max(1, width // max(1, blocks))
    usable_h = (height // block_h) * block_h
    usable_w = (width // block_w) * block_w
    view = frame[:usable_h, :usable_w].astype(np.float32)
    view = view.reshape(usable_h // block_h, block_h,
                        usable_w // block_w, block_w, -1)
    return view.mean(axis=(1, 3))


def compare_frames(got: np.ndarray, ref: np.ndarray) -> dict:
    """Estadísticas de diferencia entre dos frames uint8 (escala 0..255)."""
    stats: dict = {"shape_ok": False}
    try:
        if got is None or ref is None:
            return {**stats, "reason": "missing-output"}
        stats["shape_ok"] = tuple(got.shape) == tuple(ref.shape)
        if not stats["shape_ok"]:
            stats["got_shape"] = list(got.shape)
            stats["ref_shape"] = list(ref.shape)
            return stats
        diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        stats["max_diff"] = float(diff.max()) if diff.size else 0.0
        stats["mean_diff"] = float(diff.mean()) if diff.size else 0.0
        stats["std"] = float(got.std())
        # Medias por canal: delatan un intercambio de canales o una escala mal
        # aplicada sin que les afecte el filtro del reescalado.
        got_mean = got.reshape(-1, got.shape[-1]).mean(axis=0)
        ref_mean = ref.reshape(-1, ref.shape[-1]).mean(axis=0)
        stats["channel_mean_diff"] = float(np.abs(got_mean - ref_mean).max())
        coarse_diff = np.abs(_coarse(got) - _coarse(ref))
        stats["coarse_max_diff"] = float(coarse_diff.max()) if coarse_diff.size else 0.0
        stats["coarse_mean_diff"] = float(coarse_diff.mean()) if coarse_diff.size else 0.0
    except Exception as exc:
        stats["reason"] = f"{type(exc).__name__}: {exc}"
    return stats


def evaluate_check(stats: dict) -> tuple[bool, str]:
    """Decide si una comparación es aceptable. Devuelve ``(ok, motivo)``."""
    if not stats.get("shape_ok"):
        return False, (f"shape mismatch ({stats.get('got_shape')} vs "
                       f"{stats.get('ref_shape')})")
    if stats.get("reason"):
        return False, str(stats["reason"])
    if stats.get("std", 0.0) < MIN_STD_TOLERANCE:
        return False, "output looks flat (all one colour)"
    if stats.get("channel_mean_diff", 999.0) > CHANNEL_MEAN_TOLERANCE:
        return False, (f"channel means differ by {stats['channel_mean_diff']:.1f} "
                       f"(colour order/scale looks wrong)")
    if stats.get("coarse_max_diff", 999.0) > COARSE_MAX_TOLERANCE:
        return False, (f"coarse max diff {stats['coarse_max_diff']:.1f} > "
                       f"{COARSE_MAX_TOLERANCE:.0f}")
    if stats.get("coarse_mean_diff", 999.0) > COARSE_MEAN_TOLERANCE:
        return False, (f"coarse mean diff {stats['coarse_mean_diff']:.2f} > "
                       f"{COARSE_MEAN_TOLERANCE:.2f}")
    if stats.get("max_diff", 999.0) > MAX_DIFF_TOLERANCE:
        return False, f"max diff {stats['max_diff']:.1f} > {MAX_DIFF_TOLERANCE:.0f}"
    if stats.get("mean_diff", 999.0) > MEAN_DIFF_TOLERANCE:
        return False, f"mean diff {stats['mean_diff']:.2f} > {MEAN_DIFF_TOLERANCE:.2f}"
    return True, "ok"


def run_self_check(fast_fn, reference_fn, frames=None) -> tuple[bool, dict]:
    """Ejecuta ambos caminos sobre los frames de prueba y compara.

    ``fast_fn``/``reference_fn`` reciben un frame BGR uint8 y devuelven el
    frame final (ya con el tamaño del encoder). Cualquier excepción se
    traduce en "no fiable": el llamador usa el camino de siempre.
    """
    details: dict = {"frames": 0, "worst_max": 0.0, "worst_mean": 0.0,
                     "worst_coarse": 0.0, "worst_channel": 0.0}
    try:
        for frame in (frames if frames is not None else sample_frames()):
            reference = reference_fn(frame)
            got = fast_fn(frame)
            stats = compare_frames(got, reference)
            details["frames"] += 1
            details["worst_max"] = max(details["worst_max"],
                                       float(stats.get("max_diff", 0.0)))
            details["worst_mean"] = max(details["worst_mean"],
                                        float(stats.get("mean_diff", 0.0)))
            details["worst_coarse"] = max(details["worst_coarse"],
                                          float(stats.get("coarse_max_diff", 0.0)))
            details["worst_channel"] = max(details["worst_channel"],
                                           float(stats.get("channel_mean_diff", 0.0)))
            details["last"] = stats
            ok, reason = evaluate_check(stats)
            if not ok:
                details["reason"] = reason
                return False, details
        details["reason"] = "ok"
        return True, details
    except Exception as exc:
        details["reason"] = f"{type(exc).__name__}: {exc}"
        return False, details
