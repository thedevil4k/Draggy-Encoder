"""Baremo medido de modelos: **calidad y velocidad**, no números de folleto.

El catálogo trae `quality_score` y `popularity_score` escritos a mano. Para un
usuario que sólo quiere saber "¿cuál va rápido?" y "¿cuál se ve mejor?" esas
cifras no significan nada y no son comparables entre modelos. Este módulo mide
las dos cosas **en el equipo del usuario, con el modelo real**:

  * **velocidad**: ms/frame del benchmark existente (``src.ai.benchmark``) a una
    resolución canónica, convertido a fps y a una etiqueta en castellano
    ("muy rápida", "media", "lenta"...).
  * **calidad**: el modelo se ejecuta sobre un juego de referencia sintético
    (generado por código, idéntico en cada ejecución y sin descargas) y su
    salida se compara con la imagen original. Se miden tres cosas y se dan por
    separado, porque "calidad" no es un número único:
      - **fidelidad** (SSIM/PSNR): ¿respeta la estructura original?
      - **detalle** (ganancia de energía de bordes frente a un reescalado sin
        IA): ¿añade nitidez de verdad o sólo suaviza?
      - **artefactos** (halos/sobreoscilación y deriva de color): ¿inventa
        cosas raras?
    El índice 0-100 combina las tres con pesos fijos y **está documentado en el
    tooltip**: es un baremo repetible para comparar modelos entre sí, no una
    verdad absoluta.

Los resultados se cachean en ``bin/.session/model_ratings.json`` con la huella
del modelo, así que re-medir sólo ocurre cuando cambian los pesos, el backend o
la versión de la fórmula.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

#: Factor de degradación de la referencia: **el mismo para todos los modelos**.
#: Es lo que hace comparables modelos de escala nativa distinta: al x4 no se le
#: pide reconstruir desde 1/16 de los píxeles (tarea mucho más difícil) sino la
#: misma tarea que al x2, y su salida se lleva al tamaño final igual que hace
#: el render real cuando exportas a menos resolución que la nativa.
DEFAULT_TASK_SCALE = 2

#: Subir esto invalida las medidas cacheadas (cambia la fórmula, la tarea o el
#: juego de referencia).
RATING_VERSION = 2

RATINGS_FILENAME = "model_ratings.json"

#: Anclas de las sub-puntuaciones, calibradas con las medidas reales del
#: catálogo en una RTX 3060 (referencia a 512x288, tarea ×2):
#:   - reescalado bilineal (sin IA): ~20/100
#:   - vecino más cercano (sin IA):  ~32/100
#:   - AnimeVideo v3 (torch fp16):   ~40/100  ·  (onnx fp32): ~49/100
#:   - General v4v3 (onnx):          ~48/100
#: La escala es COMPRIMIDA a propósito: todos los modelos del catálogo son de la
#: misma familia y se parecen; el número sirve para ordenarlos, no para decir
#: que uno es el doble de bueno que otro.
SSIM_FLOOR = 0.70          # SSIM por debajo de esto = 0 en fidelidad
DETAIL_NONE = 1.05         # ganancia de detalle 1.05x (no aporta) = 0
DETAIL_GOOD = 2.60         # ganancia 2.60x (muy nítido) = 100
OVERSHOOT_PENALTY = 6.0    # puntos por cada 1% de píxeles con halo
COLOR_DRIFT_PENALTY = 2.0  # puntos por cada nivel de deriva de color

#: Pesos del índice combinado (fidelidad / detalle / ausencia de artefactos).
SCORE_WEIGHTS = {"fidelity": 0.35, "detail": 0.40, "artifacts": 0.25}


# ---------------------------------------------------------------------------
# Juego de referencia sintético (determinista, sin descargas)
# ---------------------------------------------------------------------------

def reference_pair(height: int = 288, width: int = 512, scale: int = 4):
    """Devuelve ``[(nombre, HR_uint8, LR_uint8), ...]`` para la escala pedida.

    Las imágenes altas (HR) se generan por código y la entrada del modelo (LR)
    sale de promediar bloques ``scale x scale``: es un degradado determinista,
    el mismo en cada equipo y cada ejecución, así que dos modelos se comparan
    siempre sobre exactamente lo mismo.
    """
    scale = max(1, int(scale))
    height = max(8, int(height) - int(height) % scale)
    width = max(8, int(width) - int(width) % scale)
    return [(name, hr, _downscale(hr, scale), scale)
            for name, hr in _reference_images(height, width)]


def _reference_images(height: int, width: int):
    """Imágenes HR sintéticas con características distintas a propósito."""
    yy, xx = np.indices((height, width)).astype(np.float32)
    images = []

    # 1) Line art: bordes duros y líneas finas (típico de anime).
    line_art = np.full((height, width, 3), 235, dtype=np.uint8)
    for offset, thickness in ((40, 2), (90, 1), (150, 3), (220, 1)):
        col = min(width - 1, offset)
        line_art[:, col:col + thickness] = 15
    for offset, thickness in ((30, 2), (120, 1), (200, 2)):
        row = min(height - 1, offset)
        line_art[row:row + thickness, :] = 15
    diag = (np.abs((xx - yy * 0.7) % 90 - 45) < 1.5)
    line_art[diag] = 15
    images.append(("line_art", line_art))

    # 2) Franjas sinusoidales de frecuencia creciente (aliasing y nitidez).
    freq = 0.02 + (xx / max(1.0, width)) * 0.9
    stripes = (127.5 + 127.5 * np.sin(xx * freq)).astype(np.uint8)
    stripes = np.repeat(stripes[:, :, None], 3, axis=2)
    images.append(("stripes", np.ascontiguousarray(stripes)))

    # 3) Gradientes suaves + ruido de banda limitada (zonas planas: banding,
    #    amplificación de ruido y deriva de color).
    rng = np.random.default_rng(1234)
    base = 40 + 0.5 * xx / max(1.0, width - 1) * 180 + 0.3 * yy / max(1.0, height - 1) * 120
    noise = rng.normal(0.0, 6.0, size=(height, width)).astype(np.float32)
    smooth = base + noise
    gradient = np.stack([
        np.clip(smooth, 0, 255),
        np.clip(smooth * 0.85 + 20, 0, 255),
        np.clip(255 - smooth * 0.7, 0, 255),
    ], axis=2).astype(np.uint8)
    images.append(("gradient", np.ascontiguousarray(gradient)))

    # 4) Textura multifractal (síntesis de detalle).
    texture = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    for octave in range(5):
        step = max(1, 2 ** (5 - octave))
        small = rng.normal(0.0, 1.0, size=(height // step + 1, width // step + 1)).astype(np.float32)
        big = np.repeat(np.repeat(small, step, axis=0), step, axis=1)[:height, :width]
        texture += big * amplitude
        amplitude *= 0.5
    texture = (texture - texture.min()) / max(1e-6, float(np.ptp(texture))) * 255.0
    texture_rgb = np.stack([texture, texture * 0.9 + 12, texture * 0.8 + 25], axis=2)
    images.append(("texture", np.clip(texture_rgb, 0, 255).astype(np.uint8)))

    # 5) Parches planos saturados (deriva de color / intercambio de canales).
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30),
               (200, 30, 200), (30, 200, 200), (128, 128, 128), (16, 16, 16)]
    rows, cols = 2, 4
    for index, colour in enumerate(colours):
        row, col = divmod(index, cols)
        r0, r1 = row * height // rows, (row + 1) * height // rows
        c0, c1 = col * width // cols, (col + 1) * width // cols
        chart[r0:r1, c0:c1] = colour
    images.append(("colour_chart", chart))
    return images


def _downscale(image: np.ndarray, scale: int) -> np.ndarray:
    """Promedio de bloques ``scale x scale`` (degradado determinista)."""
    height, width = image.shape[:2]
    height -= height % scale
    width -= width % scale
    cropped = image[:height, :width].astype(np.float32)
    blocks = cropped.reshape(height // scale, scale, width // scale, scale, -1)
    return np.clip(blocks.mean(axis=(1, 3)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Métricas (numpy puro: sin scipy, sin descargas)
# ---------------------------------------------------------------------------

def _luma(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _gaussian_kernel(sigma: float = 1.5, size: int = 11) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    kernel = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _sep_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolución separable 2D vectorizada (sliding_window + producto).

    Las dos pasadas usan el mismo truco: la ventana deslizante deja el kernel
    en el ÚLTIMO eje, así que basta un producto matriz-vector.
    """
    radius = len(kernel) // 2
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=1)
    out = windows @ kernel
    padded = np.pad(out, ((radius, radius), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=0)
    return windows @ kernel


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """PSNR en dB sobre luma (99.0 = prácticamente idénticas)."""
    a, b = _luma(reference), _luma(test)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(min(99.0, 10.0 * np.log10((255.0 ** 2) / mse)))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """SSIM medio (ventana gaussiana 11x11, sigma 1.5), sobre luma."""
    a, b = _luma(reference), _luma(test)
    kernel = _gaussian_kernel()
    mu_a, mu_b = _sep_filter(a, kernel), _sep_filter(b, kernel)
    aa, bb, ab = _sep_filter(a * a, kernel), _sep_filter(b * b, kernel), _sep_filter(a * b, kernel)
    sigma_a = aa - mu_a * mu_a
    sigma_b = bb - mu_b * mu_b
    sigma_ab = ab - mu_a * mu_b
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))


def _edge_energy(image: np.ndarray) -> float:
    """Energía media de bordes (gradientes) sobre luma: mide nitidez real."""
    luma = _luma(image)
    gx = np.abs(np.diff(luma, axis=1)).mean() if luma.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(luma, axis=0)).mean() if luma.shape[0] > 1 else 0.0
    return float(gx + gy)


def _baseline_upscale(low: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reescalado SIN IA (bilineal) para tener una referencia de "no aporta"."""
    try:
        from PIL import Image
        out = Image.fromarray(low).resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(out)
    except Exception:
        height_factor = max(1, height // max(1, low.shape[0]))
        width_factor = max(1, width // max(1, low.shape[1]))
        return np.repeat(np.repeat(low, height_factor, axis=0), width_factor, axis=1)[:height, :width]


def _local_min_max(reference: np.ndarray, radius: int = 1):
    """Mínimo y máximo locales (ventana ``(2r+1)^2``) sobre luma."""
    luma = _luma(reference)
    padded = np.pad(luma, radius, mode="reflect")
    offsets = [(dy, dx) for dy in range(2 * radius + 1)
               for dx in range(2 * radius + 1)]
    stack = np.stack([padded[dy:dy + luma.shape[0], dx:dx + luma.shape[1]]
                      for dy, dx in offsets])
    return stack.min(axis=0), stack.max(axis=0)


def _overshoot_ratio(reference: np.ndarray, test: np.ndarray,
                     margin: float = 12.0, flat_threshold: float = 32.0) -> float:
    """% de píxeles con halo/ringing EN ZONAS PLANAS.

    Sólo se miran zonas localmente planas (rango local pequeño): ahí un halo
    es un artefacto visible. En zonas con detalle, un valor fuera del rango
    local es simplemente "más nítido que el original", y contarlo daría por
    malo a cualquier modelo que recupere detalle.
    """
    local_min, local_max = _local_min_max(reference, radius=1)
    flat = (local_max - local_min) < flat_threshold
    if not flat.any():
        return 0.0
    test_luma = _luma(test)
    outside = (test_luma > local_max + margin) | (test_luma < local_min - margin)
    return float((outside & flat).sum() / max(1, flat.sum()) * 100.0)


def _colour_drift(reference: np.ndarray, test: np.ndarray) -> float:
    """Diferencia media por canal en zonas PLANAS (deriva de color).

    Se mide donde el original es uniforme (carta de color, degradados suaves):
    ahí cualquier cambio de color es del modelo, no del contenido. Mirado en
    todo el frame, la pérdida de textura fina se colaría como "deriva".
    """
    local_min, local_max = _local_min_max(reference, radius=1)
    flat = (local_max - local_min) < 4.0
    if not flat.any():
        return 0.0
    ref = reference.astype(np.float32)
    out = test.astype(np.float32)
    return float(np.abs(ref[flat] - out[flat]).mean())


def score_from_metrics(metrics: dict) -> dict:
    """Convierte las métricas en sub-puntuaciones 0-100 y el índice combinado."""
    ssim_value = float(metrics.get("ssim", 0.0))
    detail_gain = float(metrics.get("detail_gain", 0.0))
    overshoot = float(metrics.get("overshoot_pct", 0.0))
    drift = float(metrics.get("colour_drift", 0.0))

    fidelity = _clamp01((ssim_value - SSIM_FLOOR) / max(1e-6, 1.0 - SSIM_FLOOR)) * 100.0
    detail = _clamp01((detail_gain - DETAIL_NONE) / max(1e-6, DETAIL_GOOD - DETAIL_NONE)) * 100.0
    artifacts = _clamp01(
        1.0 - (overshoot * OVERSHOOT_PENALTY + drift * COLOR_DRIFT_PENALTY) / 100.0
    ) * 100.0
    total = (SCORE_WEIGHTS["fidelity"] * fidelity
             + SCORE_WEIGHTS["detail"] * detail
             + SCORE_WEIGHTS["artifacts"] * artifacts)
    return {
        "score": round(total, 1),
        "fidelity_score": round(fidelity, 1),
        "detail_score": round(detail, 1),
        "artifacts_score": round(artifacts, 1),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Etiquetas en castellano (para quien no quiere leer números)
# ---------------------------------------------------------------------------

def quality_tier(score: float) -> str:
    """Tramo de calidad en palabras (calibrado con las medidas reales)."""
    score = float(score or 0.0)
    if score >= 70:
        return "máxima"
    if score >= 52:
        return "muy alta"
    if score >= 40:
        return "buena"
    if score >= 25:
        return "correcta"
    return "básica"


def speed_tier(fps: float) -> str:
    fps = float(fps or 0.0)
    if fps >= 40:
        return "muy rápida"
    if fps >= 18:
        return "rápida"
    if fps >= 8:
        return "media"
    if fps >= 3:
        return "lenta"
    return "muy lenta"


def format_rating_line(rating: dict | None, bench: dict | None) -> str:
    """Línea compacta para la fila del motor: calidad + velocidad."""
    parts = []
    if rating and rating.get("score") is not None:
        parts.append(f"\u2605 {rating['score']:.0f}/100 ({quality_tier(rating['score'])})")
    if bench and bench.get("fps"):
        parts.append(f"\u26a1 {bench['fps']:.1f} fps ({speed_tier(bench['fps'])})")
    return "  ·  ".join(parts)


def rating_tooltip(rating: dict | None, bench: dict | None) -> str:
    """Explica de dónde sale cada número (el usuario no tiene por qué fiarse a ciegas)."""
    if not rating and not bench:
        return "Sin medir todavía en este equipo."
    lines = []
    if bench and bench.get("fps"):
        lines.append(f"Velocidad medida: {bench['fps']:.1f} fps "
                     f"({bench.get('ms_per_frame')} ms/frame a 512x288, "
                     f"{bench.get('device', '?')}).")
    if rating:
        lines.append(
            "Calidad medida en este equipo sobre 5 imágenes de referencia "
            "sintéticas (líneas finas, franjas, degradados con ruido, textura "
            "y carta de color), con la misma tarea para todos los modelos "
            f"(reconstruir desde una degradación ×{rating.get('task_scale', 2)} "
            "y llevar la salida al tamaño final, como el render real):\n"
            f"  · fidelidad (SSIM/PSNR): {rating.get('fidelity_score', 0):.0f}/100\n"
            f"  · detalle (nitidez real frente a un reescalado sin IA): "
            f"{rating.get('detail_score', 0):.0f}/100\n"
            f"  · ausencia de artefactos (halos y deriva de color): "
            f"{rating.get('artifacts_score', 0):.0f}/100\n"
            f"  · índice combinado: {rating.get('score', 0):.0f}/100 "
            f"({quality_tier(rating.get('score'))})\n"
            "Es un baremo repetible para comparar modelos entre sí, no una "
            "verdad absoluta: mira también tu propio vídeo."
        )
        if rating.get("measured_at"):
            lines.append(f"Medido: {rating['measured_at']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def _ratings_path() -> str:
    from src.config import ensure_runtime_dirs, get_session_dir
    ensure_runtime_dirs()
    return os.path.join(get_session_dir(), RATINGS_FILENAME)


def _load_ratings() -> dict:
    try:
        with open(_ratings_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_ratings(data: dict) -> None:
    try:
        path = _ratings_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug("could not persist model ratings: %s", exc)


def _rating_key(engine_id: str, backend: str, fingerprint: dict | None) -> str:
    fingerprint = fingerprint or {}
    stamp = f"{fingerprint.get('size')}_{fingerprint.get('mtime_ns')}"
    return f"{engine_id}|{backend}|{stamp}|v{RATING_VERSION}"


def all_ratings() -> dict:
    return _load_ratings()


def get_rating(engine_id: str, backend: str, fingerprint: dict | None = None) -> dict | None:
    """Rating cacheado de un motor/backend (o None si no se ha medido).

    Con ``fingerprint`` se busca la medida exacta de esos pesos. Sin él se
    devuelve la mejor medida disponible de ese motor: se prefieren las que
    llevan huella real (una medida suelta de un experimento no debe tapar la
    que hizo la app con los pesos instalados) y, entre esas, la más reciente.
    """
    data = _load_ratings()
    if fingerprint is not None:
        return data.get(_rating_key(engine_id, backend, fingerprint))
    prefix = f"{engine_id}|{backend}|"
    matches = [(key, value) for key, value in data.items() if key.startswith(prefix)]
    if not matches:
        return None

    def _rank(item):
        key, value = item
        parts = key.split("|")
        stamp = parts[2] if len(parts) > 2 else ""
        return (stamp not in ("", "None_None"), str(value.get("measured_at", "")))

    return max(matches, key=_rank)[1]


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------

def measure_quality(engine, scale: int | None = None,
                    height: int = 288, width: int = 512,
                    max_images: int | None = None,
                    task_scale: int | None = None) -> dict | None:
    """Ejecuta ``engine`` sobre el juego de referencia y devuelve las métricas.

    ``engine`` es cualquier objeto con ``process_frame(BGR uint8 HWC)`` (los
    motores reales de la app). Devuelve ``None`` si no se pudo medir: medir la
    calidad nunca debe romper nada.

    **Todos los modelos se miden con la MISMA tarea** (``task_scale``, por
    defecto x2): la referencia se degrada siempre igual y la salida del modelo
    se lleva al tamaño original, exactamente como hace el render real cuando
    exportas por debajo de la escala nativa. Así un x4 y un x2 son comparables
    (antes al x4 se le pedía reconstruir desde 1/16 de los píxeles, una tarea
    mucho más difícil, y parecía peor de lo que es). La escala nativa se sigue
    reportando en ``native_scale``.
    """
    if engine is None or not hasattr(engine, "process_frame"):
        return None
    try:
        task = max(1, min(8, int(task_scale or DEFAULT_TASK_SCALE)))
        native_scale = scale
        if native_scale is None:
            native_scale = int(getattr(engine, "get_scale_factor", lambda: 4)() or 4)
        pairs = reference_pair(height=height, width=width, scale=task)
        if max_images:
            pairs = pairs[:max_images]

        per_image = []
        started = time.time()
        detected_scale = None
        for name, high, low, _scale in pairs:
            output = engine.process_frame(low)
            if output is None:
                return None
            output = np.asarray(output)
            if output.ndim != 3 or output.shape[2] != 3:
                return None
            if detected_scale is None and low.shape[0]:
                detected_scale = int(round(output.shape[0] / low.shape[0]))
            # La salida se lleva al tamaño de la imagen buena (como el render
            # real, que reescala a la resolución de export).
            if output.shape[:2] != high.shape[:2]:
                output = _baseline_upscale(output, high.shape[0], high.shape[1])
            baseline = _baseline_upscale(low, high.shape[0], high.shape[1])
            detail_gain = (_edge_energy(output) / max(1e-6, _edge_energy(baseline)))
            per_image.append({
                "name": name,
                "psnr": psnr(high, output),
                "ssim": ssim(high, output),
                "detail_gain": float(detail_gain),
                "overshoot_pct": _overshoot_ratio(high, output),
                "colour_drift": _colour_drift(high, output),
            })
        if not per_image:
            return None
        metrics = {
            "psnr_db": round(float(np.mean([m["psnr"] for m in per_image])), 2),
            "ssim": round(float(np.mean([m["ssim"] for m in per_image])), 4),
            "detail_gain": round(float(np.mean([m["detail_gain"] for m in per_image])), 3),
            "overshoot_pct": round(float(np.mean([m["overshoot_pct"] for m in per_image])), 3),
            "colour_drift": round(float(np.mean([m["colour_drift"] for m in per_image])), 3),
            "images": per_image,
            "resolution": [width, height],
            "scale": task,
            "task_scale": task,
            "native_scale": detected_scale or native_scale,
            "seconds": round(time.time() - started, 2),
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "version": RATING_VERSION,
            "weights": dict(SCORE_WEIGHTS),
        }
        metrics.update(score_from_metrics(metrics))
        return metrics
    except Exception as exc:
        logger.info("quality measurement failed: %s: %s", type(exc).__name__, exc)
        return None


def measure_and_store(engine_id: str, backend: str, engine,
                      fingerprint: dict | None = None,
                      scale: int | None = None) -> dict | None:
    """Mide la calidad y la guarda en la caché de ratings."""
    metrics = measure_quality(engine, scale=scale)
    if metrics is None:
        return None
    data = _load_ratings()
    data[_rating_key(engine_id, backend, fingerprint)] = metrics
    _save_ratings(data)
    return metrics


def recommendation(ratings: dict) -> str:
    """Frase en castellano para quien no quiere interpretar números.

    ``ratings`` es ``{nombre: (rating|None, bench|None)}``.
    """
    measured = {name: (r, b) for name, (r, b) in ratings.items()
                if (r and r.get("score") is not None) or (b and b.get("fps"))}
    if not measured:
        return "Todavía no hay modelos medidos en este equipo."
    fastest = max(measured.items(),
                  key=lambda kv: (kv[1][1] or {}).get("fps", 0) or 0)
    best = max(measured.items(),
               key=lambda kv: (kv[1][0] or {}).get("score", 0) or 0)
    fastest_fps = (fastest[1][1] or {}).get("fps", 0) or 0
    best_score = (best[1][0] or {}).get("score", 0) or 0
    if fastest[0] == best[0] and fastest_fps and best_score:
        return (f"«{fastest[0]}» es el más rápido ({fastest_fps:.1f} fps) y el "
                f"de mejor calidad medida ({best_score:.0f}/100): no hay que elegir.")
    lines = []
    if fastest_fps:
        lines.append(f"Si quieres ir rápido: «{fastest[0]}» "
                     f"({fastest_fps:.1f} fps, {speed_tier(fastest_fps)}).")
    if best_score:
        lines.append(f"Si quieres la mejor imagen: «{best[0]}» "
                     f"({best_score:.0f}/100, calidad {quality_tier(best_score)}).")
    return "  ".join(lines)
