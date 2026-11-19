"""Recomendación de modelos según la GPU y la VRAM disponible.

El catálogo ya sabe qué modelos existen y cuáles están instalados, pero no
respondía a la pregunta práctica: *¿cuál debo elegir en este equipo?* Este
módulo añade un perfil de hardware barato (sin importar torch/onnxruntime en
el proceso de la GUI), una estimación conservadora de VRAM y una elección
razonada por categoría.

La estimación es deliberadamente una **aproximación**, no una promesa: los
pesos ocupan una parte y las activaciones dependen del tamaño de frame, del
tiling y del backend. Se etiqueta como "estimated" en la UI para no vender
precisión que no tenemos. La decisión de "no cabe" usa el mismo margen
conservador (95% de la VRAM libre) y solo se muestra cuando hay un dato real
de memoria; sin GPU o sin tamaño de modelo se devuelve ``unknown``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_MEMORY_RE = re.compile(r"([\d.,]+)\s*(GB|GiB|MB|MiB)", re.IGNORECASE)
_SCALE_RE = re.compile(r"(?:^|[^a-z0-9])(?:x|×)?([234])(?:x|×)?(?:[^0-9]|$)", re.IGNORECASE)

# Peso de las activaciones por categoría a 1080p. Es una constante de
# ingeniería, no una medición: sirve para ordenar y avisar, y la UI lo dice.
_ACTIVATION_MB = {
    "Upscale": 512,
    "FaceRestore": 512,
    "Denoise": 384,
    "Interpolate": 256,
    "Colorize": 256,
}
_WEIGHT_OVERHEAD = 2.5
_MIN_ESTIMATE_MB = 128


@dataclass(frozen=True)
class GpuProfile:
    """Resumen de la GPU principal, sin dependencias nativas."""

    name: str = ""
    vendor: str = ""
    total_mb: Optional[int] = None
    free_mb: Optional[int] = None
    backend: str = "CPU"

    @property
    def has_gpu(self) -> bool:
        return bool(self.name) and self.backend != "CPU"

    def available_mb(self) -> Optional[int]:
        if self.free_mb is not None:
            return max(0, int(self.free_mb))
        if self.total_mb is not None:
            return max(0, int(self.total_mb))
        return None

    def label(self) -> str:
        if not self.has_gpu:
            return "CPU only"
        mem = ""
        if self.total_mb:
            mem = f" · {self.total_mb / 1024:.1f} GB"
            free = self.available_mb()
            if free is not None and free != self.total_mb:
                mem += f" ({free / 1024:.1f} GB free)"
        return f"{self.name}{mem} · {self.backend}"


def parse_memory_mb(value) -> Optional[int]:
    """Convierte ``"12288 MiB"``/``"12 GB"``/``12288`` a MiB."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    try:
        return int(float(text))
    except ValueError:
        pass
    match = _MEMORY_RE.search(text)
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    unit = match.group(2).lower()
    if unit in ("gb", "gib"):
        number *= 1024
    return int(round(number))


def _vendor_backend(vendor: str) -> str:
    vendor = (vendor or "").strip().lower()
    if "nvidia" in vendor:
        return "CUDA"
    if "amd" in vendor or "radeon" in vendor:
        return "GPU"
    if "intel" in vendor:
        return "GPU"
    return "CPU"


def current_profile() -> GpuProfile:
    """Perfil del equipo actual, reutilizando la detección ya hecha.

    ``MainWindow`` publica el resumen de hardware al arrancar; usar
    ``get_known_hardware`` evita repetir PowerShell/nvidia-smi al abrir el
    catálogo. En un proceso sin detección previa se hace una sola.
    """
    try:
        from src.hardware.detector import get_hardware_summary, get_known_hardware

        summary = get_known_hardware() or get_hardware_summary()
    except Exception as exc:  # hardware es un extra, nunca un bloqueo
        logger.debug("hardware profile unavailable: %s", exc)
        return GpuProfile()

    gpus = list((summary or {}).get("gpus") or [])
    if not gpus:
        return GpuProfile()

    # Preferir la GPU con más VRAM; si no hay dato, la primera dedicada.
    def _rank(gpu):
        mem = parse_memory_mb(gpu.get("memory")) or 0
        vendor = str(gpu.get("vendor") or "").lower()
        dedicated = 1 if vendor in ("nvidia", "amd") else 0
        return (dedicated, mem)

    gpu = max(gpus, key=_rank)
    total = parse_memory_mb(gpu.get("memory"))
    used = parse_memory_mb(gpu.get("memory_used"))
    free = max(0, total - used) if total is not None and used is not None else total
    vendor = str(gpu.get("vendor") or "")
    return GpuProfile(
        name=str(gpu.get("name") or "").strip(),
        vendor=vendor,
        total_mb=total,
        free_mb=free,
        backend=_vendor_backend(vendor),
    )


def _model_size_mb(model: dict) -> Optional[float]:
    for key in ("size_mb", "size_estimate", "onnx_size_estimate"):
        value = model.get(key)
        parsed = parse_memory_mb(value) if not isinstance(value, (int, float)) else value
        if parsed:
            return float(parsed)
    return None


def _scale_hint(model: dict) -> int:
    text = " ".join(
        str(model.get(k) or "")
        for k in ("engine_id", "display_name", "sample_shape", "scale")
    ).lower()
    # ``x4``/``4x``; si no hay pista, un upscaler suele ser x4.
    match = _SCALE_RE.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return 4 if (model.get("category") == "Upscale") else 1


def estimate_model_vram_mb(model: dict) -> Optional[int]:
    """Estimación conservadora de VRAM (pesos + activaciones).

    Devuelve ``None`` si el catálogo/Hub no da ningún tamaño: es mejor no
    inventar una cifra que mostrar un "no cabe" falso.
    """
    size = _model_size_mb(model)
    if size is None:
        return None
    scale = max(1, _scale_hint(model))
    activation = _ACTIVATION_MB.get(str(model.get("category") or ""), 384)
    # Un x4 necesita más mapa de activaciones que un x2; normalizado a x4.
    activation = int(activation * max(0.35, (scale / 4.0) ** 2))
    estimate = size * _WEIGHT_OVERHEAD + activation
    return max(_MIN_ESTIMATE_MB, int(round(estimate)))


def _fit_for_estimate(estimate_mb: Optional[int], profile: GpuProfile) -> tuple[str, Optional[int]]:
    if estimate_mb is None:
        return "unknown", None
    available = profile.available_mb()
    if not profile.has_gpu or available is None or available <= 0:
        return "unknown", None
    if estimate_mb > available * 0.95:
        return "too_large", available
    if estimate_mb > available * 0.75:
        return "tight", available
    return "fits", available


def annotate_models(
    models: Iterable[dict],
    profile: Optional[GpuProfile] = None,
) -> tuple[list[dict], GpuProfile]:
    """Copia los modelos añadiendo estimación y encaje de VRAM."""
    profile = profile or current_profile()
    annotated: list[dict] = []
    for model in models or []:
        item = dict(model)
        estimate = estimate_model_vram_mb(item)
        source = "estimated" if estimate is not None else "unknown"
        # Si este equipo ya midió el motor con nvidia-smi, el dato real manda
        # sobre la fórmula: pesos + contexto CUDA + workspace pueden diferir
        # mucho entre modelos con el mismo tamaño de fichero.
        try:
            from src.ai import vram_calibration

            measured = vram_calibration.lookup(str(item.get("engine_id") or ""))
        except Exception:
            measured = None
        if measured and measured.get("peak_delta_mb"):
            try:
                estimate = int(round(float(measured["peak_delta_mb"])))
                source = "measured"
                item["vram_measured_resolution"] = measured.get("resolution", "")
                item["vram_measured_at_iso"] = measured.get("measured_at_iso", "")
            except (TypeError, ValueError):
                pass
        fit, available = _fit_for_estimate(estimate, profile)
        item["vram_estimate_mb"] = estimate
        item["vram_source"] = source
        item["vram_fit"] = fit
        item["gpu_available_mb"] = available
        item["gpu_total_mb"] = profile.total_mb
        item["gpu_name"] = profile.name
        annotated.append(item)
    return annotated, profile


def recommend_models(
    models: Iterable[dict],
    category: Optional[str] = None,
    profile: Optional[GpuProfile] = None,
) -> dict:
    """Elige el modelo recomendado para ``category`` en este hardware.

    Prioriza lo que ya está instalado y habilitado, descarta lo que no cabe
    y desempata por calidad/popularidad. Devuelve un dict con
    ``engine_id``, ``display_name``, ``reason`` y los datos de VRAM; si no
    hay candidatos, ``engine_id`` es ``""``.
    """
    annotated, profile = annotate_models(models, profile)
    wanted = (category or "").strip().lower()
    candidates = [
        m for m in annotated
        if not wanted or str(m.get("category") or "").lower() == wanted
    ]
    if not candidates:
        return {
            "engine_id": "",
            "display_name": "",
            "reason": "No models in this category.",
            "profile": profile,
            "models": annotated,
        }

    def _score(model: dict) -> tuple:
        fit = model.get("vram_fit")
        fit_rank = {"fits": 3, "tight": 2, "unknown": 1, "too_large": 0}.get(fit, 0)
        enabled = 1 if model.get("enabled") else 0
        installed = 1 if model.get("installed") else 0
        quality = float(model.get("quality_score") or 0)
        popularity = float(model.get("popularity_score") or 0)
        try:
            estimate = float(model.get("vram_estimate_mb") or 0)
        except (TypeError, ValueError):
            estimate = 0.0
        # Un modelo habilitado manda; después la calidad. El tamaño solo
        # desempata para preferir el más ligero cuando todo lo demás empata.
        return (fit_rank, enabled, installed, quality, popularity, -estimate)

    # Si hay candidatos que caben, no recomendamos uno que no cabe.
    viable = [m for m in candidates if m.get("vram_fit") != "too_large"] or candidates
    best = max(viable, key=_score)
    fit = best.get("vram_fit")
    if fit == "too_large":
        reason = "No model comfortably fits the detected VRAM; pick the smallest."
    elif best.get("enabled"):
        reason = "Already installed and enabled"
    elif best.get("installed"):
        reason = "Already installed"
    else:
        reason = "Best fit for this GPU"
    if fit == "tight":
        reason += " · tight VRAM"
    elif fit == "fits":
        reason += (
            " · fits measured VRAM"
            if best.get("vram_source") == "measured"
            else " · fits estimated VRAM"
        )
    if best.get("vram_source") == "measured":
        reason += " · measured on this GPU"
    return {
        "engine_id": best.get("engine_id", ""),
        "display_name": best.get("display_name", ""),
        "reason": reason,
        "vram_estimate_mb": best.get("vram_estimate_mb"),
        "vram_source": best.get("vram_source"),
        "vram_fit": fit,
        "profile": profile,
        "models": annotated,
    }


def get_catalog_recommendation(
    category: Optional[str] = None,
    profile: Optional[GpuProfile] = None,
) -> dict:
    """Recomendación usando solo el catálogo local (sin consultar HF)."""
    try:
        from src.ai.registry import iter_catalog

        models = []
        for entry in iter_catalog():
            if category and entry.get("category") != category:
                continue
            item = dict(entry)
            state = item.get("state") or {}
            item["installed"] = bool(state.get("installed"))
            item["enabled"] = bool(state.get("enabled"))
            models.append(item)
    except Exception as exc:
        logger.debug("catalog recommendation unavailable: %s", exc)
        models = []
    return recommend_models(models, category=category, profile=profile)
