"""Utilidades sobre pesos que necesitan torch, pedidas al HOST de IA.

La conversión ``.pth`` → ``.onnx`` (``torch.onnx.export``) y la lectura de los
canales de entrada de un checkpoint necesitan PyTorch. Ejecutarlas dentro del
proceso de la GUI carga sus DLL nativas ahí y Windows mata la aplicación con
una violación de acceso (0xc0000005 en python313.dll) — es el "instalado no
arranca" que perseguimos.

Estas funciones mantienen la semántica de ``src.ai.inference`` pero el trabajo
lo hace el host (``src.ai.host``), en su propio proceso. Sin host no hay
plan B: se devuelve el motivo y el llamador sigue con lo que tenga (el ``.onnx``
ya presente, o el peso sin convertir), nunca se importa torch aquí.
"""

from __future__ import annotations

import logging

from src.ai import host_client

logger = logging.getLogger(__name__)


def weight_tools_available() -> bool:
    """¿Hay host para estas operaciones? (no lo arranca)."""
    return host_client.host_available()


def convert_weight_to_onnx(pth_path: str, progress_callback=None):
    """``(onnx_path, mensaje)``: convierte el peso EN EL HOST de IA.

    Contrato idéntico a ``inference.convert_pth_to_onnx_ex``: ``onnx_path`` es
    ``None`` si no se pudo convertir y ``mensaje`` explica por qué.
    """
    if not host_client.host_available():
        return None, (
            "el host de IA no está disponible: no se puede convertir este peso "
            "a ONNX (se usará el .onnx que ya exista)"
        )
    try:
        data = host_client.get_host().call_op(
            "convert_pth_to_onnx",
            path=pth_path,
            progress_callback=progress_callback,
        )
    except host_client.HostError as exc:
        return None, str(exc)
    return data.get("onnx_path"), str(data.get("message") or "")


def model_file_scale(model_path: str) -> int:
    """Escala nativa del fichero, medida en el host (1 = no demostrada)."""
    if not host_client.host_available():
        return 1
    try:
        data = host_client.get_host().call_op("model_scale", path=model_path)
    except host_client.HostError as exc:
        logger.debug("escala de %s no disponible: %s", model_path, exc)
        return 1
    try:
        return int(data.get("scale") or 1)
    except (TypeError, ValueError):
        return 1


def arch_and_scale(weight_path: str):
    """``(arch, scale)`` de un peso, detectados en el host (usa torch)."""
    if not host_client.host_available():
        return None, None
    try:
        data = host_client.get_host().call_op("detect_arch_and_scale", path=weight_path)
    except host_client.HostError as exc:
        logger.debug("arch/escala de %s no disponibles: %s", weight_path, exc)
        return None, None
    return data.get("arch"), data.get("scale")


def benchmark(name: str, **kwargs):
    """Ejecuta ``benchmark_backend``/``benchmark_quality`` EN EL HOST.

    ``None`` si el host no está disponible o la medición falla: los llamadores
    ya tratan la ausencia de benchmark como "sin datos", no como error.
    """
    if not host_client.host_available():
        return None
    try:
        data = host_client.get_host().call_op("benchmark", name=name, kwargs=kwargs)
    except host_client.HostError as exc:
        logger.debug("benchmark %s no disponible: %s", name, exc)
        return None
    return data.get("result")


def weight_input_channels(weight_path: str):
    """Canales de entrada del checkpoint leídos por el host (``None`` si no).

    ``None`` significa "no se pudo determinar", no "3 canales": el llamador no
    debe descartar el modelo por eso (los de 12 canales sí se detectan).
    """
    if not host_client.host_available():
        return None
    try:
        data = host_client.get_host().call_op("pth_channel_count", path=weight_path)
    except host_client.HostError as exc:
        logger.debug("canales de %s no disponibles: %s", weight_path, exc)
        return None
    return data.get("channels")
