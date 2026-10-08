"""Especificaciones de arquitectura que la UI puede leer SIN cargar torch.

``src/ai/architectures.py`` importa PyTorch al importarse (para definir las
clases ``nn.Module`` de los modelos). Eso lo hace válido para el proceso de
IA, pero **veneno** para el proceso de la GUI: dos widgets de la interfaz
(``src/ui/engine_row.py`` y ``src/ui/model_search_result.py``) solo necesitan
la lista de arquitecturas convertibles a ONNX, y ese import a nivel de módulo
arrastraba torch entero (torch_cpu.dll + torch_cuda.dll) al arrancar
DraggyEncoder.exe — la violación de acceso en python313.dll que hacía que el
instalador "no arrancara".

Este módulo es la frontera: solo constantes y funciones puras, sin torch, sin
onnxruntime y sin onnx. ``architectures.py`` re-exporta estos nombres para no
romper a los llamadores existentes.
"""

from __future__ import annotations

# Arquitecturas que ``build_model_from_state_dict`` puede construir y, por
# tanto, las únicas que el conversor ONNX soporta. La UI la consulta para
# deshabilitar "Convert to ONNX" en motores no convertibles.
ONNX_CONVERTIBLE_ARCHS = frozenset({"omnisr", "rrdb", "srvgg", "cugan"})


def onnx_convertible_archs() -> frozenset:
    """Arquitecturas soportadas por el conversor ONNX (set inmutable)."""
    return ONNX_CONVERTIBLE_ARCHS


def arch_is_onnx_convertible(arch: str) -> bool:
    """True si ``arch`` puede convertirse a ONNX con el conversor propio."""
    return bool(arch) and arch in ONNX_CONVERTIBLE_ARCHS


#: Patrones (compilados de forma perezosa) que parecen una escala en el nombre.
_SCALE_NAME_RE = None


def parse_scale_from_name(fname: str) -> int:
    """Factor de upscale (2, 3, 4, 8...) que sugiere el NOMBRE del fichero.

    Cubre las convenciones del ecosistema Real-ESRGAN / Real-CUGAN /
    OmniSR en ambos órdenes ("x4plus", "x4v3", "4x-UltraSharp"). Es sólo
    una **pista**: el nombre miente (``OmniSR_X2_DIV2K.onnx`` es un x8 y
    los "x2plus" de AI-Forever son x4), así que sólo se usa cuando el
    fichero no se puede inspeccionar.
    """
    global _SCALE_NAME_RE
    if _SCALE_NAME_RE is None:
        import re
        _SCALE_NAME_RE = [
            # x2, x4, x8... seguidos de plus/vN o del final del nombre
            re.compile(r"[xX](\d+)(?=(?:plus|v\d+)?(?:$|[\s_.-]))"),
            # 2x, 4x ... por delante del nombre
            re.compile(r"(?:^|[\s_.-])(\d+)[xX](?=(?:plus|v\d+)?(?:$|[\s_.-]))"),
        ]
    best = 1
    for rx in _SCALE_NAME_RE:
        for m in rx.finditer(fname or ""):
            try:
                value = int(m.group(1))
            except (TypeError, ValueError):
                continue
            # Ignorar números absurdos que parecen versión o resolución.
            if 1 < value <= 64:
                best = max(best, value)
    return best
