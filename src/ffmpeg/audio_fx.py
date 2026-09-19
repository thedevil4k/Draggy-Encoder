"""Cadena de procesado de audio opcional que ALTERA el Audio Boost.

El Audio Boost tiene dos comportamientos según lo que haga el usuario:

1. **Casilla sola** (desplegable cerrado o sin nada activado): two-pass
   ``loudnorm`` en modo lineal, exactamente como siempre. Este módulo no
   interviene.

2. **Casilla + desplegable con algo activado**: el boost se recalcula para
   dejar trabajar a los efectos. El usuario pidió que las opciones del
   desplegable *alteren* el boost, no que se apilen encima.

ORDEN DE LA CADENA (cada paso está donde está por una razón medida)::

    EQ  ->  ganancia al objetivo  ->  compresor  ->  normalizar  ->  limitador

Por qué el compresor va DESPUÉS del boost y no antes: los umbrales de un
compresor son absolutos. Sobre material sin normalizar (típicamente -25 LUFS)
un umbral de -18 dB no se dispara nunca, así que el usuario movería los
controles y no oiría nada — medido: `acompressor=threshold=-18dB:ratio=4`
antes del boost dejaba el rango dinámico idéntico (sin efecto). Después del
boost sí trabaja.

Por qué hay una normalización DESPUÉS del compresor: un compresor baja el
nivel (medido: 7 LU con ratio 4), y el limitador no puede recuperarlo (solo
baja picos, no sube nivel). Sin esa segunda normalización el archivo acababa
en -23 LUFS, muy por debajo del objetivo. Normalizar después devuelve el nivel
a -16 LUFS **conservando** la compresión aplicada (medido: rango 8.7 -> 5.7 LU,
más comprimido que el propio boost lineal, que deja 7.6).

Por qué el limitador va al final: ``loudnorm`` limita *antes* del codificador
y el codificador añade picos entre muestras, así que el archivo entregado se
queda por encima del techo pedido. Medido en el boost actual: objetivo -1.5
dBTP, entregado -0.20 dBTP.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rangos y valores por defecto
# ---------------------------------------------------------------------------

EQ_BANDS = (
    {"key": "bass", "label": "Graves", "freq": 100, "min": -12, "max": 12},
    {"key": "mid", "label": "Medios", "freq": 1000, "min": -12, "max": 12},
    {"key": "treble", "label": "Agudos", "freq": 8000, "min": -12, "max": 12},
)

COMPRESSOR_DEFAULTS = {
    "threshold_lu": 8.0,   # LU por debajo del nivel del material
    "ratio": 3.0,
    "attack_ms": 20.0,
    "release_ms": 250.0,
}
COMPRESSOR_LIMITS = {
    "threshold_lu": (0.0, 24.0),
    "ratio": (1.0, 20.0),
    "attack_ms": (0.1, 200.0),
    "release_ms": (10.0, 2000.0),
}

LIMITER_DEFAULTS = {"enabled": False, "ceiling_dbtp": -1.5}
LIMITER_LIMITS = {"ceiling_dbtp": (-6.0, 0.0)}

# Techo del limitador que va ANTES del codificador. Se descuenta del objetivo
# pedido porque el codificador añade picos entre muestras: medido, el boost
# lineal con objetivo -1.5 entrega -0.20 dBTP, es decir ~1.3 dB de más.
ENCODER_PEAK_MARGIN_DB = 1.3

# Límite práctico del desplazamiento de nivel. Si los ajustes piden un cambio
# mayor que esto no es que la cadena esté mal: es que el material es un caso
# extremo y el resultado puede desviarse del objetivo de loudness.
MAX_TARGET_SHIFT_DB = 8.0

_TARGET_LUFS_DEFAULT = -16.0
DEFAULT_TARGET_LUFS = _TARGET_LUFS_DEFAULT


def default_effects() -> dict:
    """Efectos por defecto: TODO desactivado.

    Es deliberado. Mientras el usuario no active nada, el boost se comporta
    exactamente como siempre (loudnorm lineal two-pass), así que abrir el
    desplegable y no tocar nada no puede cambiar el sonido ni el pico del
    archivo. El limitador tampoco viene activo: aunque mejora el pico
    (-0.20 dBTP de hoy frente a -2.2 dBTP), es un cambio de comportamiento que
    debe pedir el usuario, no colarse por tener el desplegable abierto.
    """
    return {
        "enabled": False,
        "target_lufs": _TARGET_LUFS_DEFAULT,
        "eq": {"enabled": False, **{b["key"]: 0.0 for b in EQ_BANDS}},
        "compressor": {"enabled": False, **COMPRESSOR_DEFAULTS},
        "limiter": dict(LIMITER_DEFAULTS),
    }


def _clamp_number(value, low: float, high: float, default: float) -> float:
    """Convierte a float y acota al rango; ante duda devuelve el defecto."""
    if value is None or isinstance(value, bool):
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number:  # NaN
        return float(default)
    return float(max(low, min(high, number)))


def sanitize_effects(raw) -> dict:
    """Normaliza y acota una configuración de efectos.

    Cualquier valor ausente, no numérico o fuera de rango se sustituye por el
    de por defecto en vez de propagarse a FFmpeg: un filtro con un parámetro
    inválido hace fallar TODO el render, no solo el audio.
    """
    result = default_effects()
    if not isinstance(raw, dict):
        return result

    result["enabled"] = bool(raw.get("enabled", False))
    result["target_lufs"] = _clamp_number(
        raw.get("target_lufs"), -40.0, -5.0, _TARGET_LUFS_DEFAULT
    )

    eq_raw = raw.get("eq") if isinstance(raw.get("eq"), dict) else {}
    result["eq"]["enabled"] = bool(eq_raw.get("enabled", False))
    for band in EQ_BANDS:
        key = band["key"]
        result["eq"][key] = _clamp_number(
            eq_raw.get(key), band["min"], band["max"], 0.0
        )

    comp_raw = raw.get("compressor") if isinstance(raw.get("compressor"), dict) else {}
    result["compressor"]["enabled"] = bool(comp_raw.get("enabled", False))
    for key, (low, high) in COMPRESSOR_LIMITS.items():
        result["compressor"][key] = _clamp_number(
            comp_raw.get(key), low, high, COMPRESSOR_DEFAULTS[key]
        )

    lim_raw = raw.get("limiter") if isinstance(raw.get("limiter"), dict) else {}
    result["limiter"]["enabled"] = bool(
        lim_raw.get("enabled", LIMITER_DEFAULTS["enabled"])
    )
    low, high = LIMITER_LIMITS["ceiling_dbtp"]
    result["limiter"]["ceiling_dbtp"] = _clamp_number(
        lim_raw.get("ceiling_dbtp"), low, high, LIMITER_DEFAULTS["ceiling_dbtp"]
    )
    return result


def eq_is_active(cfg: dict) -> bool:
    return bool(cfg["eq"]["enabled"]) and any(
        abs(cfg["eq"][b["key"]]) > 0.01 for b in EQ_BANDS
    )


def compressor_is_active(cfg: dict) -> bool:
    return bool(cfg["compressor"]["enabled"]) and cfg["compressor"]["ratio"] > 1.0


def limiter_is_active(cfg: dict) -> bool:
    return bool(cfg["limiter"]["enabled"])


def effects_are_active(effects) -> bool:
    """True si el desplegable cambia de verdad el resultado.

    Determina el modo del boost: si esto es False, la casilla sola se comporta
    exactamente como siempre (loudnorm lineal).
    """
    cfg = sanitize_effects(effects)
    if not cfg["enabled"]:
        return False
    return (
        eq_is_active(cfg)
        or compressor_is_active(cfg)
        or limiter_is_active(cfg)
    )


def boost_uses_linear_loudnorm(effects) -> bool:
    """Modo del normalizador para el boost.

    True  -> casilla sola: two-pass loudnorm lineal (comportamiento de siempre).
    False -> con efectos: ganancia simple + normalización final, que es lo que
             deja al compresor y al limitador hacer su trabajo.
    """
    return not effects_are_active(effects)


def describe_effects(effects) -> str:
    """Resumen en una línea para el historial de tareas y los logs."""
    cfg = sanitize_effects(effects)
    if not effects_are_active(cfg):
        return "boost estandar"
    parts = []
    if eq_is_active(cfg):
        bands = ", ".join(
            f"{b['label']} {cfg['eq'][b['key']]:+.0f}dB"
            for b in EQ_BANDS
            if abs(cfg["eq"][b["key"]]) > 0.01
        )
        parts.append(f"EQ({bands})" if bands else "EQ(plano)")
    if compressor_is_active(cfg):
        c = cfg["compressor"]
        parts.append(
            f"compresor {c['ratio']:.1f}:1 @ -{c['threshold_lu']:.0f}LU"
            f" atk {c['attack_ms']:.0f}ms rel {c['release_ms']:.0f}ms"
        )
    if limiter_is_active(cfg):
        parts.append(f"limitador {cfg['limiter']['ceiling_dbtp']:.1f} dBTP")
    return f"boost + {' + '.join(parts)}" if parts else "boost estandar"


# ---------------------------------------------------------------------------
# Construcción de los filtros
# ---------------------------------------------------------------------------


def build_eq_filters(effects) -> list[str]:
    """Filtros del ecualizador (estantes en graves/agudos y banda en medios).

    Se usan ``bass``/``treble``/``equalizer`` y no ``anequalizer`` porque estos
    son multicanal sin tener que declarar el layout de canales: comprobado
    sobre audio 5.1 y estéreo real.
    """
    cfg = sanitize_effects(effects)
    if not cfg["eq"]["enabled"]:
        return []
    filters = []
    if abs(cfg["eq"]["bass"]) > 0.01:
        filters.append(f"bass=g={cfg['eq']['bass']:.2f}:f=100")
    if abs(cfg["eq"]["mid"]) > 0.01:
        filters.append(f"equalizer=f=1000:t=q:w=1:g={cfg['eq']['mid']:.2f}")
    if abs(cfg["eq"]["treble"]) > 0.01:
        filters.append(f"treble=g={cfg['eq']['treble']:.2f}:f=8000")
    return filters


def level_gain_db(effects, measured_i) -> float | None:
    """Ganancia para llevar el material al objetivo, o None si no se puede.

    Es una ganancia PURA (sin limitar ni comprimir): su único trabajo es
    colocar el material en el nivel donde el compresor trabaja.
    """
    cfg = sanitize_effects(effects)
    try:
        level = float(measured_i)
    except (TypeError, ValueError):
        return None
    if level != level or level < -70.0:  # silencio o inválido
        return None
    gain = float(cfg["target_lufs"]) - level
    if abs(gain) > MAX_TARGET_SHIFT_DB * 2:
        # Material extremo: sigue siendo válido, pero se anota.
        logger.info("ajuste de nivel grande en el boost: %.1f dB", gain)
    return gain


def compressor_threshold_db(effects, measured_i) -> float | None:
    """Umbral absoluto (dBFS) del compresor a partir del nivel del material.

    ``threshold_lu`` son LU por debajo del nivel, de modo que el compresor
    trabaja siempre, venga la fuente a -14 o a -30 LUFS.
    """
    cfg = sanitize_effects(effects)
    if not compressor_is_active(cfg):
        return None
    try:
        level = float(measured_i)
    except (TypeError, ValueError):
        return None
    if level != level or level < -70.0:
        return None
    # El compresor ve el material ya desplazado al objetivo.
    shifted = float(cfg["target_lufs"])
    threshold = shifted - float(cfg["compressor"]["threshold_lu"])
    return max(-60.0, min(0.0, threshold))


def build_compressor_filters(effects, measured_i) -> list[str]:
    """Filtros del compresor, o [] si no procede / no hay nivel medido."""
    threshold = compressor_threshold_db(effects, measured_i)
    if threshold is None:
        if compressor_is_active(sanitize_effects(effects)):
            logger.info(
                "compresor pedido sin nivel medido: se omite (un umbral "
                "absoluto a ciegas no comprimiría nada)"
            )
        return []
    cfg = sanitize_effects(effects)["compressor"]
    options = [
        f"threshold={threshold:.2f}dB",
        f"ratio={cfg['ratio']:.2f}",
        f"attack={cfg['attack_ms']:.1f}",
        f"release={cfg['release_ms']:.1f}",
    ]
    return ["acompressor=" + ":".join(options)]


def limiter_ceiling_db(effects) -> float | None:
    """Techo efectivo del limitador, o None si está desactivado.

    Descuenta ``ENCODER_PEAK_MARGIN_DB`` porque el limitador va antes del
    codificador: sin ese margen el archivo entregado supera el techo pedido,
    que es justo lo que hace hoy el boost (-0.20 dBTP con objetivo -1.5).
    """
    cfg = sanitize_effects(effects)
    if not limiter_is_active(cfg):
        return None
    ceiling = float(cfg["limiter"]["ceiling_dbtp"]) - ENCODER_PEAK_MARGIN_DB
    return max(-6.0, min(-0.1, ceiling))


def build_limiter_filters(effects) -> list[str]:
    """Filtro del limitador de pico, o [] si está desactivado.

    ``level=disabled`` es imprescindible: por defecto ``alimiter`` re-nivela la
    salida automáticamente y eso rompía el objetivo de loudness (medido:
    -16.00 LUFS con level=disabled frente a -14.50 con el auto-level activo).
    """
    ceiling = limiter_ceiling_db(effects)
    if ceiling is None:
        return []
    return [f"alimiter=limit={ceiling:.2f}dB:level=disabled"]


def build_effects_chain(effects, measured_i=None,
                        target_lufs=None) -> list[str]:
    """Cadena completa del boost con efectos, en el orden correcto.

    Devuelve [] cuando el desplegable no cambia nada: en ese caso el boost se
    hace como siempre (loudnorm lineal) y el llamador no debe usar esto.

    ``measured_i`` es el loudness integrado medido de la fuente. Si falta, el
    compresor se omite (no habría umbral con sentido) pero el resto se aplica.
    """
    cfg = sanitize_effects(effects)
    if not effects_are_active(cfg):
        return []
    if target_lufs is not None:
        cfg["target_lufs"] = _clamp_number(target_lufs, -40.0, -5.0,
                                           cfg["target_lufs"])

    chain: list[str] = []
    chain.extend(build_eq_filters(cfg))

    gain = level_gain_db(cfg, measured_i)
    if gain is not None:
        chain.append(f"volume={gain:.2f}dB")
        chain.extend(build_compressor_filters(cfg, measured_i))
        # Normalización final: devuelve el nivel que el compresor se ha
        # llevado, sin deshacer la compresión (medido: conserva el rango
        # reducido). Solo hace falta cuando el nivel se puso con una ganancia
        # pura; si abajo se normaliza directamente, sería duplicar el filtro.
        chain.append(f"loudnorm=I={cfg['target_lufs']:.1f}:TP=-3.0:LRA=11")
    else:
        # Sin medición no se puede colocar el nivel ni el umbral del
        # compresor, así que se normaliza ya de una vez (un único loudnorm,
        # no dos: dos normalizadores encadenados aplicarían la ganancia dos
        # veces).
        chain.append(f"loudnorm=I={cfg['target_lufs']:.1f}:TP=-1.5:LRA=11")

    chain.extend(build_limiter_filters(cfg))
    return chain
