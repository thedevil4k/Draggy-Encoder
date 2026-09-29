"""Veredicto de compatibilidad: ¿le sirve esta build de FFmpeg a este equipo?

Une dos datos que viven en otro sitio y no se mezclan en ninguno:

* ``src.ffmpeg.builds``: qué builds hay y qué driver mínimo exige cada una.
* ``src.hardware.codecs``: qué sabe hacer la GPU de este equipo.

Por build devuelve un veredicto plano —``{id, label, min_driver,
min_driver_note, status, reason, codecs}``— que la UI pinta tal cual, sin
tener que recalcular nada. ``installed`` NO viene aquí: que una build esté
instalada es un dato de la máquina, no del veredicto.

Los cuatro estados del semáforo:

* ``compatible``   verde  — evidencia a favor.
* ``incompatible`` rojo   — evidencia en contra (driver por debajo del mínimo).
* ``unverified``   ámbar  — no comprobable con los datos que tenemos.
* ``neutral``      gris   — la versión es irrelevante aquí (sin GPU dedicada).

Regla del semáforo: **verde sólo con evidencia**. Hacen falta los dos datos
duros (GPU reconocida en el mapa de códecs + driver legible por encima del
mínimo); si falta cualquiera de los dos, el veredicto baja a ámbar en vez de
fingir compatibilidad.
"""

from __future__ import annotations

import re

from src.ffmpeg.builds import (
    current_platform,
    get_build,
    list_builds,
    min_driver_for,
)

# Veredictos posibles.
COMPATIBLE = "compatible"        # verde  — evidencia a favor
INCOMPATIBLE = "incompatible"    # rojo   — evidencia en contra
UNVERIFIED = "unverified"        # ámbar  — no comprobable
NEUTRAL = "neutral"              # gris   — la versión es irrelevante aquí

# ---------------------------------------------------------------------------
# Versión del driver de la máquina
# ---------------------------------------------------------------------------

_WDDM_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)$")


def parse_driver_version(text):
    """Convierte la versión de driver NVIDIA a ``(major, minor)``.

    Acepta las dos formas que devuelve la detección de hardware:

    * ``nvidia-smi`` → ``"560.94"`` → ``(560, 94)``.
    * Windows/WMI  → ``"31.0.15.5222"`` (WDDM). NVIDIA codifica ahí la versión
      pública como ``(c - 10) * 100 + d / 100``: ``31.0.15.5222`` es el driver
      público 552.22.

    Devuelve ``None`` si el texto no es una versión reconocible (y sólo entonces
    el veredicto baja a "sin comprobar").
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    m = _WDDM_RE.match(raw)
    if m:
        first, _second, c, d = (int(g) for g in m.groups())
        if first < 100 and c >= 10:
            # p.ej. 31.0.15.5222 -> 552.22
            public = (c - 10) * 100 + d / 100.0
            major = int(public)
            return (major, int(round((public - major) * 100)))
        return (first, c)
    parts = re.findall(r"\d+", raw)
    if not parts:
        return None
    if len(parts) >= 2:
        return (int(parts[0]), int(parts[1]))
    return (int(parts[0]), 0)


def format_driver(version):
    """``(570, 0)`` → ``"570.0"`` (para los textos de la UI)."""
    if not version:
        return ""
    try:
        return f"{int(version[0])}.{int(version[1])}"
    except (TypeError, ValueError, IndexError):
        return str(version)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

def _gpus(hardware_info):
    return [g for g in ((hardware_info or {}).get("gpus") or []) if g]


def _gpus_by_vendor(hardware_info, vendor):
    return [g for g in _gpus(hardware_info)
            if str(g.get("vendor", "")).strip().lower() == vendor.lower()]


def _codec_labels(codecs):
    return [label for _cid, label in codecs]


def _primary_gpu(gpus, codec_fn):
    """Primera GPU de la lista a la que el mapa de códecs le reconoce encoders."""
    for gpu in gpus:
        if codec_fn(gpu.get("name", "")):
            return gpu
    return gpus[0] if gpus else None


# ---------------------------------------------------------------------------
# Veredicto
# ---------------------------------------------------------------------------

def _unverified(reason, codecs=None):
    return {"status": UNVERIFIED, "reason": reason, "codecs": list(codecs or [])}


def _amd_or_intel_verdict(hardware_info):
    """AMF y QuickSync vienen en todas las builds: la versión no decide."""
    from src.hardware.codecs import get_amd_codecs, get_intel_codecs

    if not isinstance(hardware_info, dict):
        # No lo sabemos, que no es lo mismo que "no hay GPU". Un neutro aquí
        # diría "elige la que prefieras" cuando en realidad no hay nada sobre
        # lo que elegir: se dice que falta el dato y cómo CONSEguirlo.
        return _unverified(
            "No se ha detectado el hardware de este equipo, así que no podemos "
            "decir si esta build te sirve: ni verde ni rojo, ni una "
            "recomendación. Pulsa 'Re-detect Hardware & Codecs' y vuelve a "
            "mirar el selector."
        )

    amd = _gpus_by_vendor(hardware_info, "AMD")
    if amd:
        codecs = _codec_labels(get_amd_codecs(_primary_gpu(amd, get_amd_codecs).get("name", "")))
        return _unverified(
            "AMF va incluido en todas las builds del catálogo: lo que decide "
            "es el driver de tu Radeon, no la versión de FFmpeg. Se confirma "
            "al probar códecs ('Re-detect Hardware & Codecs').",
            codecs,
        )
    intel = _gpus_by_vendor(hardware_info, "Intel")
    if intel:
        codecs = _codec_labels(
            get_intel_codecs(_primary_gpu(intel, get_intel_codecs).get("name", ""))
        )
        return _unverified(
            "QuickSync va incluido en todas las builds del catálogo: lo que "
            "decide es el driver de tu Intel, no la versión de FFmpeg. Se "
            "confirma al probar códecs ('Re-detect Hardware & Codecs').",
            codecs,
        )
    if _gpus(hardware_info):
        return _unverified(
            "No se ha identificado una GPU NVIDIA/AMD/Intel con encoder de "
            "vídeo conocido: la codificación irá por CPU (x264/x265) en "
            "cualquier build."
        )
    return {
        "status": NEUTRAL,
        "reason": "No hay GPU dedicada detectada: todas las builds codifican "
                  "igual en CPU. Elige la que prefieras.",
        "codecs": [],
    }


def _nvidia_verdict(build, hardware_info, system):
    """Veredicto con la GPU NVIDIA: aquí sí depende de la build."""
    from src.hardware.codecs import (
        get_nvidia_codecs,
        is_legacy_nvidia,
        nvidia_has_no_encoder,
    )

    nvidia = _gpus_by_vendor(hardware_info, "NVIDIA")
    gpu = _primary_gpu(nvidia, get_nvidia_codecs)
    name = gpu.get("name", "")
    codecs = get_nvidia_codecs(name)
    if not codecs:
        # Dos "no" muy distintos. Si NVIDIA publica esta GPU con cero motores
        # NVENC, no es que el programa no la conozca: es que no tiene encoder,
        # y decirlo claro es más útil que un "no lo sé" genérico.
        if nvidia_has_no_encoder(name):
            return _unverified(
                f"La matriz oficial de NVIDIA publica tu {name} sin ningún "
                f"motor NVENC (0 en la columna 'Total # of NVENC'), así que "
                f"aquí no hay codificación por GPU: da igual qué build elijas, "
                f"el vídeo se codificará en CPU (x264/x265). El botón "
                f"'Re-detect Hardware & Codecs' prueba FFmpeg en tu equipo y "
                f"es el que puede contradecir a NVIDIA si tu unidad concreta "
                f"difiere."
            )
        return _unverified(
            f"'{name}' no aparece en el mapa de GPUs con NVENC del programa, así "
            f"que no podemos afirmar nada sobre esta build. Instálala y comprueba "
            f"los códecs con 'Re-detect Hardware & Codecs'."
        )

    codec_names = _codec_labels(codecs)
    driver_raw = gpu.get("driver", "")
    driver = parse_driver_version(driver_raw)
    required = min_driver_for(build, system)

    if required is None:
        return _unverified(
            f"No hay requisito de driver documentado para esta build; "
            f"'{name}' ({', '.join(codec_names)}) sólo se puede confirmar "
            f"probando códecs.",
            codec_names,
        )
    if driver is None:
        return _unverified(
            f"No se pudo leer la versión del driver NVIDIA"
            f"{f' ({driver_raw})' if driver_raw else ''}, así que no se puede "
            f"comparar con el mínimo {format_driver(required)} que exige esta "
            f"build para usar NVENC.",
            codec_names,
        )

    legacy = ""
    if is_legacy_nvidia(name):
        legacy = (
            " Ojo: el soporte de Kepler/GTX 600-700 terminó con la rama de "
            "drivers R470, así que una actualización de driver puede haberle "
            "quitado NVENC."
        )

    if driver < required:
        return {
            "status": INCOMPATIBLE,
            "reason": (
                f"Esta build exige un driver NVIDIA {format_driver(required)} o "
                f"nuevo para {', '.join(codec_names)} en tu {name}, y el instalado "
                f"es {format_driver(driver)}: FFmpeg arrancaría pero sin "
                f"codificar por GPU.{legacy}"
            ),
            "codecs": codec_names,
        }

    return {
        "status": COMPATIBLE,
        "reason": (
            f"Tu {name} es compatible: el driver {format_driver(driver)} supera "
            f"el mínimo {format_driver(required)} que esta build exige para "
            f"{', '.join(codec_names)}.{legacy}"
        ),
        "codecs": codec_names,
    }


def evaluate_build(build, hardware_info, platform_name=None):
    """Veredicto de UNA build contra el hardware detectado.

    Devuelve ``{"status", "reason", "codecs"}`` donde ``status`` es uno de
    ``compatible`` / ``incompatible`` / ``unverified`` / ``neutral``.
    """
    build = get_build(build.get("id") if isinstance(build, dict) else build)
    system = platform_name or current_platform()
    if not _gpus_by_vendor(hardware_info, "NVIDIA"):
        return _amd_or_intel_verdict(hardware_info)
    return _nvidia_verdict(build, hardware_info, system)


def evaluate_build_list(hardware_info, platform_name=None):
    """La plataforma y la lista de builds que le pertenece, en un solo valor.

    Esto es lo que viaja del diálogo al selector, y por eso la plataforma
    va DENTRO y no se busca al otro lado: el texto que el usuario lee tiene
    que poder nombrar la plataforma cuya lista tiene delante. Si el selector
    la elbowara por su cuenta, podría decir "para Windows" sobre una lista
    de Darwin —que es lo que pasaba— y el texto y los datos acabarían
    hablando de dos sistemas distintos.

    Resolver la plataforma aquí, una sola vez, y devolver las dos cosas
    juntas es lo que hace imposible esa divergencia: el que compone el texto
    no puede elegir otra plataforma porque no tiene de dónde sacarla.
    """
    system = platform_name or current_platform()
    return {
        "platform": system,
        "entries": evaluate_all(hardware_info, system),
        "evaluation_failed": False,
    }


def failed_build_list(platform_name=None):
    """El mismo carrier cuando la evaluación ni siquiera pudo intentarse.

    Se distingue de una lista vacía a propósito: vacía es "no hay builds
    para esta plataforma" —un hecho del catálogo— y esto es "no se ha
    podido comprobar", que son dos cosas distintas y el texto del usuario
    cambia con ellas.
    """
    return {
        "platform": platform_name or current_platform(),
        "entries": [],
        "evaluation_failed": True,
    }


def evaluate_all(hardware_info, platform_name=None):
    """Veredicto de todas las builds del catálogo, listas para la UI.

    Cada entrada es un dict plano con los datos de la build (``id``, ``label``,
    ``min_driver``, ``min_driver_note``) más su veredicto (``status``,
    ``reason``, ``codecs``): es todo lo que la fila necesita para pintarse.

    A las que quedan en rojo se les añade qué versiones sí valdrían para ese
    equipo (o que no hay ninguna publicada) — sin eso el rojo no es accionable.
    """
    system = platform_name or current_platform()
    entries = []
    for build in list_builds(system):
        entry = {
            "id": build["id"],
            "label": build["label"],
            "min_driver": format_driver(min_driver_for(build, system)),
            "min_driver_note": build.get("min_driver_note", ""),
        }
        entry.update(evaluate_build(build, hardware_info, system))
        entries.append(entry)

    workable = [e["label"] for e in entries if e["status"] == COMPATIBLE]
    for entry in entries:
        if entry["status"] != INCOMPATIBLE:
            continue
        if workable:
            entry["reason"] += f" Con tu driver sí valen: {', '.join(workable)}."
        else:
            entry["reason"] += (
                " Ninguna de las versiones publicadas hoy cubre un driver tan "
                "antiguo: actualiza el driver de la GPU o usa los códecs de CPU."
            )
    return entries
