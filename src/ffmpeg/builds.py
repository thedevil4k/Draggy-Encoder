"""Catálogo de builds de FFmpeg: cuáles existen, de dónde salen y cuál se instala.

Es la raíz de la feature: no importa nada del resto de ella, y todo lo demás
importa de aquí. Las capas van en este sentido, sin vueltas atrás:

    config · preferences          de dónde salen los mirrors y la elección
        ↓
    builds (este módulo)          catálogo + elección + cómo se llama
        ↓
    build_fit · downloader · upstream
        ↓
    manager                       orquesta instalar y deja constancia
        ↓
    ui/ffmpeg_build_picker · ui/dependencies_dialog · ui/main_window

Quién es dueño de qué:

* ``BUILDS`` — los datos puros de cada build: etiqueta, driver NVIDIA mínimo
  por plataforma, nota de por qué ese mínimo, y mirrors por plataforma.
* La preferencia ``ffmpeg_build_id`` — la elección del usuario. Único dueño:
  todo el flujo la lee, nadie la pasa como parámetro, y aquí se lee por una
  sola función (``stored_build_id``).
* ``unusable_reason`` — la política de qué es una elección válida, escrita una
  sola vez. ``resolve_build_id``, ``set_preferred_build_id`` y ``build_label``
  la leen en vez de decidir cada uno por su cuenta.
* ``build_label`` / ``chosen_build_label`` — el texto que el usuario lee sobre
  una build. Vive aquí a propósito: es la presentación *de esta elección*, y
  ningún otro módulo compone texto de build. El semáforo de cada fila es otra
  cosa distinta y lo dice ``build_fit``.
* ``parse_timestamp`` / ``timestamp_age_hours`` / ``ensure_utc`` — cómo se lee
  una marca de tiempo de las que guarda la app. Hay dos reglas y cada una con
  un dueño: ``parse_timestamp`` es el único parseo de ISO y el único sitio que
  sabe qué es una ``Z`` al final; ``ensure_utc`` es el único que decide que una
  fecha sin zona se lee como UTC, y ``upstream`` lo llama en sus dos puntos de
  entrada en vez de repetir el ``if``. La edad se calcula sobre la marca ya
  leída, en vez de al lado. No es catálogo de builds, pero es el módulo que las
  dos capas que necesitan leer una fecha ya importan: la UI y el upstream.
  Copiar el parseo en las dos era justo la forma de que una ``Z`` significara
  una cosa en el banner y otra en el cache.

  Lo que NO se comparte, y conviene no contar como si sí: el parseo de las
  cabeceras RFC 1123, que es otro formato y lo hace ``upstream`` con
  ``parsedate_to_datetime``. Comparten la decisión de qué significa "sin zona",
  no la lectura.

Lo que NO está aquí: el veredicto de si una build le sirve a un equipo concreto
(``build_fit``, que sí mira hardware) ni nada de Qt.

Una excepción que conviene no confundir: ``manager`` escribe además un
``build_id`` en el manifest, y es el de **lo instalado**, no el de la elección.
Se leen parecido pero no son el mismo dato —la elección se sanea, el manifest
registra lo que hay— y por eso el manifest lleva el nombre plano, sin avisos.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import FFMPEG_DOWNLOAD_URLS

# Clave de la preferencia con la build elegida por el usuario.
BUILD_PREF = "ffmpeg_build_id"

# Build por defecto: exactamente la que instala el programa hoy.
DEFAULT_BUILD_ID = "latest"

_BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
_JVS_OLD = "https://johnvansickle.com/ffmpeg/old-releases"
_GYAN = "https://www.gyan.dev/ffmpeg/builds/packages"

# Instantánea mensual de BtbN. Desde octubre de 2024 el release rodante
# ``latest`` dejó de publicar assets de las ramas 7.1/6.1/5.1 (sólo quedan
# master, 9.0 y 8.1, reconstruidos cada día con el Video Codec SDK más
# reciente, y por tanto con el driver mínimo más alto). Esas tres ramas
# siguen publicadas como releases mensuales con assets de nombre CONGELADO
# (``ffmpeg-n7.1-16-g15035aaec0-win64-gpl-7.1.zip``): al no reconstruirse, el
# SDK con el que se compilaron no vuelve a subir y su driver mínimo se queda
# donde estaba. Son las únicas builds de Windows con NVENC y mínimo por debajo
# de 570 que publica hoy un upstream de primera mano, y se han descargado y
# probado de verdad (health check + codificación NVENC real) antes de
# ofrecerlas.
_BTBN_SNAPSHOT = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
    "autobuild-2024-10-31-12-59"
)
_SNAPSHOT_NOTE = (
    "Instantánea de BtbN de octubre de 2024: al no reconstruirse, su driver "
    "mínimo no vuelve a subir como en las builds nuevas."
)

# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
# ``min_driver`` = driver NVIDIA mínimo que exige la build para NVENC, leído de
# ``libavcodec/nvenc.c`` de la rama correspondiente. Se toma SIEMPRE la entrada
# más alta de la tabla de la rama, que es lo que imprimirá el ejecutable: en las
# builds rodantes de BtbN ese valor sube cada día (el SDK del que se compilan
# también); en los binarios congelados —johnvansickle 2021-2023 y la instantánea
# de BtbN de octubre de 2024— se queda donde estaba al compilarlos.
#
# La app **no** inventa este dato: si una build no lo declara, el veredicto baja
# a "sin comprobar" en vez de fingir compatibilidad.
#
# ``sources`` vacío significa "la de siempre": sus mirrors salen de
# ``FFMPEG_DOWNLOAD_URLS`` sin tocar el catálogo.
BUILDS = (
    {
        "id": "latest",
        "label": "Última build (master)",
        "min_driver": {"Windows": (570, 0), "Linux": (570, 0)},
        "min_driver_note": "Compilada con el Video Codec SDK 13.x (570+; 610 con SDK 13.1+).",
        "sources": {},  # se resuelve desde FFMPEG_DOWNLOAD_URLS
    },
    {
        "id": "n9.0",
        "label": "FFmpeg 9.0",
        "min_driver": {"Windows": (570, 0), "Linux": (570, 0)},
        "min_driver_note": "Reconstruida con el SDK 13.x: mismo mínimo que master.",
        "sources": {
            "Windows": [
                (f"{_BTBN}/ffmpeg-n9.0-latest-win64-gpl-9.0.zip", "zip"),
                (f"{_GYAN}/ffmpeg-9.0.2-essentials_build.zip", "zip"),
            ],
            "Linux": [
                (f"{_BTBN}/ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz", "tar.xz"),
            ],
        },
    },
    {
        "id": "n8.1",
        "label": "FFmpeg 8.1",
        "min_driver": {"Windows": (570, 0), "Linux": (570, 0)},
        "min_driver_note": "Reconstruida con el SDK 13.x: mismo mínimo que master.",
        "sources": {
            "Windows": [
                (f"{_BTBN}/ffmpeg-n8.1-latest-win64-gpl-8.1.zip", "zip"),
                (f"{_GYAN}/ffmpeg-8.1.2-essentials_build.zip", "zip"),
            ],
            "Linux": [
                (f"{_BTBN}/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz", "tar.xz"),
            ],
        },
    },
    {
        # Sólo Windows: es donde se han descargado y probado estos binarios.
        "id": "n7.1",
        "label": "FFmpeg 7.1 (BtbN oct. 2024)",
        "min_driver": {"Windows": (551, 76)},
        "min_driver_note": _SNAPSHOT_NOTE,
        "sources": {
            "Windows": [
                (f"{_BTBN_SNAPSHOT}/ffmpeg-n7.1-16-g15035aaec0-win64-gpl-7.1.zip",
                 "zip"),
            ],
        },
    },
    {
        "id": "n6.1",
        "label": "FFmpeg 6.1 (BtbN oct. 2024)",
        "min_driver": {"Windows": (522, 25)},
        "min_driver_note": _SNAPSHOT_NOTE,
        "sources": {
            "Windows": [
                (f"{_BTBN_SNAPSHOT}/ffmpeg-n6.1.2-11-g7d79d0a43b-win64-gpl-6.1.zip",
                 "zip"),
            ],
        },
    },
    {
        "id": "n5.1.6",
        "label": "FFmpeg 5.1.6 (BtbN oct. 2024)",
        "min_driver": {"Windows": (471, 41)},
        "min_driver_note": _SNAPSHOT_NOTE,
        "sources": {
            "Windows": [
                (f"{_BTBN_SNAPSHOT}/ffmpeg-n5.1.6-9-gdcdfd7fb62-win64-gpl-5.1.zip",
                 "zip"),
            ],
        },
    },
    {
        "id": "n6.0",
        "label": "FFmpeg 6.0",
        "min_driver": {"Windows": (522, 25), "Linux": (520, 56)},
        "min_driver_note": "Compilado en 2023 (SDK 12.x): no se reconstruye, el mínimo no sube.",
        "sources": {
            "Linux": [
                (f"{_JVS_OLD}/ffmpeg-6.0.1-amd64-static.tar.xz", "tar.xz"),
            ],
        },
    },
    {
        "id": "n5.1",
        "label": "FFmpeg 5.1",
        "min_driver": {"Windows": (471, 41), "Linux": (470, 57)},
        "min_driver_note": "Compilado en 2022 (SDK 11.x): admite drivers bastante viejos.",
        "sources": {
            "Linux": [
                (f"{_JVS_OLD}/ffmpeg-5.1.1-amd64-static.tar.xz", "tar.xz"),
            ],
        },
    },
    {
        "id": "n4.4",
        "label": "FFmpeg 4.4",
        "min_driver": {"Windows": (456, 71), "Linux": (455, 28)},
        "min_driver_note": "Compilado en 2021 (SDK 11.0): el mínimo más bajo del catálogo.",
        "sources": {
            "Linux": [
                (f"{_JVS_OLD}/ffmpeg-4.4.1-amd64-static.tar.xz", "tar.xz"),
            ],
        },
    },
)

_BY_ID = {entry["id"]: entry for entry in BUILDS}


def current_platform():
    """``platform.system()`` del equipo, o ``""`` si no se puede saber."""
    try:
        import platform
        return platform.system()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Las marcas de tiempo que guarda la app
# ---------------------------------------------------------------------------

def ensure_utc(dt):
    """Un ``datetime`` sin zona se lee como UTC. Con zona, no se toca.

    Es la regla repetida tres veces con el mismo ``if`` —aquí, en
    ``upstream._parse_http_date`` y en ``upstream._iso``—; las tres la llaman
    ahora. Lo que se comparte no es la lectura, que cada una hace en su
    formato y con su herramienta. Es la decisión de después: qué significa un
    ``datetime`` sin zona. Esa era la que se duplicaba.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_timestamp(iso_text):
    """Una marca ISO de las que guarda la app, como ``datetime`` con zona.

    Es el único parseo de ISO del árbol y el único sitio que decide qué hacer
    con una ``Z`` al final. Que lo diga uno solo no es elegancia: es una regla
    de la que pueden discrepar las dos caras, y si discrepan, dos marcas que
    deberían compararse de una manera se comparan al revés según quién las lea
    — el mismo texto entra, y cada capa sale con una respuesta distinta, sin
    que ninguna se entere de que la otra pensaba otra cosa.

    La ``Z`` se cambia a mano porque el proyecto admite Python 3.10, donde
    ``fromisoformat`` no la acepta, y porque en las versiones que sí la aceptan
    el resultado tiene que ser el mismo: el mismo texto da la misma marca en
    todo el rango. Lo que guarda esta app nunca trae ``Z`` —todo lo que
    escribe sale de ``.isoformat()``, que pone ``+00:00``—, así que la rama
    está para los textos que no escribió esta app: un cache de otra versión,
    un JSON restaurado o editado a mano. Sin ella, en 3.10 esos textos saldrían
    como ``None``, y ``None`` en la edad significa cache caducada, que no es lo
    que dice el texto.

    Devuelve ``None`` si no hay texto o si no se entiende. Aqui ``None`` solo
    significa "esta marca no se puede leer"; que se pinte "never", un guion o
    nada es decision de quien la muestra, no de aqui.

    Lo que no sea texto se trata igual que una fecha rota, y se comprueba
    antes de tocar la cadena: una marca guardada que llega como numero o como
    dict tiene que salir por aqui sin reventar, porque el que llama solo sabe
    prometer que pinte "never" si recibe ``None``, no que aguante una
    excepcion.

    Lo que ya viene con zona, o lo que ya es ``None``, lo decide ``ensure_utc``:
    esa regla es de las dos, no de este parseo.
    """
    if not isinstance(iso_text, str) or not iso_text:
        return None
    try:
        dt = datetime.fromisoformat(
            iso_text[:-1] + "+00:00" if iso_text.endswith("Z") else iso_text)
    except (TypeError, ValueError):
        return None
    return ensure_utc(dt)


def timestamp_age_hours(iso_text):
    """Cuantas horas lleva esa marca. ``inf`` si no hay marca utilizable.

    El centinela es ``inf`` y no ``None`` a proposito, y es la unica decision
    sobre "no hay fecha" que existe en el arbol: el consumidor que compara —el
    cache de 24 h del upstream— hace ``age < HORAS``, y con ``inf`` esa
    comparacion sale ``False`` sola, que es justo lo que quiere, porque una
    cache sin fecha utilizable esta caducada. Con ``None`` ese consumidor
    tendria que escribir el ``or`` a mano, y ahi es donde aparecen las dos
    versiones de la regla.

    La edad puede salir negativa si la marca esta en el futuro: un reloj
    atrasado, o una cache escrita por una maquina que iba rapida. Aqui no se
    recorta a cero, porque "cuanto" y "como se dice cuanto" son dos preguntas
    distintas y la segunda es de quien la pinta.
    """
    dt = parse_timestamp(iso_text)
    if dt is None:
        return float("inf")
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Consultas del catálogo
# ---------------------------------------------------------------------------

def get_build(build_id=None):
    """Devuelve la build ``build_id``; un id desconocido cae en la de siempre."""
    return _BY_ID.get(str(build_id or "").strip(), _BY_ID[DEFAULT_BUILD_ID])


def find_build(build_id=None):
    """La build ``build_id`` del catálogo, o ``None`` si no existe.

    ``get_build`` sustituye por la de siempre porque al instalar hace falta
    instalar *algo*; aquí no se sustituye nada, porque la pregunta es si lo que
    hay guardado es una build real del catálogo.
    """
    return _BY_ID.get(str(build_id or "").strip())


def chosen_build_label():
    """Cómo se le llama a la build elegida, y sólo a esa.

    Sin argumentos, a propósito: quien lo llama no elige qué identificador
    pasa. Si elige, elige el que le apetece y el aviso deja de poder dispararse
    —que es exactamente como se coló el silencio: el llamador pasaba el id ya
    saneado por ``get_preferred_build_id``, y un id saneado no puede contar que
    estaba guardado algo que no existe.

    El rótulo se compone aquí, y en ningún otro sitio, a partir de lo que hay
    REALMENTE guardado (``stored_build_id``). Con una preferencia válida, o sin
    preferencia, es su nombre. Con un id que no está en el catálogo —un
    ``preferences.json`` editado a mano, o de otra versión del programa— es el
    nombre de la build que se acabaría instalando **avisando de que no es la que
    estaba guardada**, en vez de taparlo.

    Que avise es sólo presentación: para INSTALAR, una preferencia corrupta sigue
    cayendo en la de siempre (ver ``get_preferred_build_id``), que es lo correcto.
    """
    return build_label(stored_build_id())


def build_label(raw_build_id):
    """El rótulo de una build cuyo identificador puede no servir.

    Es el único sitio donde se compone ese texto, para que el banner, el botón
    de descarga y el diálogo de reinstalar no puedan divergir entre sí. A
    diferencia de ``chosen_build_label``, aquí sí se dice qué identificador es,
    porque quien lo llama lo tiene delante y es su dueño (el manifest, por
    ejemplo).

    Cubre las dos formas en que lo guardado no es lo que se instalaría: que no
    exista en el catálogo, y que exista pero no se pueda instalar aquí (build
    sin mirror para esta plataforma). En ambos se anuncia la de siempre —que es
    la que se instalará de verdad— diciendo además por qué no es la guardada.
    """
    candidate = str(raw_build_id or "").strip()
    por_defecto = _BY_ID[DEFAULT_BUILD_ID].get("label") or DEFAULT_BUILD_ID
    motivo = unusable_reason(candidate)
    if motivo:
        return f"{por_defecto} (la versión guardada '{candidate}' {motivo})"
    if not candidate:
        return por_defecto
    build = find_build(candidate)
    return build.get("label") or build["id"]


# ---------------------------------------------------------------------------
# Versión: qué línea de FFmpeg es la que hay instalada
# ---------------------------------------------------------------------------
#
# major de libavutil -> línea de FFmpeg. Cada entrada está leída de la cabecera
# de esa rama (libavutil/version.h) al escribirla, no de memoria:
#
#   60 -> release/8.1: LIBAVUTIL_VERSION 60.26.103
#   61 -> release/9.0: 61.1.102   ·   master: 61.9.100
#
# El major basta porque es lo único que cambia de release mayor a release mayor:
# el minor sólo avanza dentro de la línea, así que una nightly de master se lee
# "9.x-dev" y no "9.1" — su minor sube con los commits y no es un número de
# versión que se pueda enseñar como tal.
#
# Sólo se consulta para NIGHTLIES (los tokens "N-..."), porque una build de
# release ya dice su propio número ("9.0.2") y no hay nada que deducir. Un
# major que no esté en la tabla NO se adivina: se devuelve el token tal cual,
# que es justo lo que se veía antes de que esto existiera.

_LINEA_POR_LIBAVUTIL = {60: "8.x", 61: "9.x"}


def descripcion_version(build_info):
    """La versión de lo INSTALADO, como la lee una persona.

    Tres casos, y ninguno pierde el token de compilación:

    * release ("9.0.2"): se devuelve tal cual, porque el binario ya dice su
      versión y no hay nada que interpretar.
    * nightly con major conocido ("N-126905-…" + libavutil 61): se devuelve
      "9.x-dev (master)", que es lo que el usuario quiere ver: si su FFmpeg
      es el 9 o el 10 o el 11.
    * nightly con major que nadie ha verificado, o sin major: se devuelve el
      token tal cual. Un major desconocido no se invierte, y la fila seguirá
      mostrando la compilación, que sí es información real.

    Devuelve ``None`` si no hay nada que describir; el que llama compone el
    texto final con su respaldo. Quién lee el binario es ``manager``, que
    devuelve el dict; qué significan esos números, este módulo.
    """
    if not isinstance(build_info, dict):
        return None
    version = str(build_info.get("version") or "").strip()
    if not version:
        return None
    if version[0].isdigit():
        return version
    linea = _LINEA_POR_LIBAVUTIL.get(build_info.get("libavutil_major"))
    return f"{linea}-dev (master)" if linea else version


def list_builds(platform_name=None):
    """Builds publicables en esta plataforma, la de siempre primero.

    Filtra con la MISMA política que decide si una elección es válida
    (``unusable_reason``), para que el catálogo y la política no se
    contradigan nunca: si una build no se puede descargar aquí, no se ofrece.

    En un sistema sin mirrors —hoy macOS y FreeBSD, porque
    ``FFMPEG_DOWNLOAD_URLS`` sólo publica Windows y Linux— la lista sale vacía
    a propósito. Antes la build de siempre estaba exenta del filtro y se
    ofrecía igual, con la descarga fallando después; ahora el selector explica
    el vacío en vez de ofrecer una opción que no se puede cumplir.
    """
    system = platform_name or current_platform()
    return [entry for entry in BUILDS
            if not unusable_reason(entry["id"], system)]


def is_installable(build_id, platform_name=None):
    """¿Se puede instalar esta build en esta plataforma?

    Un id puede existir en el catálogo y no tener mirror aquí (los binarios
    congelados de johnvansickle son sólo Linux). Instalar una build sin mirror
    deja al programa sin FFmpeg, así que ese id no vale como elección.
    """
    return bool(build_mirrors(build_id, platform_name))


def nothing_to_offer_reason(platform_name=None):
    """Por qué no hay NINGUNA build descargable aquí, o ``None`` si hay.

    Es la misma política de ``unusable_reason`` aplicada al catálogo entero:
    si ninguna build vale como elección, es que esta plataforma no tiene
    mirror para nada (hoy macOS y FreeBSD). Lo que se puede ofrecer —el
    selector— y lo que se anuncia —el banner, la confirmación de reinstalar—
    tienen que salir de la misma pregunta, o el programa dice la verdad en un
    sitio y miente en los otros dos: antes el selector se vaciaba y explicaba
    por qué, mientras el banner seguía freezerando una build que no se podía
    bajar.
    """
    system = platform_name or current_platform()
    if list_builds(system):
        return None
    plataforma = system or "esta plataforma"
    return (
        f"ninguna build del catálogo tiene mirror para {plataforma}, así que "
        f"este programa no puede descargar FFmpeg en tu sistema"
    )


def resolve_build_id(build_id=None, platform_name=None):
    """Id instalable en esta plataforma; si no lo es, la build de siempre.

    Un ``build_id`` desconocido (preferencia vieja, config traída de otra
    máquina) o sin mirror aquí cae en la build por defecto en lugar de romper
    la instalación.
    """
    candidate = str(build_id or "").strip()
    if not candidate or unusable_reason(candidate, platform_name):
        return DEFAULT_BUILD_ID
    return candidate


def unusable_reason(build_id, platform_name=None):
    """Por qué un ``build_id`` no vale como elección aquí, o ``None`` si vale.

    Ésta es la ÚNICA política del módulo sobre qué es una elección válida: que
    exista en el catálogo y que se pueda instalar en esta plataforma. Las tres
    cosas que dependen de ella la leen en vez de volver a decidir:

    * ``resolve_build_id`` — qué se instala (lo que no vale, cae en la de
      siempre, porque al instalar hay que instalar algo).
    * ``set_preferred_build_id`` — qué se guarda (no se guarda lo que no vale).
    * ``build_label`` — qué se le dice al usuario, con el motivo ya escrito.

    Devuelve la frase sin el sujeto ("no está en el catálogo"), que es lo que
    la presentación necesita para no tener que volver a distinguir los casos.
    """
    candidate = str(build_id or "").strip()
    if not candidate:
        return None
    if candidate not in _BY_ID:
        return "no está en el catálogo"
    if not is_installable(candidate, platform_name):
        plataforma = platform_name or current_platform() or "esta plataforma"
        return f"no se puede instalar en {plataforma}"
    return None


def min_driver_for(build, platform_name):
    """Driver NVIDIA mínimo declarado por ``build`` para esta plataforma."""
    if not build:
        return None
    return (build.get("min_driver") or {}).get(platform_name)


# ---------------------------------------------------------------------------
# Mirrors
# ---------------------------------------------------------------------------

def build_mirrors(build_id=None, platform_name=None):
    """Mirrors ``(url, archive_type)`` de una build, en orden de preferencia.

    Para la build por defecto devuelve exactamente lo que devuelve
    ``FFMPEG_DOWNLOAD_URLS`` (mismo orden, mismos mirrors): el flujo de
    descarga no cambia en absoluto.
    """
    build = get_build(build_id)
    system = platform_name or current_platform()
    if build["id"] == DEFAULT_BUILD_ID:
        info = FFMPEG_DOWNLOAD_URLS.get(system) or {}
        mirrors = []
        if info.get("url"):
            mirrors.append((info["url"], info.get("archive_type")))
        if info.get("backup_url"):
            mirrors.append((
                info["backup_url"],
                info.get("backup_archive_type", info.get("archive_type")),
            ))
        return mirrors
    return list((build.get("sources") or {}).get(system) or [])


def get_active_url(platform_name=None):
    """URL del upstream de la build elegida (la de siempre si no hay elección).

    Es la URL contra la que se comparan el manifest y la huella cacheada: al
    ser la de la build que se instalaría, un "hay actualización" nunca puede
    hablar de una build distinta de la que instalaría el botón.
    """
    system = platform_name or current_platform()
    mirrors = build_mirrors(get_preferred_build_id(), system)
    return mirrors[0][0] if mirrors else None


# ---------------------------------------------------------------------------
# Preferencia: la build elegida
# ---------------------------------------------------------------------------

def stored_build_id():
    """El identificador de build TAL Y COMO ESTÁ GUARDADO, sin sanearlo.

    Único dueño de esa lectura. ``get_preferred_build_id`` lo sanea (al instalar
    hay que instalar algo, y eso está bien), pero un id ya saneado no sirve
    para contar la verdad en la interfaz: si la preferencia guardada no existe,
    el rótulo tiene que poder decirlo, y para eso necesita el id original.
    """
    try:
        import src.preferences as preferences
        stored = preferences.get(BUILD_PREF, None)
    except Exception:
        stored = None
    return str(stored or "").strip()


def get_preferred_build_id():
    """Build elegida por el usuario; la de siempre si nunca eligió otra.

    También si lo que hay guardado ya no es instalable aquí (id desconocido o
    build sin mirror para esta plataforma): en ese caso se comporta como si
    nunca se hubiera elegido nada.

    Esto es lo que se INSTALA, y por eso sanea. Lo que se DICE es cosa de
    ``chosen_build_label``, que no sanea precisamente para poder avisar.
    """
    return resolve_build_id(stored_build_id())


def set_preferred_build_id(build_id):
    """Persiste la build elegida.

    No se guarda un id desconocido ni uno sin mirror para esta plataforma:
    la preferencia manda qué se descarga en el próximo arranque, y un id que
    aquí no se puede instalar dejaría al programa sin FFmpeg.
    """
    candidate = str(build_id or "").strip()
    if not candidate or unusable_reason(candidate):
        return False
    try:
        import src.preferences as preferences
        return bool(preferences.set(BUILD_PREF, candidate))
    except Exception:
        return False
