"""Selección de versión de FFmpeg: catálogo, veredicto de GPU, selector e instalación.

La pieza que más importa aquí es el semáforo: **verde sólo cuando hay
evidencia**. Un veredicto verde exige dos datos duros del equipo del usuario:
que la GPU esté identificada en el mapa de códecs de la app y que el driver
NVIDIA instalado supere el mínimo que exige esa rama de FFmpeg. Sin alguno de
los dos, el veredicto es "sin comprobar" (ámbar) — nunca verde.

Los tests siguen la estructura del código: ``src.ffmpeg.builds`` (qué hay y
dónde), ``src.ffmpeg.build_fit`` (si le sirve a este equipo) y
``src.ui.ffmpeg_build_picker`` (cómo se pinta). El selector se prueba como
widget, sin montar el panel entero.
"""

import os

import pytest

from src.config import FFMPEG_DOWNLOAD_URLS
from nvidia_matrix import all_names, DECISIONES, ESCRITURAS, expected_for, FILAS, SOURCE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_preferences(monkeypatch):
    """Preferencias en memoria: nunca tocar el preferences.json real."""
    import src.preferences as preferences

    store = {}

    monkeypatch.setattr(preferences, "get", lambda key, default=None: store.get(key, default))
    monkeypatch.setattr(preferences, "set", lambda key, value: store.__setitem__(key, value) or True)
    return store


@pytest.fixture()
def known_hardware():
    """Publica el hardware en su dueño (``src.hardware.detector``) y lo retira al
    terminar, para que ningún test dependa de la máquina donde corra la suite.

    Se usa así: ``known_hardware(_nvidia(driver="580.00"))``.
    """
    from src.hardware import detector

    detector.remember_hardware(None)
    yield detector.remember_hardware
    detector.remember_hardware(None)


def _nvidia(name="NVIDIA GeForce RTX 3060 Laptop GPU", driver="560.94"):
    return {"gpus": [{"name": name, "vendor": "NVIDIA", "driver": driver, "memory": "6 GB"}]}


def _statuses(entries):
    """``id -> status``. Acepta la lista pelada o el carrier completo.

    La mayoría de los tests de ``build_fit`` trabajan sobre la lista que
    devuelve ``evaluate_all``; los del selector pasan el carrier. Que el
    helper sepa quitarse la envoltura evita que cada test tenga que
    acordarse de si lo que tiene delante es una cosa u otra.
    """
    if isinstance(entries, dict):
        entries = entries["entries"]
    return {entry["id"]: entry["status"] for entry in entries}


# ==----------------------------------------------------------------------
# 1. Catálogo (src.ffmpeg.builds): espejos reales y comportamiento por defecto
# ==----------------------------------------------------------------------

def test_default_build_mirrors_are_exactly_the_configured_ones():
    """Sin elección el catálogo devuelve los mirrors de siempre: primary y
    backup de ``FFMPEG_DOWNLOAD_URLS``, en ese orden. El flujo de descarga no
    puede cambiar para quien no toque nada."""
    from src.ffmpeg import builds

    for system, info in FFMPEG_DOWNLOAD_URLS.items():
        mirrors = builds.build_mirrors(builds.DEFAULT_BUILD_ID, system)
        assert mirrors[0] == (info["url"], info["archive_type"])
        if info.get("backup_url"):
            assert [m[0] for m in mirrors[1:]] == [info["backup_url"]]
        else:
            assert len(mirrors) == 1


def test_unknown_build_id_falls_back_to_the_default():
    from src.ffmpeg import builds

    assert builds.get_build("no-existe")["id"] == builds.DEFAULT_BUILD_ID
    assert builds.get_build(None)["id"] == builds.DEFAULT_BUILD_ID
    assert builds.get_build("")["id"] == builds.DEFAULT_BUILD_ID


def test_every_offered_build_has_a_real_mirror_on_its_platform():
    """La lista que se ofrece al usuario sólo contiene builds descargables en
    esa plataforma, y siempre por HTTPS en un formato que el extractor abre."""
    from src.ffmpeg import builds

    for system in ("Windows", "Linux"):
        offered = builds.list_builds(system)
        assert offered, system
        for build in offered:
            mirrors = builds.build_mirrors(build["id"], system)
            assert mirrors, f"{build['id']} sin mirror en {system}"
            for url, archive_type in mirrors:
                assert url.startswith("https://")
                assert archive_type in ("zip", "tar.xz")
                assert url.endswith("." + archive_type)


def test_linux_catalog_offers_the_older_frozen_builds():
    """En Linux hay escalera de versiones: los binarios congelados de
    johnvansickle son los que aceptan drivers viejos."""
    from src.ffmpeg import builds

    linux_ids = [b["id"] for b in builds.list_builds("Linux")]
    for older in ("n6.0", "n5.1", "n4.4"):
        assert older in linux_ids
    # Esos tres no existen para Windows: no se deben ofrecer.
    windows_ids = [b["id"] for b in builds.list_builds("Windows")]
    for older in ("n6.0", "n5.1", "n4.4"):
        assert older not in windows_ids


def test_windows_catalog_offers_frozen_builds_that_beat_the_rolling_ones():
    """El caso que motiva el catálogo: en Windows también tiene que haber una
    build con NVENC por debajo del driver 570 que exigen las rodantes.

    BtbN dejó de publicar las ramas 7.1/6.1/5.1 en el release rodante ``latest``
    (sólo master/9.0/8.1, reconstruidos a diario), pero los dejó congelados en
    su release mensual de octubre de 2024. Sin esas tres, un portátil con
    driver de 2019-2021 se quedaba con tres opciones en rojo.
    """
    from src.ffmpeg import builds

    rolling = [b for b in builds.list_builds("Windows")
               if b["id"] in ("latest", "n9.0", "n8.1")]
    older = [b for b in builds.list_builds("Windows")
             if b["id"] not in ("latest", "n9.0", "n8.1")]
    assert older, "Windows necesita builds más tolerantes que el release rodante"
    for build in older:
        # Mirror real, congelado (no 'latest'), y por debajo del 570 de las rodantes.
        url = builds.build_mirrors(build["id"], "Windows")[0][0]
        assert url.startswith("https://github.com/BtbN/FFmpeg-Builds/releases/")
        assert "-latest-" not in url
        assert builds.min_driver_for(build, "Windows") < (570, 0)
    for build in rolling:
        assert builds.min_driver_for(build, "Windows") == (570, 0)
    # Y el mínimo no puede crecer bajando de versión (el semáforo no puede mentir).
    ladder = ["n5.1.6", "n6.1", "n7.1", "n8.1"]
    mins = [builds.min_driver_for(builds.get_build(b), "Windows") for b in ladder]
    assert mins == sorted(mins)
    # Sin espejo verificado para Linux no se ofrecen allí.
    assert "n5.1.6" not in {b["id"] for b in builds.list_builds("Linux")}


def test_newer_builds_never_require_an_older_driver():
    """Coherencia de la tabla de mínimos: una build más nueva no puede exigir
    un driver más viejo que la anterior (si lo hiciera, el semáforo mintiera)."""
    from src.ffmpeg import builds

    def mins(build_id):
        build = builds.get_build(build_id)
        return {
            system: builds.min_driver_for(build, system)
            for system in ("Windows", "Linux")
        }

    ladder = ["n4.4", "n5.1", "n6.0", "latest"]
    for older, newer in zip(ladder, ladder[1:]):
        for system in ("Windows", "Linux"):
            assert mins(newer)[system] >= mins(older)[system], (newer, older, system)


# ==----------------------------------------------------------------------
# 2. Preferencia: la build elegida (único dueño de esa decisión)
# ==----------------------------------------------------------------------

def test_preference_defaults_to_the_current_build(fake_preferences):
    from src.ffmpeg import builds

    assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID


def test_preference_roundtrip(fake_preferences):
    from src.ffmpeg import builds

    other = builds.list_builds()[-1]["id"]
    assert builds.set_preferred_build_id(other) is True
    assert fake_preferences[builds.BUILD_PREF] == other
    assert builds.get_preferred_build_id() == other
    # Es la misma build que instalarían el instalador y el check de upstream.
    assert builds.build_mirrors(builds.get_preferred_build_id(), builds.current_platform())
    assert builds.get_active_url() == builds.build_mirrors(
        other, builds.current_platform())[0][0]


def test_unknown_preference_is_ignored(fake_preferences):
    """Una preferencia corrupta o de una versión que ya no existe no debe
    dejar al usuario sin FFmpeg: se cae a la build de siempre."""
    from src.ffmpeg import builds

    fake_preferences[builds.BUILD_PREF] = "n0.0-inexistente"
    assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID
    assert builds.set_preferred_build_id("n0.0-inexistente") is False


def _not_installable_here():
    """Un id del catálogo que no tiene mirror en esta plataforma."""
    from src.ffmpeg import builds

    return next(
        (b["id"] for b in builds.BUILDS
         if b["id"] != builds.DEFAULT_BUILD_ID
         and not builds.build_mirrors(b["id"])),
        None,
    )


def test_preference_that_is_not_installable_here_falls_back(fake_preferences):
    """Una build sin mirror para esta plataforma no puede ser la preferencia.

    Sin esta comprobación, una preferences.json traída de otro equipo (o
    escrita a mano) con un id de Linux en Windows llevaba a
    ``download_ffmpeg`` a un RuntimeError, ``ensure_ffmpeg`` a None y
    ``src/app.py`` a cerrar la app con un error genérico: sin FFmpeg y sin
    poder abrir el panel para arreglarlo.
    """
    from src.ffmpeg import builds

    not_here = _not_installable_here()
    assert not_here, "el catálogo necesita al menos una build de otra plataforma"

    fake_preferences[builds.BUILD_PREF] = not_here
    assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID
    # Y no se acepta como elección en esta plataforma (lo que había no se
    # pisa: la preferencia se resuelve al leerla).
    assert builds.set_preferred_build_id(not_here) is False
    assert fake_preferences[builds.BUILD_PREF] == not_here


def test_resolve_build_id_only_returns_installable_builds():
    from src.ffmpeg import builds

    for system in ("Windows", "Linux"):
        for build in builds.BUILDS:
            resolved = builds.resolve_build_id(build["id"], system)
            assert resolved == builds.DEFAULT_BUILD_ID or builds.build_mirrors(
                resolved, system
            ), (build["id"], system)
    assert builds.resolve_build_id(None) == builds.DEFAULT_BUILD_ID
    assert builds.resolve_build_id("") == builds.DEFAULT_BUILD_ID
    assert builds.resolve_build_id("basura") == builds.DEFAULT_BUILD_ID


# ==----------------------------------------------------------------------
# 3. Semáforo (src.ffmpeg.build_fit): verde con evidencia, ámbar sin ella
# ==----------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("560.94", (560, 94)),
    ("471.41", (471, 41)),
    ("522.25", (522, 25)),
    # Windows/WMI: 31.0.15.5222 es el driver público 552.22
    ("31.0.15.5222", (552, 22)),
    ("31.0.15.3623", (536, 23)),
    ("", None),
    (None, None),
    ("sin datos", None),
])
def test_parse_driver_version(raw, expected):
    from src.ffmpeg import build_fit

    assert build_fit.parse_driver_version(raw) == expected


def test_recent_driver_marks_every_build_green():
    from src.ffmpeg import build_fit

    entries = build_fit.evaluate_all(_nvidia(driver="580.00"), "Windows")
    statuses = _statuses(entries)
    assert statuses and set(statuses.values()) == {build_fit.COMPATIBLE}
    for entry in entries:
        assert entry["reason"]
        assert "RTX 3060" in entry["reason"]
        # La evidencia siempre menciona los códecs que la GPU puede hacer.
        assert "H.264 NVENC" in " ".join(entry["codecs"])


def test_old_driver_marks_newer_builds_red_and_points_at_the_ones_that_work():
    """El caso que motiva la función: GPU de portátil veterana con driver
    justo por debajo del mínimo de las builds nuevas."""
    from src.ffmpeg import build_fit, builds

    hardware = _nvidia(name="NVIDIA GeForce GTX 1050 Laptop GPU", driver="500.00")
    entries = build_fit.evaluate_all(hardware, "Linux")
    by_id = {e["id"]: e for e in entries}
    red = by_id["latest"]
    assert red["status"] == build_fit.INCOMPATIBLE
    assert "GTX 1050" in red["reason"]
    # El rojo es accionable: dice qué versiones sí valdrían.
    assert "FFmpeg 5.1" in red["reason"] and "FFmpeg 4.4" in red["reason"]
    assert by_id["n4.4"]["status"] == build_fit.COMPATIBLE
    assert builds.min_driver_for(builds.get_build("n4.4"), "Linux") < (471, 0)


def test_old_windows_driver_gets_a_green_option_and_the_reds_point_at_it():
    """El caso de la auditoría: en Windows, un portátil con driver de 2019
    tenía tres opciones y las tres en rojo. Con la instantánea congelada de
    BtbN hay una verde, y los rojos dicen cuál es."""
    from src.ffmpeg import build_fit

    hardware = _nvidia(name="NVIDIA GeForce GTX 1050 Laptop GPU", driver="500.00")
    entries = {e["id"]: e for e in build_fit.evaluate_all(hardware, "Windows")}

    green = [e for e in entries.values() if e["status"] == build_fit.COMPATIBLE]
    assert [e["id"] for e in green] == ["n5.1.6"]
    for entry in entries.values():
        if entry["status"] == build_fit.INCOMPATIBLE:
            assert "FFmpeg 5.1.6" in entry["reason"]


def test_red_says_when_no_published_build_covers_the_driver():
    """Si nada encaja, el rojo lo dice: no hay build publicada que valga con
    ese driver (ni una lista de alternativas que no existen)."""
    from src.ffmpeg import build_fit

    # 460.00 está por debajo del mínimo más bajo que publica anybody: aquí ya
    # no hay nada que ofrecer y el rojo tiene que decirlo.
    hardware = _nvidia(name="NVIDIA GeForce GTX 1050 Laptop GPU", driver="460.00")
    entries = build_fit.evaluate_all(hardware, "Windows")
    assert {e["status"] for e in entries} == {build_fit.INCOMPATIBLE}
    for entry in entries:
        assert "Ninguna de las versiones publicadas" in entry["reason"]
        assert "actualiza el driver" in entry["reason"]


def test_kepler_gets_the_driver_support_end_warning():
    from src.ffmpeg import build_fit

    hardware = _nvidia(name="NVIDIA GeForce GTX 750 Ti", driver="460.98")
    entries = {e["id"]: e for e in build_fit.evaluate_all(hardware, "Linux")}
    # 460.98 sólo llega al mínimo de 4.4 (455.28); el resto queda en rojo.
    assert entries["n4.4"]["status"] == build_fit.COMPATIBLE
    assert entries["latest"]["status"] == build_fit.INCOMPATIBLE
    assert "Kepler" in entries["latest"]["reason"]


def test_unknown_gpu_is_never_green():
    """Sin GPU reconocida en el mapa de códecs no hay evidencia: ámbar."""
    from src.ffmpeg import build_fit

    hardware = _nvidia(name="NVIDIA Some Unknown Board 1234", driver="610.00")
    entries = build_fit.evaluate_all(hardware, "Windows")
    assert {e["status"] for e in entries} == {build_fit.UNVERIFIED}
    for entry in entries:
        assert "no aparece en el mapa" in entry["reason"]


def test_unreadable_driver_is_never_green():
    from src.ffmpeg import build_fit

    entries = build_fit.evaluate_all(_nvidia(driver=""), "Windows")
    assert {e["status"] for e in entries} == {build_fit.UNVERIFIED}
    assert "No se pudo leer la versión del driver" in entries[0]["reason"]


def test_amd_and_intel_are_unverified_because_the_build_is_not_the_variable():
    """AMF y QuickSync vienen en todas las builds: la versión no decide."""
    from src.ffmpeg import build_fit

    amd = {"gpus": [{"name": "AMD Radeon RX 6600", "vendor": "AMD", "driver": "31.0.21918"}]}
    intel = {"gpus": [{"name": "Intel Arc A770", "vendor": "Intel", "driver": "31.0.101.5186"}]}
    for hardware, word in ((amd, "AMF"), (intel, "QuickSync")):
        entries = build_fit.evaluate_all(hardware, "Windows")
        assert {e["status"] for e in entries} == {build_fit.UNVERIFIED}
        assert word in entries[0]["reason"]


def test_without_a_dedicated_gpu_every_build_is_neutral():
    from src.ffmpeg import build_fit

    entries = build_fit.evaluate_all({"gpus": []}, "Windows")
    assert {e["status"] for e in entries} == {build_fit.NEUTRAL}
    assert "todas las builds codifican" in entries[0]["reason"]


def test_unknown_hardware_is_not_the_same_as_no_dedicated_gpu():
    """``None`` es "no lo sabemos"; ``{"gpus": []}`` es "no hay GPU".

    Con los dos fusionados, un equipo al que no se le pudo leer el hardware
    recibía un neutro que parecía una elección del usuario.
    """
    from src.ffmpeg import build_fit

    unknown = build_fit.evaluate_all(None, "Windows")
    assert {e["status"] for e in unknown} == {build_fit.UNVERIFIED}
    assert "No se ha detectado el hardware" in unknown[0]["reason"]
    # Y sin GPU dedicada la versión sigue siendo irrelevante, como siempre.
    assert {e["status"] for e in
            build_fit.evaluate_all({"gpus": []}, "Windows")} == {build_fit.NEUTRAL}


def test_entries_are_flat_and_carry_everything_the_row_needs():
    """La entrada es un dict plano: la fila lo pinta sin recalcular nada, y
    sin que se cuele en ella nada de la plataforma (``installed`` lo decide
    la fila, no el veredicto)."""
    from src.ffmpeg import build_fit

    entry = build_fit.evaluate_all(_nvidia(), "Windows")[0]
    assert set(entry) == {
        "id", "label", "min_driver", "min_driver_note", "status", "reason", "codecs",
    }
    assert entry["min_driver"] and entry["min_driver_note"]


# ==----------------------------------------------------------------------
# 4. Flujo de instalación: lee la elección, no la recibe
# ==----------------------------------------------------------------------

def test_download_ffmpeg_uses_the_chosen_build_mirrors(tmp_path, monkeypatch,
                                                       fake_preferences):
    """Sin parámetro de build: el instalador lee la preferencia, y si el mirror
    primario falla cae al siguiente igual que hace la build de siempre."""
    import zipfile
    import src.ffmpeg.downloader as dl
    from src.ffmpeg import builds

    chosen = next(
        b["id"] for b in builds.list_builds()
        if len(builds.build_mirrors(b["id"], dl.platform.system())) >= 2
    )
    assert builds.set_preferred_build_id(chosen) is True

    bin_dir = tmp_path / "ffmpeg"
    monkeypatch.setattr(dl, "get_ffmpeg_dir", lambda: str(bin_dir))
    monkeypatch.setattr(dl, "_probe_url_metadata", lambda url: (None, False))
    mirrors = builds.build_mirrors(chosen, dl.platform.system())
    assert len(mirrors) >= 2
    hits = []

    def fake_download_file(url, archive_path, **kwargs):
        hits.append(url)
        if url == mirrors[0][0]:
            raise RuntimeError("404 Client Error: Not Found")
        with zipfile.ZipFile(archive_path, "w") as z:
            z.writestr("ffmpeg/bin/" + dl.get_ffmpeg_executable_name(), b"x")
            z.writestr("ffmpeg/bin/" + dl.get_ffprobe_executable_name(), b"x")

    monkeypatch.setattr(dl, "_download_file", fake_download_file)

    dl.download_ffmpeg(progress_callback=None)

    assert hits == [m[0] for m in mirrors]
    assert os.path.exists(os.path.join(str(bin_dir), dl.get_ffmpeg_executable_name()))


def test_download_ffmpeg_falls_back_instead_of_raising(tmp_path, monkeypatch,
                                                      fake_preferences):
    """Una preferencia que aquí no se puede instalar no puede romper la
    instalación: se descarga la de siempre en vez de lanzar RuntimeError."""
    import zipfile
    import src.ffmpeg.downloader as dl
    from src.ffmpeg import builds

    fake_preferences[builds.BUILD_PREF] = _not_installable_here()

    bin_dir = tmp_path / "ffmpeg"
    monkeypatch.setattr(dl, "get_ffmpeg_dir", lambda: str(bin_dir))
    monkeypatch.setattr(dl, "_probe_url_metadata", lambda url: (None, False))
    hits = []

    def fake_download_file(url, archive_path, **kwargs):
        hits.append(url)
        with zipfile.ZipFile(archive_path, "w") as z:
            z.writestr("ffmpeg/bin/" + dl.get_ffmpeg_executable_name(), b"x")
            z.writestr("ffmpeg/bin/" + dl.get_ffprobe_executable_name(), b"x")

    monkeypatch.setattr(dl, "_download_file", fake_download_file)
    dl.download_ffmpeg(progress_callback=None)

    expected = builds.build_mirrors(builds.DEFAULT_BUILD_ID, dl.platform.system())
    assert hits and hits[0] == expected[0][0]


def _stub_install(tmp_path, monkeypatch, manager, version="9.0.2"):
    """Deja ``ensure_ffmpeg`` con todo listo menos la descarga, y devuelve el
    registro de los kwargs con los que se llamó."""
    seen = {}
    monkeypatch.setattr(manager, "_migrate_legacy_layout", lambda: False)
    monkeypatch.setattr(manager, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(manager, "write_manifest", lambda payload: True)
    monkeypatch.setattr(manager, "get_ffmpeg_version", lambda path=None: version)
    monkeypatch.setattr(manager, "get_local_ffmpeg_path",
                        lambda: str(tmp_path / "no-existe" / "ffmpeg"))
    monkeypatch.setattr(manager, "get_local_ffprobe_path",
                        lambda: str(tmp_path / "no-existe" / "ffprobe"))
    monkeypatch.setattr(
        manager, "download_ffmpeg",
        lambda **kwargs: seen.update(kwargs) or "descargado",
    )
    return seen


def test_ensure_ffmpeg_without_a_choice_installs_the_default(tmp_path, monkeypatch,
                                                             fake_preferences):
    """Sin preferencia, ``ensure_ffmpeg`` descarga exactamente la build de
    siempre (comportamiento histórico intacto)."""
    from src.ffmpeg import builds, manager

    seen = _stub_install(tmp_path, monkeypatch, manager)
    assert manager.ensure_ffmpeg() == "descargado"
    # Ni build_id ni nada: el instalador lee la preferencia, que es la de siempre.
    assert "build_id" not in seen
    assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID


def test_ensure_ffmpeg_honours_the_chosen_build(tmp_path, monkeypatch,
                                                fake_preferences):
    from src.ffmpeg import builds, manager

    other = builds.list_builds()[-1]["id"]
    assert builds.set_preferred_build_id(other) is True
    _stub_install(tmp_path, monkeypatch, manager, version="5.1.6")
    assert manager.ensure_ffmpeg() == "descargado"
    assert builds.get_preferred_build_id() == other


def test_installing_something_uninstallable_keeps_the_default(tmp_path, monkeypatch,
                                                             fake_preferences):
    """Un id que no se puede instalar aquí no puede romper la instalación."""
    from src.ffmpeg import builds, manager

    fake_preferences[builds.BUILD_PREF] = _not_installable_here()
    _stub_install(tmp_path, monkeypatch, manager)
    assert manager.ensure_ffmpeg() == "descargado"
    assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID


def test_install_manifest_records_the_build_and_its_own_url(monkeypatch,
                                                            fake_preferences):
    """El manifest guarda versión + URL de la build instalada: sin eso, un 8.1
    se compararía contra el upstream de master y daría 'update' siempre."""
    import platform as _platform
    from src.ffmpeg import builds, manager

    chosen = builds.list_builds()[-1]["id"]
    assert builds.set_preferred_build_id(chosen) is True
    monkeypatch.setattr(manager, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(manager, "get_ffmpeg_version", lambda path=None: "5.1.6")
    monkeypatch.setattr(manager, "get_fingerprint",
                        lambda **kwargs: {"url": "x", "last_modified_iso": None})

    payload = manager._build_install_manifest("ruta/falsa", "user-reinstall")
    assert payload["build_id"] == chosen
    assert payload["build_label"] == builds.get_build(chosen)["label"]
    assert payload["version"] == "5.1.6"
    assert payload["source_url"] == builds.build_mirrors(
        chosen, _platform.system())[0][0]


def test_install_manifest_of_the_default_is_unchanged(monkeypatch, fake_preferences):
    from src.ffmpeg import builds, manager

    monkeypatch.setattr(manager, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(manager, "get_ffmpeg_version", lambda path=None: "9.0.2")
    monkeypatch.setattr(manager, "get_fingerprint",
                        lambda **kwargs: {"url": FFMPEG_DOWNLOAD_URLS["Windows"]["url"],
                                          "last_modified_iso": "2026-01-01T00:00:00+00:00",
                                          "etag": "x"})

    payload = manager._build_install_manifest("ruta/falsa", "first-download")
    assert payload["build_id"] == builds.DEFAULT_BUILD_ID
    assert payload["source_url"] == FFMPEG_DOWNLOAD_URLS["Windows"]["url"]
    assert payload["archive_type"] == FFMPEG_DOWNLOAD_URLS["Windows"]["archive_type"]
    assert payload["upstream_etag"] == "x"


def test_install_info_reports_the_installed_build_and_ignores_a_foreign_fingerprint(
        monkeypatch, fake_preferences):
    """Si la huella cacheada es de otra build no se comparan: el panel muestra
    '—' en vez de un 'update available' inventado."""
    from src.ffmpeg import builds, manager

    chosen = builds.list_builds()[-1]["id"]
    monkeypatch.setattr(manager, "read_manifest",
                        lambda: {"build_id": chosen, "version": "5.1.6",
                                 "upstream_etag": "viejo"})
    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: "/no/existe")
    monkeypatch.setattr(manager, "read_upstream_cache",
                        lambda: {"url": "https://otra/cosas", "etag": "nuevo"})

    info = manager.get_install_info()
    # Lo instalado lo dice el manifest; lo elegido, la preferencia.
    assert info["build_id"] == chosen
    assert info["build_label"] == builds.get_build(chosen)["label"]
    assert info["preferred_build_id"] == builds.DEFAULT_BUILD_ID
    assert info["update_available"] is None
    assert "build_series" not in info


def test_reinstall_needs_no_build_id(monkeypatch, fake_preferences):
    """``dependencies.reinstall`` no tiene parámetro de versión: la build la
    decide el instalador, que para FFmpeg es la preferencia."""
    import src.dependencies as depmod
    from src.ffmpeg import builds

    chosen = builds.list_builds()[-1]["id"]
    assert builds.set_preferred_build_id(chosen) is True
    seen = {}
    monkeypatch.setattr(depmod, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setitem(
        depmod.REGISTRY[0], "reinstall_fn",
        lambda **kwargs: seen.update(kwargs),
    )
    monkeypatch.setitem(
        depmod.REGISTRY[1], "reinstall_fn",
        lambda **kwargs: seen.update(kwargs),
    )

    assert depmod.reinstall("ffmpeg", progress_callback=None) is True
    assert "build_id" not in seen
    assert builds.get_preferred_build_id() == chosen
    assert not any("accepts_build_id" in entry for entry in depmod.REGISTRY)


# ==----------------------------------------------------------------------
# 5. El selector (src.ui.ffmpeg_build_picker) como widget, sin el panel
# ==----------------------------------------------------------------------

def _options(hardware=None, platform_name="Windows"):
    """El carrier que el diálogo pasa al selector: lista + su plataforma.

    Los cinco tests del selector de aquí abajo lo usan tal cual, que es
    justo lo que se quiere: se prueba el widget con el mismo valor que le
    da la app, no con una lista desnuda a la que le falta la plataforma.
    """
    from src.ffmpeg import build_fit

    return build_fit.evaluate_build_list(
        hardware if hardware is not None else _nvidia(), platform_name)


def _entries(build_list):
    """Sólo la lista, para los tests que la inspeccionan entrada a entrada."""
    return build_list["entries"]


def test_picker_colours_every_option_with_its_status(qapp, fake_preferences):
    from PySide6.QtCore import Qt

    from src.ffmpeg import build_fit
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker, STATUS_STYLE

    options = _options()
    picker = FFmpegBuildPicker(options, installed_build_id="n8.1",
                               preferred_build_id="n8.1")
    try:
        assert picker.combo.count() == len(options["entries"])
        for index in range(picker.combo.count()):
            build_id = picker.combo.itemData(index, Qt.ItemDataRole.UserRole)
            status = {o["id"]: o["status"] for o in options["entries"]}[build_id]
            color = picker.combo.itemData(index, Qt.ItemDataRole.ForegroundRole)
            assert color.name() == STATUS_STYLE[status][1].lower()
            # La marca va en el texto del item.
            assert picker.combo.itemText(index).startswith(STATUS_STYLE[status][0])
        # La instalada se marca como tal y es la que se abre.
        installed = [i for i in range(picker.combo.count())
                     if "(instalada)" in picker.combo.itemText(i)]
        assert len(installed) == 1
        assert picker.combo.itemData(installed[0],
                                     Qt.ItemDataRole.UserRole) == "n8.1"
        assert picker.selected_build_id == "n8.1"
        assert picker.selected_build_status == _statuses(options)["n8.1"]
    finally:
        picker.deleteLater()


def test_picker_opens_on_an_unknown_preference_falling_back_to_the_installed_one(
        qapp, fake_preferences):
    from src.ffmpeg import builds
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker

    options = _options()
    picker = FFmpegBuildPicker(options, installed_build_id="n5.1.6",
                               preferred_build_id="no-existe")
    try:
        assert picker.selected_build_id == "n5.1.6"
    finally:
        picker.deleteLater()


def test_picker_note_explains_the_verdict_and_the_catalog_warning(qapp,
                                                                  fake_preferences):
    """La nota lleva el veredicto, el driver mínimo y por qué ese mínimo: sin
    esto, elegir 8.1 con un driver de 2019 parece una solución y no lo es."""
    from src.ffmpeg import build_fit
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker

    hardware = _nvidia(name="NVIDIA GeForce GTX 1050 Laptop GPU", driver="500.00")
    options = {o["id"]: o for o in _options(hardware)["entries"]}
    picker = FFmpegBuildPicker(_options(hardware), installed_build_id="latest",
                               preferred_build_id="latest")
    try:
        # Una rodante: el mínimo es el mismo que master, y se dice.
        picker.combo.setCurrentIndex(picker.combo.findData("n8.1"))
        assert options["n8.1"]["min_driver_note"] in picker.note_text()
        assert "500" in picker.note_text()
        # Una congelada: el mínimo es más bajo y utilizable con este driver.
        picker.combo.setCurrentIndex(picker.combo.findData("n5.1.6"))
        assert picker.selected_build_status == build_fit.COMPATIBLE
        assert options["n5.1.6"]["min_driver"] in picker.note_text()
        assert options["n5.1.6"]["min_driver_note"] in picker.note_text()
        # El tooltip del desplegable es la misma nota.
        assert picker.combo.toolTip() == picker.note_text()
    finally:
        picker.deleteLater()


def test_picker_persists_the_choice_and_warns_only_when_it_is_a_problem(
        qapp, fake_preferences):
    from src.ffmpeg import build_fit, builds
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker

    hardware = _nvidia(name="NVIDIA GeForce GTX 1050 Laptop GPU", driver="500.00")
    options = {o["id"]: o for o in _options(hardware)["entries"]}
    picker = FFmpegBuildPicker(_options(hardware), installed_build_id="latest",
                               preferred_build_id="latest")
    try:
        # Con la de siempre (roja) avisa antes de borrar lo que funciona.
        assert picker.selected_build_status == build_fit.INCOMPATIBLE
        assert "500" in (picker.incompatibility_warning() or "")
        # Cambiar de versión es lo que persistiría el instalador.
        picker.combo.setCurrentIndex(picker.combo.findData("n5.1.6"))
        assert builds.get_preferred_build_id() == "n5.1.6"
        assert picker.incompatibility_warning() is None
    finally:
        picker.deleteLater()


def test_picker_survives_a_failing_preference_write(qapp, monkeypatch,
                                                    fake_preferences):
    """Un preferences.json que no se puede escribir no puede romper el panel."""
    import src.preferences as preferences
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker

    def boom(key, value):
        raise OSError("disco lleno")

    monkeypatch.setattr(preferences, "set", boom)
    picker = FFmpegBuildPicker(_options(), installed_build_id="latest",
                               preferred_build_id="latest")
    try:
        picker.combo.setCurrentIndex(picker.combo.findData("n8.1"))
        assert picker.selected_build_id == "n8.1"
    finally:
        picker.deleteLater()


# ==----------------------------------------------------------------------
# 6. El panel sólo monta el selector y pregunta, no sabe de builds
# ==----------------------------------------------------------------------

class _FakeMainWindow:
    """Padre mínimo, como la ventana que abre el diálogo en la app.

    No lleva hardware a propósito: el veredicto se lo pregunta a
    ``src.hardware.detector``, así que un padre con ``hardware_info`` sugeriría
    justo lo contrario de lo que hace el código.
    """

    def __new__(cls, qapp):
        from PySide6.QtWidgets import QWidget

        class _W(QWidget):
            pass

        return _W()


def _row_for(dialog, dep_id):
    for row in dialog._row_widgets:
        if getattr(row, "dep_id", None) == dep_id:
            return row
    return None


def _ffmpeg_row(**overrides):
    info = {
        "id": "ffmpeg", "display_name": "FFmpeg (encoder suite)",
        "kind": "binary", "status": "installed", "version": "8.1.3",
        "install_dir": "/tmp/ffmpeg", "source_url": "https://x",
        "build_id": "n8.1", "build_label": "FFmpeg 8.1",
        "preferred_build_id": "latest",
        "detail": "", "is_updatable": True, "deletable": True,
        "update_available": None,
    }
    info.update(overrides)
    return info


def test_ffmpeg_row_mounts_the_picker_and_other_rows_do_not(qapp, monkeypatch,
                                                            fake_preferences,
                                                            known_hardware):
    from src.ui.dependencies_dialog import DependenciesDialog

    known_hardware(_nvidia(name="NVIDIA GeForce GTX 1050 Laptop GPU",
                            driver="500.00"))

    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [
            _ffmpeg_row(),
            {"id": "torch", "display_name": "PyTorch runtime", "kind": "runtime",
             "status": "installed", "version": "2.9", "install_dir": "/tmp/rt",
             "source_url": None, "detail": "", "is_updatable": True,
             "deletable": True, "update_available": None},
        ],
    )
    parent = _FakeMainWindow(qapp)
    dialog = DependenciesDialog(parent)
    dialog._check_timer.stop()
    try:
        dialog._populate()
        row = _row_for(dialog, "ffmpeg")
        assert row is not None and row.build_picker is not None
        # La lista sale del hardware detectado (driver 500 -> master en rojo).
        options = row.build_picker._options
        assert set(options) >= {"latest", "n9.0", "n8.1"}
        # Con la versión en rojo, la confirmación avisa antes de sustituir el
        # FFmpeg que ya funciona.
        assert "500" in (dialog._build_warning_for("ffmpeg") or "")
        assert dialog._build_id_for("torch") is None
        assert dialog._build_warning_for("torch") is None
    finally:
        dialog.close()


def test_no_warning_when_the_chosen_build_is_green(qapp, monkeypatch,
                                                   fake_preferences,
                                                   known_hardware):
    from src.ui.dependencies_dialog import DependenciesDialog

    known_hardware(_nvidia(driver="580.00"))
    parent = _FakeMainWindow(qapp)
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row(build_id="latest",
                             build_label="Última build (master)",
                             preferred_build_id="latest")],
    )
    dialog = DependenciesDialog(parent)
    dialog._check_timer.stop()
    try:
        dialog._populate()
        assert dialog._build_warning_for("ffmpeg") is None
    finally:
        dialog.close()


def test_update_state_follows_the_chosen_build_not_the_installed_one(
        monkeypatch, fake_preferences):
    """"Actualizar" sólo puede ofrecer lo que el botón instala.

    La comparación con el upstream es contra la build ELEGIDA (la que instalarían
    el botón y el próximo arranque), no contra la instalada. Por eso el panel no
    necesita re-etiquetar nada: lo que dice es lo que va a pasar.
    """
    import src.dependencies as dependencies
    from src.ffmpeg import builds, manager

    chosen = builds.list_builds()[-1]["id"]
    installed = "n8.1" if chosen != "n8.1" else "n9.0"
    assert builds.set_preferred_build_id(chosen) is True

    # Instalado 8.1 (con su huella vieja) y elegida otra build.
    monkeypatch.setattr(manager, "read_manifest",
                        lambda: {"build_id": installed, "version": "8.1.3",
                                 "upstream_etag": "viejo"})
    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: "/no/existe")

    def cache_for(build_id):
        return {"url": builds.build_mirrors(build_id, "Windows")[0][0],
                "etag": "nuevo", "last_modified_iso": "2026-01-01T00:00:00+00:00"}

    # La huella es de la ELEGIDA y su manifest no la tiene: sí hay update.
    monkeypatch.setattr(manager, "read_upstream_cache", lambda: cache_for(chosen))
    info = manager.get_install_info()
    assert info["build_id"] == installed
    assert info["preferred_build_id"] == chosen
    assert info["update_available"] is True
    # Y es exactamente lo que ve el panel.
    row = dependencies._ffmpeg_status()
    assert row["build_id"] == installed
    assert row["preferred_build_id"] == chosen
    assert row["update_available"] is True

    # La huella es de la INSTALADA, no de la elegida: no hay nada comparable.
    monkeypatch.setattr(manager, "read_upstream_cache", lambda: cache_for(installed))
    assert manager.get_install_info()["update_available"] is None


def test_update_label_survives_when_the_chosen_build_is_the_checked_one(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """Con la misma build elegida e instalada, el 'Actualizar' se mantiene
    (comportamiento de siempre)."""
    from src.ffmpeg import builds, manager
    from src.ui.dependencies_dialog import DependenciesDialog

    assert builds.set_preferred_build_id("n8.1") is True
    monkeypatch.setattr(manager, "read_manifest",
                        lambda: {"build_id": "n8.1", "version": "8.1.3",
                                 "upstream_etag": "viejo"})
    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: "/no/existe")
    monkeypatch.setattr(manager, "read_upstream_cache",
                        lambda: {"url": builds.build_mirrors("n8.1", "Windows")[0][0],
                                 "etag": "nuevo"})
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row(preferred_build_id="n8.1", update_available=True)],
    )
    known_hardware(_nvidia(driver="580.00"))
    parent = _FakeMainWindow(qapp)
    dialog = DependenciesDialog(parent)
    dialog._check_timer.stop()
    try:
        dialog._populate()
        row = _row_for(dialog, "ffmpeg")
        assert row.info["update_available"] is True
        assert row.btn_primary.text() == "Actualizar"
    finally:
        dialog.close()


def test_without_known_hardware_the_dialog_does_not_choose_for_you(
        qapp, monkeypatch, known_hardware):
    """Sin hardware detectado el selector no recomienda nada.

    Este caso salía neutro ("todas las builds codifican igual en CPU, elige la
    que prefieras"), que es una afirmación falsa: nadie había mirado el
    hardware. Ahora es ámbar y el texto lo dice, sin parecerse a una elección.
    """
    from src.ffmpeg import build_fit
    from src.ui.dependencies_dialog import DependenciesDialog

    known_hardware(None)
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row(build_id="latest",
                             build_label="Última build (master)",
                             preferred_build_id="latest")],
    )
    dialog = DependenciesDialog()
    dialog._check_timer.stop()
    try:
        dialog._populate()
        row = _row_for(dialog, "ffmpeg")
        assert row.build_picker is not None
        assert row.build_picker.combo.count() > 0
        options = list(row.build_picker._options.values())
        assert {o["status"] for o in options} == {build_fit.UNVERIFIED}
        assert "No se ha detectado el hardware" in options[0]["reason"]
        assert "Elige la que prefieras" not in options[0]["reason"]
    finally:
        dialog.close()


def test_the_verdict_ignores_which_window_opened_the_dialog(
        qapp, monkeypatch, known_hardware):
    """El hardware lo pregunta su dueño, no la ventana que abrió el diálogo.

    Aquí el padre afirma una GTX 760 con driver 460.98 (todo en rojo) mientras
    que el dueño tiene publicada una RTX con driver 580.00 (todo en verde). Si
    el veredicto dependiera de la parentela, saldrían los del padre.
    """
    from src.ffmpeg import build_fit
    from src.ui.dependencies_dialog import DependenciesDialog

    known_hardware(_nvidia(driver="580.00"))
    parent = _FakeMainWindow(qapp)
    parent.hardware_info = _nvidia(name="NVIDIA GeForce GTX 760",
                                   driver="460.98")
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row(build_id="latest",
                             build_label="Última build (master)",
                             preferred_build_id="latest")],
    )
    dialog = DependenciesDialog(parent)
    dialog._check_timer.stop()
    try:
        dialog._populate()
        row = _row_for(dialog, "ffmpeg")
        statuses = {o["status"] for o in row.build_picker._options.values()}
        assert statuses == {build_fit.COMPATIBLE}
    finally:
        dialog.close()

# ==----------------------------------------------------------------------
# 7. Reinstalar nunca deja el equipo peor que antes
# ==----------------------------------------------------------------------
#
# Reemplazar un FFmpeg que funciona es una acción de rutina (con el selector de
# versiones, cada cambio de build la dispara). Por eso la reinstalación descarga
# a un staging aparte y sólo sustituye lo que ya funciona cuando tiene en la mano
# algo que arranca. Estos tests miden lo único que importa cuando algo va mal:
# que los tres ficheros de la instalación siguen ahí, byte a byte.
#
# Se ejercita ``force_redownload`` de verdad: lo único doblado es la red (o el
# health check), nunca el código que decide qué se borra y cuándo.

def _installed(tmp_path, monkeypatch, manager, downloader):
    """Apunta la app a un ``bin/ffmpeg`` temporal con una instalación falsa.

    Deja los tres ficheros que la app considera la instalación con contenido
    distinguible, para poder afirmar "byte a byte" después de un fallo.
    Devuelve ``(directorio, nombres)``.
    """
    ffmpeg_dir = tmp_path / "ffmpeg"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
    names = (manager.get_ffmpeg_executable_name(),
             manager.get_ffprobe_executable_name(),
             manager.MANIFEST_FILENAME)
    for name in names:
        (ffmpeg_dir / name).write_bytes(_old_bytes(name))

    monkeypatch.setattr(manager, "get_ffmpeg_dir", lambda: str(ffmpeg_dir))
    monkeypatch.setattr(manager, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(manager, "get_ffmpeg_version", lambda path=None: "5.1.6")
    monkeypatch.setattr(downloader, "get_ffmpeg_dir", lambda: str(ffmpeg_dir))
    # Camino single-stream: el de la red mala, sin los hilos de la descarga
    # paralela (que aquí no tienen nada que ver).
    monkeypatch.setattr(downloader, "_probe_url_metadata", lambda url: (None, False))
    return ffmpeg_dir, names


def _old_bytes(name):
    """Contenido de la instalación previa: reconocible a simple vista."""
    return ("instalado:" + name).encode("utf-8")


def _fake_archive(archive_path, manager, ffmpeg=b"NUEVO", ffprobe=b"NUEVO-PROBE"):
    """Escribe el zip que dejaría el paquete de BtbN, con sus dos ejecutables."""
    import zipfile

    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("ffmpeg/bin/" + manager.get_ffmpeg_executable_name(), ffmpeg)
        z.writestr("ffmpeg/bin/" + manager.get_ffprobe_executable_name(), ffprobe)


def _untouched(ffmpeg_dir, names):
    """Los tres ficheros de la instalación, byte a byte, como estaban."""
    return all((ffmpeg_dir / name).read_bytes() == _old_bytes(name) for name in names)


def _no_residue(ffmpeg_dir, names):
    """Ni staging a medias ni ficheros ".old" de un rollback a medias."""
    return sorted(p.name for p in ffmpeg_dir.iterdir() if p.name not in names)


def test_cancelling_a_reinstall_keeps_the_working_install(monkeypatch, tmp_path):
    """Cancelar a mitad deja el FFmpeg que funcionaba, con su manifiesto.

    El botón "Reinstalar" del panel es cancelable a propósito, y el defecto era
    justo este: cancelar era dejar el programa sin FFmpeg.
    """
    import threading

    import src.ffmpeg.downloader as downloader
    from src.ffmpeg import builds, manager

    ffmpeg_dir, names = _installed(tmp_path, monkeypatch, manager, downloader)
    abort = threading.Event()

    def user_cancels_midway(url, archive_path, **kwargs):
        abort.set()                      # el usuario pulsa Cancelar al 40%
        raise RuntimeError("cancelled by user")

    monkeypatch.setattr(downloader, "_download_file", user_cancels_midway)
    monkeypatch.setattr(manager, "get_preferred_build_id",
                        lambda: builds.DEFAULT_BUILD_ID)

    with pytest.raises(RuntimeError) as excinfo:
        manager.force_redownload(abort_event=abort)

    assert "cancelled" in str(excinfo.value)
    assert _untouched(ffmpeg_dir, names)
    assert _no_residue(ffmpeg_dir, names) == []


def test_a_reinstall_whose_mirrors_all_fail_keeps_the_working_install(
        monkeypatch, tmp_path):
    """Si se acaban los espejos tampoco se toca lo que hay instalado."""
    import src.ffmpeg.downloader as downloader
    from src.ffmpeg import builds, manager

    ffmpeg_dir, names = _installed(tmp_path, monkeypatch, manager, downloader)

    def unreachable(url, archive_path, **kwargs):
        raise RuntimeError("404 Client Error: Not Found")

    monkeypatch.setattr(downloader, "_download_file", unreachable)
    monkeypatch.setattr(manager, "get_preferred_build_id",
                        lambda: builds.DEFAULT_BUILD_ID)

    with pytest.raises(RuntimeError):
        manager.force_redownload()

    assert _untouched(ffmpeg_dir, names)
    assert _no_residue(ffmpeg_dir, names) == []


def test_a_swap_that_fails_puts_the_previous_files_back(monkeypatch, tmp_path):
    """Si el intercambio falla a mitad, se deshace entero.

    El caso real en Windows: el ejecutable está en uso y ``os.replace`` da
    ``PermissionError`` justo al Apartar el fichero viejo. Para entonces el
    ffmpeg anterior ya se había movido, así que sin rollback el usuario se
    quedaría sin binario. El bloqueo es real en el sistema de ficheros (un
    directorio donde el código quiere apartar el ffprobe viejo), no un
    ``os.replace`` doblado.
    """
    import src.ffmpeg.downloader as downloader
    from src.ffmpeg import builds, manager

    ffmpeg_dir, names = _installed(tmp_path, monkeypatch, manager, downloader)
    ffprobe = manager.get_ffprobe_executable_name()
    blocker = ffmpeg_dir / (ffprobe + ".old")
    blocker.mkdir()
    (blocker / "ocupado").write_text("bloquea el replace", encoding="utf-8")

    monkeypatch.setattr(
        downloader, "_download_file",
        lambda url, archive_path, **kw: _fake_archive(archive_path, manager),
    )
    monkeypatch.setattr(manager, "get_preferred_build_id",
                        lambda: builds.DEFAULT_BUILD_ID)
    monkeypatch.setattr("src.ffmpeg.runner.health_check_ffmpeg",
                        lambda path, **kw: (True, "5.1.6", None))

    with pytest.raises(OSError):
        manager.force_redownload()

    # Los tres siguen byte a byte como estaban, el ffmpeg incluido.
    assert _untouched(ffmpeg_dir, names)
    # Y lo único que sobra es el bloqueo que puso la propia prueba.
    assert sorted(p.name for p in ffmpeg_dir.iterdir()) == sorted(
        list(names) + [ffprobe + ".old"])


def test_a_build_that_does_not_run_never_replaces_the_working_one(
        monkeypatch, tmp_path):
    """Un binario corrupto o de otra arquitectura no toca el que sí funciona.

    Se dobla sólo el health check, que es el sujeto del test (un fichero que
    existe pero no arranca); la descarga, la extracción, el staging y el
    manifest son de verdad.
    """
    import src.ffmpeg.downloader as downloader
    from src.ffmpeg import builds, manager

    ffmpeg_dir, names = _installed(tmp_path, monkeypatch, manager, downloader)

    monkeypatch.setattr(
        downloader, "_download_file",
        lambda url, archive_path, **kw: _fake_archive(
            archive_path, manager, ffmpeg=b"MZ-basura", ffprobe=b"MZ-basura"),
    )
    monkeypatch.setattr(manager, "get_preferred_build_id",
                        lambda: builds.DEFAULT_BUILD_ID)
    monkeypatch.setattr("src.ffmpeg.runner.health_check_ffmpeg",
                        lambda path, **kw: (False, None, "no es un ejecutable"))

    with pytest.raises(RuntimeError) as excinfo:
        manager.force_redownload()

    assert "no arranca" in str(excinfo.value)
    assert _untouched(ffmpeg_dir, names)
    assert _no_residue(ffmpeg_dir, names) == []


def test_a_good_reinstall_really_replaces_the_three_files(monkeypatch, tmp_path):
    """La garantía no es "no cambiar nunca": cuando la build nueva sí sirve, se
    instala entera y no queda ni staging ni ".old"."""
    import json

    import src.ffmpeg.downloader as downloader
    from src.ffmpeg import builds, manager

    ffmpeg_dir, names = _installed(tmp_path, monkeypatch, manager, downloader)
    ffmpeg_name = manager.get_ffmpeg_executable_name()
    ffprobe_name = manager.get_ffprobe_executable_name()
    monkeypatch.setattr(manager, "get_ffmpeg_version", lambda path=None: "7.1")
    monkeypatch.setattr(
        downloader, "_download_file",
        lambda url, archive_path, **kw: _fake_archive(archive_path, manager),
    )
    monkeypatch.setattr(manager, "get_preferred_build_id", lambda: "n5.1.6")
    monkeypatch.setattr("src.ffmpeg.runner.health_check_ffmpeg",
                        lambda path, **kw: (True, "5.1.6", None))

    result = manager.force_redownload()

    assert result == str(ffmpeg_dir / ffmpeg_name)
    assert (ffmpeg_dir / ffmpeg_name).read_bytes() == b"NUEVO"
    assert (ffmpeg_dir / ffprobe_name).read_bytes() == b"NUEVO-PROBE"
    manifest = json.loads((ffmpeg_dir / manager.MANIFEST_FILENAME).read_text("utf-8"))
    assert manifest["build_id"] == "n5.1.6"
    assert manifest["version"] == "7.1"
    assert manifest["source"] == "user-reinstall"
    assert _no_residue(ffmpeg_dir, names) == []


# ==----------------------------------------------------------------------
# 8. Sin evidencia de cambio, no se ofrece "Actualizar"
# ==----------------------------------------------------------------------
#
# Una build instalada sin red deja el manifest sin huella de upstream: el HEAD
# que habría que grabarla falló. La primera comprobación con red sí encuentra
# huella del fichero de arriba... que es exactamente el mismo fichero. Ofrecer
# "Actualizar" ahí es ofrecer volver a bajar 130 MB para acabar igual. Lo
# correcto es "no lo sé", que el panel ya sabe pintar y no dispara nada.

def test_a_build_installed_offline_offers_no_update(monkeypatch, fake_preferences):
    """Manifest sin huella + upstream con huella = "no lo sé", nunca True."""
    import src.dependencies as dependencies
    from src.ffmpeg import builds, manager, upstream

    assert builds.set_preferred_build_id("n5.1.6") is True
    active_url = builds.build_mirrors("n5.1.6", manager.platform.system())[0][0]

    # Instalada sin red: el manifest se quedó sin huella del upstream.
    manifest = {"build_id": "n5.1.6", "version": "5.1.6",
                "source": "first-download", "source_url": active_url}
    monkeypatch.setattr(manager, "read_manifest", lambda: manifest)
    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: "/no/existe")
    # La primera comprobación con red: aquí sí hay huella del upstream.
    monkeypatch.setattr(manager, "read_upstream_cache", lambda: {
        "url": active_url, "etag": '"abc"',
        "last_modified_iso": "2026-01-01T00:00:00+00:00"})

    assert upstream.is_update_available(
        manifest, {"url": active_url, "etag": '"abc"'}) is None
    assert manager.get_install_info()["update_available"] is None
    # Y el panel recibe exactamente lo mismo.
    assert dependencies._ffmpeg_status()["update_available"] is None


def test_the_panel_says_reinstall_not_update_without_evidence(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """Sin huella no se ofrece volver a descargar el mismo fichero: el botón
    principal del panel no se pone en modo "Actualizar"."""
    from src.ffmpeg import builds, manager
    from src.ui.dependencies_dialog import DependenciesDialog

    assert builds.set_preferred_build_id("n5.1.6") is True
    active_url = builds.build_mirrors("n5.1.6", manager.platform.system())[0][0]
    monkeypatch.setattr(
        manager, "read_manifest",
        lambda: {"build_id": "n5.1.6", "version": "5.1.6", "source_url": active_url},
    )
    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: "/no/existe")
    monkeypatch.setattr(manager, "read_upstream_cache",
                        lambda: {"url": active_url, "etag": '"abc"'})
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row(build_id="n5.1.6", build_label="FFmpeg 5.1.6",
                             preferred_build_id="n5.1.6", update_available=None)],
    )

    known_hardware(_nvidia(driver="580.00"))
    parent = _FakeMainWindow(qapp)
    dialog = DependenciesDialog(parent)
    dialog._check_timer.stop()
    try:
        dialog._populate()
        row = _row_for(dialog, "ffmpeg")
        assert row.info["update_available"] is None
        assert row.btn_primary.text() != "Actualizar"
    finally:
        dialog.close()


def test_update_is_still_offered_once_the_upstream_really_moves(
        monkeypatch, fake_preferences):
    """La garantía es "sin evidencia no hay update", no "nunca hay update".

    Con la huella guardada, si el upstream cambia de verdad (ETag distinto) se
    sigue ofreciendo "Actualizar" exactamente igual que antes del arreglo.
    """
    from src.ffmpeg import builds, manager

    assert builds.set_preferred_build_id("n5.1.6") is True
    active_url = builds.build_mirrors("n5.1.6", manager.platform.system())[0][0]
    manifest = {"build_id": "n5.1.6", "version": "5.1.6",
                "source": "first-download", "source_url": active_url,
                "upstream_etag": '"viejo"',
                "upstream_last_modified": "2025-01-01T00:00:00+00:00"}
    monkeypatch.setattr(manager, "read_manifest", lambda: manifest)
    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: "/no/existe")
    monkeypatch.setattr(manager, "read_upstream_cache", lambda: {
        "url": active_url, "etag": '"nuevo"',
        "last_modified_iso": "2026-01-01T00:00:00+00:00"})

    assert manager.get_install_info()["update_available"] is True

# ==----------------------------------------------------------------------
# 9. El mapa de GPUs NVIDIA, contra la matriz oficial
# ==----------------------------------------------------------------------
#
# La lista de códecs por GPU no se deduce: se copia de la matriz que NVIDIA
# publica (https://developer.nvidia.com/video-encode-decode-support-matrix),
# donde cada familia viene con su número de motores NVENC y sus códecs. Estos
# tests son el recordatorio de de dónde sale cada afirmación, para que nadie
# "arregle" un hueco del mapa suponiendo lo que debería tener la GPU.
#
# El caso que los motiva: un portátil con GeForce MX 450 salía en ámbar, igual
# que una GPU que el programa no conoce. No es lo mismo, y la matriz dice cuál
# de los dos es (la MX 450 no tiene ningún motor NVENC), así que el ámbar se
# queda — con la explicación correcta — y las MX que sí lo tienen salen verdes.

@pytest.mark.parametrize("gpu_name", [
    "NVIDIA GeForce MX 450",
    "NVIDIA GeForce MX 450 Laptop GPU",
    "NVIDIA GeForce MX 350",
    "NVIDIA GeForce MX 330",
    "NVIDIA GeForce MX 130",
    "NVIDIA GeForce MX 110",
    "NVIDIA GeForce MX 570 A",
    "NVIDIA GeForce GT 1030",
    "NVIDIA Quadro P500",
    "NVIDIA Quadro P520",
])
def test_nvidia_models_the_matrix_gives_zero_encoders_claim_nothing(gpu_name):
    """"Total # of NVENC: 0" en la matriz oficial: no se ofrece ningún códec,
    aunque el nombre suene a Pascal o a Turing."""
    from src.hardware.codecs import get_nvidia_codecs, nvidia_has_no_encoder

    assert nvidia_has_no_encoder(gpu_name) is True
    assert get_nvidia_codecs(gpu_name) == []


@pytest.mark.parametrize("gpu_name", [
    "NVIDIA GeForce MX 550",      # Turing 6ª gen, 1 NVENC
    "NVIDIA GeForce MX 570",      # Ampere 7ª gen, 1 NVENC
])
def test_mx_models_that_do_have_nvenc_go_green(gpu_name):
    """La familia MX no es homogénea: la 550 y la 570 sí codifican por GPU."""
    from src.ffmpeg import build_fit
    from src.hardware.codecs import nvidia_has_no_encoder

    assert nvidia_has_no_encoder(gpu_name) is False
    entries = build_fit.evaluate_all(_nvidia(name=gpu_name, driver="580.00"), "Windows")
    assert {e["status"] for e in entries} == {build_fit.COMPATIBLE}
    assert all(gpu_name in e["reason"] for e in entries)


def test_the_mx_450_amber_says_why_instead_of_admitting_ignorance():
    """Ámbar, sí, pero el motivo es "no hay encoder", no "no lo sé".

    Un usuario con una MX 450 tiene derecho a saber que su GPU no trae
    encoder, no a deducir que el programa no sabe qué tiene delante.
    """
    from src.ffmpeg import build_fit

    hardware = _nvidia(name="NVIDIA GeForce MX 450", driver="580.00")
    entries = build_fit.evaluate_all(hardware, "Windows")
    assert {e["status"] for e in entries} == {build_fit.UNVERIFIED}
    for entry in entries:
        assert "sin ningún motor NVENC" in entry["reason"]
        assert "matriz oficial" in entry["reason"]
        # La razón genérica de "GPU desconocida" sería aquí una respuesta peor.
        assert "no aparece en el mapa" not in entry["reason"]
        assert entry["codecs"] == []


def test_a_gpu_the_matrix_does_not_list_stays_amber_and_generic():
    """Lo que NVIDIA no publica no se afirma: MX 230/250 no salen en la
    matriz, así que el veredicto es el ámbar de siempre."""
    from src.ffmpeg import build_fit
    from src.hardware.codecs import get_nvidia_codecs, nvidia_has_no_encoder

    for name in ("NVIDIA GeForce MX 230", "NVIDIA GeForce MX 250"):
        assert nvidia_has_no_encoder(name) is False
        assert get_nvidia_codecs(name) == []

    entries = build_fit.evaluate_all(
        _nvidia(name="NVIDIA GeForce MX 250", driver="580.00"), "Windows")
    assert {e["status"] for e in entries} == {build_fit.UNVERIFIED}
    assert "no aparece en el mapa" in entries[0]["reason"]


def test_maxwell_second_gen_never_claims_hevc_10bit():
    """Corrección en la dirección contraria: la matriz marca "HEVC 10-bit
    support: NO" en GM200/GM204/GM206, y el mapa lo afirmaba.

    Solo se miran los nombres que el mapa ya reconoce, para comprobar
    que se recorta una afirmacion y no se anade un reconocimiento.
    """
    from src.hardware.codecs import get_nvidia_codecs

    for name in ("NVIDIA GeForce GTX 960", "NVIDIA GeForce GTX 970",
                 "NVIDIA GeForce GTX 980 Ti", "GM206"):
        ids = [cid for cid, _label in get_nvidia_codecs(name)]
        assert "h264_nvenc" in ids and "hevc_nvenc" in ids
        assert "hevc_nvenc_10bit" not in ids, name
    # Pascal sí lo tiene: la diferencia es real y no se ha cortado de más.
    assert "hevc_nvenc_10bit" in [
        cid for cid, _l in get_nvidia_codecs("NVIDIA GeForce GTX 1070")]


def test_the_no_encoder_pattern_cannot_swallow_the_models_that_have_one():
    """El patrón de la "MX 570 A" no puede comerse la MX 550 ni la MX 570."""
    from src.hardware.codecs import nvidia_has_no_encoder

    assert nvidia_has_no_encoder("NVIDIA GeForce MX 570 A") is True
    assert nvidia_has_no_encoder("NVIDIA GeForce MX 570") is False
    assert nvidia_has_no_encoder("NVIDIA GeForce MX 550") is False
    # Y ningún patrón de "sin encoder" toca una GPU que sí codifica.
    for name in ("NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 3060 Laptop GPU",
                 "NVIDIA GeForce GTX 1650", "NVIDIA Quadro P620",
                 "NVIDIA RTX A2000", "NVIDIA RTX 4000 SFF Ada Generation"):
        assert nvidia_has_no_encoder(name) is False, name


# ==----------------------------------------------------------------------
# 10. El rótulo del banner dice la build elegida, no "la más reciente"
# ==----------------------------------------------------------------------

class _StubBanner:
    """Lo mínimo que ``_render_banner_state`` necesita para no pintar nada."""

    def __init__(self):
        self.calls = []
        self.btn_update = None

    def show_update(self, title, detail):
        self.calls.append(("update", title, detail))

    def show_offline(self, title, detail):
        self.calls.append(("offline", title, detail))

    def hide_banner(self):
        self.calls.append(("hidden",))


def _banner_stub():
    """Sustituto de ``MainWindow`` para pintar el banner sin construirlo entero.

    El rótulo ya no pasa por la ventana: lo compone ``builds``, su único
    dueño, a partir de lo que hay guardado. Así que aquí sólo hace falta el
    ``update_banner`` donde ``_render_banner_state`` pinta el resultado.
    """

    class _W:
        def __init__(self):
            self.update_banner = _StubBanner()
            self._banner_update_worker = None

    return _W()


def test_the_banner_announces_the_build_the_button_will_install(
        fake_preferences):
    """El banner se levanta comparando el upstream contra la build ELEGIDA y
    el botón instala esa misma: el rótulo tiene que nombrarla."""
    from src.ui.main_window import MainWindow
    from src.ffmpeg import builds

    assert builds.set_preferred_build_id("n5.1.6") is True
    info = {"update_available": True, "preferred_build_id": "n5.1.6",
            "latest_fingerprint": {}, "last_check_iso": None}

    stub = _banner_stub()
    MainWindow._render_banner_state(stub, info)

    kind, title, _detail = stub.update_banner.calls[-1]
    assert kind == "update"
    assert "5.1.6" in title
    assert "mas reciente" not in title


def test_a_platform_with_no_mirrors_is_offered_nothing_and_says_why(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """macOS no se puede probar en esta máquina, así que se simula entera.

    El catálogo publicaba la build de siempre en cualquier sistema, incluso sin
    mirrors, y el selector la ofrecía: instalar ahí fallaba siempre, y sin
    explicación. Ahora el catálogo y la política dicen lo mismo, así que la
    lista sale vacía y el selector explica el vacío en vez de parecer roto.
    """
    from src.ffmpeg import build_fit, builds
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker

    # 1) Windows no se toca: es la lista de siempre, byte a byte.
    assert [b["id"] for b in builds.list_builds("Windows")] == [
        "latest", "n9.0", "n8.1", "n7.1", "n6.1", "n5.1.6"]
    assert [b["id"] for b in builds.list_builds("Linux")] == [
        "latest", "n9.0", "n8.1", "n6.0", "n5.1", "n4.4"]

    # 2) Sin mirrors, no se ofrece nada — y lo que se ofrece, se puede
    #    descargar. Esa es la invariante que se rompió antes.
    for plataforma in ("Darwin", "FreeBSD"):
        assert builds.list_builds(plataforma) == []
        for build in builds.BUILDS:
            assert builds.is_installable(build["id"], plataforma) is False

    known_hardware(_nvidia(name="NVIDIA GeForce RTX 3060", driver="610.74"))
    # El hardware da igual aquí: la lista sale vacía aunque la GPU sea perfecta.
    monkeypatch.setattr(builds, "current_platform", lambda *a, **k: "Darwin")
    monkeypatch.setattr(build_fit, "current_platform", lambda *a, **k: "Darwin")

    assert build_fit.evaluate_all(_nvidia(name="NVIDIA GeForce RTX 3060",
                                         driver="610.74")) == []

    # 3) Y el selector lo dice, en vez de dejar un desplegable mudo.
    picker = FFmpegBuildPicker(build_fit.evaluate_build_list(
        _nvidia(name="NVIDIA GeForce RTX 3060", driver="610.74"), "Darwin"))
    try:
        assert picker.combo.count() == 0
        assert picker.combo.isEnabled() is False
        nota = picker.note_text()
        assert "Darwin" in nota
        assert "no hay nada que descargar" in nota.lower() or \
            "no hay nada" in nota.lower()
        # Sin nada que elegir, no hay veredicto que avisar antes de reinstalar.
        assert picker.incompatibility_warning() is None
        assert picker.selected_build_id is None
    finally:
        picker.deleteLater()


def test_the_dependencies_dialog_opens_even_if_the_build_list_cannot_be_read(
        qapp, monkeypatch, caplog):
    """El manejador de la lista de builds existe para degradar, no para caer.

    ``_annotate_build_options`` envuelve ``build_fit.evaluate_all`` en un
    ``except`` que registra y sigue con la lista vacía. Ese ``except`` llamaba a
    un ``logger`` que el módulo nunca definió, así que en vez de degradar
    lanzaba ``NameError`` y el panel de Manage Dependencies no abría. Aquí se
    fuerza justo ese fallo y se comprueba que el diálogo sube.
    """
    import logging
    from src.ffmpeg import build_fit
    from src.ui import dependencies_dialog as dd

    def revienta(*_a, **_k):
        raise RuntimeError("no se pudo leer el hardware")

    monkeypatch.setattr(dd.build_fit, "evaluate_all", revienta)
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row()],
    )

    with caplog.at_level(logging.ERROR, logger=dd.__name__):
        dialog = dd.DependenciesDialog(_FakeMainWindow(qapp))
    try:
        dialog._check_timer.stop()
        # Subió, y la fila de FFmpeg conserva la seña de que no se pudo evaluar.
        fila = next(r for r in dialog._row_widgets
                    if getattr(r, "dep_id", None) == "ffmpeg")
        picker = getattr(fila, "build_picker", None)
        assert picker is not None
        assert picker.combo.isEnabled() is False

        # Y lo que le dice al usuario es lo que pasó de verdad, NO que no haya
        # builds para su sistema: eso sería falso, y en Windows además habría
        # hecho creer que el programa no trae FFmpeg para nada.
        nota = picker.note_text()
        assert "No se ha podido comprobar" in nota
        assert "Re-detect Hardware & Codecs" in nota
        assert "no es que falten builds" in nota.lower()
        # La frase del caso sin mirrors NO debe aparecer aquí.
        assert "ninguna build del catálogo tiene mirror" not in nota

        # Y el fallo quedó registrado, que es lo que el handler prometía.
        assert any("Could not evaluate the FFmpeg build list" in r.message
                   for r in caplog.records)
    finally:
        dialog.close()


def test_a_broken_ai_engines_tab_degrades_to_its_error_widget(qapp, monkeypatch):
    """El otro manejador del mismo tipo: registrar y seguir, no romperse.

    Si el tab de AI Engines no se puede construir, la intención del código es
    mostrar ``_EnginesTabErrorWidget`` y que el diálogo abra igualmente.
    """
    from src.ui import dependencies_dialog as dd

    class _TabQueRevienta:
        def __init__(self, parent=None):
            raise RuntimeError("no se pudo construir el tab")

    monkeypatch.setattr(dd, "EnginesTab", _TabQueRevienta)
    dialog = dd.DependenciesDialog(_FakeMainWindow(qapp), initial_tab="ai_engines")
    try:
        dialog._check_timer.stop()
        assert dialog._engines_mode is True
        # La clase de error se define dentro del propio manejador, asi que se
        # comprueba por nombre y no con isinstance contra el modulo.
        assert type(dialog.engines_tab).__name__ == "_EnginesTabErrorWidget"
    finally:
        dialog.close()


def test_the_post_install_bookkeeping_failures_do_not_break_the_install_flow(
        qapp, monkeypatch, caplog, tmp_path):
    """Los tres ``logger`` del final de la instalación, uno por uno.

    Son manejadores *best-effort*: marcar el estado ONNX, enriquecer el
    registro del motor de usuario y recargar sus helpers. Su trabajo es
    exactamente que un fallo ahi no arrastre al resto. Con el ``logger`` sin
    definir, cada uno de esos fallos se convertia en un ``NameError``. Aqui se
    fuerzan los tres, uno a uno, y se comprueba que la instalación termina.
    """
    import logging
    import sys
    from unittest import mock

    from PySide6.QtWidgets import QMessageBox
    from src.ui import dependencies_dialog as dd

    # El motor no esta en el catalogo: ``_find_engine`` devuelve {} y con eso
    # el guard de "no hay pesos" no corta, que es lo que nos deja llegar abajo.
    monkeypatch.setattr("src.ui.dependencies_dialog.dependencies.list_all", lambda: [])
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(dd.DependenciesDialog, "_maybe_benchmark_backends",
                        lambda *a, **k: None)
    monkeypatch.setattr(dd.DependenciesDialog, "_refresh_engines_tab",
                        lambda *a, **k: None)

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")

    class _Progress:
        def close(self):
            pass

    # El padre se guarda en una variable: si el QWidget se recoge, Qt se lleva
    # por delante el dialogo y el ``QTimer`` de dentro queda sin objeto vivo.
    parent = _FakeMainWindow(qapp)
    dialog = dd.DependenciesDialog(parent)
    dialog._check_timer.stop()
    try:
        def instalar():
            dialog._on_engine_install_ok("motor-de-prueba", str(onnx),
                                         _Progress(), is_onnx=True)

        # A nivel WARNING, no ERROR: el guardia de la linea 1479 registra con
        # ``logger.warning``, y capturando a ERROR ese registro se filtra.
        with caplog.at_level(logging.WARNING, logger=dd.__name__):
            # Cada caso usa un parche propio que solo se revierte a si mismo:
            # con ``monkeypatch.undo()`` se cairian tambien los stubs de este
            # test y una llamada posterior llegaria a un QMessageBox de verdad,
            # que en modo tests es modal y cuelga la suite.
            def _mark(*_a, **_k):
                raise RuntimeError("no se pudo marcar ONNX")

            # 1) Marcar el estado ONNX falla.
            with mock.patch.object(dd, "engines_mark_onnx", _mark, create=True):
                instalar()
            assert any("Failed to mark ONNX state" in r.message
                       for r in caplog.records), "no se registro el fallo ONNX"
            caplog.clear()

            # 2) Enriquecer el registro del motor de usuario falla al importar.
            with mock.patch.dict(sys.modules, {"src.ai.user_models": None}):
                instalar()
            assert any("Failed to enrich user model record" in r.message
                       for r in caplog.records), "no se registro el user_models"
            caplog.clear()

            # 3) Y el guardia de imports de _refresh_user_model_record falla.
            #    Ese guardia solo se alcanza si el motor es de usuario, asi que
            #    hay que declararlo tal cual: si no, 1443 ni lo intentaria.
            import src.ai.user_models as _um
            with mock.patch.object(_um, "is_user_engine", lambda _e: True), \
                    mock.patch.dict(sys.modules, {"src.ai.architectures": None}):
                instalar()
            assert any("user-model refresh helpers unavailable" in r.message
                       for r in caplog.records), "no se registro el guardia"
    finally:
        dialog.close()


def test_the_policy_of_what_is_a_valid_choice_is_written_in_one_place(
        monkeypatch):
    """``unusable_reason`` es la única política; los demás la leen.

    No se comprueba que los tres consumidores se comporten bien (eso lo cubren
    los tests de cada uno, a través de la interfaz de verdad). Lo que se fija
    aquí es que la función tiene exactamente las tres respuestas que el resto
    del módulo da por descontadas, y que no devuelve motivo para lo que sí vale.
    """
    from src.ffmpeg import builds

    # 1) No está en el catálogo.
    assert builds.unusable_reason("no-existe") == "no está en el catálogo"

    # 2) Está, pero no se puede instalar en esta plataforma.
    guardado = "n8.1"
    real = builds.is_installable
    monkeypatch.setattr(
        builds, "is_installable",
        lambda build_id, platform_name=None: (
            False if build_id == guardado else real(build_id, platform_name)),
    )
    assert builds.find_build(guardado) is not None
    motivo = builds.unusable_reason(guardado)
    assert "no se puede instalar" in motivo
    assert "no está en el catálogo" not in motivo   # el otro motivo, no los dos

    # 3) Vale: no hay motivo. Y lo que no es elección, tampoco se acusa.
    assert builds.unusable_reason("latest") is None
    assert builds.unusable_reason("n9.0") is None
    assert builds.unusable_reason(None) is None
    assert builds.unusable_reason("") is None
    assert builds.unusable_reason("   ") is None


def test_a_chosen_build_id_that_is_not_in_the_catalog_is_said_out_loud(
        fake_preferences):
    """Una preferencia corrupta no puede anunciarse como una build válida.

    ``get_build`` cae en la de siempre porque al instalar hace falta instalar
    *algo*, y ese silencio está bien ahí. En la presentación no: si lo guardado
    no existe en el catálogo, el rótulo tiene que decirlo, o el usuario ve
    "Última build (master)" y cree que la eligió él.
    """
    from src.ffmpeg import builds

    por_defecto = builds.get_build()["label"]

    # Sin preferencia guardada, o con una válida, el rótulo es el de siempre.
    assert builds.chosen_build_label() == por_defecto
    assert builds.set_preferred_build_id("n8.1") is True
    assert builds.chosen_build_label() == builds.get_build("n8.1")["label"]

    # Con una que no está en el catálogo, el rótulo dice qué va a pasar y por
    # qué no es la que estaba guardada. Se escribe a mano en la preferencia,
    # que es como se corrompe de verdad: ``set_preferred_build_id`` no guarda
    # ids desconocidos, y el fichero puede venir editado de otra máquina.
    fake_preferences[builds.BUILD_PREF] = "no-existe"
    assert builds.find_build("no-existe") is None
    roto = builds.chosen_build_label()
    assert por_defecto in roto
    assert "no-existe" in roto
    assert "no está en el catálogo" in roto

    # Y sanear para INSTALAR sigue igual: eso no se toca, y no debe.
    assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID


def test_a_chosen_build_that_cannot_be_installed_here_is_also_said_out_loud(
        fake_preferences, monkeypatch):
    """La otra forma de no instalar lo guardado: existe, pero aquí no se puede.

    Una build del catálogo sin mirror para esta plataforma se sustituye por la
    de siempre igual que una que no existe, así que el rótulo tiene que decirlo
    igual. Si no, el banner anuncia "FFmpeg 6.0" y el botón instala la master.

    El caso se provoca con ``monkeypatch`` y no mirando la máquina: si dependiera
    de qué builds tienen mirror en este equipo, el test se iría en silencio
    (``skip``) justo donde más falta haría vigilarlo, que es donde todas las
    tienen.
    """
    from src.ffmpeg import builds

    guardado = "n8.1"
    real = builds.is_installable
    monkeypatch.setattr(
        builds, "is_installable",
        lambda build_id, platform_name=None: (
            False if build_id == guardado else real(build_id, platform_name)),
    )
    assert builds.find_build(guardado) is not None        # sí está en el catálogo
    assert builds.set_preferred_build_id(guardado) is False  # pero no se puede guardar
    fake_preferences[builds.BUILD_PREF] = guardado            # salvo a mano

    # Se instala la de siempre...
    assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID
    # ...y se dice cuál es y por qué no es la que estaba guardada.
    roto = builds.chosen_build_label()
    assert builds.get_build()["label"] in roto
    assert guardado in roto
    assert "no se puede instalar" in roto
    assert "no está en el catálogo" not in roto  # no es el otro caso


def test_a_corrupt_preference_reaches_the_banner_and_the_dialog(
        fake_preferences, monkeypatch):
    """El flujo entero, no la función suelta: preferencia guardada -> diálogo.

    Éste es el test que faltaba. El anterior llamaba a
    ``chosen_build_label`` directamente, así que pasaba en verde mientras el
    banner seguía mudo: ``get_install_info`` publicaba un id ya saneado y el
    rótulo no tenía con qué contar que lo guardado no existía. Aquí se recorre
    lo que de verdad ocurre al arrancar —lo que hay en la preferencia, la info
    que se le pasa a la ventana, y el texto que se pinta—.
    """
    from src.ffmpeg import builds, manager
    from src.ui.main_window import MainWindow

    # Una preferencia guardada a mano que no está en el catálogo.
    fake_preferences[builds.BUILD_PREF] = "ffmpeg-6.6-de-otra-maquina"
    monkeypatch.setattr(manager, "read_manifest",
                        lambda: {"build_id": builds.DEFAULT_BUILD_ID,
                                 "version": "7.1"})

    # 1. Lo que se INSTALA sigue siendo la de siempre: el saneado no se toca.
    info = manager.get_install_info()
    assert info["preferred_build_id"] == builds.DEFAULT_BUILD_ID
    assert info["build_id"] == builds.DEFAULT_BUILD_ID

    # 2. Pero lo que se DICE sale del id realmente guardado, no del saneado.
    info["update_available"] = True
    info["latest_fingerprint"] = {}
    info["last_check_iso"] = None
    stub = _banner_stub()
    MainWindow._render_banner_state(stub, info)
    kind, title, _detail = stub.update_banner.calls[-1]

    assert kind == "update"
    assert "ffmpeg-6.6-de-otra-maquina" in title, title
    assert "no está en el catálogo" in title, title

    # 3. El mismo rótulo es el que usa el botón de descarga y el diálogo de
    # reinstalar: lo compone ``builds``, no lo elige el que llama.
    assert builds.chosen_build_label() in title
    assert builds.build_label("ffmpeg-6.6-de-otra-maquina") in title

    # 4. Y sin corrupción, el banner dice el nombre de la elegida y sólo eso.
    assert builds.set_preferred_build_id("n5.1.6") is True
    limpio = _banner_stub()
    MainWindow._render_banner_state(limpio, info)
    _kind, title_limpio, _d = limpio.update_banner.calls[-1]
    assert title_limpio == f"FFmpeg update available: {builds.get_build('n5.1.6')['label']}"


def test_the_install_info_also_says_when_the_manifest_names_nothing_real(
        monkeypatch):
    """Lo mismo en la fila del panel de dependencias.

    El ``build_id`` sigue siendo el resuelto (es lo que se instalaría de
    verdad), pero el rótulo avisa de que el manifest no nombraba una build real.
    """
    from src.ffmpeg import builds, manager

    monkeypatch.setattr(manager, "read_manifest",
                        lambda: {"build_id": "basura", "version": "x"})
    info = manager.get_install_info()
    assert info["build_id"] == builds.DEFAULT_BUILD_ID
    assert "basura" in info["build_label"]
    assert "no está en el catálogo" in info["build_label"]

    monkeypatch.setattr(manager, "read_manifest",
                        lambda: {"build_id": "n8.1", "version": "x"})
    assert manager.get_install_info()["build_label"] == \
        builds.get_build("n8.1")["label"]


def test_the_update_dialog_names_the_chosen_build_and_protects_the_install(
        qapp, monkeypatch, fake_preferences):
    """El diálogo decía "descargar la build más reciente" y "el binario
    actual será reemplazado". Las dos cosas son falsas hoy: se descarga la
    elegida, y el binario actual sólo se toca si el nuevo arranca."""
    from PySide6.QtWidgets import QMessageBox
    from src.ui.main_window import MainWindow
    from src.ffmpeg import builds

    assert builds.set_preferred_build_id("n5.1.6") is True
    monkeypatch.setattr("src.ffmpeg.manager.get_install_info", lambda: {
        "preferred_build_id": "n5.1.6",
        "latest_fingerprint": {"content_length": 136 * 1024 * 1024},
    })

    asked = {}

    class _FakeBox:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def question(_parent, title, text, *a, **kw):
            asked["title"] = title
            asked["text"] = text
            return QMessageBox.StandardButton.No

    monkeypatch.setattr("src.ui.main_window.QMessageBox", _FakeBox)

    stub = _banner_stub()
    MainWindow._on_banner_update_clicked(stub)

    text = asked["text"]
    assert "5.1.6" in text
    assert "mas reciente" not in text.lower()
    assert "sigue instalado" in text
    # Y al responder "no" no se lanza ningún worker.
    assert stub._banner_update_worker is None



def test_the_reinstall_dialog_says_out_loud_what_will_actually_be_downloaded(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """El tercer sitio que anuncia una build, y el único sin nada que lo vigile.

    El banner y el rótulo tienen test; este camino —el diálogo de reinstalar del
    panel de dependencias— se cambió a ciegas y sólo se miró a mano. Con esto,
    un cambio futuro en ``dependencies_dialog.py`` que vuelva a tapar la
    sustitución se ve almediately, no en la siguiente queja de un usuario.

    Se recorre el camino real: se llama a ``_on_redownload_requested`` sobre un
    diálogo de verdad y se lee el texto de la confirmación que vería el usuario.
    """
    from PySide6.QtWidgets import QMessageBox
    from src.ffmpeg import builds
    from src.ui.dependencies_dialog import DependenciesDialog

    known_hardware(_nvidia(driver="580.00"))
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row()],
    )

    asked = {}

    class _FakeBox:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def question(_parent, title, text, *a, **kw):
            asked["title"] = title
            asked["text"] = text
            return QMessageBox.StandardButton.No  # el usuario cancela

    monkeypatch.setattr("src.ui.dependencies_dialog.QMessageBox", _FakeBox)

    parent = _FakeMainWindow(qapp)
    dialog = DependenciesDialog(parent)
    dialog._check_timer.stop()

    def confirmar():
        asked.clear()
        dialog._on_redownload_requested("ffmpeg")
        assert asked.get("text"), "no se pidio confirmacion"
        return asked["text"]

    try:
        # 1) Preferencia normal: se anuncia la elegida y nada más.
        assert builds.set_preferred_build_id("n8.1") is True
        texto = confirmar()
        assert builds.get_build("n8.1")["label"] in texto
        assert "no está en el catálogo" not in texto
        assert "no se puede instalar" not in texto

        # 2) Preferencia corrupta: se sigue instalando la de siempre, pero se
        #    dice que no es la que estaba guardada.
        corrupta = "ffmpeg-6.6-de-otra-maquina"
        fake_preferences[builds.BUILD_PREF] = corrupta
        assert builds.get_preferred_build_id() == builds.DEFAULT_BUILD_ID
        texto = confirmar()
        assert builds.get_build()["label"] in texto
        assert corrupta in texto
        assert "no está en el catálogo" in texto

        # 3) Existe en el catálogo pero no se puede instalar aquí: mismo aviso,
        #    con su propio motivo.
        guardada = "n5.1.6"
        real = builds.is_installable
        monkeypatch.setattr(
            builds, "is_installable",
            lambda build_id, platform_name=None: (
                False if build_id == guardada else real(build_id, platform_name)),
        )
        fake_preferences[builds.BUILD_PREF] = guardada
        texto = confirmar()
        assert builds.get_build()["label"] in texto
        assert guardada in texto
        assert "no se puede instalar" in texto
        assert "no está en el catálogo" not in texto

        # Y cancelar no lanza ningún worker: esto no descarga nada.
        assert dialog._active_worker is None
    finally:
        dialog.close()


# ==----------------------------------------------------------------------
# 12. El mapa completo contra la matriz transcrita, sin huecos silenciosos
# ==----------------------------------------------------------------------
#
# La transcripción de la tabla "NVENC - Encoding" de NVIDIA vive en el repo, en
# ``tests/data/nvidia_matrix.py``, con su URL, su pestaña y la fecha de la
# consulta. Vive en ``tests/`` y no en ``src/`` porque es un oráculo de
# verificación, no estado de la app: el dueño en tiempo de ejecución es
# ``src/hardware/codecs.py``, y esto comprueba que lo que ese dueño afirma
# siga diciendo NVIDIA lo que dice. Este contraste corre SIN RED y dice, fila por
# fila, si el mapa afirma lo que dice la tabla.
#
# No hay un cajón de "se me pasó": si NVIDIA publica una fila y el mapa no la
# reclama, tiene que estar en ``DECISIONES`` con su motivo, y el motivo vive en
# el dato. Cuando NVIDIA añada o cambie una fila, el trabajo es editar la tabla
# y, si el mapa aún no la reclama, escribir por qué.
#
# Las clases de códec que salen de las cuatro columnas copiadas:
#   C1 H.264 solo            (Maxwell 1ª: "HEVC 4K 4:2:0: NO")
#   C2 H.264 + HEVC 8-bit    (Maxwell 2ª; y Pascal big, sin Main10)
#   C3 H.264 + HEVC 8/10-bit (Pascal, Volta, Turing, Ampere)
#   C4 C3 + AV1 8/10-bit     (Ada Lovelace, Blackwell)


@pytest.mark.parametrize("gpu_name", all_names())
def test_el_mapa_dice_lo_que_dice_la_matriz_de_nvidia(gpu_name):
    """Cada fila publicada, una a una, contra el veredicto que da el mapa.

    Con la decisión escrita en el dato, lo esperado es la decisión; sin ella,
    lo esperado es lo que dicen las columnas de la fila. Una fila con 0 motores
    además tiene que quedar *declarada* ("no hay encoder"), no sólo en silencio.
    """
    from src.hardware.codecs import get_nvidia_codecs, nvidia_has_no_encoder

    codecs, sin_encoder = expected_for(gpu_name)
    assert [cid for cid, _l in get_nvidia_codecs(gpu_name)] == list(codecs)
    assert nvidia_has_no_encoder(gpu_name) is sin_encoder


def test_el_dato_donde_nace_esta_completo_y_cada_decision_dice_por_que():
    """El contraste sólo vale si el dato está entero y nadie se salta una fila."""
    rows = [row[0] for row in FILAS]

    # Sin productos repetidos: dos filas con el mismo nombre harían que una
    # tapase a la otra sin que se note.
    assert len(rows) == len(set(rows)), "producto repetido en FILAS"

    # La fuente y su fecha son parte del dato: sin ellas no se puede saber si
    # lo transcrito sigue siendo lo que dice NVIDIA.
    for campo in ("url", "tab", "consulted"):
        assert str(SOURCE.get(campo, "")).strip(), campo

    # Toda grafía apunta a una fila real (o a una decisión, que también es una
    # fila): si no, el contraste estaría comparando contra la nada.
    destinos = set(rows) | set(DECISIONES)
    for grafia, destino in ESCRITURAS.items():
        assert destino in destinos, (grafia, destino)

    # Y toda decisión dice por qué se aparta de la tabla. Un motivo vacío sería
    # exactamente el hueco silencioso que este trabajo existe para evitar.
    for nombre, (codecs, motivo) in DECISIONES.items():
        assert str(motivo).strip(), nombre
        assert isinstance(codecs, tuple), nombre

    # Y toda decisión es sobre una fila publicada o una grafía de una.
    for nombre in DECISIONES:
        assert nombre in destinos or any(
            destino == nombre for destino in ESCRITURAS.values()
        ), nombre


def test_la_matriz_transcrita_cubre_las_familias_que_el_programa_distingue():
    """El dato tiene que seguir siendo la tabla entera, no un resumen.

    Si alguien recorta la transcripción para que el contraste pase, este test
    se cae: comprueba que siguen estando las cinco pestañas de la tabla y las
    familias de códec que el mapa distingue.
    """
    familias = {row[1] for row in FILAS}
    for familia in ("Blackwell", "Ada", "Ampere", "Turing", "Volta",
                    "Pascal", "Pascal big", "Maxwell 1", "Maxwell 2"):
        assert familia in familias, familia

    # Las cuatro columnas copiadas de la tabla, y sólo ellas: si NVIDIA publica
    # una quinta que el programa pueda afirmar, tiene que entrar aquí.
    for fila in FILAS:
        assert len(fila) == 7, fila
        nombre, familia, motores = fila[0], fila[1], fila[2]
        assert str(nombre).strip() and str(familia).strip()
        assert isinstance(motores, int) and motores >= 0
        for columna in fila[3:]:
            assert isinstance(columna, bool), (nombre, columna)


def test_kepler_mantiene_su_afirmacion_y_no_la_disfraza_de_certidumbre():
    """Lo que se sostiene para Kepler, y por qué.

    La matriz actual no publica ninguna fila de GeForce anterior al GTX 745
    (que NVIDIA clasifica como Maxwell 1ª gen): dejó de listarlos al terminar
    el soporte con la rama R470. Eso es una omisión, no un "0 motores", así que
    el mapa mantiene su afirmación histórica de H.264 —una afirmación de qué
    códecs existen, no de rendimiento— en vez de retirarla por falta de datos.
    Ese H.264 ya está comprobado fila a fila por el contraste con la matriz
    (las filas 760/770/780 están en ``DECISIONES``, con su motivo).
    que la matriz agrupa aquí, en MATRIZ_RECONOCIDAS (cubo 1). Aquí queda lo que
    esos cubos no cubren.

    El aviso de fin de soporte sigue puesto, y con la afirmación mantenida el
    driver antiguo cae en el rojo que dice, literalmente, que ninguna build
    publicada cubre un driver tan antiguo.
    """
    from src.hardware.codecs import is_legacy_nvidia

    # El aviso de fin de soporte de Kepler sigue ahí: afirmar no es prometer.
    assert is_legacy_nvidia("NVIDIA GeForce GTX 770") is True

    from src.ffmpeg import build_fit

    hardware = _nvidia(name="NVIDIA GeForce GTX 770", driver="460.98")
    entries = build_fit.evaluate_all(hardware, "Windows")
    assert {e["status"] for e in entries} == {build_fit.INCOMPATIBLE}
    for entry in entries:
        assert "Ninguna de las versiones publicadas" in entry["reason"]
        assert "actualiza el driver" in entry["reason"]
        assert "Kepler" in entry["reason"]


def test_the_open_dependencies_panel_recovers_when_hardware_is_redetected(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """La nota manda a 'Re-detect': el panel abierto tiene que enterarse.

    El texto del selector dice "Pulsa 'Re-detect Hardware & Codecs' e
    inténtalo de nuevo". Re-detectar sí arregla la lectura del hardware, así
    que si el panel de dependencias está abierto en ese momento tiene que
    mostrar el veredicto nuevo: si no, el botón que el texto nombra parece no
    hacer nada y el usuario sólo lo nota cerrando y reabriendo el panel.

    Se recorre la costura real —el slot que el worker de redetección
    emite (``MainWindow._on_redetect_done``) con un ``DependenciesDialog``
    de verdad abierto— y no la función suelta, que es justo el hueco que
    dejó el otro test de este mismo camino.
    """
    from src.ffmpeg import build_fit
    from src.ui import dependencies_dialog as dd
    from src.ui.main_window import MainWindow

    roto = {"si": True}
    # Se captura ANTES de parchear: ``dd.build_fit`` es el mismo módulo, y
    # llamarse a sí mismo dentro delReplacement se comería la recursión.
    real = build_fit.evaluate_all

    def evaluate_a_veces(hardware, platform_name=None):
        if roto["si"]:
            raise RuntimeError("no se pudo leer el hardware")
        return real(hardware, platform_name)

    monkeypatch.setattr(dd.build_fit, "evaluate_all", evaluate_a_veces)
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row()],
    )

    # El panel se abre con el hardware aún sin leer: es el estado que pinta
    # el texto de "no se ha podido comprobar". El padre se guarda en una
    # variable porque un QWidget temporal se puede collected y llevarse por
    # delante al diálogo entero.
    padre = _FakeMainWindow(qapp)
    dialog = dd.DependenciesDialog(padre)
    try:
        dialog._check_timer.stop()
        ventana = MainWindow("C:/fake/ffmpeg.exe", {"gpus": []}, {})

        def picker_abierto():
            fila = next(r for r in dialog._row_widgets
                        if getattr(r, "dep_id", None) == "ffmpeg")
            return fila.build_picker

        # Estado inicial: el fallo, y la nota que manda a re-detectar.
        assert "No se ha podido comprobar" in picker_abierto().note_text()

        # El panel queda registrado como abierto, que es lo que hace
        # ``_open_dependencies`` y de lo que depende el repintado.
        ventana._dependencies_dialog = dialog

        # Ahora sí se puede leer el hardware, y se redetecta.
        roto["si"] = False
        known_hardware(_nvidia(driver="580.00"))
        ventana._on_redetect_done(_nvidia(driver="580.00"), {})

        # El veredicto real, en el MISMO panel: sin cerrarlo ni reabrirlo.
        picker = picker_abierto()
        nota = picker.note_text()
        assert "No se ha podido comprobar" not in nota
        assert "Re-detect Hardware & Codecs" not in nota
        assert picker.combo.isEnabled() is True
        assert picker.combo.count() > 0
        # Y es el veredicto de verdad: con la GPU leída, todas verdes.
        assert picker.selected_build_status == build_fit.COMPATIBLE

        # Sin panel abierto, la misma redetección no rompe nada ni se
        # inventa un diálogo: es el caso normal de la app.
        ventana._dependencies_dialog = None
        ventana._on_redetect_done(_nvidia(driver="580.00"), {})
    finally:
        dialog.close()
        del dialog, padre


def test_the_text_names_the_platform_of_the_list_not_the_one_of_the_machine(
        qapp, fake_preferences):
    """La plataforma viaja en los datos; el texto no puede mirar otro sitio.

    Éste es el fallo que existía: ``evaluate_all(hardware, "Darwin")`` devuelve
    la lista de Darwin, pero el selector llamaba a
    ``builds.current_platform()`` —la plataforma de la MÁQUINA— para componer
    su texto. Con la lista de Darwin en la mano y una máquina Windows, decía
    "no hay ninguna versión publicable para Windows", describiendo un sistema
    que no era el suyo. Sólo se notaba por el camino de datos equivocado, que
    hoy no ocurre porque el único llamante real pasa la plataforma por
    defecto.

    Aquí se comprueba justo ese cruce: la lista es de una plataforma y la
    máquina es otra, y el texto tiene que nombrar la de la lista.
    """
    from src.ffmpeg import builds, build_fit
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker

    # La máquina es Windows...
    monkey = None
    original = builds.current_platform
    builds.current_platform = lambda *a, **k: "Windows"
    try:
        # ...pero la lista que se le da al selector es la de FreeBSD.
        carrier = build_fit.evaluate_build_list(
            _nvidia(name="NVIDIA GeForce RTX 3060", driver="610.74"), "FreeBSD")
        assert carrier["platform"] == "FreeBSD"
        assert carrier["entries"] == []
        assert carrier["evaluation_failed"] is False

        picker = FFmpegBuildPicker(carrier, installed_build_id=None,
                                   preferred_build_id="latest")
        try:
            nota = picker.note_text()
            # Nombra la plataforma DE LA LISTA, no la de la máquina.
            assert "FreeBSD" in nota
            assert "Windows" not in nota
        finally:
            picker.deleteLater()
    finally:
        builds.current_platform = original

    # Y el carrier se puede describir sin tocar la máquina: el que compone el
    # texto no tiene de dónde sacar otra plataforma.
    assert builds.nothing_to_offer_reason(carrier["platform"]) is not None
    assert builds.nothing_to_offer_reason("Windows") is None


def test_a_failed_evaluation_still_carries_the_platform_it_failed_for(
        qapp, fake_preferences):
    """El caso "no se ha podido comprobar" también nombra una plataforma.

    Es el otro texto que se compone con la plataforma, y si al fallar la
    evaluación el carrier perdiera el nombre, el selector volvería a
    adivinarlo. Además fija que los dos vacíos siguen siendo distintos:
    lista vacía es "no hay builds", y esto es "no se ha podido comprobar".
    """
    from src.ffmpeg import build_fit
    from src.ui.ffmpeg_build_picker import FFmpegBuildPicker

    carrier = build_fit.failed_build_list("FreeBSD")
    assert carrier["platform"] == "FreeBSD"
    assert carrier["entries"] == []
    assert carrier["evaluation_failed"] is True

    picker = FFmpegBuildPicker(carrier, installed_build_id=None,
                               preferred_build_id="latest")
    try:
        nota = picker.note_text()
        assert "FreeBSD" in nota
        # Y no es el texto del catálogo vacío: son dos hechos distintos.
        assert "no se ha podido comprobar" in nota.lower()
        assert "ninguna build del catálogo tiene mirror" not in nota
    finally:
        picker.deleteLater()


def test_reinstalling_a_non_ffmpeg_dependency_is_not_blocked_by_missing_ffmpeg_mirrors(
        qapp, monkeypatch, fake_preferences):
    """Sin mirrors de FFmpeg, PyTorch y ONNX Runtime sí se reinstalan.

    El guardia del turno pasado se apoyaba en "no hay builds descargables" y
    bloqueaba TODA reinstalación, incluidas las de runtimes. Eso era falso:
    PyTorch y ONNX Runtime se reinstalan con su propio instalador (pip), no
    con los mirrors del catálogo de FFmpeg, así que en una plataforma sin
    mirrors de FFmpeg ellos sí se pueden reinstalar. Negárselo era mentir por
    el motivo contrario al que se quería evitar.
    """
    from unittest import mock

    from src.ffmpeg import builds, build_fit
    from src.ui import dependencies_dialog as dd

    monkeypatch.setattr(builds, "current_platform", lambda *a, **k: "Darwin")
    monkeypatch.setattr(build_fit, "current_platform", lambda *a, **k: "Darwin")

    ffmpeg = _ffmpeg_row()
    torch = {"id": "torch", "display_name": "PyTorch runtime", "kind": "runtime",
             "status": "installed", "version": "2.5", "install_dir": "/tmp/t",
             "source_url": "", "detail": "", "is_updatable": True,
             "deletable": True, "update_available": None}
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [dict(ffmpeg), dict(torch)],
    )

    padre = _FakeMainWindow(qapp)
    dialog = dd.DependenciesDialog(padre)
    try:
        dialog._check_timer.stop()

        # FFmpeg sí se niega: es la build que no tiene dónde bajarse.
        with mock.patch("src.ui.dependencies_dialog.QMessageBox.warning") as warn:
            with mock.patch(
                "src.ui.dependencies_dialog.QMessageBox.question"
            ) as question:
                dialog._on_redownload_requested("ffmpeg")
        assert warn.call_count == 1
        assert question.call_count == 0
        assert dialog._active_worker is None

        # PyTorch no: se reinstala con pip, así que se le pregunta como siempre.
        with mock.patch("src.ui.dependencies_dialog.QMessageBox.warning") as warn:
            with mock.patch(
                "src.ui.dependencies_dialog.QMessageBox.question",
                return_value=mock.Mock(),
            ) as question:
                dialog._on_redownload_requested("torch")
        assert warn.call_count == 0
        assert question.call_count == 1
        dialog._active_worker = None
    finally:
        dialog.close()
        del dialog, padre


def test_the_ffmpeg_row_offers_no_install_where_there_is_nothing_to_install(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """La fila de FFmpeg es la cuarta puerta, y era la que quedaba abierta.

    El selector, el banner y la confirmación de reinstalar ya se negaban en
    una plataforma sin mirrors, pero el botón principal de la fila no lo
    hacía: seguía diciendo "Instalar" y quedaba vivo. Ese botón dispara el
    mismo ``redownload_requested``, o sea que el usuario podía pulsar la
    puerta que las otras tres acababan de cerrar.

    Se comprueba sobre la fila real del diálogo, en Darwin simulada, y en
    las dos cosas que la auditoría vio: el botón y la URL.
    """
    from unittest import mock

    from src.ffmpeg import build_fit, builds
    from src.ui import dependencies_dialog as dd

    # 1) La política dice que aquí no hay nada que ofrecer.
    monkeypatch.setattr(builds, "current_platform", lambda *a, **k: "Darwin")
    monkeypatch.setattr(build_fit, "current_platform", lambda *a, **k: "Darwin")
    assert builds.nothing_to_offer_reason("Darwin") is not None

    # 2) El manifest que construye la fila lleva el motivo en los datos.
    fila = _ffmpeg_row(
        status="missing", version="", install_dir="",
        # Un manifest viejo puede tener la URL de otra plataforma; en Darwin
        # real no hay manifest y esto sería None, pero la fila tiene que
        # Resistir el caso difícil igual.
        source_url="https://example.invalid/ffmpeg-master-latest-win64-gpl.zip",
        is_updatable=True, deletable=False,
        nothing_to_install=builds.nothing_to_offer_reason("Darwin"),
    )
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all", lambda: [fila]
    )

    padre = _FakeMainWindow(qapp)
    dialog = dd.DependenciesDialog(padre)
    try:
        dialog._check_timer.stop()
        row = next(r for r in dialog._row_widgets
                   if getattr(r, "dep_id", None) == "ffmpeg")

        # El botón NO ofrece instalar, y no está vivo.
        assert row.btn_primary.text() != "Instalar"
        assert row.btn_primary.text() == "No se puede instalar"
        assert row.btn_primary.isEnabled() is False

        # Y dice por qué, con la MISMA frase que las otras tres superficies.
        assert "No se puede instalar aquí" in row.btn_primary.toolTip()
        assert "no puede descargar FFmpeg" in row.btn_primary.toolTip()
        assert "Darwin" in row.btn_primary.toolTip()

        # Y no se pinta la URL extranjera de la que no se puede hacer nada.
        from PySide6.QtWidgets import QLabel
        pintadas = [c.text() for c in row.findChildren(QLabel)
                    if "http" in c.text()]
        assert pintadas == [], f"pinta una URL que no puede usar: {pintadas}"

        # Y aunque se pulse, el camino de reinstalar se niega.
        with mock.patch("src.ui.dependencies_dialog.QMessageBox.warning") as warn:
            with mock.patch(
                "src.ui.dependencies_dialog.QMessageBox.question"
            ) as question:
                dialog._on_redownload_requested("ffmpeg")
        assert warn.call_count == 1
        assert question.call_count == 0
        assert dialog._active_worker is None
    finally:
        dialog.close()
        del dialog, padre


def test_the_ffmpeg_row_is_untouched_where_ffmpeg_can_be_installed(
        qapp, fake_preferences):
    """En Windows: el mismo botón, la misma URL, la misma etiqueta.

    El caso de macOS no se puede probar en esta máquina, así que sin este
    test un arreglo que se pasara de la raya se vería como un test de
    Darwin en verde. Aquí se fija lo de Windows con la fila real.
    """
    from PySide6.QtWidgets import QLabel

    from src.ffmpeg import builds
    from src.ui import dependencies_dialog as dd

    # Sin mirrors, la política calla; con mirrors, también.
    assert builds.nothing_to_offer_reason("Windows") is None

    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    fila = _ffmpeg_row(
        status="missing", version="", install_dir="", source_url=url,
        is_updatable=True, deletable=False,
        nothing_to_install=builds.nothing_to_offer_reason("Windows"),
    )
    import src.ui.dependencies_dialog as dd_mod
    from unittest import mock
    with mock.patch.object(dd_mod.dependencies, "list_all", lambda: [fila]):
        padre = _FakeMainWindow(qapp)
        dialog = dd.DependenciesDialog(padre)
        try:
            dialog._check_timer.stop()
            row = next(r for r in dialog._row_widgets
                       if getattr(r, "dep_id", None) == "ffmpeg")
            # Byte a byte lo de antes: botón, URL y etiqueta.
            assert row.btn_primary.text() == "Instalar"
            assert row.btn_primary.isEnabled() is True
            pintadas = [c.text() for c in row.findChildren(QLabel)
                        if "http" in c.text()]
            assert url in pintadas
        finally:
            dialog.close()
            del dialog, padre


def _simular_maquina_sin_ffmpeg(monkeypatch, plataforma):
    """Deja la máquina como estaría en ese sistema, sin tocar el productor.

    No se parchea ``list_all``, ``_ffmpeg_status`` ni ``get_install_info``:
    esos son el camino real que hay que vigilar, y el fallo anterior fue
    justo que el test se saltara esa capa. Lo que se simula es la MÁQUINA
    (sistema operativo y "FFmpeg no está instalado"), que es lo que no se
    puede tener en un runner de Windows.
    """
    import os.path

    from src.ffmpeg import build_fit, builds
    from src.ffmpeg import manager as ffmpeg_manager

    monkeypatch.setattr(builds, "current_platform", lambda *a, **k: plataforma)
    monkeypatch.setattr(build_fit, "current_platform", lambda *a, **k: plataforma)
    monkeypatch.setattr(ffmpeg_manager.platform, "system", lambda: plataforma)

    # Sin binario instalado: una máquina donde aún no se ha bajado FFmpeg.
    monkeypatch.setattr(
        ffmpeg_manager, "get_local_ffmpeg_path",
        lambda: "D:/Draggy-Encoder/bin/ffmpeg/ffmpeg.exe")
    real_exists = os.path.exists

    def sin_ffmpeg(p):
        if "ffmpeg" in str(p).lower() or "ffprobe" in str(p).lower():
            return False
        return real_exists(p)

    monkeypatch.setattr(os.path, "exists", sin_ffmpeg)


def test_the_real_ffmpeg_row_offers_no_install_where_nothing_can_be_installed(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """El PRODUCTOR real, sin monkeypatchearlo, en una plataforma sin mirrors.

    El test anterior construía la fila a mano pasándole
    ``nothing_to_install``, así que probaba el consumidor con un producto
    que le daba el test, no el que da la app. Por eso pasaba en verde con
    el defecto entero presente: ``_ffmpeg_status`` reconstruía el dict a
    mano y se le caía la clave.

    Aquí se llama a ``dependencies.list_all()`` de verdad —con el disco y el
    sistema simulados, pero sin tocar el código que produce la fila— y se
    mira lo que el usuario vería en macOS.
    """
    from src.ui import dependencies_dialog as dd
    from src.ui.dep_row import primary_label

    _simular_maquina_sin_ffmpeg(monkeypatch, "Darwin")

    filas = dd.dependencies.list_all()
    fila = next(f for f in filas if f.get("id") == "ffmpeg")

    # El productor publicó el motivo y el consumidor no lo tira.
    assert fila.get("nothing_to_install"), (
        "el motivo no llegó a la fila: _ffmpeg_status reconstruye el dict "
        "a mano y se come los campos nuevos")

    # El botón no ofrece instalar, y no está vivo.
    assert primary_label(fila) == "No se puede instalar"

    # Y el texto de la fila no manda a pulsar un botón que no funciona.
    detail = fila.get("detail") or ""
    assert "Pulsa 'Install'" not in detail
    assert "no puede descargar FFmpeg" in detail

    # Y la fila montada de verdad: sin botón vivo, sin URL extranjera.
    import src.ui.dependencies_dialog as dd_mod
    from unittest import mock
    from PySide6.QtWidgets import QLabel
    with mock.patch.object(dd_mod.dependencies, "list_all", lambda: [fila]):
        padre = _FakeMainWindow(qapp)
        dialog = dd.DependenciesDialog(padre)
        try:
            dialog._check_timer.stop()
            row = next(r for r in dialog._row_widgets
                       if getattr(r, "dep_id", None) == "ffmpeg")
            assert row.btn_primary.text() == "No se puede instalar"
            assert row.btn_primary.isEnabled() is False
            pintadas = [c.text() for c in row.findChildren(QLabel)
                        if "http" in c.text()]
            assert pintadas == [], f"pinta una URL que no puede usar: {pintadas}"
        finally:
            dialog.close()
            del dialog, padre


def test_the_real_ffmpeg_row_is_untouched_where_ffmpeg_can_be_installed(
        qapp, monkeypatch, fake_preferences, known_hardware):
    """El mismo productor real, en Windows: se ofrece instalar, con su URL.

    El caso de macOS no se puede ejecutar en esta máquina, así que sin este
    test cualquier corrección que se pasara de la raya se vería como el test
    de Darwin en verde.
    """
    from src.ui import dependencies_dialog as dd
    from src.ui.dep_row import primary_label

    _simular_maquina_sin_ffmpeg(monkeypatch, "Windows")

    fila = next(f for f in dd.dependencies.list_all()
                if f.get("id") == "ffmpeg")

    # Con mirrors, la política calla y la fila ofrece instalar.
    assert fila.get("nothing_to_install") is None
    assert primary_label(fila) == "Instalar"
    # El detalle es el de siempre, byte a byte: sí se puede instalar aquí.
    assert (fila.get("detail") or "").startswith(
        "FFmpeg no está instalado todavía. Pulsa 'Install'")

    # Y la fila montada de verdad ofrece el botón vivo. Se monta porque
    # ``primary_enabled`` consulta el soporte del runtime, que es una
    # pregunta al sistema y no sale del dict de la fila.
    import src.ui.dependencies_dialog as dd_mod
    from unittest import mock
    from PySide6.QtWidgets import QLabel
    with mock.patch.object(dd_mod.dependencies, "list_all", lambda: [fila]):
        padre = _FakeMainWindow(qapp)
        dialog = dd.DependenciesDialog(padre)
        try:
            dialog._check_timer.stop()
            row = next(r for r in dialog._row_widgets
                       if getattr(r, "dep_id", None) == "ffmpeg")
            assert row.btn_primary.text() == "Instalar"
            assert row.btn_primary.isEnabled() is True
            # Y no se calla nada: sin mirrors la fila se calla, con mirrors
            # cualquier URL que traiga se sigue pintando. Se comprueba con
            # una fila que sí lleva URL, porque en esta máquina el manifest
            # puede no tenerla y eso no sería un defecto del código.
            con_url = dict(fila)
            con_url["source_url"] = ("https://example.invalid/"
                                     "ffmpeg-master-latest-win64-gpl.zip")
            with mock.patch.object(dd_mod.dependencies, "list_all",
                                   lambda: [con_url]):
                dialog2 = dd.DependenciesDialog(padre)
                try:
                    dialog2._check_timer.stop()
                    row2 = next(r for r in dialog2._row_widgets
                                if getattr(r, "dep_id", None) == "ffmpeg")
                    pintadas = [c.text() for c in row2.findChildren(QLabel)
                                if "http" in c.text()]
                    assert con_url["source_url"] in pintadas
                finally:
                    dialog2.close()
        finally:
            dialog.close()
            del dialog, padre


def test_a_runtime_row_carries_only_row_fields_and_not_manager_internals(
        qapp, fake_preferences):
    """La fila de un runtime no puede arrastrar las claves del sensor.

    ``_runtime_status`` recibe de ``runtimes.manager`` el dict de su SENSOR
    (``installed``, ``system``, ``gpu``), que no es una fila: son claves
    internas. Al pasarlo entero con ``dict(info)`` la fila se las llevaba, y
    un cambio futuro en el manager contaminaría la fila sin que nadie lo
    pidiera. Aquí se fija el conjunto de claves, para que una clave que se
    cuele se vea en el test y no en la máquina del usuario.
    """
    import src.dependencies as deps

    fila = deps._pytorch_status()

    # Exactamente estas, ni una más ni una menos. ``gpu_available`` es el
    # campo de fila equivalente al ``gpu`` del sensor.
    assert set(fila) == {
        "id", "display_name", "kind", "status", "version",
        "install_dir", "source_url", "is_updatable", "size_bytes",
        "detail", "deletable", "local", "gpu_available",
    }

    # Y explícitamente: las tres del sensor no están, ni renombradas ni
    # como estaban.
    for clave_interna in ("installed", "system", "gpu"):
        assert clave_interna not in fila, (
            f"la fila arrastra la clave interna '{clave_interna}' del manager")

    # El valor observable no cambia: el veredicto de GPU sigue llegando,
    # renombrado al nombre que la fila usa.
    assert fila["gpu_available"] in (True, False)
    assert fila["local"] in (True, False)


def test_nothing_is_announced_as_downloadable_where_nothing_can_be_downloaded(
        qapp, monkeypatch, fake_preferences):
    """El banner y el diálogo de reinstalar no pueden seguir mintiendo.

    El selector ya se vaciaba y explicaba el por qué en un sistema sin
    mirrors, pero el banner seguía anunciando "FFmpeg update available:
    Última build (master)" con el botón vivo, y la confirmación de reinstalar
    decía "Se descargará la versión: Última build (master)". Los tres salían
    de preguntas distintas, así que el programa decía la verdad en un sitio y
    mentía en los otros dos: la descarga no puede ocurrir porque no hay
    mirror para nada en esa plataforma.

    Sale de la misma política que el selector (``nothing_to_offer_reason``,
    que es ``list_builds`` vacía), así que los tres tienen que discrepar juntos
    o no pueden discrepar. El camino de descarga, además, ya se negaba solo
    sin tocar la red; aquí se comprueba que ni siquiera se llega a preguntar.
    """
    import builtins
    from unittest import mock

    from src.ffmpeg import builds
    from src.ui import dependencies_dialog as dd
    from src.ui.main_window import MainWindow

    # 1) La política: motivo en Darwin/FreeBSD, nada que decir en Windows.
    assert builds.nothing_to_offer_reason("Windows") is None
    assert builds.nothing_to_offer_reason("Linux") is None
    for plataforma in ("Darwin", "FreeBSD"):
        motivo = builds.nothing_to_offer_reason(plataforma)
        assert motivo is not None
        assert plataforma in motivo
        assert "no puede descargar FFmpeg" in motivo

    # La plataforma entera, como en el test de macOS del selector.
    monkeypatch.setattr(builds, "current_platform", lambda *a, **k: "Darwin")

    ventana = MainWindow("C:/fake/ffmpeg.exe", {"gpus": []}, {})

    # 2) El banner: dice que no hay nada, y el botón deja de ofrecer la
    #    descarga que no puede ocurrir.
    ventana._render_banner_state({
        "update_available": True,
        "last_check_iso": None,
        "latest_fingerprint": {"last_modified_iso": "2026-09-20T10:00:00Z"},
    })
    banner = ventana.update_banner
    titulo = banner.title_label.text()
    detalle = banner.detail_label.text()
    assert "update available" not in titulo.lower()
    assert "no puede descargar" in detalle
    assert "Darwin" in detalle
    # "FFmpeg" y "Darwin" no hunde sus mayúsculas por el camino.
    assert "FFmpeg" in detalle
    assert banner.btn_update.isEnabled() is False

    # 3) Ni siquiera pulsar Update alcanza a preguntar por la descarga.
    with mock.patch.object(builtins, "open", side_effect=AssertionError("red")):
        with mock.patch("src.ui.main_window.QMessageBox.information") as info:
            ventana._on_banner_update_clicked()
    assert info.call_count == 1
    assert "no puede descargar" in info.call_args[0][2]
    # Y no se lanzó ningún worker de descarga.
    assert ventana._banner_update_worker is None

    # 4) La confirmación de reinstalar, en el diálogo real.
    monkeypatch.setattr(
        "src.ui.dependencies_dialog.dependencies.list_all",
        lambda: [_ffmpeg_row()],
    )
    padre = _FakeMainWindow(qapp)
    dialog = dd.DependenciesDialog(padre)
    try:
        dialog._check_timer.stop()
        with mock.patch("src.ui.dependencies_dialog.QMessageBox.warning") as warn:
            with mock.patch(
                "src.ui.dependencies_dialog.QMessageBox.question"
            ) as question:
                dialog._on_redownload_requested("ffmpeg")
        # Avisa de que no se puede, y no ofrece la reinstalación.
        assert warn.call_count == 1
        texto = warn.call_args[0][2]
        assert "No se puede reinstalar" in texto
        assert "no puede descargar" in texto
        assert "Darwin" in texto
        # Y no llegó a preguntar "¿volver a descargar?" ni a lanzar el worker.
        assert question.call_count == 0
        assert dialog._active_worker is None
    finally:
        dialog.close()
        del dialog, padre


def test_the_announcements_about_downloading_are_untouched_where_it_works(
        qapp, fake_preferences):
    """Lo de antes, en Windows: byte a byte, como estaban.

    El motivo de este test es que la Platform-sin-mirrors no se puede probar
    en esta máquina: si el arreglo se pasara de la raya y cambiara también el
    texto del caso normal, aquí se vería, y no en el test de macOS.
    """
    from unittest import mock

    from src.ffmpeg import builds
    from src.ui import dependencies_dialog as dd
    from src.ui.main_window import MainWindow

    # El rótulo de la build elegida, con sus dos avisos, sin tocar.
    assert builds.build_label("latest") == "Última build (master)"
    assert builds.build_label("") == "Última build (master)"
    assert builds.build_label("n8.1") == "FFmpeg 8.1"
    assert builds.build_label("basura") == (
        "Última build (master) (la versión guardada 'basura' no está en el catálogo)"
    )

    ventana = MainWindow("C:/fake/ffmpeg.exe", {"gpus": []}, {})
    ventana._render_banner_state({
        "update_available": True,
        "last_check_iso": None,
        "latest_fingerprint": {"last_modified_iso": "2026-09-20T10:00:00Z"},
    })
    banner = ventana.update_banner
    # Sigue anunciando la actualización, y con el botón vivo.
    assert banner.title_label.text() == (
        "FFmpeg update available: Última build (master)"
    )
    assert banner.detail_label.text() == "upstream build 2026-09-20"
    assert banner.btn_update.isEnabled() is True

    # Y la confirmación de reinstalar sigue siendo la de siempre, con la
    # versión exacta que se va a bajar.
    import src.ui.dependencies_dialog as dd_mod
    with mock.patch.object(
        dd_mod.dependencies, "list_all", lambda: [_ffmpeg_row()]
    ):
        padre = _FakeMainWindow(qapp)
        dialog = dd.DependenciesDialog(padre)
        try:
            dialog._check_timer.stop()
            with mock.patch(
                "src.ui.dependencies_dialog.QMessageBox.warning"
            ) as warn:
                with mock.patch(
                    "src.ui.dependencies_dialog.QMessageBox.question"
                ) as question:
                    dialog._on_redownload_requested("ffmpeg")
            # Nada de avisos: en Windows sí se puede reinstalar.
            assert warn.call_count == 0
            assert question.call_count == 1
            texto = question.call_args[0][2]
            assert "Se descargará la versión: Última build (master)." in texto
            assert "no puede descargar" not in texto
        finally:
            dialog.close()
            del dialog, padre

# ---------------------------------------------------------------------------
# La política de la fila, probada sin widget
# ---------------------------------------------------------------------------
#
# Estas cinco funciones antes eran metodos de un ``QFrame`` y eso
# hacia imposible llamarlas: habia que montar un dialogo entero para preguntar
# "quepondria este boton". Ahora son funciones puras del dict de la fila, y por
# eso los tests de abajo no piden ``qapp`` ni construyen nada: son la prueba de
# que la extraccion sirvio para algo.
#
# Ninguno de ellos necesita el ``None`` que antes hacia falta en el sitio del
# ``self``: eso era la seña de que la funcion no tocaba el widget, y ya no hay
# widget al que pasarle nada.


def test_the_primary_button_says_what_the_row_can_actually_do():
    """``primary_label`` recorre el estado y dice la unica accion posible.

    La tabla esta entera, incluidos los dos finales que no dependen del
    estado: que no se actualice desde este panel, y que aqui no haya nada que
    bajar. El caso que se persigue desde el principio de este hilo es el
    segundo: en una plataforma sin mirrors, la fila dice "No se puede
    instalar" en vez de prometer una descarga que va a fallar.
    """
    from src.ui.dep_row import primary_label

    fila = {"is_updatable": True, "nothing_to_install": None}

    def con(**cambios):
        return {**fila, **cambios}

    casos = [
        # (lo que cambia, lo que tiene que decir)
        ({"status": "missing"}, "Instalar"),
        ({"status": "installed"}, "Reinstalar"),
        ({"status": "installed", "update_available": True}, "Actualizar"),
        # ``update_available`` a False NO es una señal de nada: solo el
        # True Promotion es un Actualizar. Aqui sigue siendo Reinstalar.
        ({"status": "installed", "update_available": False}, "Reinstalar"),
        ({"status": "error"}, "Reintentar"),
        ({"status": "coming_soon"}, "No disponible"),
        # Aqui no hay build de la que bajar: el motivo llega en los datos.
        ({"status": "missing", "nothing_to_install": "sin mirrors para esta plataforma"},
         "No se puede instalar"),
        ({"status": "installed", "nothing_to_install": "sin mirrors para esta plataforma"},
         "No se puede instalar"),
        # Los modelos se gestionan en la pestaña de motores, no aqui.
        ({"is_updatable": False, "status": "installed"}, "No disponible"),
        ({"is_updatable": False, "status": "missing"}, "No disponible"),
    ]
    for cambios, esperado in casos:
        assert primary_label(con(**cambios)) == esperado, cambios

    # Sin nada en la fila tampoco se inventa una accion.
    assert primary_label({}) == "No disponible"


def test_the_primary_button_is_alive_for_exactly_the_three_things_that_kill_it():
    """``primary_enabled`` tiene tres motivos para estar muerto, y solo tres.

    Se separan de ``primary_label`` a proposito: la etiqueta dice que se puede
    hacer, y esta funcion dice si se puede. today coinciden salvo en un caso
    raro, que esta en el ultimo test de aqui.
    """
    from src.ui.dep_row import primary_enabled

    base = {"is_updatable": True, "status": "missing", "nothing_to_install": None}

    def con(**cambios):
        return {**base, **cambios}

    # Vive: se puede instalar, y no hay nada que lo impida.
    assert primary_enabled(con()) is True
    assert primary_enabled(con(status="installed")) is True
    assert primary_enabled(con(status="error")) is True
    # Una actualizacion disponible no desactiva el boton: lo cambia de texto.
    assert primary_enabled(con(status="installed", update_available=True)) is True
    # Ni la deja en False por estar al dia.
    assert primary_enabled(con(status="installed", update_available=False)) is True

    # Muere por motivo 1: esta dependencia no se gestiona desde este panel.
    assert primary_enabled(con(is_updatable=False)) is False
    # Muere por motivo 2: aqui no hay nada bajable.
    assert primary_enabled(con(nothing_to_install="sin mirrors")) is False
    # Muere por motivo 3: este Python no tiene ruedas para el runtime.
    assert primary_enabled(con(kind="runtime", id="torch"),
                           support={"supported": False}) is False
    # Y si el soporte viene bien, el mismo runtime sigue vivo.
    assert primary_enabled(con(kind="runtime", id="torch"),
                           support={"supported": True}) is True


def test_a_button_that_says_it_is_not_available_is_still_clickable():
    """LaCombination rara que conviene tener a la vista: ``coming_soon``.

    ``primary_label`` dice "No disponible" pero ``primary_enabled`` dice que
    si. No es lo que se quiere, y este test lo deja escrito para que no se
    pierda de vista: hoy no hay ninguna fila en coming_soon —las siete
    fabricas del registro la fijan—, pero ``list_all`` usa ese estado como
    valor por defecto, asi que la primera dependencia nueva que se olvide de
    ponerlo nace con un boton vivo que dice que no se puede.
    """
    from src.ui.dep_row import primary_enabled, primary_label

    fila = {"is_updatable": True, "status": "coming_soon"}
    assert primary_label(fila) == "No disponible"
    assert primary_enabled(fila) is True   # <- la contradiccion, a proposito


def test_the_badge_says_the_status_in_words():
    """``badge_text``: los cuatro estados conocidos y lo que no se sabe."""
    from src.ui.dep_row import badge_text

    assert badge_text("installed") == "INSTALLED"
    assert badge_text("missing") == "MISSING"
    assert badge_text("error") == "ERROR"
    assert badge_text("coming_soon") == "COMING SOON"
    # Un estado que no se conoce se enseña tal cual, no inventado.
    assert badge_text("descargando") == "descargando"
    # Y sin estado, un guion: la insignia nunca sale vacia.
    assert badge_text(None) == "—"


def test_humanize_age_writes_integers_and_cuts_where_it_says_it_does():
    """``humanize_age``: el unico sitio donde se decide el formato.

    Enteros, y los tres cortes son los de siempre. Se comprueban los valores
    que rompen una cuenta —justo antes de una hora, justo despues de un dia—
    porque un formato con cortes mal puestos no se ve en los numeros redondos.
    """
    from datetime import datetime, timedelta, timezone
    from src.ui.dep_row import humanize_age

    ahora = datetime.now(timezone.utc)

    def hace(**kw):
        return (ahora - timedelta(**kw)).isoformat()

    # Menos de una hora, en minutos, y sin decimales que inventen precision.
    assert humanize_age(hace(minutes=42)) == "42 min ago"
    assert humanize_age(hace(seconds=90)) == "1 min ago"
    # Justo antes de la hora sigue siendo minutos, no "0.9 h ago".
    assert humanize_age(hace(minutes=59)) == "59 min ago"
    # Justo despues, ya son horas enteras: el corte cae en la hora.
    assert humanize_age(hace(hours=1)) == "1 h ago"
    assert humanize_age(hace(hours=3, minutes=10)) == "3 h ago"
    # Un dia y algo mas: 25 horas son un dia, no "1.0 d ago" ni 1.04.
    assert humanize_age(hace(hours=25)) == "1 d ago"
    # Y 47 horas siguen siendo un dia: truncar es lo que impide que se pinten
    # como "2.0 d ago" media hora antes de tiempo.
    assert humanize_age(hace(hours=47)) == "1 d ago"
    assert humanize_age(hace(hours=48)) == "2 d ago"


def test_humanize_age_says_nothing_when_there_is_no_date_to_measure():
    """Sin fecha, o con una fecha que no se entiende, no se inventa una edad."""
    from src.ui.dep_row import humanize_age

    assert humanize_age(None) is None
    assert humanize_age("") is None
    assert humanize_age("ayer") is None
    # Un reloj atrasado no es una edad negativa: se recorta a cero.
    from datetime import datetime, timedelta, timezone
    futuro = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    assert humanize_age(futuro) == "0 min ago"
    # Y una fecha sin zona horaria se lee como UTC, no como local.
    naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
    assert humanize_age(naive.isoformat()) == "5 min ago"


def test_the_row_and_the_banner_say_the_same_thing_about_the_same_check(
        fake_preferences):
    """El motivo de que la edad tenga un dueño: las dos superficies coinciden.

    La fila y el banner leen el mismo instante —el ``checked_at`` del cache,
    que viaja una vez como ``last_check_iso``— asi que tienen que decirlo con
    las mismas palabras. Antes cada una hacia su cuenta y el usuario leia
    "checked 3.1 h ago" en el banner y "3.1 h ago" en la fila, y ademas el
    banner redondeaba donde la fila truncaba.
    """
    from datetime import datetime, timedelta, timezone
    from src.ui.main_window import MainWindow
    from src.ui.dep_row import format_latest_line, humanize_age

    ahora = datetime.now(timezone.utc)

    for delta, esperado in (
        (timedelta(minutes=42), "42 min ago"),
        (timedelta(hours=3, minutes=10), "3 h ago"),
        (timedelta(hours=47), "1 d ago"),
    ):
        check = (ahora - delta).isoformat()
        # La fila lo pinta en su columna Latest.
        fila = {"id": "ffmpeg",
                "latest_fingerprint": {"last_modified_iso": check,
                                       "checked_at": check}}
        linea = format_latest_line(fila)
        # Y el banner lo pinta en su frase, desde la misma fila real.
        info = {"update_available": True, "preferred_build_id": "n5.1.6",
                "latest_fingerprint": fila["latest_fingerprint"],
                "last_check_iso": check}
        stub = _banner_stub()
        MainWindow._render_banner_state(stub, info)
        _kind, _title, detail = stub.update_banner.calls[-1]

        assert f"last check: {esperado}" in linea, linea
        assert f"checked {esperado}" in detail, detail
        # Y las dos frases son la misma edad, no dos que se parecen.
        assert humanize_age(check) == esperado

    # Sin fecha, la celda pone su guion y el banner se calla: cada superficie
    # decide que hacer con la ausencia, pero ninguna inventa una edad.
    sin_fecha = {"id": "ffmpeg",
                 "latest_fingerprint": {"last_modified_iso": "2026-09-20T00:00:00Z"}}
    assert format_latest_line(sin_fecha).endswith("last check: —")
    stub = _banner_stub()
    MainWindow._render_banner_state(stub, {
        "update_available": True, "preferred_build_id": "n5.1.6",
        "latest_fingerprint": sin_fecha["latest_fingerprint"],
        "last_check_iso": None})
    _kind, _title, detail = stub.update_banner.calls[-1]
    assert "checked" not in detail


def test_one_parse_reads_every_timestamp_the_app_stores():
    """``parse_timestamp`` es el unico que sabe que es una 'Z' y que sin zona es UTC.

    Estas dos reglas vivian en dos sitios —la edad del check y la comparacion
    de ``last_modified``— y lo que se rompe si se separan no es un texto mal
    puesto: es que dos marcas que deberian compararse de una manera se
    comparan al reves segun quien las lea, sin error en ninguna.
    """
    from datetime import datetime
    from src.ffmpeg.builds import parse_timestamp

    con_z = "2026-09-20T10:00:00Z"
    con_offset = "2026-09-20T10:00:00+00:00"
    # La 'Z' y el '+00:00' son la misma marca, no dos horas de diferencia.
    assert parse_timestamp(con_z) == parse_timestamp(con_offset)
    # Y una fecha sin zona se lee como UTC, que es como las escribe el cache.
    assert parse_timestamp("2026-09-20T10:00:00") == parse_timestamp(con_offset)
    assert parse_timestamp(datetime(2026, 9, 20, 10, 0, 0).isoformat()) == \
        parse_timestamp(con_offset)
    # Un offset que no es cero se respeta, no se aplana a UTC.
    assert parse_timestamp("2026-09-20T15:30:00+05:30") == \
        parse_timestamp("2026-09-20T10:00:00+00:00")
    # Y lo que no es una marca se queda sin leer, sin reventar: el que llama
    # promete aguantar un None, no una excepcion.
    for malo in (None, "", "ayer", 12345, {"a": 1}, b"x", True, []):
        assert parse_timestamp(malo) is None


def test_no_timestamp_is_one_sentinel_and_it_is_infinity():
    """Lo que no hay fecha se dice una sola vez, y es ``inf``.

    El cache de 24 h compara la edad con su umbral, y con ``inf`` esa
    comparacion sale ``False`` sola —que es lo que quiere, porque una cache
    sin fecha utilizable esta caducada. Con ``None`` ese consumidor tendria que
    comprobarlo a mano, y ahi es donde aparecen las dos reglas.
    """
    from src.ffmpeg.builds import timestamp_age_hours

    for malo in (None, "", "ayer", 12345, {"a": 1}):
        assert timestamp_age_hours(malo) == float("inf")
    # Una edad real si se compara bien, y una negativa no se recorta aqui:
    # "cuanto" y "como se dice cuanto" son dos preguntas, y la segunda la
    # contesta humanize_age.
    from datetime import datetime, timedelta, timezone
    from src.ffmpeg.builds import parse_timestamp
    hace_una_hora = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert 0.99 < timestamp_age_hours(hace_una_hora) < 1.01
    futuro = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    assert timestamp_age_hours(futuro) < 0
    # Que el recorte ocurra en la presentacion y no aqui es lo que permite
    # que el cache diga "caducada" de una marca del futuro sin mentir sobre
    # la edad que devuelve.
    assert parse_timestamp(futuro) is not None


def test_the_stale_cache_needs_no_check_because_of_the_sentinel(monkeypatch):
    """El consumidor del centinela no comprueba nada, y se comprueba de verdad.

    La misma cache, con la misma URL y sin errores, se reutiliza o no segun si
    su ``checked_at`` se puede leer. Que sea el centinela el que lo decida, y
    no un ``if`` del consumidor, es justo lo que evita las dos reglas: aqui
    no hay ningun ``or`` escrito, y aun asi una cache sin fecha utilizable se
    da por caducada.
    """
    from src.ffmpeg import upstream
    from src.config import UPSTREAM_CHECK_CACHE_HOURS

    class _Resp:
        status_code = 0
        text = "sin red en el test"

    heads = []

    def _fake_head(url):
        heads.append(url)
        return _Resp()

    monkeypatch.setattr(upstream, "read_cache", dict)
    monkeypatch.setattr(upstream, "write_cache", lambda data: True)
    monkeypatch.setattr(upstream, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(upstream, "_head", _fake_head)
    monkeypatch.setattr(upstream, "get_active_url", lambda *a, **k: "https://x/y.zip")

    # 1. Reciente: se devuelve tal cual, sin tocar la red. La fecha se calcula
    #    desde ahora, no fija, o el test empezaria a fallar solo con el reloj.
    from datetime import datetime, timedelta, timezone
    hace_un_minuto = (datetime.now(timezone.utc)
                      - timedelta(minutes=1)).isoformat(timespec="seconds")
    fresco = {"url": "https://x/y.zip", "last_error": None,
              "checked_at": hace_un_minuto}
    monkeypatch.setattr(upstream, "read_cache", lambda: dict(fresco))
    mal = upstream.get_fingerprint()
    assert mal == fresco, "una cache reciente no deberia tocarse"
    assert heads == [], "no deberia haber hecho un HEAD con la cache al dia"

    # 2. Sin fecha utilizable: se da por caducada y sale a preguntar. Mismo
    #    consumidor, misma comparacion, sin una sola comprobacion extra.
    for inservible in (None, "", "ayer", 12345):
        heads.clear()
        monkeypatch.setattr(
            upstream, "read_cache",
            lambda v=inservible: {"url": "https://x/y.zip", "last_error": None,
                                  "checked_at": v})
        saliente = upstream.get_fingerprint()
        assert heads == ["https://x/y.zip"], (
            "con checked_at=%r deberia haber preguntado al upstream" % (inservible,))
        # Preguntar y no poder, es lo que deja el error anotado: prueba de
        # que no se devolvio la cache como si estuviera al dia.
        assert saliente.get("last_error"), (
            "con checked_at=%r no se pudo comprobar nada" % (inservible,))

    # 3. Y el umbral sigue siendo el de siempre, no uno nuevo puesto aqui.
    assert UPSTREAM_CHECK_CACHE_HOURS > 0


def test_the_runtime_support_question_is_only_asked_of_the_runtime_rows(monkeypatch):
    """``ai_runtime_support_info`` solo le pregunta al sistema por runtimes.

    Las filas de FFmpeg y de modelos no tienen nada que ver con las ruedas de
    Python, así que no se pregunta. Para no depender de la máquina donde
    corra el test, el "¿este Python tiene ruedas?" se sustituye aqui: lo que
    se comprueba es a quien se le pregunta, no lo que responde el Python.
    """
    from src.ui import dep_row

    llamadas = []

    def_support = {"supported": True, "message": "ok"}
    monkeypatch.setattr(
        dep_row, "_get_python_support_info",
        lambda: (llamadas.append(1), def_support)[1])

    assert dep_row.ai_runtime_support_info({"kind": "binary", "id": "ffmpeg"}) is None
    assert dep_row.ai_runtime_support_info({"kind": "model", "id": "upscale_models"}) is None
    assert llamadas == [], "preguntó por unos tipos de fila que no son runtimes"

    # Este es el caso que de verdad distingue una guarda de la otra: una fila
    # que NO es runtime pero cuyo id SÍ es el de un runtime. Con la guarda de
    # ``kind`` no se pregunta nada; sin ella, aquí se le preguntaría al
    # sistema por unas ruedas de Python que esta fila no puede usar. Con solo
    # comprobar filas de otros ids, este hueco pasaría inadvertido: el ``None``
    # de arriba lo daba el ``id``, no el ``kind``.
    assert dep_row.ai_runtime_support_info({"kind": "binary", "id": "torch"}) is None
    assert llamadas == [], "una fila que no es runtime llegó a preguntar por ruedas"

    assert dep_row.ai_runtime_support_info({"kind": "runtime", "id": "torch"}) == def_support
    assert dep_row.ai_runtime_support_info({"kind": "runtime", "id": "onnxruntime"}) == def_support
    assert len(llamadas) == 2

    # Un runtime que no es ninguno de los dos conocidos no se pregunta.
    assert dep_row.ai_runtime_support_info({"kind": "runtime", "id": "otro"}) is None
    assert len(llamadas) == 2


def test_the_latest_line_says_the_date_and_how_old_the_check_is():
    """``format_latest_line``: sin cache pone que nunca se ha comprobado.

    Los tres tramos de edad —minutos, horas, días— se comprueban con horas
    relativas a ahora, no con fechas fijas: si el test dijera "3 d ago"
    hardcodeado, empezaria a fallar solo por el paso del tiempo, y un test
    que se rompe solo no informa de nada.
    """
    from datetime import datetime, timedelta, timezone

    from src.ui.dep_row import format_latest_line

    # Sin cache y sin error: no se ha comprobado nunca.
    assert format_latest_line({}) == "Latest: — · last check: never"
    # Sin cache pero con error de red: se dice que fue offline, no que no hay
    # dato, que son dos cosas distintas.
    assert format_latest_line({"upstream_error": "timeout"}) == (
        "Latest: — (offline) · last check: never")
    # Un upstream_error sin fecha no inventa fecha.
    assert format_latest_line({"upstream_error": "timeout",
                               "latest_fingerprint": {"checked_at": None}}) == (
        "Latest: — (offline) · last check: never")

    def linea_de(hace):
        ahora = datetime.now(timezone.utc)
        return format_latest_line({"latest_fingerprint": {
            "last_modified_iso": "2026-09-20T10:00:00+00:00",
            "checked_at": (ahora - timedelta(hours=hace)).isoformat(),
        }})

    # Solo la fecha, no la hora entera: la linea no debe saturarse.
    assert linea_de(0.5).startswith("Latest upstream build: 2026-09-20")
    assert "30 min ago" in linea_de(0.5)
    # Las edades van enteras, y es el formato compartido que dice
    # ``humanize_age`` — el mismo que pinta el banner. Antes esta linea sacaba
    # "5.0 h ago" y el banner "checked 5 h ago" para el mismo instante.
    assert "5 h ago" in linea_de(5)
    assert "3 d ago" in linea_de(72)

    # Una fecha de comprobacion ilegible no rompe la linea: se queda sin edad.
    rota = format_latest_line({"latest_fingerprint": {
        "last_modified_iso": "2026-09-20T10:00:00+00:00",
        "checked_at": "no-es-una-fecha",
    }})
    assert rota == "Latest upstream build: 2026-09-20  ·  last check: —"

# ---------------------------------------------------------------------------
# El contrato de la fila
# ---------------------------------------------------------------------------


# Quien lee una fila. Se declara aqui, en un sitio, porque el contrato tiene
# que saber el alcance de "la fila": una fila no la lee solo la fila, la leen
# cinco sitios, y si la lista de consumidores se queda corta el contrato
# declara que esta entero cuando no lo esta.
#
# (modulo, funcion que recibe la fila —None para todo el modulo—, variables
#  que ahi contienen una fila)
#
# El ambito de funcion no es un detalle: ``main_window`` usa la variable
# ``info`` para media docena de cosas que NO son filas (el resultado de
# ffprobe, los datos del encoder). Sin acotar a la funcion que pinta el
# banner, el contrato se llenaria de campos que no son de fila y dejaria de
# significar nada.
CONSUMIDORES_DE_FILA = (
    ("src/ui/dep_row.py", None, ("info",)),
    ("src/ui/ffmpeg_row.py", None, ("info",)),
    ("src/ui/dependencies_dialog.py", None, ("info",)),
    # Esta no pasa por list_all: lee el dict crudo de un status_fn, pero es
    # una fila en todo lo que importa, y por eso tambien la vigila el contrato.
    ("src/ui/update_chain.py", None, ("info",)),
    ("src/ui/main_window.py", "_render_banner_state", ("info",)),
    # El propio registro tambien consume filas: ``pending_updatable_dependencies``
    # llama a cada status_fn y lee de su dict. Y necesita ambito de funcion por
    # el motivo opuesto al de main_window: en ``_runtime_status`` la variable
    # ``info`` es el SENSOR del manager (installed, system, gpu), no una fila.
    ("src/dependencies.py", "pending_updatable_dependencies", ("info",)),
)

# El que PRODUCE las filas no es un consumidor de ellas, asi que la guarda de
# abajo lo salta por el nombre y no por un caso especial en el codigo.
MODULO_PRODUCTOR = "src/dependencies.py"


def _campos_que_lee_una_fila() -> set:
    """Claves que los consumidores declarados piden de una fila.

    Se saca leyendo el AST de cada consumidor, NO de una lista escrita aqui.
    Eso es lo que hace que el test no se_limits a repetirse a si mismo: si
    manana alguien empieza a leer ``info.get("nuevo_campo")`` en cualquiera de
    los cinco, este conjunto lo recoge solo.
    """
    import ast
    from pathlib import Path

    raiz = Path(__file__).resolve().parent.parent

    def clave(nodo):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            return nodo.value
        return None

    def base(nodo):
        if isinstance(nodo, ast.Name):
            return nodo.id
        if isinstance(nodo, ast.Attribute) and isinstance(nodo.value, ast.Name):
            return nodo.value.id
        return None

    def recoge(arbol, variables):
        out = set()
        for n in ast.walk(arbol):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and n.args
                    and base(n.func.value) in variables and clave(n.args[0])):
                out.add(clave(n.args[0]))
            if (isinstance(n, ast.Subscript) and base(n.value) in variables
                    and clave(n.slice)):
                out.add(clave(n.slice))
            if (isinstance(n, ast.Compare) and len(n.ops) == 1
                    and isinstance(n.ops[0], ast.In) and n.comparators
                    and base(n.comparators[0]) in variables and clave(n.left)):
                out.add(clave(n.left))
        return out

    leidos = set()
    for modulo, funcion, variables in CONSUMIDORES_DE_FILA:
        arbol = ast.parse((raiz / modulo).read_text(encoding="utf-8"))
        if funcion is not None:
            elegido = [n for n in ast.walk(arbol)
                       if isinstance(n, ast.FunctionDef) and n.name == funcion]
            if not elegido:
                sys.exit("ABORTA: %s ya no tiene la funcion %s" % (modulo, funcion))
            arbol = elegido[0]
            # Los parametros de la funcion son las variables de la fila.
            variables = {a.arg for a in elegido[0].args.args}
        leidos |= recoge(arbol, set(variables))
    return leidos


def _consumidores_sin_declarar(raiz) -> set:
    """Modulos que reciben filas y no estan en la lista de arriba.

    Se deriva de quien llama a las dos funciones que producen filas. Asi, un
    sexto consumidor que se cuele sin declararse aqui no se pierde en
    silencio: esto lo nombra.
    """
    import ast

    declarados = {m for m, _f, _v in CONSUMIDORES_DE_FILA}
    ausentes = set()
    for ruta in (raiz / "src").rglob("*.py"):
        relativo = ruta.relative_to(raiz).as_posix()
        if relativo == MODULO_PRODUCTOR or relativo in declarados:
            continue
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        for n in ast.walk(arbol):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            nombre = (f.attr if isinstance(f, ast.Attribute)
                      else f.id if isinstance(f, ast.Name) else None)
            if nombre in ("list_all", "pending_updatable_dependencies"):
                ausentes.add(relativo)
                break
    return ausentes


@pytest.fixture(scope="module")
def filas_reales():
    """Una sola foto de las filas para los dos tests del contrato.

    ``list_all()`` sondea el disco de verdad —el binario de FFmpeg, el store de
    runtimes, las cuatro carpetas de modelos— y cuesta mas que todo el resto
    de estos tests juntos. Dos llamadas eran dos sondeos para mirar lo mismo.
    """
    import src.dependencies as deps

    return {f["id"]: f for f in deps.list_all()}


@pytest.fixture(scope="module")
def campos_leidos():
    """El conjunto de campos que leen los consumidores, recorrido una vez."""
    return _campos_que_lee_una_fila()


def test_a_real_row_carries_everything_the_row_reads(filas_reales, campos_leidos):
    """El contrato, ejecutado: lo que la fila lee sale de su codigo.

    Tres cosas, y las tres tienen que cumplirse a la vez:

    1. Todo lo que la fila lee esta en ``ROW_FIELDS``. Si la fila empieza a
       leer un campo que nadie ha declarado, el contrato esta roto y aqui se
       ve — que es justo lo que no pasaba antes, cuando la lista vivia en
       prosa y nadie la contrastaba con nada.
    2. Una ``list_all()`` de verdad trae todos los campos que esta capa
       garantiza. No se comprueba la constante contra si misma: se comprueba
       contra filas de verdad, con el disco y los managers de verdad.
    3. Los campos que la fila lee y que pone otra capa son los que estan
       declarados como tales. Ni uno de mas, ni uno de menos.
    """
    import src.dependencies as deps

    from pathlib import Path
    raiz = Path(__file__).resolve().parent.parent

    sin_declarar = _consumidores_sin_declarar(raiz)
    assert not sin_declarar, (
        f"{sorted(sin_declarar)} reciben filas y no estan en "
        "CONSUMIDORES_DE_FILA: anadelos, o el contrato miente sobre su alcance")

    leidos = campos_leidos
    # Guarda contra un extractor roto: si no lee nada, el test pasaria solo.
    assert len(leidos) >= 10, (
        f"el extractor solo ha visto {len(leidos)} campos: se ha roto y este "
        "test pasaria sin comprobar nada")

    declarados = {campo: dueno for campo, _, dueno in deps.ROW_FIELDS}

    # 1. Lo que la fila lee, declarado.
    sin_declarar = sorted(leidos - set(declarados))
    assert not sin_declarar, (
        f"la fila lee {sin_declarar} y no estan en ROW_FIELDS: anadelos a la "
        "tabla de src/dependencies.py, con su dueno")

    # 2. Una fila de verdad trae lo que esta capa garantiza.
    garantizados = [c for c, _, d in deps.ROW_FIELDS
                    if d in (deps.DUENO_LIST_ALL, deps.DUENO_REGISTRY)]
    filas = list(filas_reales.values())
    assert filas, "list_all() no devolvio ninguna fila"
    filas_f = filas_reales
    for f in filas:
        faltan = [c for c in garantizados if c not in f]
        assert not faltan, (
            f"la fila de {f.get('id')!r} no trae {faltan}, y esta capa dice "
            f"que los garantiza (de {len(garantizados)})")

    # 3. Lo que la fila lee y no llega a todas las filas, es lo declarado
    #    "fuera". Y se contrasta contra las filas REALES, no contra la tabla:
    #    lo que solo trae FFmpeg se saca de ahi, no de lo declarado. Asi el
    #    reparto entre "lo garantiza esta capa" y "lo trae otra" no puede
    #   _falsearse_ en ninguna de las dos direcciones — degradar un campo de
    #    "otra capa" a "esta capa" tambien lo peta, y eso antes pasaba.
    en_todas = set.intersection(*(set(f) for f in filas))
    solo_ffmpeg = set(filas_f["ffmpeg"]) - en_todas
    declarados_fuera = {c for c, d in declarados.items() if d == deps.DUENO_FUERA}
    esperados_fuera = (leidos & solo_ffmpeg) | {"build_options"}

    fuera_de_su_capa = sorted(esperados_fuera - declarados_fuera)
    assert not fuera_de_su_capa, (
        f"{fuera_de_su_capa} los lee un consumidor y no llegan a todas las "
        "filas, pero la tabla no los declara como de otra capa: anadelos a "
        "ROW_FIELDS con DUENO_FUERA, o deja de leerlos")

    sin_usar = sorted(declarados_fuera - esperados_fuera)
    assert not sin_usar, (
        f"la tabla declara {sin_usar} como 'otra capa' pero llegan a todas "
        "las filas: Promoteilos a garantizados, o no los declares")

    # Limite conocido de esta comprobacion, y a proposito: degradar un campo
    # de "otra capa" a "esta capa" (por ejemplo, declarar build_label como
    # garantizado) NO lo pete. Lo que haria es que list_all escribiera esa
    # clave en filas donde antes no estaba, y como la fila la consulta con
    # ``.get`` y solo la pinta si es truthy, no se ve nada. Se podria cazar
    # fijando los ocho nombres, pero eso seria exactamente lo que este test
    # no debe ser: una constante comprobandose contra si misma. Se deja
    # dicho aqui para que quien lo lea sepa donde acaba el contrato.


def test_the_fields_of_other_layers_arrive_where_they_are_declared_to(
        filas_reales):
    """Los campos que no pone ``list_all`` llegan, y llegan de donde dicen.

    Los siete de ``ffmpeg.manager`` viajan en la fila de FFmpeg y SOLO en esa.
    ``build_options`` es el caso raro: no lo pone nadie de este lado, lo anade
    el dialogo, asi que no debe estar en ninguna fila de ``list_all`` — si lo
    estuviera, el dialogo montaria el selector de build en torch y en las
    ranuras de modelos, que es justo lo que el ``in`` de ahi decide.
    """
    import src.dependencies as deps

    filas = filas_reales
    ffmpeg = filas["ffmpeg"]
    declarados = {c: d for c, _, d in deps.ROW_FIELDS}

    del_gestor = [c for c, d in declarados.items()
                  if d == deps.DUENO_FUERA and c != "build_options"]

    assert del_gestor, "la tabla ya no declara ningun campo del gestor"
    for campo in del_gestor:
        assert campo in ffmpeg, (
            f"{campo} lo lee la fila de FFmpeg pero la fila real no lo trae")
    for dep_id, f in filas.items():
        if dep_id == "ffmpeg":
            continue
        sobran = [c for c in del_gestor if c in f]
        assert not sobran, (
            f"{dep_id} trae {sobran}, que son campos de la fila de FFmpeg")

    # Y el que anade el dialogo no puede estar en ninguna.
    assert "build_options" in declarados
    assert declarados["build_options"] == deps.DUENO_FUERA
    for dep_id, f in filas.items():
        assert "build_options" not in f, (
            f"list_all le esta dando build_options a {dep_id}: el dialogo "
            "montaria el selector de build ahi")

    # El agujero que cierra esto: si un campo garantizado se degrada a "otra
    # capa" y su fabrica deja de devolverlo, las dos comprobaciones de arriba
    # se lo comen: la tabla lo declara, y el gestor lo comprueba en la fila
    # de FFmpeg, que resultaria ser la unica que lo trae. Por eso la fila de
    # FFmpeg tiene que traer ABSOLUTAMENTE todo lo declarado salvo el campo del
    # dialogo. Un campo que caiga por el camino, salta aqui.
    ausentes = [c for c in declarados if c != "build_options" and c not in ffmpeg]
    assert not ausentes, (
        f"la fila de FFmpeg no trae {ausentes}, y la tabla los declara: o "
        "vuelve el manager a publicarlos, o se quitan de la tabla")

def test_the_naive_is_utc_rule_is_one_function_with_three_entrances():
    """"Sin zona es UTC" no estaba en tres sitios: estaba en uno, tres veces.

    Las tres copias eran el mismo ``if``, pegado. Ahora los tres caminos entran
    por el mismo ``ensure_utc``, y que sea el mismo objeto y no tres copias
    con el mismo cuerpo se comprueba con ``is``: si alguien vuelve a pegarlo
    dentro de ``upstream``, esta comparacion lo dice sola.

    Y las tres entradas de verdad normalizan. Una cabecera RFC 1123 sin zona
    tambien sale con zona, porque ``parsedate_to_datetime`` la devuelve
    ingenua: ahi se llega, no es una rama de adorno.
    """
    from datetime import datetime as real_datetime, timedelta, timezone
    from src.ffmpeg import builds, upstream

    # El mismo objeto, no una copia con el mismo cuerpo.
    assert upstream.ensure_utc is builds.ensure_utc

    sin_zona = real_datetime(2026, 9, 20, 10, 0)
    en_utc = sin_zona.replace(tzinfo=timezone.utc)
    assert builds.parse_timestamp("2026-09-20T10:00:00") == en_utc
    assert upstream._parse_http_date("Thu, 20 Sep 2026 10:00:00") == en_utc
    assert upstream._iso(sin_zona) == "2026-09-20T10:00:00+00:00"

    # Y con zona no se toca: un offset que no es cero se respeta en los tres,
    # porque aplanar a UTC seria mentir sobre la hora del texto.
    mas_tarde = real_datetime(2026, 9, 20, 15, 30,
                              tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert builds.parse_timestamp("2026-09-20T15:30:00+05:30") == mas_tarde
    assert upstream._parse_http_date("Thu, 20 Sep 2026 15:30:00 +0530") == mas_tarde
    assert upstream._iso(mas_tarde) == "2026-09-20T15:30:00+05:30"


def test_the_z_is_not_for_the_marks_this_app_writes_and_is_not_decoration():
    """La ``Z`` se mide en vez de suponerla, en las dos direcciones.

    Lo que esta app guarda sale de ``.isoformat()``, que escribe ``+00:00``, y
    la cabecera del upstream entra por ``parsedate_to_datetime`` y se vuelve a
    serializar con ``+00:00``. O sea que ninguna marca de la app trae ``Z``, y
    la justificacion de la rama no puede ser "la que llega de las cabeceras",
    porque esa cabecera nunca llega asi. La rama esta por el rango de Python
    -- en 3.10 ``fromisoformat`` rechaza la ``Z`` -- y por los textos que no
    escribio esta app.

    Y se mira en las dos direcciones, porque una rama que no hace nada es
    indistinguible de una que no hace falta.
    """
    from datetime import datetime as real_datetime, timezone
    from src.ffmpeg import builds

    con_z = "2026-09-20T10:00:00Z"
    con_offset = "2026-09-20T10:00:00+00:00"
    esperado = real_datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)

    # En la Python que corre, la rama no cambia nada: la 'Z' ya se sabe.
    assert builds.parse_timestamp(con_z) == builds.parse_timestamp(con_offset)

    # Y lo que la app escribe de verdad nunca trae 'Z', ni que se lo busque.
    assert "Z" not in real_datetime.now(timezone.utc).isoformat()

    # En una 3.10, sin la rama el texto se pierde, y perderlo no es inocuo:
    # None en la edad significa cache caducada. La 3.10 se simula sombreando
    # el nombre `datetime` dentro del modulo, porque datetime.datetime es
    # inmutable y al stdlib no se le hace monkeypatch.
    class _Sin_Z:
        @staticmethod
        def fromisoformat(text):
            if text.endswith("Z"):
                raise ValueError("Invalid isoformat string")
            return real_datetime.fromisoformat(text)

        @staticmethod
        def now(*args, **kwargs):
            return real_datetime.now(*args, **kwargs)

    original = builds.datetime
    try:
        builds.datetime = _Sin_Z
        # Con la rama, el mismo texto da la misma marca que en 3.14.
        assert builds.parse_timestamp(con_z) == esperado
        # Y lo que escribe la app, ni en 3.10 necesita la rama.
        assert builds.parse_timestamp(con_offset) == esperado
        assert builds.parse_timestamp(real_datetime.now(
            timezone.utc).isoformat()) is not None
    finally:
        builds.datetime = original


# ==----------------------------------------------------------------------
# La version que se LEE: "9.x-dev (master)" en vez de "N-126905-..."
# ==----------------------------------------------------------------------
#
# Antes, la fila de FFmpeg decia solo el token que imprime el binario. En una
# nightly de master ese token es un `git describe` ("N-126905-gb87602a63a-2026
# 0927"), que identifica la compilacion pero no dice la version: el usuario
# no tenia forma de saber si su FFmpeg era el 9, el 10 o el 11.
#
# Ahora viajan DOS numeros en `get_install_info`, y son cosas distintas:
# `version` es el token tal cual (sirve para comparar, para el log y para
# deduplicar compilaciones) y `version_text` es lo que se pinta. La fila
# ensena los dos, y solo anade el token cuando de verdad aporta algo.


def _version_block(token, libavutil="61.  8.100"):
    """El bloque que devuelve `ffmpeg -version`, con los numeros alineados.

    ffmpeg imprime los numeros de las libs con los puntos separados por
    espacios ("libavutil      61.  8.100"), no "61.8.100". El texto de aqui
    es el real, para que el lector se pruebe con el espacio de verdad.
    """
    return (
        "ffmpeg version %s\n"
        "built with gcc 15.1.0 (GCC)\n"
        "configuration: --enable-gpl --enable-version3\n"
        "libavutil      %s\n"
        "libavcodec     63. 14.100\n"
        "libavformat     63.  7.100\n" % (token, libavutil)
    )


def test_una_release_dice_su_propio_numero_y_no_hay_que_adivinarlo():
    """Un binario de release ya dice cual es; interpretarlo seria inventar.

    Ni el major de libavutil ni la tabla de `builds` tienen nada que hacer
    ahi: la respuesta es el token, sin tocarlo.
    """
    from src.ffmpeg import builds, manager

    texto = _version_block("9.0.2", "61.  1.102")
    info = manager.parse_ffmpeg_build(texto)
    assert info == {"version": "9.0.2", "libavutil_major": 61}
    assert builds.descripcion_version(info) == "9.0.2"


def test_una_nightly_dice_la_linea_que_lee_una_persona():
    """El caso que motiva el cambio: `N-126905-...` con libavutil 61 es 9.x.

    El major de libavutil es lo que de verdad sitúa una compizacion en una
    linea de FFmpeg; el token no lo dice. Lo que sale es la linea, mas la
    marca de que es master, que es como se distingue de una release.
    """
    from src.ffmpeg import builds, manager

    texto = _version_block("N-126905-gb87602a63a-20260927")
    info = manager.parse_ffmpeg_build(texto)
    assert info["libavutil_major"] == 61
    assert builds.descripcion_version(info) == "9.x-dev (master)"


def test_una_nightly_de_la_linea_anterior_suene_a_8():
    """La tabla no es una constante de FFmpeg 9: master retrocede y avanza.

    release/8.1 va con libavutil 60, asi que la misma forma de token da otra
    linea distinta. Es la misma regla aplicada al major que traiga el binario.
    """
    from src.ffmpeg import builds, manager

    texto = _version_block("N-113402-g8e7c1f2a3b-20250310", "60. 26.103")
    info = manager.parse_ffmpeg_build(texto)
    assert info["libavutil_major"] == 60
    assert builds.descripcion_version(info) == "8.x-dev (master)"


def test_un_major_que_nadie_ha_comprobado_no_se_inventa():
    """Si BtbN sube master a libavutil 62, aqui no hay respuesta y se dice.

    La regla acordada: un major desconocido devuelve el token tal cual. No se
    escribe "62.x-dev" a ojo, porque si esa linea no existe el programa
    habria afirmado una version que no existe. El token sigue siendo
    informacion real, asi que no se pierde nada.
    """
    from src.ffmpeg import builds, manager

    texto = _version_block("N-999999-gcafe", "62.  1.100")
    info = manager.parse_ffmpeg_build(texto)
    assert info["libavutil_major"] == 62
    assert 62 not in builds._LINEA_POR_LIBAVUTIL

    texto = builds.descripcion_version(info)
    assert texto == "N-999999-gcafe"
    assert "62" not in texto
    assert "dev" not in texto


def test_una_salida_vacia_o_ilegible_no_dice_nada():
    """Sin binario, o con una salida que no es de ffmpeg, `None` y punto.

    `None` no es un caso raro: es lo que pasa cuando el archivo falta o cuando
    el proceso no arranca. Quien llama tiene su propio respaldo (el manifest),
    y por eso esta funcion devuelve `None` en vez de adivinar.
    """
    from src.ffmpeg import builds, manager

    assert manager.parse_ffmpeg_build("") is None
    assert manager.parse_ffmpeg_build(None) is None
    assert manager.parse_ffmpeg_build("basura que no es la salida de ffmpeg") is None
    # Y del lado de la descripcion: sin dict, o con dict sin version.
    assert builds.descripcion_version(None) is None
    assert builds.descripcion_version({}) is None
    assert builds.descripcion_version({"version": "", "libavutil_major": 61}) is None
    assert builds.descripcion_version("no soy un dict") is None


def test_el_espaciado_de_ffmpeg_no_engana_al_lector():
    """ffmpeg alinea los numeros de las libs; el regex tiene que notarlo.

    "libavutil      61.  8.100" y "libavutil 61.8.100" son el mismo numero.
    Sin tolerar los espacios detras de cada punto, el segundo se leia como
    major 8 (o directamente no se leia), y la fila diria 8.x-dev para un
    FFmpeg que es de la linea 9.
    """
    from src.ffmpeg import manager

    for texto in (
        "ffmpeg version N-1-gabc\nlibavutil      61.  8.100\n",
        "ffmpeg version N-1-gabc\nlibavutil 61.8.100\n",
        "ffmpeg version N-1-gabc\nlibavutil 61. 8.100\n",
    ):
        assert manager.parse_ffmpeg_build(texto)["libavutil_major"] == 61, texto


def test_una_sola_vuelta_al_binario_da_los_dos_numeros(
        fake_preferences, tmp_path, monkeypatch):
    """Leer version y linea cuesta UN `-version`, no dos procesos.

    Antes se llamaba a `get_ffmpeg_version` y ahora hace falta tambien el
    major; si eso se resuelto con una segunda vuelta al binario, cada vez que
    se abria el panel habria lanzado ffmpeg dos veces. Aqui se cuenta.
    """
    from src.ffmpeg import manager

    binario = tmp_path / "ffmpeg"
    binario.write_text("no se ejecuta de verdad", encoding="utf-8")
    llamadas = []

    class _Resultado:
        stdout = _version_block("N-126905-gb87602a63a-20260927")
        stderr = ""

    def _run(args, *resto, **kwargs):
        llamadas.append(list(args))
        return _Resultado()

    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: str(binario))
    monkeypatch.setattr(manager.subprocess, "run", _run)
    monkeypatch.setattr(manager, "read_manifest", lambda: {"build_id": "latest"})

    info = manager.get_install_info()
    assert llamadas == [[str(binario), "-version"]], llamadas
    assert info["version"] == "N-126905-gb87602a63a-20260927"
    assert info["version_text"] == "9.x-dev (master)"


def test_get_ffmpeg_version_sigue_dando_el_token_que_siempre_dio(
        fake_preferences, tmp_path, monkeypatch):
    """La firma no se toca porque `update_chain` y el manifest la usan.

    Lo que se anadio al lado no puede cambiar lo que yahabia: quien llama
    esperando el token tal cual tiene que seguir recibiendolo.
    """
    from src.ffmpeg import manager

    binario = tmp_path / "ffmpeg"
    binario.write_text("x", encoding="utf-8")
    llamadas = []

    class _Resultado:
        stdout = _version_block("N-126905-gb87602a63a-20260927")
        stderr = ""

    def _run(args, *resto, **kwargs):
        llamadas.append(list(args))
        return _Resultado()

    monkeypatch.setattr(manager.subprocess, "run", _run)
    assert manager.get_ffmpeg_version(str(binario)) == "N-126905-gb87602a63a-20260927"
    # Y sigue siendo una capa fina: no vuelve a lanzar el proceso.
    assert llamadas == [[str(binario), "-version"]], llamadas

    # Sin binario, `None` como siempre.
    assert manager.get_ffmpeg_version(str(tmp_path / "no-existe")) is None


def test_sin_binario_no_se_inventa_una_linea_de_ffmpeg(fake_preferences, monkeypatch):
    """Si el binario no esta, manda lo que guardo el manifest, y nada mas.

    Un manifest viejo con "8.1" es una release: se devuelve tal cual, sin
    colarsele un "-dev (master)" que nadie ha comprobado.
    """
    from src.ffmpeg import manager

    monkeypatch.setattr(manager, "get_local_ffmpeg_path", lambda: None)
    monkeypatch.setattr(manager, "read_manifest",
                        lambda: {"build_id": "latest", "version": "8.1"})
    info = manager.get_install_info()
    assert info["status"] == "missing"
    assert info["version_text"] == "8.1"
    assert "dev" not in info["version_text"]

    # Y si el manifest tampoco sabe nada, "unknown" y se acabó.
    monkeypatch.setattr(manager, "read_manifest", lambda: {"build_id": "latest"})
    assert manager.get_install_info()["version_text"] == "unknown"


def test_el_ffmpeg_real_dice_a_que_linea_pertenece_sin_depender_del_dia():
    """Contra el binario de `bin/`, pero sin clavarse a la linea de hoy.

    Este test es el que se rompe solo si el cambio esta mal. Se comprueba la
    FORMA de la respuesta, nunca el numero: cuando BtbN suba master a
    libavutil 62 (o lo que sea dentro de dos meses) este test tiene que
    seguir en verde, porque lo unico que se afirma es que la linea que dice la
    tabla es la que el binario declara de verdad -- o, si ya no la declara
    ninguna, que sale el token sin inventar.
    """
    import re

    from src.ffmpeg import builds, manager

    ruta = manager.get_local_ffmpeg_path()
    if not (ruta and os.path.exists(ruta)):
        pytest.skip("No hay FFmpeg local en bin/ en este checkout")

    info = manager.get_ffmpeg_build_info(ruta)
    assert info is not None, "el binario de bin/ no responde a -version"
    assert info["version"]
    assert isinstance(info["libavutil_major"], int), info
    texto = builds.descripcion_version(info)

    if info["version"][0].isdigit():
        # Release: el token es la respuesta y no hay nada que interpretar.
        assert texto == info["version"]
        return

    # Nightly. La forma tiene que ser una linea, nunca un numero suelto.
    assert re.match(r"^(\d+\.x-dev \(master\)|N-\d+)", texto), texto

    if info["libavutil_major"] in builds._LINEA_POR_LIBAVUTIL:
        assert texto == "%s-dev (master)" % builds._LINEA_POR_LIBAVUTIL[
            info["libavutil_major"]]
    else:
        # Major fuera de la tabla: el token, y ningun numero de linea.
        assert texto == info["version"]


def test_la_fila_dice_la_version_gorda_y_el_token_de_compilacion(qapp):
    """Lo que el usuario no podia leer, en la fila de verdad.

    Las dos mitades son datos ciertos y distintos: una dice en que linea esta
    el FFmpeg y la otra WHICH compilacion es exactamente. Se ven las dos.
    """
    from PySide6.QtWidgets import QLabel

    from src.ui.dep_row import DepRow

    fila = DepRow(_ffmpeg_row(
        version="N-126905-gb87602a63a-20260927",
        version_text="9.x-dev (master)",
        build_label="FFmpeg (master, nightly)",
    ))
    try:
        meta = next(c.text() for c in fila.findChildren(QLabel)
                    if c.text().startswith("BINARY"))
        assert "version: 9.x-dev (master)" in meta
        assert "build: N-126905-gb87602a63a-20260927" in meta
        assert "selected: FFmpeg (master, nightly)" in meta
    finally:
        fila.deleteLater()


def test_una_release_no_repite_el_token_dos_veces_porque_es_lo_mismo(qapp):
    """Con una release, token y linea coinciden: se dice una vez.

    Poner "9.0.2" y despues "build: 9.0.2" es ruido, y hace creer que hay
    dos datos donde hay uno. El campo `build` solo aparece cuando aporta.
    """
    from PySide6.QtWidgets import QLabel

    from src.ui.dep_row import DepRow

    fila = DepRow(_ffmpeg_row(
        version="9.0.2", version_text="9.0.2", build_label="FFmpeg 9.0.2",
    ))
    try:
        meta = next(c.text() for c in fila.findChildren(QLabel)
                    if c.text().startswith("BINARY"))
        assert "version: 9.0.2" in meta
        assert "  \u00b7  build:" not in meta
        assert "selected: FFmpeg 9.0.2" in meta
    finally:
        fila.deleteLater()


def test_las_filas_que_no_son_ffmpeg_dicen_su_version_tal_cual(qapp):
    """PyTorch y ONNX no se han tocado, y esto lo fija.

    La logica nueva es solo de FFmpeg: esas versiones son suyas, sin linea ni
    token de compilacion que interpretar. Si alguien las pasara por aqui, el
    "-dev (master)" se les colaria donde no significa nada.
    """
    from PySide6.QtWidgets import QLabel

    from src.ui.dep_row import DepRow

    for dep_id, version in (("torch", "2.14.0+cu130"), ("onnxruntime", "1.30.0")):
        fila = DepRow({
            "id": dep_id, "display_name": dep_id, "kind": "runtime",
            "status": "installed", "version": version, "install_dir": "/tmp/x",
            "source_url": None, "detail": "", "is_updatable": False,
            "deletable": False, "update_available": None,
        })
        try:
            meta = next(c.text() for c in fila.findChildren(QLabel)
                        if c.text().startswith("RUNTIME"))
            assert "version: %s" % version in meta
            assert "dev (master)" not in meta
            assert "selected:" not in meta
        finally:
            fila.deleteLater()
