"""El selector de versión dentro de la fila de FFmpeg.

Existe porque la fila de FFmpeg es la única que deja elegir *qué* build se
descarga: el resto de dependencias se instalan y ya está, y FFmpeg tiene
varias builds publicadas, cada una con su mínimo de driver, así que la
elección es del usuario. El selector se monta aunque la lista llegue vacía
—en un sistema sin mirrors no hay nada publicable— porque entonces el picker
va desactivado y explica por qué no se puede elegir versión, que es mejor
que no dejar el selector ausente y en silencio.

Vive aquí y no en el diálogo por lo mismo que vive ``FFmpegBuildPicker`` en
su propio módulo: es la parte de la fila que hay que abrir para tocarla, y
no debería obligar a leer el panel entero. No trae lógica propia —recibe las
opciones ya evaluadas por ``src.ffmpeg.build_fit`` y las monta—; su trabajo
es solo colocar el picker en el layout de la fila.

El diálogo conserva la decisión de *cuándo* montarlo (``"build_options" in
info``, que solo la fila de FFmpeg cumple).
"""

from __future__ import annotations

from src.ui.ffmpeg_build_picker import FFmpegBuildPicker


def mount_build_picker(row, layout, info):
    """Monta el selector de versión en ``layout`` y lo devuelve.

    ``row`` es el padre (la propia fila) y ``info`` la dependencia ya
    anotada con su ``build_options``, que es el carrier de
    ``build_fit.evaluate_build_list``: la lista de builds y la plataforma
    a la que pertenecen. Se pasa tal cual, sin separarlo en partes, porque
    el selector compone su texto con la plataforma que viene dentro — si
    aquí se desarmara el carrier, el texto volvería a tener que buscar la
    plataforma por su cuenta.
    """
    picker = FFmpegBuildPicker(
        info["build_options"],
        installed_build_id=info.get("build_id"),
        preferred_build_id=info.get("preferred_build_id"),
        parent=row,
    )
    layout.addWidget(picker)
    return picker
