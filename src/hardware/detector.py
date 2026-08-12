import base64
import os
import platform
import re
import subprocess
import sys
import psutil


def get_cpu_info():
    """Devuelve información básica de la CPU."""
    info = {
        "name": platform.processor() or "Unknown CPU",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
    }
    # Intentar obtener nombre más descriptivo en Windows
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            info["name"] = name.strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["name"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return info


def _run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result.stdout
    except Exception:
        return ""


def _detect_gpus_windows():
    """Detecta GPUs en Windows usando múltiples métodos.

    Orden de intentos:
    1. nvidia-smi con paths conocidos (System32, NVSMI, PATH)
    2. PowerShell Get-CimInstance (con EncodedCommand para evitar problemas de comillas)
    3. WMIC (legacy, puede fallar en Windows 11)
    4. Registro de Windows (HKLM\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968...})
    """
    gpus = []
    existing_names = set()

    def _add_gpu(name, memory="", vendor=None, driver="", memory_used="",
                 core_clock="", temperature=""):
        name_clean = name.strip()
        if not name_clean:
            return
        name_lower = name_clean.lower()
        if name_lower in existing_names:
            return
        existing_names.add(name_lower)
        vnd = vendor or _guess_vendor(name_clean)
        gpus.append({
            "name": name_clean,
            "memory": memory,
            "vendor": vnd,
            "driver": driver,
            "memory_used": memory_used,
            "core_clock": core_clock,
            "temperature": temperature,
        })

    # ── 1. nvidia‑smi con rutas conocidas ──
    smi_paths = [
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Windows\Sysnative\nvidia-smi.exe",
        os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe",
        ),
        "nvidia-smi",  # último intento: PATH del sistema
    ]
    # Campos extra: driver, memoria usada/total, reloj del núcleo y temperatura.
    _SMI_QUERY = (
        "name,driver_version,memory.total,memory.used,"
        "clocks.max.graphics,temperature.gpu"
    )
    for smi_path in smi_paths:
        try:
            smi_out = _run_cmd(
                f'"{smi_path}" --query-gpu={_SMI_QUERY} --format=csv,noheader,nounits'
            )
        except Exception:
            smi_out = ""

        if smi_out and "is not recognized" not in smi_out and "," in smi_out:
            # Salida CSV válida: "Nombre GPU, 560.94, 12288, 3144, 2520, 68"
            for line in smi_out.strip().splitlines():
                if "," in line:
                    parts = [p.strip() for p in line.split(",")]
                    name = parts[0]
                    driver = parts[1] if len(parts) > 1 else ""
                    memory = parts[2] if len(parts) > 2 else ""
                    memory_used = parts[3] if len(parts) > 3 else ""
                    core_clock = parts[4] if len(parts) > 4 else ""
                    temperature = parts[5] if len(parts) > 5 else ""
                    _add_gpu(
                        name, memory + " MiB", "NVIDIA",
                        driver=driver, memory_used=memory_used + " MiB",
                        core_clock=core_clock + " MHz",
                        temperature=temperature + " °C",
                    )
            break
        elif smi_out and "used" not in smi_out.lower():
            # nvidia-smi contestó pero sin --query-format (versión vieja);
            # intentamos parsear líneas que contengan el nombre de GPU.
            for line in smi_out.strip().splitlines():
                if any(
                    tag in line.lower()
                    for tag in ("nvidia", "rtx", "gtx", "quadro", "tesla")
                ):
                    # Intentar extraer solo el nombre (sin métricas de uso)
                    clean = line.strip()
                    if "%" in clean or "W/" in clean or "MiB" in clean.lower():
                        # Tomar solo la primera parte significativa
                        parts = clean.split()
                        clean = " ".join(parts[:4])  # primeras 4 palabras típicamente el nombre
                    _add_gpu(clean, "", "NVIDIA")
            if any(g["vendor"] == "NVIDIA" for g in gpus):
                break

    # ── 2. PowerShell con EncodedCommand (evita el infierno de comillas anidadas) ──
    ps_script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name, AdapterRAM, VideoProcessor, AdapterCompatibility, "
        "DriverVersion, Status | "
        "ConvertTo-Csv -NoTypeInformation"
    )
    try:
        b64_cmd = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
        output = _run_cmd(
            f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {b64_cmd}"
        )
    except Exception:
        output = ""

    # ── 3. Fallback a wmic (puede estar deprecado en Win 11) ──
    if not output or "Name" not in output:
        output = _run_cmd(
            "wmic path win32_VideoController get name,AdapterRAM /format:csv"
        )

    for gpu in _parse_video_controller_csv(output):
        _add_gpu(
            gpu["name"], gpu["memory"], gpu["vendor"],
            driver=gpu.get("driver", ""),
        )

    # ── 4. Fallback último: Registro de Windows ──
    if not gpus:
        try:
            import winreg

            # GUID de la clase "Display Adapters"
            key_path = (
                r"SYSTEM\CurrentControlSet\Control\Class"
                "\\"
                r"{4d36e968-e325-11ce-bfc1-08002be10318}"
            )
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                for i in range(20):
                    try:
                        with winreg.OpenKey(key, f"{i:04d}") as sub_key:
                            name, _ = winreg.QueryValueEx(sub_key, "DriverDesc")
                            _add_gpu(name)
                    except OSError:
                        continue
        except Exception:
            pass

    return gpus


def _parse_video_controller_csv(output):
    """Parsea la salida CSV de Win32_VideoController.

    Usa el módulo ``csv`` del stdlib: los campos quoted pueden contener
    comas (p.ej. ``"Advanced Micro Devices, Inc."``) y el split naive los
    rompía, desplazando la columna DriverVersion.
    """
    import csv as _csv
    import io as _io

    gpus = []
    if not output:
        return gpus
    try:
        reader = _csv.reader(_io.StringIO(output))
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    except Exception:
        return gpus
    if not rows:
        return gpus

    # La primera línea son los headers
    headers = [h.strip().strip('"').lower() for h in rows[0]]
    for row in rows[1:]:
        parts = [p.strip() for p in row]
        if len(parts) < len(headers):
            continue
        data = dict(zip(headers, parts))
        name = data.get("name", "")
        if not name:
            continue
        ram = data.get("adapterram", "")
        memory = _format_ram(ram)
        vendor = _guess_vendor(name)
        if data.get("adaptercompatibility"):
            vendor = _guess_vendor(data["adaptercompatibility"] + " " + name)
        gpus.append({
            "name": name,
            "memory": memory,
            "vendor": vendor,
            "driver": data.get("driverversion", ""),
        })
    return gpus


def _format_ram(ram_value):
    """Convierte un valor de AdapterRAM a formato legible."""
    try:
        ram_bytes = int(ram_value)
        if ram_bytes <= 0:
            return ""
        ram_mb = ram_bytes / (1024 * 1024)
        if ram_mb >= 1024:
            return f"{ram_mb / 1024:.1f} GB"
        return f"{int(ram_mb)} MB"
    except (ValueError, TypeError):
        return ""


def _detect_gpus_linux():
    gpus = []

    # nvidia-smi: driver + VRAM + clocks + temperature (NVIDIA only).
    try:
        smi_out = _run_cmd(
            "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,"
            "clocks.max.graphics,temperature.gpu --format=csv,noheader,nounits"
        )
        if smi_out and "," in smi_out and "not recognized" not in smi_out.lower():
            for line in smi_out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if not parts or not parts[0]:
                    continue
                gpus.append({
                    "name": parts[0],
                    "memory": (parts[2] + " MiB") if len(parts) > 2 else "",
                    "vendor": "NVIDIA",
                    "driver": parts[1] if len(parts) > 1 else "",
                    "memory_used": (parts[3] + " MiB") if len(parts) > 3 else "",
                    "core_clock": (parts[4] + " MHz") if len(parts) > 4 else "",
                    "temperature": (parts[5] + " °C") if len(parts) > 5 else "",
                })
            if gpus:
                return gpus
    except Exception:
        pass

    output = _run_cmd("lspci -nn | grep -iE 'vga|3d|video'")
    for line in output.splitlines():
        if not line.strip():
            continue
        # Ejemplo: 01:00.0 VGA compatible controller: NVIDIA Corporation ...
        match = re.search(r":\s*(.+?)\[", line)
        if match:
            name = match.group(1).strip()
        else:
            name = line.split(":", 2)[-1].strip()
        gpus.append({
            "name": name, "memory": "", "vendor": _guess_vendor(name),
            "driver": "", "memory_used": "", "core_clock": "",
            "temperature": "",
        })
    return gpus


def _guess_vendor(name):
    name_lower = name.lower()
    if "nvidia" in name_lower:
        return "NVIDIA"
    if "amd" in name_lower or "radeon" in name_lower:
        return "AMD"
    if "intel" in name_lower:
        return "Intel"
    return "Unknown"


def get_gpu_info():
    """Devuelve lista de GPUs detectadas."""
    if platform.system() == "Windows":
        return _detect_gpus_windows()
    return _detect_gpus_linux()


def get_recommended_device(gpus):
    """Devuelve el dispositivo recomendado según las GPUs detectadas.

    Prefiere GPUs dedicadas (NVIDIA/AMD) sobre integradas Intel y CPU.
    """
    preferred_order = ["NVIDIA", "AMD", "Intel"]
    by_vendor = {gpu.get("vendor", ""): gpu for gpu in gpus}
    for vendor in preferred_order:
        if vendor in by_vendor:
            return f"{vendor} GPU"
    return "CPU"


def get_system_summary():
    """Versión legible del sistema operativo y del intérprete de Python.

    Pensada para soporte: el usuario informa el entorno exacto (build del
    OS + arquitectura + intérprete concreto) sin adivinar. Funciona en
    Windows (nombre de marketing 10/11 por build) y en Linux/macOS
    (``platform.system()`` + ``release()``).
    """
    system = platform.system() or "Unknown"
    release = platform.release() or ""
    machine = platform.machine() or ""

    os_name = system
    win_build = ""
    if system == "Windows":
        try:
            # win32_ver() -> (version, service_pack, type); release() solo
            # devuelve "10". El build completo sale de version(), p.ej.
            # "10.0.19045" (Win10) / "10.0.26100" (Win11).
            win_ver = platform.win32_ver() or ()
            family = win_ver[0] if len(win_ver) > 0 else ""
            win_build = platform.version() or ""
            try:
                build_num = int(float(win_build.split(".")[-1])) if win_build else 0
            except (ValueError, TypeError):
                build_num = 0
            if family == "10":
                # Build >= 22000 == Windows 11 (los builds 10.x anteriores
                # son Windows 10, incluido 10.0.19041/19045).
                os_name = "Windows 11" if build_num >= 22000 else "Windows 10"
            elif family:
                os_name = f"Windows {family}"
        except Exception:
            pass

    os_display = os_name
    if release:
        if system == "Windows":
            # p.ej. "Windows 10 (10.0.19045)" — build completo de version().
            os_display = f"{os_name} ({win_build or release})"
        else:
            os_display = f"{os_name} {release}"    # p.ej. "Linux 6.8.0-45"

    try:
        python_version = sys.version.split()[0]  # p.ej. "3.11.15"
    except Exception:
        python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    return {
        "os": os_display,
        "os_name": os_name,
        "os_release": release,
        "machine": machine,
        "python_version": python_version,
        "python_executable": sys.executable or "",
        "python_bits": 64 if sys.maxsize > 2**32 else 32,
    }


def get_hardware_summary():
    cpu = get_cpu_info()
    gpus = get_gpu_info()
    system = get_system_summary()
    return {
        "cpu": cpu,
        "gpus": gpus,
        "recommended_device": get_recommended_device(gpus),
        "os": system["os"],
        "machine": system["machine"],
        "python": {
            "version": system["python_version"],
            "executable": system["python_executable"],
            "bits": system["python_bits"],
        },
    }


if __name__ == "__main__":
    print(get_hardware_summary())
