import base64
import os
import platform
import re
import subprocess
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

    def _add_gpu(name, memory="", vendor=None):
        name_clean = name.strip()
        if not name_clean:
            return
        name_lower = name_clean.lower()
        if name_lower in existing_names:
            return
        existing_names.add(name_lower)
        vnd = vendor or _guess_vendor(name_clean)
        gpus.append({"name": name_clean, "memory": memory, "vendor": vnd})

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
    for smi_path in smi_paths:
        try:
            smi_out = _run_cmd(
                f'"{smi_path}" --query-gpu=name,memory.total --format=csv,noheader'
            )
        except Exception:
            smi_out = ""

        if smi_out and "is not recognized" not in smi_out and "," in smi_out:
            # Salida CSV válida: "Nombre GPU, 12288 MiB"
            for line in smi_out.strip().splitlines():
                if "," in line:
                    name, mem = line.split(",", 1)
                    _add_gpu(name.strip(), mem.strip(), "NVIDIA")
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
        "Select-Object Name, AdapterRAM, VideoProcessor, AdapterCompatibility, Status | "
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
        _add_gpu(gpu["name"], gpu["memory"], gpu["vendor"])

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
    """Parsea la salida CSV de Win32_VideoController."""
    gpus = []
    if not output:
        return gpus
    lines = output.strip().splitlines()
    if not lines:
        return gpus

    # La primera línea son los headers
    headers = [h.strip().strip('"').lower() for h in lines[0].split(",")]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
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
        gpus.append({"name": name, "memory": memory, "vendor": vendor})
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
        gpus.append({"name": name, "memory": "", "vendor": _guess_vendor(name)})
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


def get_hardware_summary():
    cpu = get_cpu_info()
    gpus = get_gpu_info()
    return {
        "cpu": cpu,
        "gpus": gpus,
        "recommended_device": get_recommended_device(gpus),
    }


if __name__ == "__main__":
    print(get_hardware_summary())
