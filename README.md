# DRAGGY ENCODER

**Cross-platform video compression app with AI-powered workflow pipeline.**

DRAGGY ENCODER combines FFmpeg-based encoding with ONNX/PyTorch AI models for video upscaling, frame interpolation (RIFE/FILM), colorization, and denoising — all in a visual drag-and-drop workflow.

## Features

- GPU-accelerated AI upscaling (Real-ESRGAN, OmniSR, CUGAN, SwinIR) via ONNX Runtime (CUDA/TensorRT)
- Frame interpolation (RIFE, FILM) via PyTorch/ONNX
- Frame-by-frame decode → AI → encode pipeline with audio passthrough
- Auto-downloading FFmpeg (latest BtbN gpl build) and AI models from HuggingFace
- NVENC / AMF / QuickSync GPU encoder support
- Two-pass EBU R128 loudness normalization (audio boost)
- Side-by-side preview comparison with zoom + pan
- Multi-language UI (English / Spanish)
- Auto-update chain for dependencies
- PyInstaller-ready for portable Windows/Linux distribution

## Quick Start

```bash
git clone https://github.com/YOUR_USER/DRAGGY_ENCODER.git
cd DRAGGY_ENCODER

python -m venv venv
venv/Scripts/activate   # Windows
# source venv/bin/activate   # Linux
pip install -r requirements.txt
python main.py
```

On first launch the app auto-detects your hardware, downloads FFmpeg + ONNX Runtime + PyTorch if needed, and tests all compatible GPU/CPU codecs — no configuration required.

## Project Structure

```
DRAGGY_ENCODER/
├── main.py                  # Entry point
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT
├── assets/                   # UI icons (7 PNGs)
├── bin/                      # Downloaded binaries (auto-generated, gitignored)
│   ├── .session/             # Preferences & engine state
│   ├── ffmpeg/               # FFmpeg (auto-downloaded on first run)
│   ├── runtimes/             # PyTorch + ONNX Runtime (pip install --target)
│   ├── models/               # Downloaded AI model weights
│   └── logs/                 # Render logs & FFmpeg logs
├── personal/                 # HuggingFace tokens (gitignored)
└── src/
    ├── ai/                   # AI engine catalog, inference, architectures, HF search
    ├── ffmpeg/               # FFmpeg manager, downloader, runner, codec testing
    ├── hardware/             # CPU/GPU detection + codec compatibility maps
    ├── processing/           # Video pipeline (AI frame-by-frame + standard encode)
    ├── runtimes/             # PyTorch/ONNX Runtime local install manager
    ├── ui/                   # PySide6 GUI components (~16 widgets)
    └── utils/                # Time/format helpers
```

## Requirements

- Python 3.10–3.14
- GPU recommended for AI upscaling (NVIDIA CUDA or AMD Vulkan)
- No Git required: model downloads use the bundled `huggingface_hub`
  (the packaged app no longer falls back to `git clone`, since the
  Windows installer does not ship Git).

See `requirements.txt` for the full Python dependency list.

### AI runtimes: PyTorch vs ONNX Runtime

- **Converting a `.pth`/`.pt`/`.safetensors` checkpoint to `.onnx` requires
  PyTorch** — the app uses `torch.onnx.export` to rebuild the model from the
  checkpoint and export it.
- **Running an existing `.onnx` model only needs ONNX Runtime.** Models that
  already ship as `.onnx` (many HuggingFace repos do) don't require PyTorch
  at all.

On the first launch the app automatically downloads FFmpeg, ONNX Runtime,
and PyTorch — no manual setup required. ONNX models work immediately;
PyTorch is ready for `.pth` import/conversion without extra steps.
A failed runtime download never blocks the app from starting:
retry from *Manage Dependencies → AI Runtimes → Install*.

**The PyTorch build is resolved, not hardcoded.** Before downloading, the app
queries the PyTorch wheel indices and PyPI metadata and picks the
(variant, version) pair that actually publishes wheels for your CPython, your
platform and your NVIDIA driver — for example CUDA 13.0 builds on a recent
driver, or an older CUDA/CPU build otherwise. The progress dialog shows the
resolved choice (`CUDA 13.0 · torch 2.14.0 + torchvision 0.29.0 + …`).
If a download is interrupted or fails validation, the previously working
runtime is kept: the new files are assembled and verified in a temporary
directory and only then swapped in.

**Installing and uninstalling each runtime is independent.** PyTorch and ONNX
Runtime share one folder, but the Uninstall button on each row removes only
that runtime (plus its own satellites) and leaves the other one — and the
shared dependencies such as numpy — untouched. Read the confirmation dialog:
it tells you exactly what will be removed. On Windows, if the runtime is
loaded in memory its DLLs cannot be deleted immediately; the app schedules the
removal for the next launch and tells you so. Reinstalling before that restart
is safe: the pending cleanup recognises the new installation and leaves it
alone.

- Installing **ONNX Runtime** also installs `onnx`/`onnxscript`, which is what
  `torch.onnx.export` needs to convert `.pth` checkpoints.
- Installing **PyTorch** installs them too, so a `.pth`-only workflow works
  after installing either one.

### Audio Boost: two levels

- **Quick:** just the *Audio Boost (normalize volume)* checkbox. It applies the
  usual two-pass EBU R128 normalization (`loudnorm` `I=-16` / `TP=-1.5` /
  `LRA=11`) and raises the volume to the streaming standard with nothing to
  configure.
- **Detailed:** the *Avanzado* tab next to the checkbox opens an equalizer
  (bass/mid/treble), a dynamic-range compressor (threshold, ratio, attack,
  release, with presets) and a peak limiter.

The options in the dropdown **alter the boost itself**, they are not stacked on
top of it: the normalization is recalculated to let the equalizer and the
compressor do their job. With nothing enabled the boost behaves exactly as it
does with the dropdown closed, so merely opening it never changes the sound.

*Escuchar efecto* plays a short excerpt of the track with the current settings
applied; while it is playing, moving any control returns updated audio so you
can tune by ear, and *Parar* stops it.

## Portable builds

GitHub Actions publishes one package per platform:

- **Windows (x64):** `DraggyEncoder-Setup-*.exe`, built with PyInstaller and Inno Setup.
- **Linux (x86_64):** `DraggyEncoder-Linux-x86_64-*.AppImage`, directly executable.

The AppImage can be marked executable and launched from the file manager.

### Downloads — latest build

After every push to `main`, the rolling **Latest build** release is updated, so
these links always point to the newest build without waiting for a version tag:

- **Windows installer** (Inno Setup):
  [DraggyEncoder-Setup-1.exe](https://github.com/thedevil4k/Draggy-Encoder/releases/latest/download/DraggyEncoder-Setup-1.exe)
- **Linux AppImage**:
  [DraggyEncoder-Linux-x86_64-1.AppImage](https://github.com/thedevil4k/Draggy-Encoder/releases/latest/download/DraggyEncoder-Linux-x86_64-1.AppImage)

The AppImage is also attached to tagged releases (`v*`), which are the permanent
downloads; the rolling `latest` release always points at the newest push.

Installed Linux formats store writable data under
`${XDG_DATA_HOME:-~/.local/share}/draggy-encoder` so `/opt` and a read-only
AppImage mount are never modified. The first launch downloads FFmpeg, AI
runtimes (ONNX Runtime + PyTorch), and models into that user data directory.

> Flatpak (with a signed OSTree repository published to GitHub Pages) is
> planned for a later release.

All bundles include a private Python 3.13 interpreter used only to install the
optional PyTorch and ONNX Runtime packages on demand; the large AI runtimes are
not shipped inside the release artifact. On Linux, use a distribution with a
recent glibc and make the executable runnable if your extraction tool did not
preserve permissions (`chmod +x DraggyEncoder/DraggyEncoder`).

For source development on Linux:

```bash
source venv/bin/activate
python main.py
```

## Usage

1. **Import a video** — Drag and drop or browse
2. **Configure AI workflow** — Select upscale (x2‑x4) and/or interpolation (double FPS)
3. **Adjust export settings** — Codec, bitrate, resolution, format, audio
4. **Preview** — Compare original vs compressed frame
5. **Compress** — GPU accelerated encode with audio passthrough

## License

MIT License — see `LICENSE` for details.