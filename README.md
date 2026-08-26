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

- Python 3.9–3.14
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

## Portable builds

GitHub Actions publishes these platform-specific artifacts:

- **Windows:** `DraggyEncoder-Setup-*.exe`, built with PyInstaller and Inno Setup.
- **Linux x86_64:** `DraggyEncoder-Linux-x86_64-*.tar.gz`, portable archive.
- **Linux desktop:** `DraggyEncoder-Linux-x86_64-*.AppImage`, directly executable.

The Linux archive is portable: extract it and run `./DraggyEncoder/DraggyEncoder`.
The AppImage can be marked executable and launched from the file manager.

### Downloads — latest build

After every push to `main`, the rolling **Latest build** release is updated, so
these links always point to the newest build without waiting for a version tag:

- **Windows installer** (Inno Setup):
  [DraggyEncoder-Setup-1.exe](https://github.com/thedevil4k/Draggy-Encoder/releases/latest/download/DraggyEncoder-Setup-1.exe)
- **Linux AppImage**:
  [DraggyEncoder-Linux-x86_64-1.AppImage](https://github.com/thedevil4k/Draggy-Encoder/releases/latest/download/DraggyEncoder-Linux-x86_64-1.AppImage)

The portable Linux `tar.gz` is attached to tagged releases (`v*`) and to manual
workflow runs (`workflow_dispatch`).

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