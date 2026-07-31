# DRAGGY ENCODER

**Cross-platform video compression app with AI-powered workflow pipeline.**

DRAGGY ENCODER combines FFmpeg-based encoding with ONNX/PyTorch AI models for video upscaling, frame interpolation (FSR 3.1), colorization, and denoising — all in a visual drag-and-drop workflow.

## Features

- GPU-accelerated AI upscaling (Real-ESRGAN, OmniSR, CUGAN, SwinIR) via ONNX Runtime (CUDA/TensorRT)
- AMD FSR 1.0 (CPU), FSR 2.0 (Vulkan/libplacebo), FSR 3.1 (upscale + frame interpolation)
- Frame-by-frame decode → AI → encode pipeline with audio passthrough
- Auto-downloading FFmpeg (including libplacebo builds) and AI models from HuggingFace
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

On first launch the app auto-detects your hardware, downloads FFmpeg if needed, and tests all compatible GPU/CPU codecs — no configuration required.

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
│   ├── ffmpeg-full/          # FFmpeg with libplacebo (FSR 3.1)
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
- Git (for model downloads; fallback to huggingface_hub if unavailable)

See `requirements.txt` for the full Python dependency list.

## Usage

1. **Import a video** — Drag and drop or browse
2. **Configure AI workflow** — Select upscale (x2‑x4) and/or interpolation (double FPS)
3. **Adjust export settings** — Codec, bitrate, resolution, format, audio
4. **Preview** — Compare original vs compressed frame
5. **Compress** — GPU accelerated encode with audio passthrough

## License

MIT License — see `LICENSE` for details.