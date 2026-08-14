# DRAGGY ENCODER

**Cross-platform video compression app with AI-powered workflow pipeline.**

DRAGGY ENCODER combines FFmpeg-based encoding with ONNX/PyTorch AI models for video upscaling, frame interpolation (FSR 3.1), colorization, and denoising — all in a visual drag-and-drop workflow.

## Features

- GPU-accelerated AI upscaling (Real-ESRGAN, OmniSR, CUGAN, SwinIR) via ONNX Runtime (CUDA/TensorRT)
- AMD FSR 3.1 (libplacebo upscale + frame interpolation)
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

## Portable builds

GitHub Actions publishes these platform-specific artifacts:

- **Windows:** `DraggyEncoder-Setup-*.exe`, built with PyInstaller and Inno Setup.
- **Linux x86_64:** `DraggyEncoder-Linux-x86_64-*.tar.gz`, portable archive.
- **Debian/Ubuntu:** `DraggyEncoder-Linux-x86_64-*.deb`, with a desktop launcher.
- **Fedora/RHEL/openSUSE:** `DraggyEncoder-Linux-x86_64-*.rpm`, with a desktop launcher.
- **Linux desktop:** `DraggyEncoder-Linux-x86_64-*.AppImage`, directly executable.
- **Flatpak:** `DraggyEncoder-Linux-x86_64-*.flatpak`, installable with Flatpak.

The Linux archive is portable: extract it and run `./DraggyEncoder/DraggyEncoder`.
The `.deb` and `.rpm` packages install a desktop entry and command
`draggy-encoder`; the AppImage can be marked executable and launched from the
file manager. The standalone Flatpak can be installed with `flatpak install --user
DraggyEncoder-Linux-x86_64-*.flatpak`. Installed Linux
formats store writable data under `${XDG_DATA_HOME:-~/.local/share}/draggy-encoder`
so `/opt` and a read-only AppImage mount are never modified. The first launch
downloads FFmpeg and models into that user data directory.

All bundles include a private Python 3.13 interpreter used only to install the
optional PyTorch and ONNX Runtime packages on demand; the large AI runtimes are
not shipped inside the release artifact. On Linux, use a distribution with a
recent glibc and make the executable runnable if your extraction tool did not
preserve permissions (`chmod +x DraggyEncoder/DraggyEncoder`).

### Flatpak repository and automatic updates

GitHub Actions publishes a GPG-signed Flatpak OSTree repository to GitHub
Pages. Import the project key once, then add the remote without
`--no-gpg-verify`:

```bash
curl -fsSL https://thedevil4k.github.io/Draggy-Encoder/draggy-encoder.gpg \
  -o draggy-encoder.gpg
flatpak remote-add --if-not-exists --gpg-import=draggy-encoder.gpg draggy-encoder \
  https://thedevil4k.github.io/Draggy-Encoder/
flatpak install --user draggy-encoder com.draggyencoder.DraggyEncoder
flatpak update
```

After the remote is added, normal `flatpak update` detects new commits from
releases published by the project; users no longer need to download individual
`.flatpak` files. The repository is deployed by the Linux workflow, so GitHub
Pages must be enabled once in the repository settings with **Source: GitHub
Actions**. The standalone `.flatpak` bundle attached to each release is signed
with the same key, so `flatpak install` verifies it the same way.

#### Setting up the signing key (maintainers)

The repository is signed with a dedicated, passphrase-less GPG key stored in
GitHub Actions secrets. To (re)create it and store it:

```bash
gpg --batch --passphrase '' --quick-gen-key \
  'Draggy Encoder Flatpak <flatpak@draggyencoder.io>' ed25519 sign never
FINGERPRINT=$(gpg --list-keys --with-colons --fingerprint \
  'Draggy Encoder Flatpak <flatpak@draggyencoder.io>' | awk -F: '$1=="fpr"{print $10; exit}')
gpg --armor --export-secret-keys "$FINGERPRINT" > flatpak-signing-key.asc
gh secret set FLATPAK_GPG_PRIVATE_KEY --repo thedevil4k/Draggy-Encoder \
  < flatpak-signing-key.asc
# Opcional: fijar el fingerprint explícitamente (si no, se deriva solo).
gh secret set FLATPAK_GPG_KEY_ID --repo thedevil4k/Draggy-Encoder --body "$FINGERPRINT"
```

`FLATPAK_GPG_KEY_ID` (the key fingerprint) is optional — the workflow derives
it from the imported key. Without `FLATPAK_GPG_PRIVATE_KEY`, the build still
runs but leaves the repository unsigned (forks and PRs).

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