"""AI engine catalog, state persistence, and fetch utilities.

This package hosts the glue between DRAGGY ENCODER and external neural
network models (upscalers, frame interpolators, colorizers, denoisers,
face restorers, etc.). The design rules of this package mirror those
of ``src/dependencies.py``:

- Engines are declared declaratively in ``src.ai.registry.CATALOG``.
- State (installed, enabled, ONNX-converted) lives in
  ``bin/.session/engines.json`` and is read/written through
  ``src.ai.engines_state``.
- Heavy work (cloning a HuggingFace repo, converting to ONNX) runs in
  a ``QThread`` so the GUI stays responsive.

Files:
    registry.py       - catalog of known engines + path helpers.
    engines_state.py  - JSON-backed state for every known engine.
    fetcher.py        - git-clone QThread workers + ONNX stub.
"""

from __future__ import annotations
