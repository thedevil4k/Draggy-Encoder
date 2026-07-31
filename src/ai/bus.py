"""Process-wide event bus for AI engine state changes.

Why a singleton QObject?

    The AI Engines Manager (inside ``DependenciesDialog``) and the
    workflow nodes (inside ``MainWindow``) need to react when an
    engine's installed/enabled/converted flag flips. Options we
    rejected:

    - Polling: wasteful at 500 ms; UI redraws become jittery when
      many rows exist; also pollutes the diff between "user-initiated
      change" and "consistency check".
    - QAbstractItemModel: overkill for a 9-entry catalog; the engine
      list is not a tree.
    - Per-instance pub/sub: requires a registry of subscribers that
      must be cleaned up on dialog/worker destruction; easy to leak.

    A singleton QObject has a clean lifecycle (process-lifetime,
    destroyed at interpreter shutdown) and Qt signal semantics that
    naturally support cross-thread ``emit`` and ``connect``.

Signal contract:

    state_changed = Signal(str)
        Emitted by ``engines_state.patch`` after every successful
        state write. ``engine_id`` is the slug of the engine whose
        state changed (``"upscale-realesrgan-x4plus"`` etc.). Empty
        string means "global sweep" (e.g. on a future migrate/reset);

        listeners should re-query ``registry.list_enabled`` rather
        than tracking fine-grained diffs themselves.

Connections should use ``Qt.ConnectionType.UniqueConnection`` so
re-creating ``MainWindow`` in unit tests does not stack duplicate
handlers. Use:

    from src.ai.bus import bus
    bus.state_changed.connect(handler, Qt.ConnectionType.UniqueConnection)
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class _AIBus(QObject):
    """The bus singleton. Use the module-level ``bus`` instance."""

    state_changed = Signal(str)   # engine_id; "" = global sweep


# Module-level singleton. Qt guarantees safe cross-thread emit/connect.
bus = _AIBus()


__all__ = ["bus"]
