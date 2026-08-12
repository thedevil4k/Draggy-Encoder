"""Cross-platform lifecycle management for external render processes.

Every FFmpeg/auxiliary child spawned by a render can be registered here.  The
supervisor terminates process groups where possible, escalates to kill, and
waits for every child so a cancelled Qt worker cannot leave orphan processes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Optional


def process_group_popen_kwargs() -> dict:
    """Return Popen kwargs that isolate a child process tree."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


class ProcessSupervisor:
    """Track and stop all children belonging to one render operation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processes: dict[int, tuple[object, str]] = {}
        self._stopping = False

    def register(self, process, role: str = "process") -> None:
        if process is None:
            return
        pid = getattr(process, "pid", None)
        if pid is None:
            return
        with self._lock:
            self._processes[int(pid)] = (process, str(role or "process"))

    def unregister(self, process) -> None:
        pid = getattr(process, "pid", None)
        if pid is None:
            return
        with self._lock:
            self._processes.pop(int(pid), None)

    def snapshot(self) -> list[tuple[object, str]]:
        with self._lock:
            return list(self._processes.values())

    def alive(self) -> list[tuple[object, str]]:
        alive = []
        for process, role in self.snapshot():
            try:
                if process.poll() is None:
                    alive.append((process, role))
            except Exception:
                alive.append((process, role))
        return alive

    @staticmethod
    def _signal(process, force: bool = False) -> None:
        """Signal a process and its descendants when the platform permits."""
        pid = getattr(process, "pid", None)
        if not pid:
            return
        if os.name == "nt":
            if force:
                # taskkill /T reaches descendants created by FFmpeg filters.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        check=False,
                    )
                    return
                except Exception:
                    pass
            try:
                process.terminate() if not force else process.kill()
            except Exception:
                pass
            return

        try:
            pgid = os.getpgid(pid)
            # Children are launched with start_new_session=True, so this does
            # not touch the GUI process group.
            os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
        except Exception:
            try:
                process.kill() if force else process.terminate()
            except Exception:
                pass

    @staticmethod
    def _wait(process, timeout: float) -> bool:
        try:
            process.wait(timeout=max(0.0, timeout))
            return True
        except Exception:
            return False

    def stop_all(self, grace_seconds: float = 3.0,
                 kill_seconds: float = 2.0) -> list[str]:
        """Terminate every registered child and return roles still alive."""
        with self._lock:
            self._stopping = True
        processes = self.snapshot()
        for process, _role in processes:
            self._signal(process, force=False)

        deadline = time.monotonic() + max(0.0, grace_seconds)
        for process, _role in processes:
            remaining = max(0.0, deadline - time.monotonic())
            self._wait(process, remaining)

        remaining = self.alive()
        for process, _role in remaining:
            self._signal(process, force=True)

        deadline = time.monotonic() + max(0.0, kill_seconds)
        for process, _role in remaining:
            remaining_time = max(0.0, deadline - time.monotonic())
            self._wait(process, remaining_time)

        still_alive = [role for _process, role in self.alive()]
        with self._lock:
            for process, _role in processes:
                pid = getattr(process, "pid", None)
                if pid is not None and process.poll() is not None:
                    self._processes.pop(int(pid), None)
        return still_alive

    def clear_finished(self) -> None:
        with self._lock:
            for pid, (process, _role) in list(self._processes.items()):
                try:
                    if process.poll() is not None:
                        self._processes.pop(pid, None)
                except Exception:
                    pass

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping
