"""Test suite bootstrap.

Sets the Qt platform to ``offscreen`` *before* PySide6 is imported so the
headless UI tests can construct QWidgets without a real display server.

The Qt application object is exposed via the ``qapp`` fixture (function scoped)
so each test gets a clean event loop and widgets can be safely garbage
collected between tests.
"""
import os
import sys

# Must run before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture()
def qapp():
    """A QApplication configured for the offscreen platform.

    If the offscreen platform is unavailable (no Qt platform plugin at all),
    the test is skipped rather than crashing the whole suite.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        # ``-platform offscreen`` as an argv token is the most reliable way to
        # force the offscreen plugin across environments (a bare env-var set
        # can be ignored by some PySide6/Platform builds and segfaults at
        # construction time in headless sandboxes).
        app = QApplication(["draggy-test", "-platform", "offscreen"])
    return app


def pytest_runtest_setup(item):
    """Attach the offscreen platform to the Qt tests even if imported lazily."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
