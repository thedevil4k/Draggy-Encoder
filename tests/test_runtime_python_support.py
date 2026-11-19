"""La app no debe intentar wheels cp314 que ONNX Runtime no publica."""

import types

from src.runtimes import manager


def _version(major, minor):
    return types.SimpleNamespace(major=major, minor=minor, micro=0)


def test_python_313_is_supported(monkeypatch):
    monkeypatch.setattr(manager.sys, "version_info", _version(3, 13))
    assert manager._is_python_version_supported() is True


def test_python_314_is_rejected_before_pip(monkeypatch):
    monkeypatch.setattr(manager.sys, "version_info", _version(3, 14))
    assert manager._is_python_version_supported() is False
    message = manager.get_python_support_info()["message"]
    assert "3.13" in message
    assert "supported" in message.lower()


def test_python_39_is_rejected(monkeypatch):
    monkeypatch.setattr(manager.sys, "version_info", _version(3, 9))
    assert manager._is_python_version_supported() is False
