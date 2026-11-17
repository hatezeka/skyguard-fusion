from unittest.mock import patch

import pytest

from app.detection.capabilities import (
    inspect_model,
    require_drone_bird_model,
)


class FakeModel:
    def __init__(self, names):
        self.names = names


def test_drone_bird_model(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"test")

    with patch(
        "app.detection.capabilities.YOLO",
        return_value=FakeModel({0: "drone", 1: "bird"}),
    ):
        result = inspect_model(str(path))

    assert result.supports_drone
    assert result.supports_bird
    assert result.ready_for_drone_bird


def test_bird_only_model(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"test")

    with patch(
        "app.detection.capabilities.YOLO",
        return_value=FakeModel({0: "bird", 1: "airplane"}),
    ):
        result = inspect_model(str(path))

    assert not result.supports_drone
    assert result.supports_bird
    assert not result.ready_for_drone_bird


def test_drone_alias(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"test")

    with patch(
        "app.detection.capabilities.YOLO",
        return_value=FakeModel({0: "UAV", 1: "Bird"}),
    ):
        result = inspect_model(str(path))

    assert result.ready_for_drone_bird


def test_missing_model():
    with pytest.raises(FileNotFoundError):
        inspect_model("missing_model_123456.pt")


def test_require_rejects_incomplete_model(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"test")

    with patch(
        "app.detection.capabilities.YOLO",
        return_value=FakeModel({0: "bird"}),
    ):
        with pytest.raises(ValueError, match="both drone and bird"):
            require_drone_bird_model(str(path))
