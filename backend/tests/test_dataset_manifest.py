import pytest
from pydantic import ValidationError

from app.datasets.manifest import (
    VideoManifest,
    validate_video_file,
)


def sample_manifest(**overrides):
    data = {
        "video_id": "VIDEO-001",
        "file_path": "raw/drone_test.mp4",
        "source_type": "real",
        "camera_profile": "standard",
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "split": "test",
        "scenario_tags": ["drone", "small_object"],
        "license": "authorized-test-use",
    }

    data.update(overrides)
    return VideoManifest(**data)


def test_valid_manifest():
    manifest = sample_manifest()
    assert manifest.video_id == "VIDEO-001"
    assert manifest.fps == 30.0


def test_invalid_fps():
    with pytest.raises(ValidationError):
        sample_manifest(fps=0)


def test_unknown_scenario_tag():
    with pytest.raises(ValidationError):
        sample_manifest(
            scenario_tags=["unknown_tag"]
        )


def test_missing_video(tmp_path):
    manifest = sample_manifest()

    with pytest.raises(FileNotFoundError):
        validate_video_file(manifest, tmp_path)


def test_existing_video_path(tmp_path):
    video = tmp_path / "raw" / "drone_test.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"test")

    manifest = sample_manifest()

    assert validate_video_file(manifest, tmp_path)


def test_path_traversal_blocked(tmp_path):
    manifest = sample_manifest(
        file_path="../outside.mp4"
    )

    with pytest.raises(ValueError):
        validate_video_file(manifest, tmp_path)
