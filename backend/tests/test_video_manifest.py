import pytest

from app.datasets.video_manifest import (
    VideoRecord,
    inspect_video,
    sha256_file,
    save_manifest,
)


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"skyguard")
    assert sha256_file(path) == sha256_file(path)


def test_rejects_unknown_category(tmp_path):
    with pytest.raises(ValueError):
        inspect_video(
            tmp_path / "missing.mp4",
            "airplane",
            "https://example.org/video",
            "unknown",
        )


def test_rejects_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_video(
            tmp_path / "missing.mp4",
            "drone",
            "https://example.org/video",
            "unknown",
        )


def test_manifest_does_not_overwrite(tmp_path):
    record = VideoRecord(
        video_id="abc",
        category="drone",
        source_url="https://example.org/video",
        license_name="test",
        license_verified=False,
        sha256="abc",
        width=640,
        height=480,
        fps=30.0,
        frame_count=300,
        duration_seconds=10.0,
    )

    path = tmp_path / "manifest.json"
    save_manifest(record, path)

    assert path.exists()

    with pytest.raises(FileExistsError):
        save_manifest(record, path)