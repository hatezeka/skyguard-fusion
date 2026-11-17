import numpy as np
import pytest

from app.datasets.camera_emulator import (
    CameraMetadata,
    emulate_camera,
    select_camera_profile,
)


@pytest.fixture
def sample_frame():
    frame = np.full(
        (120, 160, 3),
        180,
        dtype=np.uint8,
    )

    frame[40:80, 50:100] = 255
    return frame


@pytest.mark.parametrize(
    "profile",
    [
        "standard",
        "low_light",
        "low_resolution",
        "noisy",
        "blurred",
    ],
)
def test_all_profiles_preserve_dimensions(
    sample_frame, profile
):
    result = emulate_camera(
        sample_frame,
        profile=profile,
    )

    assert result.image.shape == sample_frame.shape
    assert result.image.dtype == np.uint8
    assert result.profile == profile


def test_standard_profile_unchanged(sample_frame):
    result = emulate_camera(
        sample_frame,
        profile="standard",
    )

    assert np.array_equal(
        result.image,
        sample_frame,
    )


def test_low_light_reduces_brightness(sample_frame):
    result = emulate_camera(
        sample_frame,
        profile="low_light",
    )

    assert result.image.mean() < sample_frame.mean()


def test_noisy_profile_is_deterministic(sample_frame):
    first = emulate_camera(
        sample_frame,
        profile="noisy",
        seed=10,
    )

    second = emulate_camera(
        sample_frame,
        profile="noisy",
        seed=10,
    )

    assert np.array_equal(
        first.image,
        second.image,
    )


def test_invalid_profile(sample_frame):
    with pytest.raises(ValueError):
        emulate_camera(
            sample_frame,
            profile="unknown",
        )


def test_invalid_frame():
    with pytest.raises(ValueError):
        emulate_camera(
            np.zeros((0, 0, 3), dtype=np.uint8)
        )


def test_unknown_camera_fallback():
    assert select_camera_profile(None) == "standard"


def test_low_resolution_camera_profile():
    metadata = CameraMetadata(
        width=320,
        height=240,
    )

    assert (
        select_camera_profile(metadata)
        == "low_resolution"
    )


def test_high_resolution_camera_profile():
    metadata = CameraMetadata(
        width=1920,
        height=1080,
    )

    assert (
        select_camera_profile(metadata)
        == "standard"
    )
