from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


CameraProfile = Literal[
    "standard",
    "low_light",
    "low_resolution",
    "noisy",
    "blurred",
]


@dataclass(frozen=True)
class CameraMetadata:
    manufacturer: str | None = None
    model: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class FrameTransform:
    image: np.ndarray
    profile: CameraProfile
    source_width: int
    source_height: int


def select_camera_profile(
    metadata: CameraMetadata | None,
) -> CameraProfile:
    if metadata is None:
        return "standard"

    if metadata.width is not None and metadata.width <= 640:
        return "low_resolution"

    return "standard"


def emulate_camera(
    frame: np.ndarray,
    profile: CameraProfile = "standard",
    seed: int = 42,
) -> FrameTransform:
    allowed = {
        "standard",
        "low_light",
        "low_resolution",
        "noisy",
        "blurred",
    }

    if profile not in allowed:
        raise ValueError("Unknown camera profile")

    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
    ):
        raise ValueError("Expected non-empty uint8 BGR frame")

    height, width = frame.shape[:2]
    result = frame.copy()

    if profile == "low_light":
        result = cv2.convertScaleAbs(
            result,
            alpha=0.35,
            beta=0,
        )

    elif profile == "low_resolution":
        reduced_width = max(1, width // 4)
        reduced_height = max(1, height // 4)

        reduced = cv2.resize(
            result,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        )

        result = cv2.resize(
            reduced,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    elif profile == "noisy":
        rng = np.random.default_rng(seed)

        noise = rng.normal(
            0,
            25,
            result.shape,
        )

        result = np.clip(
            result.astype(np.float32) + noise,
            0,
            255,
        ).astype(np.uint8)

    elif profile == "blurred":
        result = cv2.GaussianBlur(
            result,
            (9, 9),
            sigmaX=2,
        )

    return FrameTransform(
        image=result,
        profile=profile,
        source_width=width,
        source_height=height,
    )
