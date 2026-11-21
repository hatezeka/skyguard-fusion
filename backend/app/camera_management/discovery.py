"""Opt-in local USB camera probing. Never stores profiles or accesses GPS."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable
import math
import cv2

@dataclass(frozen=True)
class CameraProbe:
    index: int
    available: bool
    width_px: int | None = None
    height_px: int | None = None
    fps_reported: float | None = None
    fps_verified: bool = False
    error: str | None = None

    def as_dict(self):
        return asdict(self)


def _positive_int(value):
    try:
        n = float(value)
        return int(round(n)) if math.isfinite(n) and n >= 1 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_float(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def probe_usb(index: int, capture_factory: Callable | None = None, *, read_frame: bool = True) -> CameraProbe:
    """Probe a *user-selected* local index; may activate the camera LED.

    No scanning, frame storage, network access, or registration occurs.
    FPS is driver-reported and must not be treated as measured throughput.
    """
    if type(index) is not int or not 0 <= index <= 32:
        raise ValueError('USB index must be an integer from 0 to 32')
    factory = capture_factory or cv2.VideoCapture
    capture = None
    try:
        capture = factory(index)
        if not capture.isOpened():
            return CameraProbe(index, False, error='device_unavailable')
        width = _positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _positive_float(capture.get(cv2.CAP_PROP_FPS))
        if read_frame:
            ok, frame = capture.read()
            if not ok or frame is None or getattr(frame, 'ndim', None) not in (2, 3):
                return CameraProbe(index, False, error='frame_unavailable')
            height, width = frame.shape[:2]
        if width is None or height is None:
            return CameraProbe(index, False, error='resolution_unavailable')
        return CameraProbe(index, True, width, height, fps)
    except (OSError, RuntimeError, cv2.error) as exc:
        return CameraProbe(index, False, error=f'probe_failed:{type(exc).__name__}')
    finally:
        if capture is not None:
            capture.release()


def probe_selected(indices, capture_factory=None, *, read_frame=True):
    if len(indices) > 8 or len(set(indices)) != len(indices):
        raise ValueError('Select at most 8 distinct camera indices')
    return [probe_usb(i, capture_factory, read_frame=read_frame) for i in indices]
