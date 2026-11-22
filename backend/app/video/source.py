from dataclasses import dataclass
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import cv2
import numpy as np


SourceType = Literal["file", "usb", "rtsp"]


@dataclass(frozen=True)
class VideoSourceConfig:
    source_type: SourceType
    location: str | int
    usb_mode: tuple[int, int] | None = None
    usb_fourcc: str | None = None
    stereo_right_index: int | None = None


@dataclass(frozen=True)
class VideoFrame:
    index: int
    image: np.ndarray
    width: int
    height: int


class UniversalVideoSource:
    def __init__(self, config: VideoSourceConfig):
        self.config = config
        self.capture = None
        self.frame_index = 0

    def _resolve_location(self):
        kind = self.config.source_type
        location = self.config.location

        if kind == "file":
            if not isinstance(location, str):
                raise ValueError("File source requires a path")

            path = Path(location)

            if not path.is_file():
                raise FileNotFoundError(path)

            return str(path)

        if kind == "usb":
            if type(location) is not int or location < 0:
                raise ValueError("USB source requires camera index")

            return location

        if kind == "rtsp":
            if not isinstance(location, str):
                raise ValueError("RTSP source requires URL")

            parsed = urlsplit(location)

            if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
                raise ValueError("Invalid RTSP URL")

            return location

        raise ValueError("Unsupported source type")

    def open(self):
        if self.capture is not None:
            raise RuntimeError("Video source already open")

        location = self._resolve_location()

        capture = (
            cv2.VideoCapture(location, cv2.CAP_DSHOW)
            if self.config.usb_mode else cv2.VideoCapture(location)
        )

        if not capture.isOpened():
            capture.release()
            raise RuntimeError("USB camera unavailable" if self.config.usb_mode else "Unable to open video source")

        if self.config.usb_mode:
            width, height = self.config.usb_mode
            codec = cv2.VideoWriter_fourcc(*(self.config.usb_fourcc or "MJPG"))
            capture.set(cv2.CAP_PROP_FOURCC, codec)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Some DirectShow drivers do not expose FOURCC readback.
            reported_codec = capture.get(cv2.CAP_PROP_FOURCC)
            accepted = (codec,) if self.config.usb_fourcc else (-1, 0, codec)
            if reported_codec not in accepted:
                capture.release()
                raise RuntimeError(f"USB camera reported a different codec ({reported_codec:g})")

        self.capture = capture
        self.frame_index = 0

    def read(self) -> VideoFrame | None:
        if self.capture is None:
            raise RuntimeError("Video source is not open")

        if self.config.usb_mode:
            image = self._read_configured_usb()
            ok = True
        else:
            ok, image = self.capture.read()

        if not ok or image is None:
            return None

        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("Invalid decoded frame")

        height, width = image.shape[:2]

        frame = VideoFrame(
            index=self.frame_index,
            image=image,
            width=width,
            height=height,
        )

        self.frame_index += 1
        return frame

    def _read_configured_usb(self):
        width, height = self.config.usb_mode
        for attempt in range(2):
            usable = 0
            required = 3 if self.frame_index == 0 or attempt else 1
            for _ in range(12):
                ok, image = self.capture.read()
                if ok and isinstance(image, np.ndarray) and image.dtype == np.uint8:
                    if image.ndim == 3 and image.shape[2] == 3 and image.size:
                        if image.shape[:2] != (height, width):
                            raise RuntimeError(
                                f"USB resolution mismatch: requested {width}x{height}, "
                                f"received {image.shape[1]}x{image.shape[0]}"
                            )
                        if np.max(image) > 8:
                            usable += 1
                            if usable >= required:
                                return image
                            continue
                usable = 0
                time.sleep(0.04)
            if attempt == 0:
                frame_index = self.frame_index
                self.close()
                self.open()
                self.frame_index = frame_index
        raise RuntimeError("USB camera returned black or invalid frames after reconnect")

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
