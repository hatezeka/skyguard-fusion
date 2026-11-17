from dataclasses import dataclass
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

        capture = cv2.VideoCapture(location)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Unable to open video source")

        self.capture = capture
        self.frame_index = 0

    def read(self) -> VideoFrame | None:
        if self.capture is None:
            raise RuntimeError("Video source is not open")

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

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
