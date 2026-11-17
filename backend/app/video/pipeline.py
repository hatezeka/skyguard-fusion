from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from app.datasets.camera_emulator import (
    CameraProfile,
    emulate_camera,
)
from app.video.source import (
    UniversalVideoSource,
    VideoFrame,
    VideoSourceConfig,
)


@dataclass(frozen=True)
class ProcessedFrame:
    index: int
    image: np.ndarray
    profile: CameraProfile
    width: int
    height: int


@dataclass(frozen=True)
class PipelineResult:
    frames_processed: int
    elapsed_seconds: float
    processing_fps: float


class VideoProcessingPipeline:
    def __init__(
        self,
        config: VideoSourceConfig,
        profile: CameraProfile = "standard",
    ):
        self.config = config
        self.profile = profile

    def run(
        self,
        on_frame: Callable[[ProcessedFrame], None] | None = None,
        max_frames: int | None = None,
    ) -> PipelineResult:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")

        frames_processed = 0
        started = perf_counter()

        with UniversalVideoSource(self.config) as source:
            while True:
                if (
                    max_frames is not None
                    and frames_processed >= max_frames
                ):
                    break

                frame: VideoFrame | None = source.read()

                if frame is None:
                    break

                transformed = emulate_camera(
                    frame.image,
                    profile=self.profile,
                    seed=frame.index,
                )

                processed = ProcessedFrame(
                    index=frame.index,
                    image=transformed.image,
                    profile=self.profile,
                    width=frame.width,
                    height=frame.height,
                )

                if on_frame is not None:
                    on_frame(processed)

                frames_processed += 1

        elapsed = perf_counter() - started

        return PipelineResult(
            frames_processed=frames_processed,
            elapsed_seconds=elapsed,
            processing_fps=(
                frames_processed / elapsed
                if elapsed > 0
                else 0.0
            ),
        )
