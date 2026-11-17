from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VideoManifest(BaseModel):
    video_id: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    source_type: Literal["real", "synthetic"]
    camera_profile: str = "standard"

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)

    annotations_path: str | None = None
    split: Literal["train", "validation", "test"]
    scenario_tags: list[str] = Field(default_factory=list)
    license: str = Field(min_length=1)

    @field_validator("scenario_tags")
    @classmethod
    def validate_tags(cls, tags: list[str]) -> list[str]:
        allowed = {
            "drone",
            "bird",
            "other",
            "crossing",
            "occlusion",
            "small_object",
            "low_light",
            "multiple_drones",
        }

        unknown = set(tags) - allowed

        if unknown:
            raise ValueError(
                f"Unknown scenario tags: {sorted(unknown)}"
            )

        return tags


def validate_video_file(
    manifest: VideoManifest,
    dataset_root: Path,
) -> bool:
    root = dataset_root.resolve()
    video_path = (root / manifest.file_path).resolve()

    if not video_path.is_relative_to(root):
        raise ValueError("Video path escapes dataset root")

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    return True
