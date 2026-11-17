from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass(frozen=True)
class ModelCapabilities:
    model_path: str
    classes: tuple[str, ...]
    supports_drone: bool
    supports_bird: bool
    ready_for_drone_bird: bool


def inspect_model(model_path: str) -> ModelCapabilities:
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    model = YOLO(str(path))
    names = model.names

    if isinstance(names, dict):
        classes = tuple(str(value).strip().lower() for value in names.values())
    else:
        classes = tuple(str(value).strip().lower() for value in names)

    drone_aliases = {"drone", "uav", "quadcopter"}

    supports_drone = any(name in drone_aliases for name in classes)
    supports_bird = "bird" in classes

    return ModelCapabilities(
        model_path=str(path),
        classes=classes,
        supports_drone=supports_drone,
        supports_bird=supports_bird,
        ready_for_drone_bird=supports_drone and supports_bird,
    )


def require_drone_bird_model(model_path: str) -> ModelCapabilities:
    capabilities = inspect_model(model_path)

    if not capabilities.ready_for_drone_bird:
        raise ValueError(
            "Model does not support both drone and bird classes"
        )

    return capabilities
