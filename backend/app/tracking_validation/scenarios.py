from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticObject:
    identity: str
    x: float
    y: float
    object_class: str = "drone"


@dataclass(frozen=True)
class SyntheticFrame:
    frame_index: int
    objects: tuple[SyntheticObject, ...]


def crossing_scenario(
    frames: int = 21,
    separation: float = 100.0,
) -> list[SyntheticFrame]:
    if frames < 2:
        raise ValueError("frames must be at least 2")
    if separation <= 0:
        raise ValueError("separation must be positive")

    sequence = []

    for index in range(frames):
        progress = index / (frames - 1)

        first = SyntheticObject(
            identity="DRONE_A",
            x=20.0 + separation * progress,
            y=50.0,
        )

        second = SyntheticObject(
            identity="DRONE_B",
            x=20.0 + separation * (1.0 - progress),
            y=50.0,
        )

        sequence.append(
            SyntheticFrame(
                frame_index=index,
                objects=(first, second),
            )
        )

    return sequence


def parallel_scenario(
    object_count: int,
    frames: int = 20,
) -> list[SyntheticFrame]:
    if object_count < 1:
        raise ValueError("object_count must be positive")
    if frames < 1:
        raise ValueError("frames must be positive")

    return [
        SyntheticFrame(
            frame_index=index,
            objects=tuple(
                SyntheticObject(
                    identity=f"DRONE_{number:03d}",
                    x=20.0 + index * 2.0,
                    y=20.0 + number * 35.0,
                )
                for number in range(object_count)
            ),
        )
        for index in range(frames)
    ]
