import json

import pytest

from app.datasets.scenario_generator import generate_scenario
from app.datasets.video_inspector import inspect_video


@pytest.mark.parametrize(
    "scenario,expected_objects",
    [
        ("crossing", 2),
        ("occlusion", 1),
        ("multiple_drones", 4),
    ],
)
def test_generate_scenario(
    tmp_path, scenario, expected_objects
):
    video, annotations = generate_scenario(
        tmp_path, scenario, frames=20
    )

    assert video.is_file()
    assert annotations.is_file()

    metadata = inspect_video(video)

    assert metadata.width == 640
    assert metadata.height == 360
    assert metadata.frame_count == 20

    payload = json.loads(
        annotations.read_text(encoding="utf-8")
    )

    assert len(payload["annotations"]) == 20

    for frame in payload["annotations"]:
        assert len(frame["objects"]) == expected_objects


def test_occlusion_has_hidden_frames(tmp_path):
    _, annotations = generate_scenario(
        tmp_path, "occlusion", frames=100
    )

    payload = json.loads(
        annotations.read_text(encoding="utf-8")
    )

    visibility = [
        frame["objects"][0]["visible"]
        for frame in payload["annotations"]
    ]

    assert any(not value for value in visibility)
    assert any(visibility)


def test_invalid_scenario(tmp_path):
    with pytest.raises(ValueError):
        generate_scenario(tmp_path, "invalid")
