import json

import pytest

from app.datasets.benchmark_validator import (
    validate_annotations,
    validate_split_integrity,
)
from app.datasets.scenario_generator import generate_scenario


@pytest.mark.parametrize(
    "scenario",
    ["crossing", "occlusion", "multiple_drones"],
)
def test_valid_synthetic_annotations(tmp_path, scenario):
    _, path = generate_scenario(
        tmp_path, scenario, frames=30
    )

    result = validate_annotations(path)

    assert result["valid"]
    assert result["frame_count"] == 30
    assert result["unique_tracks"] >= 1


def test_occlusion_has_hidden_instances(tmp_path):
    _, path = generate_scenario(
        tmp_path, "occlusion", frames=100
    )

    result = validate_annotations(path)

    assert result["hidden_instances"] > 0


def test_duplicate_track_rejected(tmp_path):
    _, path = generate_scenario(
        tmp_path, "crossing", frames=10
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    data["annotations"][0]["objects"][1]["track_id"] = 1

    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate track"):
        validate_annotations(path)


def test_frame_sequence_rejected(tmp_path):
    _, path = generate_scenario(
        tmp_path, "crossing", frames=10
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    data["annotations"][0]["frame_index"] = 99

    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="frame sequence"):
        validate_annotations(path)


def test_valid_split_integrity():
    records = [
        {"source_id": "camera-A", "split": "train"},
        {"source_id": "camera-A", "split": "train"},
        {"source_id": "camera-B", "split": "validation"},
        {"source_id": "camera-C", "split": "test"},
    ]

    result = validate_split_integrity(records)

    assert result["valid"]
    assert result["sources"] == 3


def test_source_leakage_rejected():
    records = [
        {"source_id": "camera-A", "split": "train"},
        {"source_id": "camera-A", "split": "test"},
    ]

    with pytest.raises(ValueError, match="leakage"):
        validate_split_integrity(records)


def test_invalid_split_rejected():
    records = [
        {"source_id": "camera-A", "split": "unknown"},
    ]

    with pytest.raises(ValueError, match="split"):
        validate_split_integrity(records)
