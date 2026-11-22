import json

import cv2
import numpy as np
import pytest

from app.api.media_upload import _summary
from skyguard_fix.cli import evaluate_file
from skyguard_fix.core import Detection, merge_object_proposals


@pytest.mark.parametrize("labels", [
    ["DRONE", "BIRD"], ["DRONE"], ["BIRD"], ["UNKNOWN"], [],
    ["DRONE", "DRONE"], ["BIRD"] * 3, ["DRONE", "BIRD", "BIRD"],
    ["DRONE", "DRONE", "BIRD", "BIRD", "UNKNOWN", "UNKNOWN"],
])
def test_image_summary_from_final_detections(tmp_path, labels):
    path = tmp_path / "detections.jsonl"
    rows = [
        {"label": label, "confidence": .9, "xyxy": [i * 10, 2, i * 10 + 5, 7]}
        for i, label in enumerate(labels)
    ]
    path.write_text(json.dumps({"source_frame_index": 0, "detections": rows}))
    result = _summary(path, image=True)
    assert result["counts"] == {label: labels.count(label) for label in ("DRONE", "BIRD", "UNKNOWN")}
    assert result["total_detected_objects"] == len(labels)
    assert all(d["track_id"] is None for d in result["detections"])
    assert result["track_paths_px"] == []


def test_image_compares_high_resolution_even_if_first_detection_is_confident(tmp_path):
    class Detector:
        names = {0: "DRONE", 1: "BIRD"}

        def predict(self, image, imgsz, conf):
            results = [(0, .95, [10, 10, 80, 80])]
            if imgsz > 640:
                results += [(1, .82, [45, 25, 100, 80])]
            return results

    source = tmp_path / "image.png"
    assert cv2.imwrite(str(source), np.zeros((128, 128, 3), dtype=np.uint8))
    folder = tmp_path / "results"
    evaluate_file(source, folder, Detector())
    result = _summary(folder / "detections.jsonl", image=True)
    assert result["counts"] == {"DRONE": 1, "BIRD": 1, "UNKNOWN": 0}


def test_near_identical_conflicting_proposals_count_as_one_unknown(tmp_path):
    proposals = merge_object_proposals([
        Detection("DRONE", .81, (1, 1, 45, 43), "full"),
        Detection("BIRD", .8, (1, 1, 44, 43), "tile"),
    ])
    assert len(proposals) == 1
    path = tmp_path / "detections.jsonl"
    path.write_text(json.dumps({"source_frame_index": 0, "detections": [
        {"label": proposals[0].label, "xyxy": proposals[0].xyxy, "verdict_status": proposals[0].verdict_status}
    ]}))
    result = _summary(path, image=True)
    assert result["counts"] == {"DRONE": 0, "BIRD": 0, "UNKNOWN": 1}
    assert result["detections"][0]["confidence"] is None


def test_missing_checkpoint_returns_service_unavailable(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from app.api.media_upload import _require_model_file

    monkeypatch.setenv("SKYGUARD_MODEL_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(HTTPException) as failure:
        _require_model_file()
    assert failure.value.status_code == 503


def test_overlapping_views_can_replace_a_composite_box():
    from skyguard_fix.core import multiscale

    class Detector:
        def predict(self, image, imgsz, conf):
            if image.shape[1] == 120:
                return [(0, .87, [0, 12, 120, 85])]
            if image[0, 0, 0] == 0:
                return [(1, .89, [3, 14, 51, 72])]
            return [(0, .84, [30, 38, 78, 80])]

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, 40:, 0] = 1
    result = multiscale(image, Detector(), {0:"DRONE", 1:"BIRD"}, mode="full")
    assert {d.label: d.xyxy for d in result} == {
        "DRONE": (70, 38, 118, 80), "BIRD": (3, 14, 51, 72)
    }
