"""Frame/optics identity, geometry and non-inference safety contracts."""
import math

import pytest

from app.localization.camera_geometry import CameraIntrinsics
from app.localization.camera_profile import CameraGeoProfile
from app.localization.checkerboard_calibration import CheckerboardCalibration
from app.localization.world_bearing import CameraOrientation
from app.localization.live_direction_adapter import project_frame_detections


def camera_profile():
    return CameraGeoProfile(
        camera_id="camera-A", latitude=42.0, longitude=21.0,
        heading_degrees=90.0, horizontal_fov_degrees=90.0,
        coverage_radius_m=1000.0,
    )


def intrinsics(provenance="measured"):
    return CameraIntrinsics(
        width_px=1280, height_px=720,
        fx_px=640.0, fy_px=640.0,
        cx_px=640.0, cy_px=360.0,
        provenance=provenance,
    )


def orientation():
    return CameraOrientation(
        heading_degrees=90.0,
        north_reference="true_north",
        orientation_source="manual_unverified",
    )


def detection(*, label="drone", center=(640.0, 360.0), confidence=0.9):
    return {
        "class": label, "confidence": confidence,
        "bbox": {
            "x": center[0] - 5, "y": center[1] - 5,
            "width": 10.0, "height": 10.0,
        },
    }


def frame(detections=None):
    return {
        "camera_id": "camera-A", "session_id": "session-1",
        "frame_index": 12, "state": "running",
        "image_width_px": 1280, "image_height_px": 720,
        "detections": [detection()] if detections is None else detections,
    }


def calculate(data=None, **options):
    return project_frame_detections(
        frame() if data is None else data,
        profile=options.pop("profile", camera_profile()),
        orientation=options.pop("orientation", orientation()),
        optics=options.pop("optics", intrinsics()),
        **options,
    )


def test_center_drone_bearing_east_without_target_gps_or_range():
    result = calculate()
    item = result.directions[0]
    assert result.camera_id == "camera-A"
    assert result.frame_index == 12
    assert result.session_id == "session-1"
    assert item.object_class == "drone"
    assert item.direction.azimuth_degrees == pytest.approx(90)
    assert item.direction.elevation_degrees == pytest.approx(0)
    assert item.direction.orientation_source == "manual_unverified"
    assert item.direction.optics_source == "measured"
    assert not hasattr(item, "distance_m")
    assert not hasattr(item, "latitude")
    assert not hasattr(result, "target_gps")


def test_two_detections_produce_separate_bearings():
    data = frame([
        detection(center=(640, 360)),
        detection(label="bird", center=(960, 360)),
    ])
    result = calculate(data)
    assert len(result.directions) == 2
    assert [item.detection_index for item in result.directions] == [0, 1]
    assert [item.object_class for item in result.directions] == ["drone", "bird"]
    assert result.directions[0].direction.azimuth_degrees == pytest.approx(90)
    assert result.directions[1].direction.azimuth_degrees == pytest.approx(116.565051, abs=1e-5)


def test_unknown_is_not_promoted_to_confirmed_drone():
    item = calculate(frame([detection(label="unknown")])).directions[0]
    assert item.object_class == "unknown"
    assert not hasattr(item, "confirmed_drone")
    assert not hasattr(item, "track_id")


def test_empty_detections_are_supported():
    result = calculate(frame([]))
    assert result.directions == ()
    assert result.frame_index == 12


def test_caller_declared_atomic_provenance_is_preserved_not_verified():
    result = calculate(frame_source="declared_atomic_frame")
    assert result.frame_source == "declared_atomic_frame"
    assert not hasattr(result, "verified_atomic")


def test_nominal_fov_and_magnetic_north_not_reclassified():
    result = calculate(
        optics=intrinsics("nominal_fov"),
        orientation=CameraOrientation(90, "magnetic_north"),
    )
    direction = result.directions[0].direction
    assert direction.optics_source == "nominal_fov"
    assert direction.north_reference == "magnetic_north"
    assert not hasattr(direction, "true_north_azimuth_degrees")


def test_pixel_center_uses_bbox_midpoint_not_top_left():
    data = frame([detection(center=(640, 360))])
    data["detections"][0]["bbox"]["x"] = 630
    data["detections"][0]["bbox"]["width"] = 20
    item = calculate(data).directions[0]
    assert item.center_x_px == pytest.approx(640)
    assert item.direction.azimuth_degrees == pytest.approx(90)


def test_rejects_mismatched_profile_camera():
    data = frame()
    data["camera_id"] = "camera-B"
    with pytest.raises(ValueError, match="camera_id"):
        calculate(data)


def test_rejects_heading_mismatch_without_implicit_rotation():
    with pytest.raises(ValueError, match="heading"):
        calculate(orientation=CameraOrientation(89, "true_north"))


def test_rejects_optical_resolution_mismatch():
    data = frame()
    data["image_width_px"] = 640
    with pytest.raises(ValueError, match="dimensions"):
        calculate(data)


def test_rejects_stale_detection_frame():
    item = detection()
    item["frame_index"] = 11
    with pytest.raises(ValueError, match="frame_index"):
        calculate(frame([item]))


def test_rejects_wrong_detection_camera():
    item = detection()
    item["camera_id"] = "camera-B"
    with pytest.raises(ValueError, match="camera_id"):
        calculate(frame([item]))


@pytest.mark.parametrize("bad", [
    {"x": 1275, "y": 350, "width": 15, "height": 10},
    {"x": -2, "y": 350, "width": 10, "height": 10},
    {"x": 630, "y": 350, "width": 0, "height": 10},
    {"x": 630, "y": 350, "width": float("nan"), "height": 10},
    {"x": 630, "y": True, "width": 10, "height": 10},
])
def test_invalid_boxes_fail_closed(bad):
    item = detection()
    item["bbox"] = bad
    with pytest.raises(ValueError, match="bbox"):
        calculate(frame([item]))


@pytest.mark.parametrize("invalid", [float("nan"), -0.1, 1.1, True])
def test_invalid_confidence_fails_closed(invalid):
    with pytest.raises(ValueError, match="confidence"):
        calculate(frame([detection(confidence=invalid)]))


def test_bad_label_rejected():
    with pytest.raises(ValueError, match="class"):
        calculate(frame([detection(label="confirmed_enemy")]))


def test_nonrunning_frame_rejected():
    data = frame()
    data["state"] = "stopped"
    with pytest.raises(ValueError, match="non-running"):
        calculate(data)


def test_missing_frame_index_rejected():
    data = frame()
    del data["frame_index"]
    with pytest.raises(ValueError, match="frame_index"):
        calculate(data)


def test_invalid_source_rejected():
    with pytest.raises(ValueError, match="frame_source"):
        calculate(frame_source="unproven_verified")


def test_atomic_projection_is_still_only_angular():
    result = calculate()
    item = result.directions[0]
    direction = item.direction
    assert math.hypot(direction.east, direction.north, direction.up) == pytest.approx(1)
    assert not hasattr(direction, "distance_m")
    assert not hasattr(direction, "latitude")
    assert not hasattr(direction, "longitude")


def test_distortion_calibration_changes_off_axis_angle():
    # k1 > 0 means observed radius grows: undistortion returns a smaller yaw.
    calibration = CheckerboardCalibration(
        intrinsics=intrinsics(),
        distortion=(0.2, 0.0, 0.0, 0.0, 0.0),
        rms_px=0.2,
        per_view_rmse_px=(0.2,) * 6,
        views=6,
        board_columns=7,
        board_rows=5,
    )
    data = frame([detection(center=(960, 360))])
    undistorted = calculate(data, optics=calibration).directions[0].direction
    uncorrected = calculate(data).directions[0].direction
    assert undistorted.azimuth_degrees < uncorrected.azimuth_degrees
    assert undistorted.optics_source == "measured"


def test_nonhashable_label_is_rejected_as_validation_error():
    data = frame([detection()])
    data["detections"][0]["class"] = ["drone"]
    with pytest.raises(ValueError, match="class"):
        calculate(data)


def test_invalid_session_id_is_rejected():
    data = frame()
    data["session_id"] = " "
    with pytest.raises(ValueError, match="session_id"):
        calculate(data)
