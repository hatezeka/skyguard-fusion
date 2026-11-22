"""Read the original calibration profile and collect an optional local camera probe."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.localization.stereo_calibration import StereoCalibration
from app.video.stereo_capture import capture_stereo_probe


def static_checkerboards(profile, directory, reference_mm, max_epipolar_px):
    samples = []
    matrices = profile.matrix
    midpoint_rectified_m = (
        matrices["R1"] @ (-matrices["R"].T @ matrices["T_mm_nominal"].reshape(3))
    ) / 2000
    for left_file in sorted(Path(directory).glob("*_laptop.png")):
        right_file = left_file.with_name(left_file.name.replace("_laptop.png", "_usb.png"))
        row = {"sample": left_file.stem.removesuffix("_laptop")}
        try:
            corners = []
            for path in (left_file, right_file):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape[::-1] != profile.resolution:
                    raise ValueError("Missing image or incompatible resolution")
                found, points = cv2.findChessboardCornersSB(image, (9, 6))
                if not found:
                    raise ValueError("Checkerboard not found")
                corners.append(points.reshape(-1, 2))
            # OpenCV may index the same symmetric board in opposite directions.
            reconstructions = []
            for right in (corners[1], corners[1][::-1]):
                try:
                    cloud = [profile.triangulate_rectified(a, b, max_epipolar_px=max_epipolar_px)
                             for a, b in zip(corners[0], right, strict=True)]
                    reconstructions.append(cloud)
                except ValueError:
                    continue
            if len(reconstructions) != 1:
                raise ValueError("No unique checkerboard correspondence within the epipolar gate")
            center = np.mean(reconstructions[0], axis=0)
            range_mm = float(np.linalg.norm(center - midpoint_rectified_m) * 1000)
            row.update(range_mm=range_mm, signed_error_mm=range_mm - reference_mm)
        except (ValueError, cv2.error) as exc:
            row["rejection"] = str(exc)
        samples.append(row)
    return {"reference_mm": reference_mm, "max_epipolar_px": max_epipolar_px,
            "reference_origin": "OPTICAL_CENTER_MIDPOINT_TO_STATIC_BOARD_CENTER",
            "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--checkerboards")
    parser.add_argument("--reference-mm", type=float)
    parser.add_argument("--max-epipolar-px", type=float)
    args = parser.parse_args()
    if args.checkerboards and not (
        args.reference_mm and args.max_epipolar_px
        and np.isfinite([args.reference_mm, args.max_epipolar_px]).all()
        and min(args.reference_mm, args.max_epipolar_px) > 0
    ):
        parser.error("Checkerboard replay requires positive reference and epipolar threshold")
    profile = StereoCalibration.load(args.profile)
    result = {
        "profile_sha256": profile.sha256, "resolution": profile.resolution,
        "baseline_mm": profile.baseline_mm, "holdout_epipolar_mean_px": profile.holdout_epipolar_px,
        "operational_status": "CALIBRATION_SUSPECT",
    }
    if args.capture:
        result["capture"] = capture_stereo_probe(args.left, args.right, 10)
    if args.checkerboards:
        result["static_checkerboard_replay"] = static_checkerboards(
            profile, args.checkerboards, args.reference_mm, args.max_epipolar_px
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
