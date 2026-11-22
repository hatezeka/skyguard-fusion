"""Strict import of the existing V3 stereo profile and rectified point geometry."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    sha256: str
    resolution: tuple[int, int]
    matrix: dict[str, np.ndarray]
    baseline_mm: float
    holdout_epipolar_px: float

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        if data["resolution_px"] != [640, 480]:
            raise ValueError("Expected an exact 640x480 stereo profile")
        sizes = {"K1": (3, 3), "K2": (3, 3), "R": (3, 3), "R1": (3, 3),
                 "R2": (3, 3), "P1": (3, 4), "P2": (3, 4), "Q": (4, 4)}
        matrix = {}
        for key in (*sizes, "D1", "D2", "T_mm_nominal"):
            value = np.asarray(data[key], dtype=float)
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {key}")
            if key in sizes and value.shape != sizes[key]:
                raise ValueError(f"Invalid shape for {key}")
            if key in ("D1", "D2") and (value.ndim > 2 or value.size not in (4, 5, 8, 12, 14)):
                raise ValueError(f"Invalid distortion for {key}")
            if key == "T_mm_nominal" and value.shape not in ((3,), (3, 1), (1, 3)):
                raise ValueError("Invalid translation")
            value.setflags(write=False)
            matrix[key] = value
        for key in ("K1", "K2", "P1", "P2"):
            if min(matrix[key][0, 0], matrix[key][1, 1]) <= 0:
                raise ValueError(f"Invalid focal length in {key}")
        for key in ("R", "R1", "R2"):
            rotation = matrix[key]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
                raise ValueError(f"Invalid rotation in {key}")
        baseline = float(data["physical_baseline_mm"])
        holdout = float(data["holdout_epipolar_mean_px"])
        projected_baseline = abs(matrix["P2"][0, 3] / matrix["P2"][0, 0])
        if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(holdout) or holdout < 0:
            raise ValueError("Invalid baseline or holdout error")
        if not all(np.isclose(v, baseline, rtol=0.02) for v in (
            projected_baseline, np.linalg.norm(matrix["T_mm_nominal"]),
        )):
            raise ValueError("Baseline inconsistent with translation or rectified projection")
        return cls(hashlib.sha256(raw).hexdigest(), (640, 480), matrix, baseline, holdout)

    def triangulate_rectified(self, left, right, *, max_epipolar_px):
        """Reconstruct a verified homologous point in the rectified left frame, in metres."""
        if not np.isfinite(max_epipolar_px) or max_epipolar_px <= 0:
            raise ValueError("Invalid epipolar tolerance")
        rectified = []
        for point, side in ((left, "1"), (right, "2")):
            pixel = np.asarray(point, dtype=float)
            if pixel.shape != (2,) or not np.isfinite(pixel).all() or not (
                0 <= pixel[0] < self.resolution[0] and 0 <= pixel[1] < self.resolution[1]
            ):
                raise ValueError("Invalid image point")
            rectified.append(cv2.undistortPoints(
                pixel.reshape(1, 1, 2), self.matrix["K" + side], self.matrix["D" + side].ravel(),
                R=self.matrix["R" + side], P=self.matrix["P" + side],
            ).reshape(2))
        a, b = rectified
        if not np.isfinite(rectified).all() or abs(a[1] - b[1]) > max_epipolar_px:
            raise ValueError("Epipolar residual exceeded")
        p1, p2 = self.matrix["P1"], self.matrix["P2"]
        disparity = (a[0] - p1[0, 2]) - (b[0] - p2[0, 2])
        if abs(disparity) < 1:
            raise ValueError("Insufficient disparity")
        homogeneous = cv2.triangulatePoints(p1, p2, a.reshape(2, 1), b.reshape(2, 1)).ravel()
        if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-10:
            raise ValueError("Unstable triangulation")
        point = homogeneous / homogeneous[3]
        if min((p1 @ point)[2], (p2 @ point)[2]) <= 0 or not np.isfinite(point).all():
            raise ValueError("Non-positive or invalid depth")
        return point[:3] / 1000
