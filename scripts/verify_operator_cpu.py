"""Compare CPU thread counts on the existing adaptive inference pipeline."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from skyguard_fix.core import UltralyticsPredictor, multiscale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "video", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    try:
        frames = []
        for _ in range(8):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Expected at least eight real frames")
            frames.append(frame)
    finally:
        capture.release()
    predictor = UltralyticsPredictor(args.weights)
    reports = []
    reference = None
    for threads in (4, 2, 1):
        torch.set_num_threads(threads)
        multiscale(frames[0], predictor, predictor.names, include_tiles=True)
        times, signatures = [], []
        for frame in frames:
            start = time.perf_counter()
            detections = multiscale(frame, predictor, predictor.names, include_tiles=True)
            times.append((time.perf_counter() - start) * 1000)
            signatures.append([
                [d.label, d.verdict_status, round(d.confidence, 4), [round(n, 1) for n in d.xyxy]]
                for d in detections
            ])
        if reference is None:
            reference = signatures
        row = dict(threads=threads, frames=len(times), mean_ms=statistics.mean(times),
                   median_ms=statistics.median(times), same_rounded_output=signatures == reference)
        reports.append(row)
        print(json.dumps(row), flush=True)
    args.report.write_text(json.dumps({"source": args.video.name, "runs": reports}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
