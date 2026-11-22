"""Probe the configured OpenCV capture path without saving camera images."""
import argparse
import json
from statistics import median

import cv2
from time import perf_counter

from app.video.source import UniversalVideoSource, VideoSourceConfig


def measure(index, mode):
    try:
        start = perf_counter()
        with UniversalVideoSource(VideoSourceConfig("usb", index, mode)) as source:
            frame = source.read()
            timestamps = [perf_counter()]
            codec = source.capture.get(cv2.CAP_PROP_FOURCC)
            for _ in range(19):
                source.read()
                timestamps.append(perf_counter())
        intervals = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]
        return {
            "samples": len(timestamps),
            "actual_resolution": [frame.width, frame.height],
            "mjpg_confirmed_by_readback": codec == cv2.VideoWriter_fourcc(*"MJPG"),
            "first_frame_ms": (timestamps[0] - start) * 1000,
            "capture_fps": len(intervals) / (timestamps[-1] - timestamps[0]),
            "median_interframe_ms": median(intervals),
        }
    except RuntimeError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    modes = ((640, 480), (1280, 720), (1920, 1080))
    print(json.dumps({
        "index": args.index,
        "modes": {f"{w}x{h}": measure(args.index, (w, h)) for w, h in modes},
    }, indent=2))
