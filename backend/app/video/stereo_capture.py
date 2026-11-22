"""Independent camera workers and bounded pairing by host read-completion time."""
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from time import perf_counter_ns
from zlib import crc32

import cv2

from app.video.source import UniversalVideoSource, VideoFrame, VideoSourceConfig


@dataclass(frozen=True)
class Captured:
    frame: VideoFrame
    timestamp_ns: int
    read_ms: float


class StereoCapture:
    size = (640, 480)

    def __init__(self, left_index, right_index, *, skew_ms=35, max_age_ms=150):
        if left_index == right_index or min(left_index, right_index) < 0:
            raise ValueError("Two distinct nonnegative camera indices required")
        self.indices = (left_index, right_index)
        self.skew_ns = int(skew_ms * 1e6)
        self.max_age_ns = int(max_age_ms * 1e6)
        self.condition = Condition()
        self.open_lock = Lock()
        self.stop = Event()
        self.queues = [deque(maxlen=4), deque(maxlen=4)]
        self.metrics = [{"index": index, "valid_frames": 0, "queue_drops": 0,
                         "unpaired_drops": 0, "frozen_drops": 0, "read_total_ms": 0.0}
                        for index in self.indices]
        self.error = None
        self.workers = []

    def __enter__(self):
        for side in range(2):
            worker = Thread(target=self._capture, args=(side,), daemon=True)
            self.workers.append(worker)
            worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        for worker in self.workers:
            worker.join(timeout=1)

    def _capture(self, side):
        index = self.indices[side]
        source = UniversalVideoSource(VideoSourceConfig("usb", index, self.size, "YUY2"))
        metrics = self.metrics[side]
        previous = None
        repeated = 0
        try:
            with ExitStack() as stack:
                with self.open_lock:
                    if self.stop.is_set():
                        return
                    stack.enter_context(source)
                with self.condition:
                    metrics.update(backend=source.capture.getBackendName(),
                                   fourcc=int(source.capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii"))
                while not self.stop.is_set():
                    start = perf_counter_ns()
                    frame = source.read()
                    timestamp = perf_counter_ns()
                    signature = crc32(frame.image.tobytes())
                    repeated = repeated + 1 if signature == previous else 0
                    previous = signature
                    with self.condition:
                        if repeated >= 2:
                            metrics["frozen_drops"] += 1
                            continue
                        read_ms = (timestamp - start) / 1e6
                        metrics["read_total_ms"] += read_ms
                        metrics["valid_frames"] += 1
                        metrics.setdefault("first_ns", timestamp)
                        metrics["last_ns"] = timestamp
                        metrics["actual_size"] = [frame.width, frame.height]
                        queue = self.queues[side]
                        if len(queue) == queue.maxlen:
                            metrics["queue_drops"] += 1
                        queue.append(Captured(frame, timestamp, read_ms))
                        self.condition.notify_all()
        except (RuntimeError, ValueError, cv2.error) as exc:
            with self.condition:
                self.error = f"Camera {index}: {exc}"
                self.condition.notify_all()

    def next_pair(self, timeout_s=15, cancel=None):
        deadline = perf_counter_ns() + int(timeout_s * 1e9)
        with self.condition:
            while not self.stop.is_set() and not self.error and not (cancel and cancel.is_set()):
                now = perf_counter_ns()
                for queue, metrics in zip(self.queues, self.metrics):
                    while queue and now - queue[0].timestamp_ns > self.max_age_ns:
                        queue.popleft()
                        metrics["unpaired_drops"] += 1
                left, right = self.queues
                while left and right:
                    skew = left[0].timestamp_ns - right[0].timestamp_ns
                    if abs(skew) <= self.skew_ns:
                        return left.popleft(), right.popleft()
                    older = 0 if skew < 0 else 1
                    self.queues[older].popleft()
                    self.metrics[older]["unpaired_drops"] += 1
                remaining = (deadline - now) / 1e9
                if remaining <= 0:
                    break
                self.condition.wait(min(remaining, .1) if cancel else remaining)
        if self.error:
            raise RuntimeError(self.error)
        raise TimeoutError("No fresh host-time stereo pair within tolerance")

    def report(self):
        with self.condition:
            cameras = {}
            for side, recorded in zip(("left", "right"), self.metrics):
                metrics = recorded.copy()
                count = metrics["valid_frames"]
                elapsed = metrics.pop("last_ns", 0) - metrics.pop("first_ns", 0)
                metrics["observed_fps"] = (count - 1) * 1e9 / elapsed if elapsed > 0 else None
                total = metrics.pop("read_total_ms")
                metrics["mean_read_ms"] = total / count if count else None
                cameras[side] = metrics
            return cameras


def capture_stereo_probe(left_index, right_index, samples):
    if samples < 1:
        raise ValueError("Positive sample count required")
    result = {"timestamp_semantics": "HOST_READ_RETURN_MONOTONIC_NOT_EXPOSURE",
              "exposure_sync_verified": False, "cameras": {}, "pairs": []}
    with StereoCapture(left_index, right_index) as capture:
        try:
            for _ in range(samples):
                left, right = capture.next_pair()
                result["pairs"].append({
                    "left_frame_id": left.frame.index, "right_frame_id": right.frame.index,
                    "left_capture_timestamp_ns": left.timestamp_ns,
                    "right_capture_timestamp_ns": right.timestamp_ns,
                    "pair_timestamp_ns": max(left.timestamp_ns, right.timestamp_ns),
                    "timestamp_skew_ms": abs(left.timestamp_ns - right.timestamp_ns) / 1e6,
                    "left_read_ms": left.read_ms, "right_read_ms": right.read_ms,
                    "left_frame_valid": True, "right_frame_valid": True,
                    "host_pair_valid": True, "frame_pair_valid": False,
                })
        except (TimeoutError, RuntimeError) as exc:
            result["error"] = str(exc)
    result["cameras"] = capture.report()
    return result
