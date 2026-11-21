"""Experimental motion-aware multi-object tracker.

Coordinates and velocities are in image pixels and pixels/frame.
This is not a physical 3D motion estimator.
"""

from dataclasses import dataclass
from math import hypot, isfinite

from app.detection.engine import Detection


@dataclass(frozen=True)
class MotionObservation:
    track_id: int
    detection_index: int
    detection: Detection


@dataclass
class _Track:
    track_id: int
    object_class: str
    x: float
    y: float
    vx: float
    vy: float
    last_frame: int
    missing: int = 0


def _hungarian(cost):
    """Minimum-cost rectangular assignment, rows <= columns."""
    n = len(cost)
    if not n:
        return []

    m = len(cost[0])
    if m < n:
        raise ValueError("Assignment requires columns >= rows")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, m + 1):
                if used[j]:
                    continue

                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]

                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0

                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1

    return result


class MotionAwareTracker:
    def __init__(
        self,
        max_distance=50.0,
        max_missing_frames=5,
        velocity_alpha=1.0,
    ):
        if not isfinite(max_distance) or max_distance <= 0:
            raise ValueError("Invalid max_distance")
        if isinstance(max_missing_frames, bool) or (
            not isinstance(max_missing_frames, int)
            or max_missing_frames < 0
        ):
            raise ValueError("Invalid max_missing_frames")
        if not isfinite(velocity_alpha) or not 0 < velocity_alpha <= 1:
            raise ValueError("Invalid velocity_alpha")

        self.max_distance = float(max_distance)
        self.max_missing_frames = max_missing_frames
        self.velocity_alpha = float(velocity_alpha)
        self.reset()

    def reset(self):
        self._tracks = {}
        self._next_id = 1
        self._last_frame = None

    @staticmethod
    def _center(detection):
        box = detection.bbox
        return (
            box.x + box.width / 2.0,
            box.y + box.height / 2.0,
        )

    def update(self, frame_index: int, detections: list[Detection]):
        if isinstance(frame_index, bool) or (
            not isinstance(frame_index, int) or frame_index < 0
        ):
            raise ValueError("Invalid frame index")

        if self._last_frame is not None:
            if frame_index != self._last_frame + 1:
                raise ValueError("Frames must be consecutive")

        for detection in detections:
            detection.validate(
                image_width=10**9,
                image_height=10**9,
            )

        centers = [self._center(d) for d in detections]
        tracks = sorted(
            self._tracks.values(),
            key=lambda t: t.track_id,
        )

        n = len(tracks)
        k = len(detections)
        penalty = self.max_distance + 1.0
        invalid = penalty * 1000000.0

        costs = []
        for track in tracks:
            dt = frame_index - track.last_frame
            px = track.x + track.vx * dt
            py = track.y + track.vy * dt

            row = []
            for detection, (cx, cy) in zip(detections, centers):
                distance = hypot(cx - px, cy - py)

                if (
                    detection.object_class != track.object_class
                    or distance > self.max_distance
                ):
                    row.append(invalid)
                else:
                    # Prefer recently observed tracks in near-ties.
                    # Keep the penalty small relative to pixel distance.
                    row.append(distance + 0.01 * track.missing)

            # One dummy column per track permits unmatched tracks.
            row.extend([penalty] * n)
            costs.append(row)

        assignments = _hungarian(costs)
        matched_detections = set()
        observations = []

        for i, track in enumerate(tracks):
            j = assignments[i] if assignments else -1

            if 0 <= j < k and costs[i][j] < penalty:
                cx, cy = centers[j]
                dt = frame_index - track.last_frame

                measured_vx = (cx - track.x) / dt
                measured_vy = (cy - track.y) / dt

                alpha = self.velocity_alpha
                track.vx = (
                    alpha * measured_vx
                    + (1 - alpha) * track.vx
                )
                track.vy = (
                    alpha * measured_vy
                    + (1 - alpha) * track.vy
                )

                track.x = cx
                track.y = cy
                track.last_frame = frame_index
                track.missing = 0

                matched_detections.add(j)
                observations.append(
                    MotionObservation(
                        track.track_id, j, detections[j]
                    )
                )
            else:
                track.missing = frame_index - track.last_frame

        for track in tracks:
            if track.missing > self.max_missing_frames:
                del self._tracks[track.track_id]

        for j, detection in enumerate(detections):
            if j in matched_detections:
                continue

            cx, cy = centers[j]
            track_id = self._next_id
            self._next_id += 1

            self._tracks[track_id] = _Track(
                track_id=track_id,
                object_class=detection.object_class,
                x=cx,
                y=cy,
                vx=0.0,
                vy=0.0,
                last_frame=frame_index,
            )

            observations.append(
                MotionObservation(track_id, j, detection)
            )

        self._last_frame = frame_index
        return sorted(observations, key=lambda o: o.track_id)
