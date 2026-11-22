"""Visual correspondence for host-time pairs. Metric output requires physical validation."""
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.localization.stereo_calibration import StereoCalibration


@dataclass(frozen=True)
class StereoObservation:
    track_id: int
    classification: str
    right_bbox: tuple[float, float, float, float] | None
    matched_points: int
    range_status: str
    xyz_m: None = None
    range_m: None = None
    wgs84: None = None


def load_stereo_profile(path):
    if not path:
        return None
    return StereoCalibration.load(Path(path))


def stereo_observations(left, right, tracks, right_detections, profile):
    # Use one identity namespace anchored in the primary view. Right-view
    # evidence enriches an existing observation and never creates an ID.
    extractor = cv2.ORB_create(nfeatures=120, edgeThreshold=8, patchSize=15)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(image, box):
        x, y = int(box.x), int(box.y)
        crop = image[y:int(box.y + box.height), x:int(box.x + box.width)]
        if min(crop.shape[:2]) < 16:
            return [], None
        points, descriptors = extractor.detectAndCompute(crop, None)
        return [(p.pt[0] + x, p.pt[1] + y) for p in points], descriptors

    candidates = []
    peers = [(d, features(right, d.bbox)) for d in right_detections
             if profile is not None and d.object_class in ('drone', 'bird')]
    for track in tracks:
        detection = track.detection
        if detection.object_class not in ('drone', 'bird') or not peers:
            continue
        lp, ld = features(left, detection.bbox)
        if ld is None:
            continue
        for index, (peer, (rp, rd)) in enumerate(peers):
            if peer.object_class != detection.object_class or rd is None:
                continue
            forward = matcher.knnMatch(ld, rd, k=2)
            reverse = {m.queryIdx: m.trainIdx for m in matcher.match(rd, ld)}
            inliers = 0
            for neighbors in forward:
                if len(neighbors) != 2:
                    continue
                a, b = neighbors
                if a.distance >= .75 * b.distance or reverse.get(a.trainIdx) != a.queryIdx:
                    continue
                try:
                    profile.triangulate_rectified(lp[a.queryIdx], rp[a.trainIdx], max_epipolar_px=3)
                    inliers += 1
                except ValueError:
                    continue
            if inliers >= 4:
                candidates.append((track.track_id, index, inliers))

    output = []
    for track in tracks:
        label = track.detection.object_class
        if label not in ('drone', 'bird'):
            continue
        options = [item for item in candidates if item[0] == track.track_id]
        box, support = None, 0
        if len(options) == 1:
            _, index, points = options[0]
            if sum(item[1] == index for item in candidates) == 1:
                bbox = peers[index][0].bbox
                box, support = (bbox.x, bbox.y, bbox.width, bbox.height), points
        status = ('CALIBRATION_UNAVAILABLE' if profile is None else
                  'RANGE_UNAVAILABLE' if box is None else 'CALIBRATION_SUSPECT')
        output.append(StereoObservation(track.track_id, label.upper(), box, support, status))
    return output
