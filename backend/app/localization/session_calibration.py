"""Immutable, minimal, per-session registry-calibration provenance.

This is **software identity/provenance** only. A registry's CALIBRATED label
is NOT independent proof of the camera's physical calibration or orientation.
The frozen record omits location, source URLs, credentials, and all non-optical
fields; the camera's field of view/range is never fabricated here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class FrozenSessionCalibration:
    camera_id: str
    session_id: str
    canonical_json: str
    sha256: str

    def record(self) -> dict:
        """Return a fresh record, never the session's authoritative payload."""
        if hashlib.sha256(self.canonical_json.encode('utf-8')).hexdigest() != self.sha256:
            raise ValueError('Frozen calibration checksum mismatch')
        result = json.loads(self.canonical_json)
        if result.get('camera_id') != self.camera_id:
            raise ValueError('Frozen calibration camera identity mismatch')
        return result


def freeze_session_calibration(
    registry_record: Mapping | None,
    camera_id: str | None,
    session_id: str,
) -> FrozenSessionCalibration | None:
    """Snapshot only the explicitly registered optical/pose data.

    Accepts UNCALIBRATED records: the strict projection endpoint will reject
    them, even if the mutable registry is calibrated later in the session.
    """
    if registry_record is None:
        return None
    if not isinstance(registry_record, Mapping):
        raise ValueError('Camera registry record must be a mapping')
    if not isinstance(camera_id, str) or not camera_id or registry_record.get('camera_id') != camera_id:
        raise ValueError('Registry camera identity mismatch at session start')
    if not isinstance(session_id, str) or len(session_id) != 32:
        raise ValueError('Session identity unavailable for calibration binding')

    # Nested allowlists matter too: a future registry extension must never
    # make a camera URL or credential part of a session calibration snapshot.
    def fields(value: object, keys: tuple[str, ...]) -> dict | None:
        if not isinstance(value, Mapping):
            return None
        return {key: value.get(key) for key in keys}

    calibration = registry_record.get('calibration')
    clean_cal = fields(calibration, ('status', 'resolution', 'verified_at', 'profile_id', 'validation_status'))
    if clean_cal is not None:
        clean_cal['intrinsics'] = fields(
            calibration.get('intrinsics'),
            ('fx', 'fy', 'cx', 'cy', 'distortion'),
        )
    slim = {
        'camera_id': camera_id,
        'stream': fields(registry_record.get('stream'), ('width_px', 'height_px')),
        'calibration': clean_cal,
        'orientation': fields(
            registry_record.get('orientation'),
            ('yaw_deg', 'pitch_deg', 'roll_deg', 'source', 'north_reference', 'evidence_ref'),
        ),
    }
    try:
        payload = json.dumps(
            slim, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError('Registry calibration cannot be frozen safely') from exc
    return FrozenSessionCalibration(
        camera_id=camera_id,
        session_id=session_id,
        canonical_json=payload,
        sha256=hashlib.sha256(payload.encode('utf-8')).hexdigest(),
    )
