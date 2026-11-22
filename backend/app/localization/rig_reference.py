"""Local rig reference and WGS84 projection for independently validated stereo XYZ."""
from datetime import datetime, timedelta, timezone
import math
from typing import Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat

from app.localization.dual_map_projection_v1 import SiteWGS84Reference, _enu_to_wgs84


class RigReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: FiniteFloat = Field(ge=-90, le=90)
    longitude: FiniteFloat = Field(ge=-180, le=180)
    altitude_m: FiniteFloat | None = None
    altitude_datum: Literal["ellipsoidal", "orthometric"] | None = None
    source: Literal["manual", "phone", "gnss", "other"]
    horizontal_accuracy_m: FiniteFloat | None = Field(default=None, ge=0)
    vertical_accuracy_m: FiniteFloat | None = Field(default=None, ge=0)
    heading_deg: FiniteFloat | None = Field(default=None, ge=0, lt=360)
    heading_uncertainty_deg: FiniteFloat | None = Field(default=None, ge=0, le=180)
    pitch_uncertainty_deg: FiniteFloat | None = Field(default=None, ge=0, le=180)
    roll_uncertainty_deg: FiniteFloat | None = Field(default=None, ge=0, le=180)
    pitch_deg: FiniteFloat | None = Field(default=None, ge=-90, le=90)
    roll_deg: FiniteFloat | None = Field(default=None, ge=-180, le=180)
    antenna_offset_xyz_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    captured_at: AwareDatetime
    valid_until: AwareDatetime | None = None
    evidence_ref: str | None = Field(default=None, max_length=160)

    def status(self, now=None):
        now = now or datetime.now(timezone.utc)
        if self.valid_until and self.valid_until <= now:
            return "GEO_REFERENCE_STALE"
        if (self.altitude_datum != "ellipsoidal" or
            any(value is None for value in (self.altitude_m, self.heading_deg, self.pitch_deg,
                                            self.roll_deg, self.antenna_offset_xyz_m))):
            return "GEO_REFERENCE_INCOMPLETE"
        if (not (self.evidence_ref or "").strip() or not self.valid_until
            or self.captured_at > now + timedelta(minutes=1)
            or self.valid_until <= self.captured_at):
            return "GEO_REFERENCE_UNVERIFIED"
        return "GEO_REFERENCE_VALID"


def camera_xyz_to_enu(xyz, heading, pitch, roll):
    """Left optical axes right/down/forward, pitch up and roll right-down."""
    x, y, z = np.asarray(xyz, dtype=float)
    h, p, r = map(math.radians, (heading, pitch, roll))
    if not all(map(math.isfinite, (x, y, z, h, p, r))):
        raise ValueError("Non-finite rig input")
    sh, ch = math.sin(h), math.cos(h)
    sp, cp = math.sin(p), math.cos(p)
    sr, cr = math.sin(r), math.cos(r)
    right = np.array((ch, -sh, 0.))
    down = np.array((sh*sp, ch*sp, -cp))
    forward = np.array((sh*cp, ch*cp, sp))
    return (cr*x-sr*y)*right + (sr*x+cr*y)*down + z*forward


def project_verified_position(xyz, reference: RigReference, *, stereo_valid: bool, covariance_xyz_m2=None,
                              transform_sigma_m=None, gps_accuracy_is_one_sigma=False):
    if not stereo_valid or reference.status() != "GEO_REFERENCE_VALID":
        return None
    point = np.asarray(xyz, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all() or point[2] <= 0:
        return None
    # GPS coordinates describe the antenna. XYZ and antenna offset share
    # the left optical origin, so subtract the offset before rotation.
    angles = (reference.heading_deg, reference.pitch_deg, reference.roll_deg)
    offset_point = point - reference.antenna_offset_xyz_m
    enu = camera_xyz_to_enu(offset_point, *angles)
    site = SiteWGS84Reference(reference.latitude, reference.longitude, reference.altitude_m,
                             "stereo-rig", reference.source, True)
    latitude, longitude, altitude = _enu_to_wgs84(site, *enu)
    uncertainty = position_uncertainty(offset_point, angles, reference, covariance_xyz_m2,
                                       transform_sigma_m if gps_accuracy_is_one_sigma else None)
    return {"latitude": latitude, "longitude": longitude, "altitude_m": altitude,
            "altitude_datum": "ellipsoidal", "enu_m": enu.tolist(),
            **uncertainty}


def position_uncertainty(point, angles, ref, stereo_covariance, transform_sigma):
    """First-order ENU covariance for supplied independent 1-sigma inputs."""
    missing = {"uncertainty_status": "UNCERTAINTY_UNAVAILABLE"}
    sigmas = (ref.heading_uncertainty_deg, ref.pitch_uncertainty_deg, ref.roll_uncertainty_deg,
              ref.horizontal_accuracy_m, ref.vertical_accuracy_m, transform_sigma)
    if stereo_covariance is None or any(v is None or not math.isfinite(v) or v < 0 for v in sigmas):
        return missing
    cov = np.asarray(stereo_covariance, dtype=float)
    if (cov.shape != (3, 3) or not np.isfinite(cov).all() or
        not np.allclose(cov, cov.T, atol=1e-10) or np.linalg.eigvalsh(cov)[0] < -1e-10):
        return missing
    rotation = np.column_stack([camera_xyz_to_enu(axis, *angles) for axis in np.eye(3)])
    # Derivatives are per degree, matching the configured angular sigmas.
    step = .0001
    derivatives = []
    for direction in np.eye(3):
        plus = camera_xyz_to_enu(point, *(angles + step * direction))
        minus = camera_xyz_to_enu(point, *(angles - step * direction))
        derivatives.append((plus - minus) / (2 * step))
    jacobian = np.column_stack(derivatives)
    heading, pitch, roll, horizontal, vertical, transform = sigmas
    position = rotation @ cov @ rotation.T
    position += jacobian @ np.diag(np.square([heading, pitch, roll])) @ jacobian.T
    position += np.diag(np.square([horizontal, horizontal, vertical]) + transform**2)
    return {
        "uncertainty_status": "PROPAGATED_INDEPENDENT_1_SIGMA",
        "covariance_enu_m2": position.tolist(),
        "horizontal_rms_m": math.sqrt(max(0., position[0, 0] + position[1, 1])),
        "vertical_sigma_m": math.sqrt(max(0., position[2, 2])),
    }