from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class ObjectClass(str, Enum):
    drone = 'drone'
    bird = 'bird'
    other = 'other'
    unknown = 'unknown'

class CameraProfile(BaseModel):
    camera_id: str = Field(min_length=1)
    manufacturer: str | None = None
    model: str | None = None
    profile_id: str = 'standard'
    profile_source: Literal['fallback', 'model_specific', 'manual'] = 'fallback'
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    calibration_available: bool = False

class Detection(BaseModel):
    detection_id: str
    camera_id: str
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    class_name: ObjectClass
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[float, float, float, float]

    @model_validator(mode='after')
    def valid_bbox(self):
        x1,y1,x2,y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Bounding box must have positive width and height')
        return self

class PositionEstimate(BaseModel):
    method: Literal['unknown', 'bearing_only', 'calibrated', 'estimated'] = 'unknown'
    bearing_deg: float | None = None
    distance_m: float | None = Field(default=None, ge=0)
    distance_uncertainty_m: float | None = Field(default=None, ge=0)
    x_m: float | None = None
    y_m: float | None = None

    @model_validator(mode='after')
    def no_fake_coordinates(self):
        if self.method in ('unknown', 'bearing_only') and any(v is not None for v in (self.distance_m, self.x_m, self.y_m)):
            raise ValueError('Unknown/bearing-only methods cannot assert distance or position')
        if (self.x_m is None) != (self.y_m is None):
            raise ValueError('x_m and y_m must both be set or both be absent')
        return self

class TrackedObject(BaseModel):
    track_id: str
    camera_id: str
    class_name: ObjectClass
    tracking_state: Literal['tentative', 'confirmed', 'lost', 'ended']
    last_seen_ms: int = Field(ge=0)
    position: PositionEstimate = Field(default_factory=PositionEstimate)
