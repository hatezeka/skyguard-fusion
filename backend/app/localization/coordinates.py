"""Geographic calculations for SkyGuard FUSION."""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8


def destination_point(
    latitude: float,
    longitude: float,
    bearing_degrees: float,
    distance_m: float,
) -> tuple[float, float]:
    """Return destination latitude/longitude using a spherical Earth model."""

    if not (-90.0 <= latitude <= 90.0):
        raise ValueError("Latitude must be between -90 and 90")

    if not (-180.0 <= longitude <= 180.0):
        raise ValueError("Longitude must be between -180 and 180")

    if not all(
        math.isfinite(value)
        for value in (latitude, longitude, bearing_degrees, distance_m)
    ):
        raise ValueError("All inputs must be finite")

    if distance_m < 0:
        raise ValueError("Distance cannot be negative")

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)
    bearing = math.radians(bearing_degrees % 360.0)
    angular_distance = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing)
        * math.sin(angular_distance)
        * math.cos(lat1),
        math.cos(angular_distance)
        - math.sin(lat1) * math.sin(lat2),
    )

    destination_lat = math.degrees(lat2)
    destination_lon = (
        (math.degrees(lon2) + 180.0) % 360.0
    ) - 180.0

    return destination_lat, destination_lon


def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great-circle distance between two GPS positions."""

    for latitude in (lat1, lat2):
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("Invalid latitude")

    for longitude in (lon1, lon2):
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("Invalid longitude")

    latitude1 = math.radians(lat1)
    latitude2 = math.radians(lat2)

    delta_lat = latitude2 - latitude1
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(latitude1)
        * math.cos(latitude2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2.0
        * EARTH_RADIUS_M
        * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    )
