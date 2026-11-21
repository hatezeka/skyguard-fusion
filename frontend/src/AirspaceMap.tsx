import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polygon,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import CameraGeoForm from "./CameraGeoForm";
import type { LiveStatus } from "./api/liveCamera";

type GeoPoint = {
  latitude: number;
  longitude: number;
};

type CameraGeoProfile = {
  camera_id: string;
  latitude: number;
  longitude: number;
  heading_degrees: number;
  horizontal_fov_degrees: number;
  coverage_radius_m: number;
  position_source: "manual" | "gnss";
  position_accuracy_m: number | null;
  updated_at: string | null;
  coverage_polygon: GeoPoint[];
};

const cameraIcon = L.divIcon({
  className: "skyguard-camera-marker",
  html: `
    <div style="
      width:28px;
      height:28px;
      border-radius:50%;
      background:#10b981;
      border:3px solid white;
      box-shadow:0 0 12px rgba(16,185,129,.6);
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:14px;
    ">📷</div>
  `,
  iconSize: [28, 28],
  iconAnchor: [14, 14],
});

function CameraBounds({
  cameras,
}: {
  cameras: CameraGeoProfile[];
}) {
  const map = useMap();

  useEffect(() => {
    if (cameras.length === 0) return;

    const bounds = L.latLngBounds(
      cameras.map((camera) => [
        camera.latitude,
        camera.longitude,
      ])
    );

    cameras.forEach((camera) => {
      camera.coverage_polygon.forEach((point) => {
        bounds.extend([
          point.latitude,
          point.longitude,
        ]);
      });
    });

    map.fitBounds(bounds, {
      padding: [35, 35],
      maxZoom: 16,
    });
  }, [cameras, map]);

  return null;
}

type AirspaceMapProps = {
  liveStatus: LiveStatus;
  connected: boolean;
};

export default function AirspaceMap({
  liveStatus,
  connected,
}: AirspaceMapProps) {
  const [cameras, setCameras] = useState<CameraGeoProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadCameras() {
      try {
        const response = await fetch(
          "/api/v1/geo/cameras",
          { cache: "no-store" }
        );

        if (!response.ok) {
          throw new Error(
            `Geo API returned HTTP ${response.status}`
          );
        }

        const data: CameraGeoProfile[] =
          await response.json();

        if (active) {
          setCameras(data);
          setError(null);
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load camera GPS data"
          );
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    void loadCameras();

    const interval = window.setInterval(
      loadCameras,
      5000
    );

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [refreshKey]);

  return (
    <section
      style={{
        background: "#102226",
        border: "1px solid #294448",
        borderRadius: 14,
        padding: 16,
        color: "#e2f6ee",
      }}
    >
      <header style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>
          Live Airspace Map
        </h2>
        <small style={{ color: "#94aaa7" }}>
          Camera GPS and theoretical coverage
        </small>
      </header>

      <div
        style={{
          background: "#071512",
          border: "1px solid #294448",
          borderRadius: 10,
          padding: 14,
          marginBottom: 16,
        }}
      >
        <h3 style={{ marginTop: 0 }}>Live AI Surveillance</h3>

        <p>
          Backend: {connected ? "Connected" : "Disconnected"}
          {" | "}
          Session: {liveStatus.state.toUpperCase()}
        </p>

        <p>
          Frame: {liveStatus.frame_index ?? "N/A"}
          {" | "}
          Detections: {
            Array.isArray(liveStatus.detections)
              ? liveStatus.detections.length
              : 0
          }
          {" | "}
          Tracks: {
            Array.isArray(liveStatus.tracks)
              ? liveStatus.tracks.length
              : 0
          }
        </p>

        {Array.isArray(liveStatus.tracks) &&
          liveStatus.tracks.map((track: unknown, index: number) => {
            if (
              typeof track !== "object" ||
              track === null ||
              !("track_id" in track) ||
              !("class" in track)
            ) {
              return null;
            }

            return (
              <div key={index} style={{ fontSize: 13, marginBottom: 5 }}>
                Object #{String(track.track_id)}
                {" — "}
                {String(track.class)}
                {" — "}
                {("state" in track) ? String(track.state) : "unknown"}
              </div>
            );
          })}

        <small style={{ color: "#94aaa7" }}>
          Detections are camera-frame observations.
          Object GPS coordinates are not yet validated.
        </small>
      </div>

      <CameraGeoForm
        onSaved={() => setRefreshKey((previous) => previous + 1)}
      />

      {loading && <p>Loading camera GPS data...</p>}

      {error && (
        <p role="alert" style={{ color: "#fca5a5" }}>
          {error}
        </p>
      )}

      {!loading && !error && cameras.length === 0 && (
        <p>
          No camera GPS profiles configured yet.
        </p>
      )}

      {!loading && !error && cameras.length > 0 && (
        <MapContainer
          center={[
            cameras[0].latitude,
            cameras[0].longitude,
          ]}
          zoom={13}
          style={{
            height: 420,
            width: "100%",
            borderRadius: 10,
          }}
        >
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <CameraBounds cameras={cameras} />

          {cameras.map((camera) => (
            <Marker
              key={camera.camera_id}
              position={[
                camera.latitude,
                camera.longitude,
              ]}
              icon={cameraIcon}
            >
              <Popup>
                <strong>{camera.camera_id}</strong>
                <div>
                  GPS: {camera.latitude.toFixed(6)},
                  {" "}
                  {camera.longitude.toFixed(6)}
                </div>
                <div>
                  Source: {camera.position_source}
                </div>
                <div>
                  Heading: {camera.heading_degrees}°
                </div>
                <div>
                  FOV: {camera.horizontal_fov_degrees}°
                </div>
                <div>
                  Configured coverage radius:
                  {" "}
                  {camera.coverage_radius_m} m
                </div>
              </Popup>
            </Marker>
          ))}

          {cameras.map((camera) => (
            <Polygon
              key={`${camera.camera_id}-coverage`}
              positions={camera.coverage_polygon.map(
                (point): [number, number] => [
                  point.latitude,
                  point.longitude,
                ]
              )}
              pathOptions={{
                color: "#10b981",
                fillColor: "#10b981",
                fillOpacity: 0.13,
                weight: 2,
              }}
            />
          ))}
        </MapContainer>
      )}

      <p
        style={{
          fontSize: 12,
          color: "#94aaa7",
          marginBottom: 0,
        }}
      >
        Coverage sectors are theoretical. Drone positions
        and measured ranges are not yet available.
      </p>
    </section>
  );
}



