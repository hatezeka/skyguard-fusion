import CameraDeviceDiscovery from "./CameraDeviceDiscovery";
import { useState, type FormEvent } from "react";

type CameraType = "laptop" | "usb" | "ip" | "ptz";

export type CameraSelection =
  | { source_type: "usb"; source_location: number }
  | { source_type: "rtsp"; source_location: string };

type CameraManagerProps = {
  onSelectCamera?: (source: CameraSelection) => void;
  disabled?: boolean;
};

type Camera = {
  id: string;
  name: string;
  type: CameraType;
  connection: string;
};

const cameraTypes: CameraType[] = [
  "laptop",
  "usb",
  "ip",
  "ptz",
];

export default function CameraManager({ onSelectCamera, disabled = false }: CameraManagerProps) {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [name, setName] = useState("");
  const [type, setType] = useState<CameraType>("laptop");
  const [connection, setConnection] = useState("");
  const [usbIndex, setUsbIndex] = useState(0);
  const [error, setError] = useState("");

  function addCamera(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    const cameraName = name.trim();
    const cameraConnection = connection.trim();

    if (!cameraName) {
      setError("Camera name is required.");
      return;
    }

    if ((type === "ip" || type === "ptz") && !cameraConnection) {
      setError("RTSP address is required for IP/PTZ cameras.");
      return;
    }

    if (
      (type === "ip" || type === "ptz") &&
      !/^rtsps?:\/\/[^/\s]+(?:\/[^\s]*)?$/i.test(cameraConnection)
    ) {
      setError("Enter a valid RTSP or RTSPS address.");
      return;
    }

    const camera: Camera = {
      id: crypto.randomUUID(),
      name: cameraName,
      type,
      connection:
        type === "ip" || type === "ptz"
          ? cameraConnection
          : String(usbIndex),
    };

    setCameras((previous) => [...previous, camera]);
    setName("");
    setConnection("");
  }

  function removeCamera(id: string) {
    setCameras((previous) =>
      previous.filter((camera) => camera.id !== id)
    );
  }

  return (
    <section
      style={{
        background: "#0d1912",
        border: "1px solid #294234",
        borderRadius: 7,
        padding: 20,
        color: "#c6e6d0",
      }}
    >
      <h2 style={{ color: "#5af36d", marginTop: 0 }}>
        Universal Camera Manager
      </h2>

      <p style={{ color: "#88aa96", fontSize: 13 }}>
        Configure laptop, USB, IP and PTZ cameras.
        Connections are not activated in this frontend-only step.
      </p>

      <form
        onSubmit={addCamera}
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: 12,
          marginBottom: 20,
        }}
      >
        <label>
          Camera Name
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Example: North Tower Camera"
            style={inputStyle}
          />
        </label>

        <label>
          Camera Type
          <select
            value={type}
            onChange={(event) =>
              setType(event.target.value as CameraType)
            }
            style={inputStyle}
          >
            {cameraTypes.map((cameraType) => (
              <option key={cameraType} value={cameraType}>
                {cameraType.toUpperCase()}
              </option>
            ))}
          </select>
        </label>

        {(type === "laptop" || type === "usb") && (
          <label>
            Backend USB Index (OpenCV)
            <select
              value={usbIndex}
              disabled={disabled}
              onChange={event => setUsbIndex(Number(event.target.value))}
              style={inputStyle}
            >
              {[0, 1, 2, 3, 4].map(index => (
                <option key={index} value={index}>
                  Camera {index}
                </option>
              ))}
            </select>
          </label>
        )}

        {(type === "ip" || type === "ptz") && (
          <label>
            RTSP Address
            <input
              value={connection}
              onChange={(event) =>
                setConnection(event.target.value)
              }
              placeholder="rtsp://camera-host/stream"
              style={inputStyle}
            />
          </label>
        )}

        <button
          type="submit"
          disabled={disabled}
          style={{
            padding: 12,
            background: "#166534",
            color: "white",
            border: 0,
            borderRadius: 6,
            cursor: "pointer",
            alignSelf: "end",
          }}
        >
          Add Camera
        </button>
      </form>

      {error && (
        <p role="alert" style={{ color: "#fca5a5" }}>
          {error}
        </p>
      )}

      <CameraDeviceDiscovery />

      <h3>Registered Cameras ({cameras.length})</h3>

      {cameras.length === 0 ? (
        <p style={{ color: "#88aa96" }}>
          No cameras configured yet.
        </p>
      ) : (
        <div style={{ display: "grid", gap: 10 }}>
          {cameras.map((camera) => (
            <div
              key={camera.id}
              style={{
                border: "1px solid #294234",
                borderRadius: 6,
                padding: 14,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 12,
                flexWrap: "wrap",
              }}
            >
              <div>
                <strong>{camera.name}</strong>
                <div
                  style={{
                    fontSize: 12,
                    color: "#88aa96",
                    marginTop: 5,
                  }}
                >
                  {camera.type.toUpperCase()} · Configured only
                </div>
              </div>

              {onSelectCamera && (
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => {
                    if (camera.type === "ip" || camera.type === "ptz") {
                      onSelectCamera({
                        source_type: "rtsp",
                        source_location: camera.connection,
                      });
                    } else {
                      onSelectCamera({
                        source_type: "usb",
                        source_location: Number(camera.connection),
                      });
                    }
                  }}
                  style={{
                    background: "#166534",
                    color: "white",
                    border: 0,
                    borderRadius: 5,
                    padding: "8px 12px",
                    cursor: "pointer",
                  }}
                >
                  Select for AI
                </button>
              )}
              <button
                type="button"
                disabled={disabled}
                onClick={() => removeCamera(camera.id)}
                style={{
                  background: "#321b1b",
                  color: "#ff7777",
                  border: "1px solid #613535",
                  padding: "8px 12px",
                  borderRadius: 5,
                  cursor: "pointer",
                }}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

const inputStyle = {
  display: "block",
  width: "100%",
  boxSizing: "border-box" as const,
  padding: 10,
  marginTop: 7,
  background: "#07120b",
  border: "1px solid #34543e",
  color: "#c6e6d0",
  borderRadius: 5,
};
