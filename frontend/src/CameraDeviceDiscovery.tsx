import { useEffect, useRef, useState } from "react";

type CameraDevice = {
  deviceId: string;
  label: string;
};

export default function CameraDeviceDiscovery() {
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [previewing, setPreviewing] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  function stopPreview() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setPreviewing(false);
  }

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  async function discover() {
    setBusy(true);
    setError("");
    setMessage("");

    try {
      if (!navigator.mediaDevices?.enumerateDevices) {
        throw new Error(
          "Camera discovery requires localhost or a secure HTTPS context."
        );
      }

      const allDevices = await navigator.mediaDevices.enumerateDevices();

      const cameras = allDevices
        .filter((device) => device.kind === "videoinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${index + 1}`,
        }));

      setDevices(cameras);

      setSelectedId((previous) =>
        cameras.some((camera) => camera.deviceId === previous)
          ? previous
          : cameras[0]?.deviceId ?? ""
      );

      setMessage(
        `${cameras.length} video device(s) visible to this browser.`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function requestPermission() {
    setBusy(true);
    setError("");
    setMessage("");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is unavailable in this browser.");
      }

      if (streamRef.current) {
        throw new Error("Stop the current preview first.");
      }

      const temporaryStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      temporaryStream.getTracks().forEach((track) => track.stop());

      setMessage("Camera permission request completed.");
      await discover();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function startPreview() {
    if (busy || previewing) return;

    setBusy(true);
    setError("");
    setMessage("");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera preview is unavailable.");
      }

      if (!selectedId) {
        throw new Error("Select a camera first.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: selectedId },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setPreviewing(true);
    } catch (err) {
      stopPreview();
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
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
        Local Camera Discovery
      </h2>

      <p style={{ color: "#88aa96", fontSize: 13 }}>
        Discover and preview cameras available to this browser.
        Preview video stays local and is not sent to SkyGuard AI.
      </p>

      <div
        style={{
          display: "flex",
          gap: 10,
          flexWrap: "wrap",
          marginBottom: 15,
        }}
      >
        <button onClick={() => void discover()} disabled={busy}>
          Discover Cameras
        </button>

        <button
          onClick={() => void requestPermission()}
          disabled={busy || previewing}
        >
          Allow Camera Access
        </button>
      </div>

      <label style={{ display: "block", marginBottom: 12 }}>
        Available Cameras

        <select
          value={selectedId}
          onChange={(event) => {
            stopPreview();
            setSelectedId(event.target.value);
          }}
          disabled={busy || previewing}
          style={{
            display: "block",
            marginTop: 8,
            width: "100%",
            padding: 10,
            background: "#07120b",
            border: "1px solid #34543e",
            color: "#c6e6d0",
          }}
        >
          {devices.length === 0 && (
            <option value="">No cameras discovered</option>
          )}

          {devices.map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label}
            </option>
          ))}
        </select>
      </label>

      <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
        <button
          onClick={() => void startPreview()}
          disabled={busy || previewing || !selectedId}
        >
          Start Preview
        </button>

        <button
          onClick={stopPreview}
          disabled={!previewing}
        >
          Stop Preview
        </button>
      </div>

      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{
          display: previewing ? "block" : "none",
          width: "100%",
          maxWidth: 700,
          aspectRatio: "16 / 9",
          objectFit: "contain",
          background: "#030806",
          borderRadius: 6,
        }}
      />

      {message && (
        <p role="status" style={{ color: "#86efac" }}>
          {message}
        </p>
      )}

      {error && (
        <p role="alert" style={{ color: "#fca5a5" }}>
          {error}
        </p>
      )}
    </section>
  );
}
