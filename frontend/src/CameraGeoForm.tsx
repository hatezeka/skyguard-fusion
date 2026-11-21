import { useState, type FormEvent } from "react";

type Props = {
  onSaved: () => void;
};

type FormValues = {
  camera_id: string;
  latitude: string;
  longitude: string;
  heading_degrees: string;
  horizontal_fov_degrees: string;
  coverage_radius_m: string;
  position_accuracy_m: string;
};

const initialValues: FormValues = {
  camera_id: "camera-0",
  latitude: "",
  longitude: "",
  heading_degrees: "",
  horizontal_fov_degrees: "",
  coverage_radius_m: "",
  position_accuracy_m: "",
};

export default function CameraGeoForm({ onSaved }: Props) {
  const [values, setValues] = useState<FormValues>(initialValues);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  function update(field: keyof FormValues, value: string) {
    setValues((previous) => ({
      ...previous,
      [field]: value,
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (saving) return;

    setError("");
    setMessage("");

    const cameraId = values.camera_id.trim();

    const required = [
      "latitude",
      "longitude",
      "heading_degrees",
      "horizontal_fov_degrees",
      "coverage_radius_m",
    ] as const;

    if (!cameraId || required.some((field) => !values[field].trim())) {
      setError("Please complete all required fields.");
      return;
    }

    const latitude = Number(values.latitude);
    const longitude = Number(values.longitude);
    const heading = Number(values.heading_degrees);
    const fov = Number(values.horizontal_fov_degrees);
    const radius = Number(values.coverage_radius_m);

    const accuracyText = values.position_accuracy_m.trim();
    const accuracy = accuracyText === "" ? null : Number(accuracyText);

    const valid =
      cameraId.length <= 128 &&
      !cameraId.includes("/") &&
      !cameraId.includes("?") &&
      !cameraId.includes("#") &&
      [latitude, longitude, heading, fov, radius].every(Number.isFinite) &&
      latitude >= -90 &&
      latitude <= 90 &&
      longitude >= -180 &&
      longitude <= 180 &&
      heading >= 0 &&
      heading < 360 &&
      fov > 0 &&
      fov <= 180 &&
      radius > 0 &&
      radius <= 100000 &&
      (accuracy === null ||
        (Number.isFinite(accuracy) && accuracy >= 0));

    if (!valid) {
      setError("Invalid GPS or camera configuration values.");
      return;
    }

    const payload = {
      camera_id: cameraId,
      latitude,
      longitude,
      heading_degrees: heading,
      horizontal_fov_degrees: fov,
      coverage_radius_m: radius,
      position_source: "manual" as const,
      position_accuracy_m: accuracy,
    };

    setSaving(true);

    try {
      const response = await fetch(
        `/api/v1/geo/cameras/${encodeURIComponent(cameraId)}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;

        try {
          const body = await response.json();
          if (typeof body.detail === "string") {
            detail = body.detail;
          }
        } catch {
          // Keep HTTP status if response is not JSON.
        }

        throw new Error(detail);
      }

      setMessage(`Camera ${cameraId} saved successfully.`);
      onSaved();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to save camera configuration."
      );
    } finally {
      setSaving(false);
    }
  }

  const fields: {
    key: keyof FormValues;
    label: string;
    placeholder: string;
  }[] = [
    {
      key: "camera_id",
      label: "Camera ID",
      placeholder: "camera-0",
    },
    {
      key: "latitude",
      label: "Latitude",
      placeholder: "Camera latitude",
    },
    {
      key: "longitude",
      label: "Longitude",
      placeholder: "Camera longitude",
    },
    {
      key: "heading_degrees",
      label: "Heading (degrees)",
      placeholder: "0 to less than 360",
    },
    {
      key: "horizontal_fov_degrees",
      label: "Horizontal FOV (degrees)",
      placeholder: "Greater than 0, up to 180",
    },
    {
      key: "coverage_radius_m",
      label: "Theoretical radius (meters)",
      placeholder: "Configured radius, not measured range",
    },
    {
      key: "position_accuracy_m",
      label: "Position accuracy (meters, optional)",
      placeholder: "Leave blank if unknown",
    },
  ];

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        display: "grid",
        gap: 12,
        marginBottom: 20,
      }}
    >
      <h3 style={{ margin: 0 }}>Camera GPS Configuration</h3>

      <p style={{ margin: 0, color: "#94aaa7", fontSize: 12 }}>
        Enter the actual camera location manually.
        Do not enter the laptop location unless the camera
        is physically installed there.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
          gap: 12,
        }}
      >
        {fields.map((field) => (
          <label
            key={field.key}
            style={{
              display: "grid",
              gap: 5,
              fontSize: 12,
            }}
          >
            {field.label}

            <input
              type="text"
              inputMode={
                field.key === "camera_id" ? "text" : "decimal"
              }
              value={values[field.key]}
              placeholder={field.placeholder}
              onChange={(event) =>
                update(field.key, event.target.value)
              }
              disabled={saving}
              style={{
                padding: 10,
                background: "#071512",
                border: "1px solid #294448",
                borderRadius: 6,
                color: "#e2f6ee",
                minWidth: 0,
              }}
            />
          </label>
        ))}
      </div>

      <button
        type="submit"
        disabled={saving}
        style={{
          justifySelf: "start",
          padding: "10px 18px",
          background: "#166534",
          color: "#ffffff",
          border: 0,
          borderRadius: 6,
          cursor: saving ? "wait" : "pointer",
        }}
      >
        {saving ? "Saving..." : "Save Camera GPS"}
      </button>

      {error && (
        <p role="alert" style={{ color: "#fca5a5", margin: 0 }}>
          {error}
        </p>
      )}

      {message && (
        <p role="status" style={{ color: "#86efac", margin: 0 }}>
          {message}
        </p>
      )}
    </form>
  );
}
