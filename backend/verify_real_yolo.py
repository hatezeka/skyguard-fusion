from pathlib import Path
import shutil
import numpy as np

from ultralytics import YOLO
from app.detection.yolo import YOLODetectionEngine

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "yolo11n.pt"

if not model_path.is_file():
    print("Downloading YOLO model...")
    model = YOLO("yolo11n.pt")
    source = Path(model.ckpt_path).resolve()
    if source != model_path.resolve():
        shutil.copy2(source, model_path)

print("Model:", model_path)
print("Model size:", model_path.stat().st_size, "bytes")

engine = YOLODetectionEngine(
    model_path=str(model_path),
    confidence_threshold=0.25,
)

names = engine.model.names
classes = set(names.values()) if isinstance(names, dict) else set(names)

print("Bird class available:", "bird" in classes)
print("Drone class available:", "drone" in classes)

image = np.zeros((320, 320, 3), dtype=np.uint8)
detections = engine.detect(image)

print("Inference completed successfully")
print("Detections:", len(detections))

for detection in detections:
    print(detection)

assert "bird" in classes
assert "drone" not in classes
