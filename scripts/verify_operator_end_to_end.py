"""Verify the local operator HTTP stack against a SQLite backup and real media."""
import argparse
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import math
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import httpx
import uvicorn
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


def wait_until(check, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.25)
    raise TimeoutError("Verification condition did not complete")


def excerpt(source, output, count):
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or fps <= 0:
            raise ValueError(f"Cannot read source clip or its frame rate: {source}")
        for index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Only {index} source frames could be decoded")
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Could not write the verification excerpt")
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def verify(args, temp):
    video = args.video
    if args.expected_video_frames is not None and args.video_frames:
        raise ValueError("Frame-count assertions require the original, untrimmed input")
    if args.video_frames:
        video = temp / "source-excerpt.mp4"
        excerpt(args.video, video, args.video_frames)

    database = temp / "verification.sqlite3"
    with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(database)) as target:
            source.backup(target)
            assert target.execute("SELECT version_num FROM alembic_version").fetchone() == ("0002",)
            assert target.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert not target.execute("PRAGMA foreign_key_check").fetchall()

    sys.path.insert(0, str(BACKEND))
    os.environ["SKYGUARD_DATABASE_URL"] = "sqlite:///" + database.as_posix()
    os.environ["SKYGUARD_DB_API_KEY"] = os.environ["SKYGUARD_API_KEY"] = key = secrets.token_hex(32)
    os.environ["SKYGUARD_MODEL_PATH"] = str(args.weights.resolve())
    os.environ["YOLO_CONFIG_DIR"] = str(temp / "ultralytics")
    rtsp_url = os.environ.get("SKYGUARD_VERIFY_RTSP_URL") if args.rtsp else None
    if args.rtsp:
        if not rtsp_url or not rtsp_url.lower().startswith("rtsp://"):
            raise ValueError("Set SKYGUARD_VERIFY_RTSP_URL to a configured physical RTSP stream")
        try:
            target = urlsplit(rtsp_url)
            if not target.hostname:
                raise ValueError("Missing host")
            with socket.create_connection((target.hostname, target.port or 554), timeout=5):
                pass
        except (OSError, ValueError):
            raise RuntimeError("The configured RTSP endpoint did not accept a TCP connection") from None
        os.environ["SKYGUARD_ENABLE_RTSP_LIVE"] = "1"
        os.environ["OPENCV_LOG_LEVEL"] = "OFF"
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    from app.api import media_upload
    from app.skyguard_persistence.composed import app

    media_upload._REPORT = temp / "media"
    backend_socket = socket.socket()
    backend_socket.bind(("127.0.0.1", 0))
    backend_port = backend_socket.getsockname()[1]
    with closing(socket.socket()) as candidate:
        candidate.bind(("127.0.0.1", 0))
        vite_port = candidate.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=backend_port, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [backend_socket]})
    thread.start()
    vite = None
    with args.weights.open("rb") as weights:
        model_sha = hashlib.file_digest(weights, "sha256").hexdigest()
    result = {"sqlite_revision": "0002", "model_sha256": model_sha, "omp_num_threads": os.getenv("OMP_NUM_THREADS")}
    try:
        wait_until(lambda: server.started, 30)
        env = dict(os.environ)
        env["SKYGUARD_BACKEND_PORT"] = str(backend_port)
        vite = subprocess.Popen(
            [shutil.which("node"), "node_modules/vite/bin/vite.js", "--config", "operator.vite.config.ts", "--port", str(vite_port), "--strictPort"],
            cwd=ROOT / "frontend", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        auth = {"Authorization": "Bearer " + key}
        with httpx.Client(base_url=f"http://127.0.0.1:{vite_port}", timeout=90, trust_env=False) as client:
            def ready():
                try:
                    return client.get("/operator.html").status_code == 200
                except httpx.ConnectError:
                    return False

            wait_until(ready, 30)
            assert key not in client.get("/operator.html").text
            for path in ("/health", "/cameras", "/sessions", "/benchmarks", "/audit", "/live/frame", "/live/status", "/live/health"):
                assert client.get(f"http://127.0.0.1:{backend_port}/api/persistence" + path).status_code == 401, path
                response = client.get("/api/persistence" + path, headers=auth)
                response.raise_for_status()
                result[path] = response.status_code
                if path == "/live/frame":
                    frame = response.json()
                    assert frame["image_base64"] is None
                    assert frame["detections"] == []
                    result["live_state"] = frame["state"]

            camera = client.post("/api/persistence/cameras", headers=auth, json={"name": "HTTP verification", "source_kind": "FILE"})
            assert camera.status_code == 201, camera.text
            session = client.post("/api/persistence/sessions", headers=auth,
                                  json={"camera_id": camera.json()["id"], "source_kind": "IMAGE"})
            assert session.status_code == 201, session.text
            session_id = session.json()["id"]
            for path in ("/detections", "/tracks", "/geospatial", "/alerts", "/performance", "/evidence"):
                client.get(f"/api/persistence{path}", headers=auth, params={"session_id": session_id}).raise_for_status()
            client.get(f"/api/persistence/sessions/{session_id}/summary", headers=auth).raise_for_status()
            result["dashboard_session_reads"] = 7
            assert client.get(f"http://127.0.0.1:{backend_port}/api/media/health").status_code == 401
            assert client.get("/api/media/health").json()["model_found"]
            assert client.get(f"http://127.0.0.1:{backend_port}/api/persistence/health", headers={"Authorization": "Bearer invalid"}).status_code == 403
            os.environ.pop("SKYGUARD_API_KEY")
            try:
                missing = client.post("/api/media/upload")
                assert missing.status_code == 503 and missing.json()["detail"] == "API authentication is not configured"
            finally:
                os.environ["SKYGUARD_API_KEY"] = key
            result["auth"] = {"missing_server_key": 503, "missing_bearer": 401, "invalid_bearer": 403}
            if args.map_only or args.responsive:
                browser = subprocess.run(
                    [shutil.which("node"), str(ROOT / "scripts/verify_operator_browser.mjs"),
                     str(args.browser.resolve()), f"http://127.0.0.1:{vite_port}/operator.html",
                     "", "", "--responsive" if args.responsive else "--map-only"], capture_output=True, text=True, timeout=110,
                )
                if browser.returncode:
                    raise RuntimeError(f"Browser verification failed: {browser.stderr}")
                result["headless_browser"] = json.loads(browser.stdout)
                return result

            if args.tracking_replay:
                from app.api.live_camera import manager
                from app.detection.yolo import YOLODetectionEngine
                from app.video.source import VideoSourceConfig

                detector = YOLODetectionEngine(model_path=str(args.weights.resolve()), confidence_threshold=0.35)
                manager.start(VideoSourceConfig("file", str(video.resolve())), detector)
                live_id = manager.status().session_id
                try:
                    def confirmed_history():
                        sessions = client.get("/api/persistence/sessions", headers=auth).raise_for_status().json()
                        archived = next((item for item in sessions if item["dataset_ref"] == "skyguard-live:" + live_id), None)
                        if archived is None:
                            return None
                        tracks = client.get("/api/persistence/tracks", headers=auth,
                                            params={"session_id": archived["id"]}).raise_for_status().json()
                        for track in tracks:
                            history = client.get("/api/persistence/tracks/" + track["id"] + "/history",
                                                 headers=auth).raise_for_status().json()
                            if len(history) >= 3:
                                assert track["classification"] in ("DRONE", "BIRD")
                                samples = [
                                    {
                                        "frame_index": item["observation"]["frame_index"],
                                        "bbox_xyxy": [item["detection"][key] for key in ("x1", "y1", "x2", "y2")],
                                        "confidence": item["detection"]["confidence"],
                                    }
                                    for item in sorted(history, key=lambda row: row["observation"]["frame_index"])
                                ]
                                for item in samples:
                                    x1, y1, x2, y2 = item["bbox_xyxy"]
                                    assert all(map(math.isfinite, (x1, y1, x2, y2))) and 0 <= x1 < x2 and 0 <= y1 < y2
                                return dict(classification=track["classification"], observations=len(samples),
                                            frame_indexes=[item["frame_index"] for item in samples], samples=samples)
                        return None
                    result["tracking_replay"] = wait_until(confirmed_history, 120)
                finally:
                    manager.stop()
                if args.browser:
                    browser = subprocess.run(
                        [shutil.which("node"), str(ROOT / "scripts/verify_operator_browser.mjs"),
                         str(args.browser.resolve()), f"http://127.0.0.1:{vite_port}/operator.html",
                         "", "", "--tracking"], capture_output=True, text=True, timeout=100,
                    )
                    if browser.returncode:
                        raise RuntimeError(f"Tracking browser verification failed: {browser.stderr}")
                    result["tracking_dashboard"] = json.loads(browser.stdout)
                return result
            if args.usb_index is not None or rtsp_url:
                route = "/api/persistence/live"
                request = ({"source_type": "rtsp", "source_location": rtsp_url} if rtsp_url
                           else {"camera_index": args.usb_index})
                client.post(route + "/start", headers=auth, json=request).raise_for_status()
                try:
                    def read_live():
                        frame = client.get(route + "/frame", headers=auth).raise_for_status().json()
                        if frame["state"] in ("failed", "error"):
                            raise RuntimeError("The live camera API reported a failed stream")
                        if frame["image_base64"] and frame["recorder"]["ingested_frames"] >= 3:
                            return frame
                        return None
                    frame = wait_until(read_live, 75)
                    assert all(d["track_id"] is None for d in frame["detections"] if d["classification"] == "UNKNOWN")
                finally:
                    client.post(route + "/stop", headers=auth).raise_for_status()
                def persisted():
                    with closing(sqlite3.connect(database)) as db:
                        return db.execute(
                            "SELECT COUNT(*) FROM sg_frames JOIN sg_sessions ON sg_frames.session_id = sg_sessions.id "
                            "WHERE sg_sessions.dataset_ref = ?", ("skyguard-live:" + frame["session_id"],)
                        ).fetchone()[0]
                result["physical_rtsp" if rtsp_url else "physical_usb"] = {
                    **({"local_tcp_connected": True, "credentials_supplied": target.username is not None,
                        "stream_access_verified_by_decoded_frames": True} if rtsp_url else {"index": args.usb_index}),
                    "frame_index": frame["frame_index"], "real_jpeg": True,
                    "latency_ms": frame["latency_ms"], "publication_fps": frame.get("fps"), "fps_source": frame.get("fps_source"),
                    "persisted_frames": wait_until(persisted, 15),
                    "detections_in_sample": len(frame["detections"]), "tracks_in_sample": len(frame["tracks"]),
                    "recorder": frame["recorder"],
                }
                return result

            media_results = []
            for kind, file in (("IMAGE", args.image), ("VIDEO", video)):
                payload = file.read_bytes()
                start = time.perf_counter()
                response = client.post("/api/media/upload", headers={"X-Skyguard-Media-Type": kind}, content=payload)
                assert response.status_code == 202, response.text
                accepted_ms = (time.perf_counter() - start) * 1000
                job_id = response.json()["job_id"]
                route = f"/api/media/jobs/{job_id}"

                running_progress = 0
                def complete():
                    nonlocal running_progress
                    job = client.get(route).raise_for_status().json()
                    if job["status"] == "RUNNING":
                        running_progress = max(running_progress, job["frames_processed"])
                    return job if job["status"] in ("DONE", "FAILED", "INTERRUPTED") else None

                if kind == "VIDEO" and args.concurrent_frame:
                    client.post("/api/media/browser/start").raise_for_status()
                    try:
                        def video_running():
                            state = client.get(route).raise_for_status().json()
                            return state["status"] == "RUNNING" and state["frames_processed"] > 0
                        wait_until(video_running, 60)
                        with ThreadPoolExecutor(max_workers=1) as worker:
                            analysis = worker.submit(client.post, "/api/media/browser/frame", content=args.image.read_bytes())
                            health_started = time.perf_counter()
                            client.get("/api/persistence/health", headers=auth).raise_for_status()
                            health_ms = (time.perf_counter() - health_started) * 1000
                            frame = analysis.result(timeout=80).raise_for_status().json()
                        current = client.get(route).raise_for_status().json()
                        assert current["status"] == "RUNNING", current
                        assert frame["preview_jpeg_base64"]
                        result["concurrent_camera_and_upload"] = {
                            "camera_frame": frame["frame_index"], "camera_latency_ms": frame["latency_ms"],
                            "persistence_health_ms": round(health_ms, 2), "video_status": current["status"],
                            "video_frames_processed": current["frames_processed"],
                        }
                    finally:
                        client.post("/api/media/browser/stop").raise_for_status()
                job = wait_until(complete, args.job_timeout)
                assert job["status"] == "DONE", job
                if kind == "VIDEO" and args.expected_video_frames is not None:
                    assert job["frames_processed"] == args.expected_video_frames, job
                summary = job["summary"]
                assert job["input_sha256"] == hashlib.sha256(payload).hexdigest()
                artifact = client.get(route + "/prediction").raise_for_status()
                assert artifact.content
                rows = [json.loads(line) for line in
                        (media_upload._REPORT / job_id / "inference" / "detections.jsonl").read_text().splitlines()]
                assert len(rows) == job["frames_processed"]
                assert all(detection["track_id"] is None for row in rows for detection in row["detections"]
                           if detection["label"] == "UNKNOWN")
                with closing(sqlite3.connect(database)) as db:
                    stored = db.execute("SELECT status, frames_processed FROM sg_media_jobs WHERE id=?", (job_id,)).fetchone()
                    assert stored == (
                        "DONE", job["frames_processed"]), stored
                media_results.append({
                    "kind": kind, "file": file.name, "job_id": job_id,
                    "frames_observed_while_running": running_progress,
                    "frames": job["frames_processed"], "upload_ms": round(accepted_ms, 2),
                    "total_ms": round((time.perf_counter() - start) * 1000, 2),
                    "artifact_bytes": len(artifact.content), "artifact_type": artifact.headers["content-type"],
                    **{k: v for k, v in summary.items() if k != "preview_jpeg_base64"},
                })
                if kind == "IMAGE":
                    assert summary.get("preview_jpeg_base64")
                    detections = rows[0]["detections"]
                    media_results[-1]["image_detections"] = detections
                    if args.image_labels:
                        frame = cv2.imread(str(file))
                        height, width = frame.shape[:2]
                        expected = []
                        for line in args.image_labels.read_text().splitlines():
                            cls, cx, cy, w, h = map(float, line.split())
                            box = [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
                            expected.append({"classification": ("DRONE", "BIRD")[int(cls)], "bbox": box})
                        def iou(a, b):
                            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                            return overlap / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap)
                        for target in expected:
                            target["max_correct_class_iou"] = max(
                                (iou(target["bbox"], detection["xyxy"]) for detection in detections
                                 if detection["label"] == target["classification"]
                                 and detection["verdict_status"] != "CONFLICT_REVIEW"), default=0)
                        media_results[-1]["image_annotation_comparison"] = expected
                print(json.dumps(media_results[-1]), flush=True)
            result["media"] = media_results
            if args.browser:
                browser = subprocess.run(
                    [shutil.which("node"), str(ROOT / "scripts/verify_operator_browser.mjs"),
                     str(args.browser.resolve()), f"http://127.0.0.1:{vite_port}/operator.html",
                     str(args.image.resolve()), str(video.resolve())],
                    capture_output=True, text=True, timeout=300,
                )
                if browser.returncode:
                    raise RuntimeError(f"Browser verification failed: {browser.stderr}")
                result["headless_browser"] = json.loads(browser.stdout)
            if args.video_frames:
                result["video_excerpt"] = {"source_file": str(args.video), "first_frames": args.video_frames}
            client.post("/api/media/browser/start").raise_for_status()
            try:
                browser_frame = client.post("/api/media/browser/frame", content=args.image.read_bytes()).raise_for_status().json()
                assert browser_frame["preview_jpeg_base64"]
                assert all(d["track_id"] is None for d in browser_frame["detections"] if d["classification"] == "UNKNOWN")
                result["browser_frame_api"] = {key: browser_frame[key] for key in ("frame_index", "latency_ms", "detections")}
            finally:
                client.post("/api/media/browser/stop").raise_for_status()
    finally:
        if vite is not None:
            vite.terminate()
            vite.wait(timeout=10)
        server.should_exit = True
        thread.join(timeout=15)
        backend_socket.close()
        media_upload._executor.shutdown(wait=True, cancel_futures=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("database", "weights", "image", "video", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--concurrent-frame", action="store_true")
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--responsive", action="store_true")
    parser.add_argument("--tracking-replay", action="store_true", help="Verify tracking on recorded input through the actual live manager")
    parser.add_argument("--usb-index", type=int, choices=range(5), help="Verify the physical USB path instead of media uploads")
    parser.add_argument("--image-labels", type=Path)
    parser.add_argument("--expected-video-frames", type=int)
    parser.add_argument("--video-frames", type=int, default=0, help="Upload only the first N real source frames")
    parser.add_argument("--rtsp", action="store_true", help="Verify a real RTSP stream configured by SKYGUARD_VERIFY_RTSP_URL")
    args = parser.parse_args()
    if args.rtsp and args.usb_index is not None:
        parser.error("Choose RTSP or USB for a live verification run")
    if (args.map_only or args.responsive) and not args.browser:
        parser.error("--map-only requires --browser")
    if args.video_frames < 0:
        parser.error("--video-frames must be nonnegative")
    with tempfile.TemporaryDirectory(prefix="skyguard-verification-") as folder:
        result = verify(args, Path(folder))
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
