"""Phase 42.4: bounded, opt-in camera/video throughput and latency profiler.

No automatic hardware access.  No GPS/depth, tracking accuracy or certified FPS.
Synthetic traces are always labeled as synthetic and never used as real field data.
This is an OpenCV decode/Ultralytics.predict REFERENCE benchmark, not the
SkyGuard production capture, tracker, or frontend rendering pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping

VERSION = 'skyguard.phase42.camera_performance.v1'
_VIDEO_EXT = frozenset({'.mp4', '.avi', '.mov', '.mkv', '.m4v'})
_FIELDS = frozenset({'frame_index', 'wall_end_s', 'read_ms', 'inference_ms',
                     'total_ms', 'width', 'height', 'detections', 'source_video_msec'})
_BUNDLED_SYNTHETIC_SHA256 = 'd6ea5fb39426dd237900245a943edd5f326e92d441dad49836142a925eb7b892'
_MAX_FRAMES = 50000
_MAX_TRACE_BYTES = 48 * 1024 * 1024
_MAX_LINE_BYTES = 2048


class PerformanceInputError(ValueError):
    """An unsafe, incomplete, or non-reproducible performance input."""


def _number(value: Any, name: str, lo: float = 0, hi: float = 1e9) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
        raise PerformanceInputError(f'{name} must be finite and in [{lo}, {hi}]')
    return float(value)


def _integer(value: Any, name: str, lo: int = 0, hi: int = 1_000_000) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise PerformanceInputError(f'{name} must be integer in [{lo}, {hi}]')
    return value


def validate_trace(rows: Iterable[Mapping[str, Any]], *, max_frames: int = _MAX_FRAMES) -> list[dict[str, Any]]:
    """Reject malformed, inconsistent, nonmonotonic and mixed-mode frames."""
    _integer(max_frames, 'max_frames', 1, _MAX_FRAMES)
    result: list[dict[str, Any]] = []
    prev_end: float | None = None
    mode: bool | None = None
    dimensions: tuple[int, int] | None = None
    for row in rows:
        if len(result) >= max_frames:
            raise PerformanceInputError('trace exceeds frame limit')
        if not isinstance(row, dict) or set(row) != _FIELDS:
            raise PerformanceInputError('each frame must have exactly the documented timing fields')
        index = _integer(row['frame_index'], 'frame_index', 0, _MAX_FRAMES - 1)
        if index != len(result):
            raise PerformanceInputError('frame_index must start at zero and be sequential; these are profiler read indices')
        end = _number(row['wall_end_s'], 'wall_end_s', 0.0000001, 1e7)
        read = _number(row['read_ms'], 'read_ms', 0, 1e7)
        total = _number(row['total_ms'], 'total_ms', 0, 1e7)
        inf = row['inference_ms']
        has_model = inf is not None
        if has_model:
            inf = _number(inf, 'inference_ms', 0, 1e7)
        count = row['detections']
        if has_model:
            _integer(count, 'detections', 0, 100_000)
        elif count is not None:
            raise PerformanceInputError('decode-only trace must not invent detection counts')
        if mode is not None and mode != has_model:
            raise PerformanceInputError('mixed model/decode-only timings in a single capture are not comparable')
        mode = has_model
        width = _integer(row['width'], 'width', 1, 16_384)
        height = _integer(row['height'], 'height', 1, 16_384)
        if dimensions is not None and dimensions != (width, height):
            raise PerformanceInputError('resolution changed within trace; benchmark separately')
        dimensions = (width, height)
        source_ms = row['source_video_msec']
        if source_ms is not None:
            source_ms = _number(source_ms, 'source_video_msec', 0, 1e11)
        # The read/predict intervals must be included in the measured frame cycle.
        if read + (inf or 0) > total + 0.02:
            raise PerformanceInputError('total_ms must include read_ms and inference_ms')
        if prev_end is not None and end <= prev_end:
            raise PerformanceInputError('wall_end_s must increase strictly')
        prev_end = end
        result.append({**row, 'wall_end_s': end, 'read_ms': read, 'inference_ms': inf,
                       'total_ms': total, 'source_video_msec': source_ms})
    if not result:
        raise PerformanceInputError('at least one captured frame is required')
    return result


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size > _MAX_TRACE_BYTES:
        raise PerformanceInputError('trace must be a local file up to 48 MiB')
    rows: list[dict[str, Any]] = []
    with path.open('rb') as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            if len(raw) > _MAX_LINE_BYTES:
                raise PerformanceInputError(f'line {line_no} is too large')
            try:
                obj = json.loads(raw.decode('utf-8'), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            except (UnicodeError, ValueError) as exc:
                raise PerformanceInputError(f'bad JSON at line {line_no}') from exc
            rows.append(obj)
            if len(rows) > _MAX_FRAMES:
                raise PerformanceInputError('trace exceeds frame limit')
    return validate_trace(rows)


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def nth(p: float) -> float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return {'mean': round(sum(values) / len(values), 4), 'p50': round(nth(.50), 4),
            'p95': round(nth(.95), 4), 'p99': round(nth(.99), 4),
            'max': round(ordered[-1], 4)}


def summarize_trace(rows: Iterable[Mapping[str, Any]], *, source_kind: str,
                    synthetic: bool, target_fps: float = 15.0,
                    wall_elapsed_s: float | None = None, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if source_kind not in ('synthetic_trace', 'independent_trace', 'local_video', 'opt_in_camera'):
        raise PerformanceInputError('unknown source_kind')
    if type(synthetic) is not bool:
        raise PerformanceInputError('synthetic flag must be boolean')
    if synthetic != (source_kind == 'synthetic_trace'):
        raise PerformanceInputError('synthetic flag and source_kind disagree')
    fps_budget = _number(target_fps, 'target_fps', 1, 240)
    parsed = validate_trace(rows)
    elapsed = (parsed[-1]['wall_end_s'] if wall_elapsed_s is None else
               _number(wall_elapsed_s, 'wall_elapsed_s', 0.0000001, 1e7))
    if elapsed + 1e-7 < parsed[-1]['wall_end_s']:
        raise PerformanceInputError('elapsed wall clock cannot be less than last sampled timestamp')
    model_mode = parsed[0]['inference_ms'] is not None
    ends = [x['wall_end_s'] for x in parsed]
    gaps = [(b - a) * 1000 for a, b in zip(ends, ends[1:])]
    median_gap = _stats(gaps)['p50'] if gaps else None
    long_gaps = (sum(g > max(2.5 * median_gap, 1000 / fps_budget) for g in gaps)
                 if median_gap is not None else 0)
    actual_elapsed = elapsed
    provenance_fields = dict(provenance or {})
    if 'synthetic' in provenance_fields or 'source_kind' in provenance_fields:
        raise PerformanceInputError('provenance cannot override verified origin classification')
    return {
        'schema_version': VERSION,
        'provenance': {'source_kind': source_kind, 'synthetic': synthetic,
                       'pipeline': 'ultralytics_predict_reference_not_skyguard_live' if model_mode
                       else 'opencv_capture_or_trace_decode_only', **provenance_fields},
        'measurement': {
            'processed_frames': len(parsed),
            'resolution': {'width': parsed[0]['width'], 'height': parsed[0]['height']},
            'elapsed_wall_s': round(actual_elapsed, 5),
            'wall_throughput_fps': round(len(parsed) / actual_elapsed, 4),
            'interarrival_fps': round((len(parsed)-1) / (ends[-1]-ends[0]), 4)
                                if len(parsed) > 1 else None,
            'target_fps_budget': fps_budget,
            'over_budget_frame_count': sum(x['total_ms'] > 1000 / fps_budget for x in parsed),
            'long_interarrival_gap_count_not_proven_drops': long_gaps,
            'read_latency_ms': _stats([x['read_ms'] for x in parsed]),
            'processing_cycle_ms': _stats([x['total_ms'] for x in parsed]),
            'inference_latency_ms': _stats([x['inference_ms'] for x in parsed]) if model_mode else None,
            'total_model_box_detections_unverified': sum(x['detections'] for x in parsed) if model_mode else None,
            'actual_dropped_frames': None,
            'actual_dropped_frames_reason': 'unavailable_without_independent_source_frame_sequence_numbers',
        },
        'limitations': [
            'Not a precision/recall result; independent labeled ground truth is required',
            'Not measured SkyGuard production capture/tracker/UI latency',
            'Camera/video frame indices are profiler read counters, not source hardware sequence numbers',
            'Interarrival gaps do not prove lost camera frames',
            'No measured target depth, GPS, distance from facility or geofence violation',
            'Hardware thermal throttling and long-run endurance are not established by short runs',
        ],
    }


def _sha256(file: Path) -> str:
    h = hashlib.sha256()
    with file.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _model_box_count(predictions: Any) -> int:
    if not isinstance(predictions, (list, tuple)) or len(predictions) != 1:
        raise PerformanceInputError('YOLO.predict must return exactly one result per input image')
    boxes = getattr(predictions[0], 'boxes', None)
    if boxes is None:
        return 0
    count = len(boxes)
    return _integer(count, 'model boxes', 0, 100_000)


def sample_capture(capture: Any, *, max_frames: int, max_seconds: float,
                   is_camera: bool, cv2_module: Any, model: Any = None,
                   conf: float = .25, imgsz: int = 640, device: str = 'cpu',
                   clock: Callable[[], float] = time.perf_counter,
                   sync_gpu: Callable[[], None] | None = None) -> tuple[list[dict[str, Any]], float]:
    """Measure complete capture-read+optional reference-model cycles, bounded.

    A live-camera read failure is fatal (no partial success); video EOF is normal.
    No raw frames are saved; ID-switch metrics are NOT measured here.
    """
    _integer(max_frames, 'max_frames', 1, 50000)
    _number(max_seconds, 'max_seconds', 0.1, 3600)
    _number(conf, 'conf', 0.000001, 1)
    _integer(imgsz, 'imgsz', 64, 4096)
    started = clock()
    rows: list[dict[str, Any]] = []
    dimensions = None
    while len(rows) < max_frames:
        if clock() - started >= max_seconds:
            break
        before_read = clock()
        ok, image = capture.read()
        after_read = clock()
        if not ok:
            if is_camera or not rows:
                raise PerformanceInputError('camera read failed' if is_camera else 'video has no decodable frames')
            break
        try:
            height, width = image.shape[:2]
        except (AttributeError, TypeError, ValueError) as exc:
            raise PerformanceInputError('capture returned an invalid image') from exc
        _integer(width, 'width', 1, 16384)
        _integer(height, 'height', 1, 16384)
        if dimensions and dimensions != (width, height):
            raise PerformanceInputError('source resolution changed during measurement')
        dimensions = (width, height)
        inference_ms = None
        detections = None
        if model is not None:
            if sync_gpu is not None:
                sync_gpu()
            pred_start = clock()
            results = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)
            detections = _model_box_count(results)
            if sync_gpu is not None:
                sync_gpu()
            inference_ms = (clock() - pred_start) * 1000
        after_frame = clock()
        source_msec = None
        if not is_camera:
            try:
                value = capture.get(cv2_module.CAP_PROP_POS_MSEC)
                if type(value) in (int, float) and math.isfinite(value) and value > 0:
                    source_msec = float(value)
            except (AttributeError, TypeError, ValueError):
                pass
        rows.append({
            'frame_index': len(rows), 'wall_end_s': after_frame - started,
            'read_ms': (after_read - before_read) * 1000,
            'inference_ms': inference_ms,
            'total_ms': (after_frame - before_read) * 1000,
            'width': int(width), 'height': int(height), 'detections': detections,
            'source_video_msec': source_msec,
        })
    if not rows:
        raise PerformanceInputError('no frame captured before measurement timeout')
    elapsed = clock() - started
    return validate_trace(rows), elapsed


def run_capture(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (args.video is None) == (args.camera_index is None):
        raise PerformanceInputError('choose exactly one: --video or --camera-index')
    _integer(args.max_frames, 'max_frames', 1, 50000)
    _number(args.max_seconds, 'max_seconds', .1, 3600)
    _number(args.target_fps, 'target_fps', 1, 240)
    _number(args.confidence, 'confidence', 0.000001, 1)
    _integer(args.imgsz, 'imgsz', 64, 4096)
    if not isinstance(args.device, str) or not re.fullmatch(r'(?:cpu|mps|[0-9]{1,2})', args.device):
        raise PerformanceInputError('unsupported device, use cpu, mps, or CUDA index 0-99')
    camera_mode = args.camera_index is not None
    if camera_mode:
        _integer(args.camera_index, 'camera_index', 0, 32)
        if not args.confirm_camera:
            raise PerformanceInputError('physical camera requires explicit --confirm-camera')
        if args.hash_video:
            raise PerformanceInputError('--hash-video is applicable only to a local video file')
    else:
        video = Path(args.video).expanduser().resolve()
        if not args.confirm_license:
            raise PerformanceInputError('local video requires --confirm-license')
        if not video.is_file() or video.suffix.lower() not in _VIDEO_EXT:
            raise PerformanceInputError('local video missing or unsupported extension')
    if args.weights is not None and not args.confirm_trusted_checkpoint:
        raise PerformanceInputError('local .pt weights require --confirm-trusted-checkpoint')
    if args.confirm_trusted_checkpoint and args.weights is None:
        raise PerformanceInputError('checkpoint confirmation supplied without weights')
    if args.weights is not None:
        weights = Path(args.weights).expanduser().resolve()
        if not weights.is_file() or weights.suffix.lower() != '.pt':
            raise PerformanceInputError('checkpoint must be an existing, trusted local .pt file')
    # Optional external dependencies are imported ONLY after explicit input validation.
    try:
        import cv2
    except ImportError as exc:
        raise PerformanceInputError('OpenCV missing in backend venv; no packages installed') from exc
    model = None
    ultralytics_version = None
    sync_gpu = None
    if args.weights is not None:
        os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise PerformanceInputError('Ultralytics missing in backend venv; no packages installed') from exc
        model = YOLO(str(weights))
        ultralytics_version = getattr(ultralytics, '__version__', None)
        if args.device.isdecimal():
            try:
                import torch
            except ImportError as exc:
                raise PerformanceInputError('torch required for CUDA synchronization') from exc
            if not torch.cuda.is_available():
                raise PerformanceInputError('requested CUDA device is not available')
            gpu_index = int(args.device)
            if gpu_index >= torch.cuda.device_count():
                raise PerformanceInputError('requested CUDA GPU index does not exist')
            sync_gpu = lambda: torch.cuda.synchronize(gpu_index)
    source = args.camera_index if camera_mode else str(video)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise PerformanceInputError('OpenCV cannot open the requested camera/video')
    try:
        fps_meta = cap.get(cv2.CAP_PROP_FPS)
        fps_meta = float(fps_meta) if type(fps_meta) in (float, int) and math.isfinite(fps_meta) and fps_meta > 0 else None
        rows, elapsed = sample_capture(cap, max_frames=args.max_frames, max_seconds=args.max_seconds,
                                       is_camera=camera_mode, cv2_module=cv2, model=model,
                                       conf=args.confidence, imgsz=args.imgsz, device=args.device,
                                       sync_gpu=sync_gpu)
    finally:
        cap.release()
    provenance: dict[str, Any] = {
        'source_identifier': f'camera_index_{args.camera_index}' if camera_mode else video.name,
        'opencv_version': cv2.__version__, 'python_version': platform.python_version(),
        'fps_reported_by_capture_metadata_not_measured': fps_meta,
        'content_real_world_origin_not_independently_verified': not camera_mode,
        'max_frames_requested': args.max_frames, 'max_seconds_requested': args.max_seconds,
        'ultralytics_version': ultralytics_version,
    }
    if not camera_mode:
        provenance['source_file_size_bytes'] = video.stat().st_size
        if args.hash_video:
            provenance['video_sha256'] = _sha256(video)
    if args.weights is not None:
        provenance['checkpoint_sha256'] = _sha256(weights)
        provenance['model_confidence'] = args.confidence
        provenance['model_imgsz'] = args.imgsz
        provenance['model_device'] = args.device
    report = summarize_trace(rows, source_kind='opt_in_camera' if camera_mode else 'local_video',
                             synthetic=False, target_fps=args.target_fps,
                             wall_elapsed_s=elapsed, provenance=provenance)
    return report, rows


def _write_new_report(path: Path, report: Mapping[str, Any], *, trace: list[dict[str, Any]] | None = None,
                      trace_output: Path | None = None) -> None:
    if not path.parent.is_dir() or path.exists():
        raise PerformanceInputError('output parent must exist; existing reports must never be overwritten')
    if trace_output is not None and (trace_output == path or not trace_output.parent.is_dir() or trace_output.exists()):
        raise PerformanceInputError('trace output must be distinct, new and in an existing folder')
    created: list[Path] = []
    try:
        if trace_output is not None:
            if trace is None:
                raise PerformanceInputError('trace output requires captured trace data')
            with trace_output.open('x', encoding='utf-8', newline='\n') as output:
                created.append(trace_output)
                for row in trace:
                    output.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
        with path.open('x', encoding='utf-8') as output:
            created.append(path)
            json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write('\n')
    except Exception:
        for item in created:
            item.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='42.4 bounded offline/opt-in video-camera timing profiler; no GPS/range/physical accuracy claims')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--trace', help='previously captured JSONL trace, never touches camera')
    group.add_argument('--video', help='local licensed video path (explicit opt-in)')
    group.add_argument('--camera-index', type=int, help='physical camera index; explicit --confirm-camera required')
    parser.add_argument('--output', required=True, help='new JSON path, never overwrite')
    parser.add_argument('--trace-output', help='optional new JSONL for actual capture timing; no image data saved')
    parser.add_argument('--synthetic-self-test', action='store_true', help='required for bundled artificial trace only')
    parser.add_argument('--confirm-camera', action='store_true')
    parser.add_argument('--confirm-license', action='store_true')
    parser.add_argument('--weights', help='optional trusted local .pt for Ultralytics reference predict, NOT SkyGuard production')
    parser.add_argument('--confirm-trusted-checkpoint', action='store_true')
    parser.add_argument('--hash-video', action='store_true', help='compute optional SHA256 over video file (may take longer)')
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--max-seconds', type=float, default=30)
    parser.add_argument('--target-fps', type=float, default=15)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        trace_output = Path(args.trace_output).expanduser().resolve() if args.trace_output else None
        if args.trace is not None:
            if any((args.confirm_camera, args.confirm_license, args.weights, args.confirm_trusted_checkpoint, args.hash_video)):
                raise PerformanceInputError('trace mode must not mix physical-camera/model options')
            if trace_output is not None:
                raise PerformanceInputError('trace input cannot also write a capture trace')
            trace_path = Path(args.trace).expanduser().resolve()
            trace_hash = _sha256(trace_path)
            synthetic_sample = trace_hash == _BUNDLED_SYNTHETIC_SHA256
            if synthetic_sample != args.synthetic_self_test:
                raise PerformanceInputError('bundled synthetic trace requires --synthetic-self-test; never label arbitrary data synthetic')
            rows = read_trace(trace_path)
            report = summarize_trace(rows, source_kind='synthetic_trace' if synthetic_sample else 'independent_trace',
                                     synthetic=synthetic_sample, target_fps=args.target_fps,
                                     provenance={'trace_sha256': trace_hash, 'trace_filename': trace_path.name,
                                                 'provenance_not_independently_verified': not synthetic_sample,
                                                 'synthetic_flag_only_identifies_bundled_fixture': True})
        else:
            if args.synthetic_self_test:
                raise PerformanceInputError('synthetic-self-test is valid only for bundled artificial trace')
            report, rows = run_capture(args)
        _write_new_report(output, report, trace=rows if args.trace is None else None,
                          trace_output=trace_output)
    except (PerformanceInputError, OSError, ValueError) as exc:
        print('PHASE_42_4_REJECTED: ' + str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print('PHASE_42_4_EXECUTION_FAILED: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 3
    print('PHASE_42_4_CAMERA_PERFORMANCE_REPORT_WRITTEN: ' + str(output))
    print('NOTE: FPS is observed pipeline throughput; physical dropped frames, real tracker accuracy and drone GPS/range are NOT measured')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
