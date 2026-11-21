"""Phase 42.4 timing validation: all hardware, YOLO, and clock tests are faked.

No physical camera, checkpoint, network, or video is accessed by this suite.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from app.evaluation.camera_performance import (
    PerformanceInputError, _model_box_count, _stats, _write_new_report,
    main, read_trace, run_capture, sample_capture, summarize_trace, validate_trace,
)


FIXTURE = Path(__file__).resolve().parents[1] / 'examples' / 'phase42_4_SYNTHETIC_timing.jsonl'


def frame(i=0, end=.04, *, infer=None, boxes=None, width=640, height=480, read=5., total=10., source=None):
    return {'frame_index': i, 'wall_end_s': end, 'read_ms': read, 'inference_ms': infer,
            'total_ms': total, 'width': width, 'height': height,
            'detections': boxes, 'source_video_msec': source}


def two():
    return [frame(), frame(1, .08)]


def test_synthetic_fixture_is_valid():
    assert len(read_trace(FIXTURE)) == 6


def test_metadata_is_explicit_synthetic():
    r = summarize_trace(read_trace(FIXTURE), source_kind='synthetic_trace', synthetic=True)
    assert r['provenance']['synthetic'] is True
    assert r['measurement']['processed_frames'] == 6
    assert r['measurement']['actual_dropped_frames'] is None
    assert r['measurement']['inference_latency_ms'] is None
    assert r['measurement']['long_interarrival_gap_count_not_proven_drops'] == 1


def test_declared_video_fps_vs_actual_fps():
    r = summarize_trace(two(), source_kind='local_video', synthetic=False,
                        provenance={'fps_reported_by_capture_metadata_not_measured': 120.})
    assert r['measurement']['interarrival_fps'] == 25.
    assert r['provenance']['fps_reported_by_capture_metadata_not_measured'] == 120.
    assert r['measurement']['wall_throughput_fps'] == 25.


def test_inference_stats_without_detection_claims():
    rows = [frame(infer=6, boxes=2, total=12), frame(1, .08, infer=7, boxes=3, total=15)]
    r = summarize_trace(rows, source_kind='local_video', synthetic=False)
    assert r['measurement']['inference_latency_ms']['p95'] == 7
    assert r['measurement']['total_model_box_detections_unverified'] == 5
    assert 'ultralytics_predict' in r['provenance']['pipeline']


def test_single_frame_interarrival_unknown():
    r = summarize_trace([frame()], source_kind='opt_in_camera', synthetic=False)
    assert r['measurement']['interarrival_fps'] is None


def test_over_budget_count():
    rows = [frame(total=90), frame(1, .12, total=85)]
    r = summarize_trace(rows, source_kind='independent_trace', synthetic=False, target_fps=15)
    assert r['measurement']['over_budget_frame_count'] == 2


def test_percentiles_are_nearest_rank():
    assert _stats([1, 2, 3, 4, 100]) == {'mean': 22., 'p50': 3., 'p95': 100., 'p99': 100., 'max': 100.}


@pytest.mark.parametrize('changes', [
    {'frame_index': True}, {'frame_index': 1}, {'wall_end_s': float('nan')},
    {'wall_end_s': float('inf')}, {'wall_end_s': -2}, {'read_ms': -1},
    {'read_ms': True}, {'total_ms': -1}, {'width': 0}, {'width': True},
    {'height': 0}, {'detections': 2}, {'source_video_msec': -1},
    {'inference_ms': float('nan')}, {'inference_ms': True},
    {'inference_ms': -2}, {'inference_ms': 2},
])
def test_bad_input_rejected(changes):
    with pytest.raises(PerformanceInputError):
        validate_trace([{**frame(), **changes}])


@pytest.mark.parametrize('rows', [
    [], [{'frame_index': 0}],
    [frame(), frame(2, .08)], [frame(), frame(1, .03)],
    [frame(), frame(1, .08, width=320)],
    [frame(), frame(1, .08, infer=1, boxes=1, total=8)],
])
def test_bad_sequences_rejected(rows):
    with pytest.raises(PerformanceInputError):
        validate_trace(rows)


def test_excess_max_frame_rejected():
    with pytest.raises(PerformanceInputError, match='frame limit'):
        validate_trace(two(), max_frames=1)


def test_extra_fields_rejected():
    with pytest.raises(PerformanceInputError):
        validate_trace([{**frame(), 'target_gps': [0, 0]}])


@pytest.mark.parametrize('source,synthetic', [('local_video', True), ('synthetic_trace', False), ('unknown', False)])
def test_provenance_classification_not_forgeable(source, synthetic):
    with pytest.raises(PerformanceInputError):
        summarize_trace(two(), source_kind=source, synthetic=synthetic)


def test_provenance_override_rejected():
    with pytest.raises(PerformanceInputError):
        summarize_trace(two(), source_kind='local_video', synthetic=False,
                        provenance={'synthetic': True})


def test_invalid_elapsed_rejected():
    with pytest.raises(PerformanceInputError):
        summarize_trace(two(), source_kind='local_video', synthetic=False, wall_elapsed_s=.03)


def test_invalid_target_fps_rejected():
    with pytest.raises(PerformanceInputError):
        summarize_trace(two(), source_kind='local_video', synthetic=False, target_fps=0)


def test_read_trace_rejects_nan_and_oversize(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"foo":NaN}\n', encoding='utf-8')
    with pytest.raises(PerformanceInputError):
        read_trace(path)
    path.write_text('x' * 4096, encoding='utf-8')
    with pytest.raises(PerformanceInputError):
        read_trace(path)


def test_read_trace_rejects_missing_path(tmp_path):
    with pytest.raises(PerformanceInputError):
        read_trace(tmp_path / 'missing.jsonl')


def test_trace_main_no_camera_and_no_overwrite(tmp_path):
    output = tmp_path / 'out.json'
    assert main(['--trace', str(FIXTURE), '--synthetic-self-test', '--output', str(output)]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['provenance']['synthetic'] is True
    assert report['provenance']['source_kind'] == 'synthetic_trace'
    assert report['measurement']['actual_dropped_frames'] is None
    assert main(['--trace', str(FIXTURE), '--synthetic-self-test', '--output', str(output)]) == 2


def test_bundled_synthetic_hash_even_when_renamed(tmp_path):
    renamed = tmp_path / 'renamed.jsonl'
    renamed.write_bytes(FIXTURE.read_bytes())
    output = tmp_path / 'out.json'
    assert main(['--trace', str(renamed), '--output', str(output)]) == 2
    assert main(['--trace', str(renamed), '--synthetic-self-test', '--output', str(output)]) == 0


def test_ordinary_trace_cannot_be_marked_synthetic(tmp_path):
    trace = tmp_path / 'data.jsonl'
    trace.write_text(json.dumps(frame()) + '\n', encoding='utf-8')
    output = tmp_path / 'x.json'
    assert main(['--trace', str(trace), '--synthetic-self-test', '--output', str(output)]) == 2
    assert main(['--trace', str(trace), '--output', str(output)]) == 0
    r = json.loads(output.read_text())
    assert r['provenance']['synthetic'] is False
    assert r['provenance']['provenance_not_independently_verified'] is True


def test_trace_rejects_unused_camera_options(tmp_path):
    output = tmp_path / 'x.json'
    assert main(['--trace', str(FIXTURE), '--synthetic-self-test', '--output', str(output), '--confirm-camera']) == 2
    assert not output.exists()


def test_invalid_write_parent_and_trace_collision(tmp_path):
    with pytest.raises(PerformanceInputError):
        _write_new_report(tmp_path / 'missing' / 'x.json', {})
    with pytest.raises(PerformanceInputError):
        _write_new_report(tmp_path / 'x.json', {}, trace=[frame()], trace_output=tmp_path / 'x.json')


def test_write_new_report_can_write_jsonl_trace(tmp_path):
    output, trace = tmp_path / 'result.json', tmp_path / 'trace.jsonl'
    _write_new_report(output, {'ok': True}, trace=[frame()], trace_output=trace)
    assert json.loads(output.read_text())['ok']
    assert len(trace.read_text().splitlines()) == 1
    with pytest.raises(PerformanceInputError):
        _write_new_report(tmp_path / 'new.json', {}, trace=[frame()], trace_output=trace)


def test_model_box_count_is_strict():
    class FakeBoxes:
        def __len__(self): return 3
    assert _model_box_count([SimpleNamespace(boxes=FakeBoxes())]) == 3
    assert _model_box_count([SimpleNamespace(boxes=None)]) == 0
    with pytest.raises(PerformanceInputError):
        _model_box_count([])


class Clock:
    def __init__(self, step=.005):
        self.now = -step
        self.step = step
    def __call__(self):
        self.now += self.step
        return self.now


class Image:
    shape = (480, 640, 3)


class Capture:
    def __init__(self, n=2, fail_at=None):
        self.n, self.counter, self.fail_at = n, 0, fail_at
        self.released = False
    def read(self):
        if self.counter == self.fail_at:
            return False, None
        self.counter += 1
        return (self.counter <= self.n, Image() if self.counter <= self.n else None)
    def get(self, prop):
        return 33.33 * self.counter
    def isOpened(self):
        return True
    def release(self):
        self.released = True


class Model:
    def predict(self, image, **kwargs):
        return [SimpleNamespace(boxes=[1, 2])]


def test_fake_video_capture_counts_and_does_not_save_frames():
    cap = Capture(n=3)
    rows, elapsed = sample_capture(cap, max_frames=3, max_seconds=10,
                                   is_camera=False, cv2_module=SimpleNamespace(CAP_PROP_POS_MSEC=1),
                                   clock=Clock())
    assert len(rows) == 3
    assert rows[0]['frame_index'] == 0
    assert rows[0]['detections'] is None
    assert rows[-1]['source_video_msec'] is not None
    assert elapsed > 0


def test_fake_camera_read_failure_is_not_silent():
    with pytest.raises(PerformanceInputError, match='camera read failed'):
        sample_capture(Capture(n=2, fail_at=1), max_frames=5, max_seconds=10,
                       is_camera=True, cv2_module=SimpleNamespace(CAP_PROP_POS_MSEC=1),
                       clock=Clock())


def test_fake_video_eof_is_normal():
    rows, elapsed = sample_capture(Capture(n=2), max_frames=5, max_seconds=10,
                                  is_camera=False, cv2_module=SimpleNamespace(CAP_PROP_POS_MSEC=1),
                                  clock=Clock())
    assert len(rows) == 2
    assert elapsed > 0


def test_no_frames_fails_closed():
    with pytest.raises(PerformanceInputError, match='no decodable frames'):
        sample_capture(Capture(n=0), max_frames=5, max_seconds=10,
                       is_camera=False, cv2_module=SimpleNamespace(CAP_PROP_POS_MSEC=1),
                       clock=Clock())


def test_fake_model_mode_counts_predictions():
    rows, _ = sample_capture(Capture(n=2), max_frames=2, max_seconds=10,
                             is_camera=False, cv2_module=SimpleNamespace(CAP_PROP_POS_MSEC=1),
                             clock=Clock(), model=Model())
    assert [x['detections'] for x in rows] == [2, 2]
    assert all(x['inference_ms'] is not None for x in rows)
    assert all(x['total_ms'] >= x['read_ms'] + x['inference_ms'] for x in rows)


def test_gpu_sync_injection():
    calls = []
    sample_capture(Capture(n=1), max_frames=1, max_seconds=10, is_camera=False,
                   cv2_module=SimpleNamespace(CAP_PROP_POS_MSEC=1),
                   model=Model(), clock=Clock(), sync_gpu=lambda: calls.append(1))
    assert calls == [1, 1]


@pytest.mark.parametrize('args', [
    {'video': None, 'camera_index': 0, 'confirm_camera': False},
    {'video': None, 'camera_index': -1, 'confirm_camera': True},
    {'video': None, 'camera_index': 33, 'confirm_camera': True},
    {'video': None, 'camera_index': None},
    {'video': 'fake.mp4', 'camera_index': 0, 'confirm_camera': True},
    {'video': 'fake.mp4', 'camera_index': None, 'confirm_license': False},
    {'video': 'fake.mp4', 'camera_index': None, 'confirm_license': True},
    {'video': None, 'camera_index': 0, 'confirm_camera': True, 'weights': 'unknown.pt'},
    {'video': None, 'camera_index': 0, 'confirm_camera': True, 'target_fps': 0},
    {'video': None, 'camera_index': 0, 'confirm_camera': True, 'device': 'remote'},
    {'video': None, 'camera_index': 0, 'confirm_camera': True, 'hash_video': True},
])
def test_run_capture_rejects_unsafe_or_unavailable_inputs_before_camera_open(args, monkeypatch):
    default = dict(video=None, camera_index=None, max_frames=5, max_seconds=10.,
                   target_fps=15., confidence=.25, imgsz=640, device='cpu',
                   confirm_camera=False, confirm_license=False, weights=None,
                   confirm_trusted_checkpoint=False, hash_video=False)
    monkeypatch.setitem(sys.modules, 'cv2', None)  # invalid requests must fail before even importing cv2
    with pytest.raises(PerformanceInputError):
        run_capture(SimpleNamespace(**{**default, **args}))


def test_run_capture_fake_cv2_camera_branch(monkeypatch):
    cap = Capture(n=3)
    cv2 = SimpleNamespace(VideoCapture=lambda index: cap, CAP_PROP_POS_MSEC=1,
                          CAP_PROP_FPS=2, __version__='fake-cv2')
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    args = SimpleNamespace(video=None, camera_index=0, max_frames=3, max_seconds=10.,
                           target_fps=15., confidence=.25, imgsz=640, device='cpu',
                           confirm_camera=True, confirm_license=False, weights=None,
                           confirm_trusted_checkpoint=False, hash_video=False)
    report, rows = run_capture(args)
    assert report['provenance']['source_kind'] == 'opt_in_camera'
    assert report['provenance']['synthetic'] is False
    assert report['measurement']['processed_frames'] == 3
    assert cap.released
    assert len(rows) == 3


def test_run_capture_fake_cv2_video_branch(monkeypatch, tmp_path):
    video = tmp_path / 'licensed.mp4'
    video.write_bytes(b'fake_video_for_unit_test')
    cap = Capture(n=2)
    cv2 = SimpleNamespace(VideoCapture=lambda path: cap, CAP_PROP_POS_MSEC=1,
                          CAP_PROP_FPS=2, __version__='fake-cv2')
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    args = SimpleNamespace(video=str(video), camera_index=None, max_frames=4, max_seconds=10.,
                           target_fps=15., confidence=.25, imgsz=640, device='cpu',
                           confirm_camera=False, confirm_license=True, weights=None,
                           confirm_trusted_checkpoint=False, hash_video=True)
    report, rows = run_capture(args)
    assert report['provenance']['source_kind'] == 'local_video'
    assert len(report['provenance']['video_sha256']) == 64
    assert report['provenance']['source_identifier'] == 'licensed.mp4'
    assert cap.released
    assert len(rows) == 2


def test_video_not_drones_and_no_tracking_claims():
    report = summarize_trace(two(), source_kind='local_video', synthetic=False)
    assert report['measurement']['actual_dropped_frames'] is None
    assert 'tracker' in ' '.join(report['limitations'])
    assert not any('drone_gps' in k for k in report['measurement'])
