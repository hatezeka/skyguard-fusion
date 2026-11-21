"""Phase 42.2 offline runner unit tests; simulated model/video, NOT actual field metrics."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.evaluation.real_video_adapter import (
    VERSION, RealVideoError, class_map_from_json, normalize_names,
    validate_class_map, read_independent_annotations, parse_yolo_predictions,
    merge_frame, process_frames, digest, main,
)
from app.evaluation.mot_benchmark import evaluate_frames

SID = "test_clip"
GT = {"sequence_id": SID, "frame_index": 0, "width": 100, "height": 80,
      "ground_truth": [{"id": "d1", "class": "drone", "bbox_xyxy": [1, 2, 20, 30]}]}
NAMES = {0: "drone", 1: "bird"}
MAPPING = {"drone": "drone", "bird": "bird"}


class FakeImage:
    shape = (80, 100, 3)


class FakeCapture:
    def __init__(self, n):
        self.n = n
        self.read_count = 0
    def read(self):
        if self.read_count == self.n:
            return False, None
        self.read_count += 1
        return True, FakeImage()


class FakeBoxes:
    def __init__(self, xyxy=None, conf=None, cls=None, ids=None):
        self.xyxy = [] if xyxy is None else xyxy
        self.conf = [] if conf is None else conf
        self.cls = [] if cls is None else cls
        self.id = ids


def result(xyxy=None, cls=None, ids=None, conf=None):
    return SimpleNamespace(boxes=FakeBoxes(xyxy, conf, cls, ids))


class FakeModel:
    def __init__(self, predictions=None):
        self.calls = []
        self.predictions = predictions or []
    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        return [self.predictions[index] if index < len(self.predictions) else result()]


def write_gt(tmp_path, rows):
    file = tmp_path / 'annotations.jsonl'
    file.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    return file


def test_version_and_digest(tmp_path):
    assert VERSION == 'skyguard.phase42.real_video_adapter.v1'
    file = tmp_path / 'hash.bin'
    file.write_bytes(b'hello')
    assert digest(file) == '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'


def test_explicit_class_mapping(tmp_path):
    file = tmp_path / 'map.json'
    file.write_text(json.dumps(MAPPING), encoding='utf-8')
    assert class_map_from_json(file) == MAPPING
    validate_class_map(NAMES, MAPPING)


@pytest.mark.parametrize('mapping', [{}, {'drone': 'DRONE'}, {'drone': 'drone', 'bird': 'bad space'},
                                    {'drone': 1}, [], None])
def test_reject_invalid_class_mapping(tmp_path, mapping):
    file = tmp_path / 'map.json'
    file.write_text(json.dumps(mapping), encoding='utf-8')
    with pytest.raises(RealVideoError):
        class_map_from_json(file)


def test_class_mapping_must_cover_all_model_names():
    with pytest.raises(RealVideoError, match='unmapped'):
        validate_class_map(NAMES, {'drone': 'drone'})
    with pytest.raises(RealVideoError, match='unknown'):
        validate_class_map(NAMES, {**MAPPING, 'uav': 'drone'})


def test_model_names_dict_and_array():
    assert normalize_names(['drone','bird']) == NAMES
    assert normalize_names({0:'drone',1:'bird'}) == NAMES


@pytest.mark.parametrize('names', [None, [], {True:'drone'}, {0:''}, {'zero':'drone'}])
def test_reject_bad_names(names):
    with pytest.raises(RealVideoError):
        normalize_names(names)


def test_read_annotations_and_duplicate(tmp_path):
    file = write_gt(tmp_path, [GT])
    assert list(read_independent_annotations(file, SID)) == [0]
    file = write_gt(tmp_path, [GT, GT])
    with pytest.raises(RealVideoError, match='duplicated'):
        read_independent_annotations(file, SID)


@pytest.mark.parametrize('change', [
    {'predictions': []}, {'target_gps': 'fabricated'}, {'range_m': 10},
    {'sequence_id': 'another_clip'}, {'width': 0}, {'ground_truth': [] , 'frame_index': True},
])
def test_reject_nonindependent_or_invalid_annotations(tmp_path, change):
    file = write_gt(tmp_path, [{**GT, **change}])
    with pytest.raises((RealVideoError, ValueError)):
        read_independent_annotations(file, SID)


def test_reject_annotation_outside_frame(tmp_path):
    gt = copy.deepcopy(GT)
    gt['ground_truth'][0]['bbox_xyxy'] = [1,2,101,30]
    with pytest.raises(ValueError):
        read_independent_annotations(write_gt(tmp_path, [gt]), SID)


def test_reject_missing_annotations(tmp_path):
    file = write_gt(tmp_path, [])
    with pytest.raises(RealVideoError, match='no independent'):
        read_independent_annotations(file, SID)


def test_prediction_extracts_real_tracking_id_and_class():
    raw = result(xyxy=[[1., 2., 20., 30.]], conf=[.9], cls=[0], ids=[44.])
    output = parse_yolo_predictions(raw, NAMES, MAPPING, 100, 80)
    assert output == [{'class':'drone', 'bbox_xyxy':[1.,2.,20.,30.], 'confidence':.9, 'track_id':44}]


def test_missing_tracking_id_is_not_fabricated():
    raw = result(xyxy=[[1,2,20,30]], conf=[.5], cls=[0], ids=None)
    assert 'track_id' not in parse_yolo_predictions(raw, NAMES, MAPPING, 100, 80)[0]


def test_yolo_boxes_clipped_to_source_pixels():
    raw = result(xyxy=[[-2,-2,105,85]], conf=[.5], cls=[0], ids=[1])
    assert parse_yolo_predictions(raw, NAMES, MAPPING, 100, 80)[0]['bbox_xyxy'] == [0,0,100,80]


@pytest.mark.parametrize('prediction', [
    result(xyxy=[[1,2,30,40]], conf=[1.1], cls=[0], ids=[3]),
    result(xyxy=[[1,2,30,40]], conf=[.5], cls=[3], ids=[3]),
    result(xyxy=[[1,2,30,40]], conf=[.5], cls=[0], ids=[-1]),
    result(xyxy=[[1,2,30,40]], conf=[.5], cls=[0], ids=[1.2]),
    result(xyxy=[[2,2,2,40]], conf=[.5], cls=[0], ids=[1]),
    result(xyxy=[[1,2,30,40]], conf=[float('nan')], cls=[0], ids=[1]),
    result(xyxy=[[1,2,30,float('inf')]], conf=[.5], cls=[0], ids=[1]),
    result(xyxy=[[1,2,30,40]], conf=[], cls=[0], ids=[1]),
    result(xyxy=[[1,2,30,40],[20,20,40,40]], conf=[.5,.7], cls=[0,1], ids=[1,1]),
])
def test_invalid_yolo_predictions_fail_closed(prediction):
    with pytest.raises(RealVideoError):
        parse_yolo_predictions(prediction, NAMES, MAPPING, 100, 80)


def test_merge_frame_strict_resolution():
    with pytest.raises(RealVideoError, match='resolution'):
        merge_frame(GT, [], 101, 80)
    assert merge_frame(GT, [], 100, 80)['predictions'] == []


def test_continuous_tracking_all_frames_and_score_only_annotated():
    first = result([[1,2,20,30]], [0], [123], [.9])
    third = result([[1,2,20,30]], [0], [123], [.8])
    model = FakeModel([first, result(), third])
    gt3 = {**GT, 'frame_index': 2}
    rows, perf = process_frames(FakeCapture(3), model, {0: GT, 2:gt3}, NAMES, MAPPING,
                 max_frames=5, confidence=.25, imgsz=640, device='cpu', tracker='bytetrack.yaml', allow_partial=True)
    assert len(model.calls) == 3
    assert all(c['persist'] for c in model.calls)
    assert all(c['verbose'] is False for c in model.calls)
    assert [r['frame_index'] for r in rows] == [0, 2]
    assert perf['decoded_frames'] == 3 and perf['annotated_frames'] == 2
    assert perf['partial_annotation_coverage'] is True
    scores = evaluate_frames(rows)
    assert scores['frames'] == 2 and scores['detection_classification']['tp'] == 2
    assert scores['drone_tracking']['id_switches'] == 0


def test_unannotated_frame_rejected_without_opt_in():
    with pytest.raises(RealVideoError, match='no independent GT'):
        process_frames(FakeCapture(2), FakeModel(), {0:GT}, NAMES, MAPPING,
            max_frames=5, confidence=.25, imgsz=640, device='cpu', tracker='bytetrack.yaml', allow_partial=False)


def test_truncated_video_and_missing_gt_detected():
    with pytest.raises(RealVideoError, match='beyond decoded'):
        process_frames(FakeCapture(1), FakeModel(), {2:{**GT, 'frame_index':2}}, NAMES, MAPPING,
            max_frames=5, confidence=.25, imgsz=640, device='cpu', tracker='bytetrack.yaml', allow_partial=True)


def test_excess_frames_abort_without_claiming_result():
    with pytest.raises(RealVideoError, match='exceeds --max-frames'):
        process_frames(FakeCapture(2), FakeModel(), {0:GT}, NAMES, MAPPING,
            max_frames=1, confidence=.25, imgsz=640, device='cpu', tracker='bytetrack.yaml', allow_partial=True)


def test_no_frames_decode_rejected():
    with pytest.raises(RealVideoError, match='No frames'):
        process_frames(FakeCapture(0), FakeModel(), {0:GT}, NAMES, MAPPING,
            max_frames=5, confidence=.25, imgsz=640, device='cpu', tracker='bytetrack.yaml', allow_partial=True)


def test_cli_rejects_without_local_files_and_no_output(tmp_path):
    target = tmp_path / 'out'
    status = main(['--video', str(tmp_path / 'missing.mp4'), '--weights', str(tmp_path / 'missing.pt'),
                   '--ground-truth', str(tmp_path / 'missing.jsonl'), '--class-map', str(tmp_path / 'map.json'),
                   '--sequence-id', SID, '--output-dir', str(target), '--confirm-license'])
    assert status == 2 and not target.exists()


def test_end_to_end_mocked_model_video_exports_evaluable_files(tmp_path, monkeypatch):
    """Mocked integration; does not establish live YOLO accuracy."""
    import sys
    from app.evaluation.real_video_adapter import run_real_benchmark
    from app.evaluation.mot_benchmark import read_jsonl
    video, weights, class_map = (tmp_path / name for name in ('video.mp4', 'checkpoint.pt', 'map.json'))
    video.write_bytes(b'fake local video')
    weights.write_bytes(b'fake checkpoint')
    class_map.write_text(json.dumps(MAPPING), encoding='utf-8')
    annotation = write_gt(tmp_path, [GT, {**GT, 'frame_index': 1}])
    model = FakeModel([result([[1,2,20,30]], [0], [1], [.95]),
                       result([[1,2,20,30]], [0], [1], [.9])])
    model.names = NAMES
    fake_cv2 = SimpleNamespace(VideoCapture=lambda path: FakeCapture(2), __version__='fake',
                               # We must support the interface the runner calls.
                               )
    class RealisticCapture(FakeCapture):
        def isOpened(self): return True
        def release(self): pass
    fake_cv2.VideoCapture = lambda path: RealisticCapture(2)
    fake_yolo = SimpleNamespace(YOLO=lambda path: model, __version__='fake')
    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)
    monkeypatch.setitem(sys.modules, 'ultralytics', fake_yolo)
    options = SimpleNamespace(video=str(video), weights=str(weights), ground_truth=str(annotation),
        class_map=str(class_map), output_dir=str(tmp_path / 'new_output'), sequence_id=SID,
        confirm_license=True, confirm_trusted_checkpoint=True,
        max_frames=4, confidence=.25, iou_threshold=.5, max_switch_gap=30,
        imgsz=640, device='cpu', tracker='bytetrack.yaml', allow_partial=False)
    completed = run_real_benchmark(options)
    assert completed['frames_scored'] == completed['decoded_frames'] == 2
    report = json.loads((tmp_path / 'new_output' / 'metrics.json').read_text())
    assert report['detection_classification']['tp'] == 2
    assert report['drone_tracking']['id_switches'] == 0
    rows = list(read_jsonl(tmp_path / 'new_output' / 'annotated_predictions.jsonl'))
    assert rows[0]['predictions'][0]['track_id'] == 1
    metadata = json.loads((tmp_path / 'new_output' / 'manifest.json').read_text())
    assert metadata['baseline'] == 'ultralytics_yolo_track_not_skyguard_live_tracker'
    assert metadata['performance']['annotated_frames'] == 2
    assert metadata['performance']['model_latency_p95_ms'] >= 0
    assert metadata['checkpoint_sha256'] == digest(weights)
    assert len(model.calls) == 2
    with pytest.raises(RealVideoError, match='exists'):
        run_real_benchmark(options)


def test_trusted_checkpoint_confirmation_is_mandatory_before_model_import(tmp_path, monkeypatch):
    from app.evaluation.real_video_adapter import run_real_benchmark
    video, weights, class_map = (tmp_path / name for name in ('clip.mp4', 'best.pt', 'map.json'))
    video.write_bytes(b'local')
    weights.write_bytes(b'local')
    class_map.write_text(json.dumps(MAPPING), encoding='utf-8')
    annotations = write_gt(tmp_path, [GT])
    options = SimpleNamespace(video=str(video), weights=str(weights), ground_truth=str(annotations),
        class_map=str(class_map), output_dir=str(tmp_path / 'reports'), sequence_id=SID,
        confirm_license=True, confirm_trusted_checkpoint=False,
        max_frames=4, confidence=.25, iou_threshold=.5, max_switch_gap=30,
        imgsz=640, device='cpu', tracker='bytetrack.yaml', allow_partial=False)
    with pytest.raises(RealVideoError, match='trusted'):
        run_real_benchmark(options)
    assert not (tmp_path / 'reports').exists()
