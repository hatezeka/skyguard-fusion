"""Phase 42.1 self-contained offline evaluation tests; not field-validation claims."""
import json
import math
import pytest
from app.evaluation.mot_benchmark import (
    BenchmarkInputError, VERSION, _optimal_match, bbox_iou, evaluate_frames, read_jsonl, main,
)

BOX_A = [10, 10, 30, 30]
BOX_B = [60, 10, 80, 30]


def gt(id='d1', cls='drone', box=None):
    return {'id': id, 'class': cls, 'bbox_xyxy': BOX_A if box is None else box}


def pred(id='t1', cls='drone', box=None, conf=.9):
    obj = {'class': cls, 'bbox_xyxy': BOX_A if box is None else box, 'confidence': conf}
    if id != 'omit':
        obj['track_id'] = id
    return obj


def frame(idx=0, truth=None, predictions=None, seq='clip_1'):
    return {'sequence_id': seq, 'frame_index': idx, 'width': 100, 'height': 100,
            'ground_truth': [gt()] if truth is None else truth,
            'predictions': [pred()] if predictions is None else predictions}


def result(rows, **kwargs):
    return evaluate_frames(rows, **kwargs)


def test_perfect_detection_and_ground_truth_association():
    r = result([frame(0), frame(1), frame(2)])
    assert r['schema_version'] == VERSION
    assert r['frames'] == 3
    assert r['detection_classification'] == {'tp': 3, 'fp': 0, 'fn': 0, 'precision': 1., 'recall': 1., 'f1': 1.}
    assert r['drone_tracking']['id_switches'] == 0
    assert 'range_m' in r['not_measured']


def test_two_simultaneous_drones_remain_distinct_when_paths_cross():
    rows = [frame(i, [gt('d1', box=a), gt('d2', box=b)],
                  [pred('t1', box=a), pred('t2', box=b)])
            for i, (a,b) in enumerate([
                (BOX_A, BOX_B), ([30,10,50,30], [40,10,60,30]), (BOX_B, BOX_A)])]
    r = result(rows)
    assert r['detection_classification']['tp'] == 6
    assert r['drone_tracking']['id_switches'] == 0
    assert r['drone_tracking']['max_simultaneous_ground_truth_drones'] == 2
    assert r['drone_tracking']['distinct_predicted_drone_track_ids'] == 2


def test_id_swaps_are_detected_at_crossing():
    r = result([frame(0, [gt('d1', box=BOX_A), gt('d2', box=BOX_B)],
                     [pred('t1', box=BOX_A), pred('t2', box=BOX_B)]),
                frame(1, [gt('d1', box=BOX_B), gt('d2', box=BOX_A)],
                     [pred('t1', box=BOX_A), pred('t2', box=BOX_B)])])
    assert r['drone_tracking']['id_switches'] == 2


def test_id_reuse_across_sequences_does_not_switch():
    r = result([frame(0, seq='clip1'), frame(0, seq='clip2')])
    assert r['drone_tracking']['id_switches'] == 0
    assert r['drone_tracking']['distinct_predicted_drone_track_ids'] == 2


def test_id_switch_gap_is_configurable():
    rows = [frame(0), frame(100, predictions=[pred('t2')])]
    assert result(rows)['drone_tracking']['id_switches'] == 0
    assert result(rows, max_switch_gap=100)['drone_tracking']['id_switches'] == 1


def test_dropped_classification_is_counted_twice():
    r = result([frame(0, predictions=[pred('t1', 'bird')])])
    assert r['detection_classification']['tp'] == 0
    assert r['detection_classification']['fp'] == 1
    assert r['detection_classification']['fn'] == 1
    assert r['confusion']['drone']['bird'] == 1
    assert r['per_class']['bird']['fp'] == 1
    assert r['per_class']['drone']['fn'] == 1


def test_unmatched_gt_and_false_positive():
    r = result([frame(0, [gt()], [pred(box=BOX_B)])])
    assert r['detection_classification']['tp'] == 0
    assert r['detection_classification']['fn'] == 1
    assert r['detection_classification']['fp'] == 1
    assert r['confusion']['drone']['__missed__'] == 1
    assert r['confusion']['__spurious__']['drone'] == 1


def test_confidence_threshold_filters_predictions():
    r = result([frame(0, predictions=[pred(conf=.1)])], confidence_threshold=.5)
    assert r['retained_predictions'] == 0
    assert r['predictions_below_threshold'] == 1
    assert r['detection_classification']['fn'] == 1


def test_missing_track_id_is_reported_without_inventing_identity():
    r = result([frame(0, predictions=[pred('omit')])])
    assert r['detection_classification']['tp'] == 1
    assert r['drone_tracking']['matched_drones_without_track_id'] == 1
    assert r['drone_tracking']['distinct_predicted_drone_track_ids'] == 0


def test_empty_frame_is_valid_and_yields_na_metrics():
    r = result([frame(0, [], [])])
    assert r['detection_classification']['precision'] is None
    assert r['detection_classification']['recall'] is None
    assert r['localization']['mean_matched_iou'] is None


def test_all_frames_absent_rejected():
    with pytest.raises(BenchmarkInputError, match='no frames'):
        result([])


@pytest.mark.parametrize('modified', [
    lambda r: {**r, 'frame_index': True},
    lambda r: {**r, 'frame_index': -1},
    lambda r: {**r, 'width': 0},
    lambda r: {**r, 'height': 32769},
    lambda r: {**r, 'gps': {'lat': 1, 'lon': 2}},
    lambda r: {**r, 'ground_truth': 'invalid'},
    lambda r: {**r, 'predictions': 'invalid'},
    lambda r: {**r, 'sequence_id': '../invalid'},
    lambda r: {**r, 'ground_truth': [gt(box=[0,0,0,1])]},
    lambda r: {**r, 'ground_truth': [gt(box=[-1,0,1,1])]},
    lambda r: {**r, 'ground_truth': [gt(box=[0,0,101,1])]},
    lambda r: {**r, 'ground_truth': [gt(box=[0,0,math.inf,1])]},
    lambda r: {**r, 'ground_truth': [gt(), gt()]},
    lambda r: {**r, 'predictions': [pred('t1'), pred('t1')]},
    lambda r: {**r, 'predictions': [pred(conf=float('nan'))]},
    lambda r: {**r, 'predictions': [pred(conf=True)]},
    lambda r: {**r, 'predictions': [pred(cls='Drone')]},
    lambda r: {**r, 'predictions': [pred(), {**pred('t2'), 'range_m': 120.}]},
    lambda r: {**r, 'predictions': [pred(), {**pred('t2'), 'gps_lat': 42.}]},
    lambda r: {**r, 'ground_truth': [gt(), {**gt('two'), 'distance_m': 100}]},
])
def test_invalid_data_and_unverified_distance_claims_rejected(modified):
    with pytest.raises(BenchmarkInputError):
        result([modified(frame())])


def test_non_increasing_frames_rejected_even_when_interleaved():
    with pytest.raises(BenchmarkInputError, match='strictly'):
        result([frame(2, seq='one'), frame(0, seq='two'), frame(2, seq='one')])


def test_increasing_frame_indices_per_interleaved_sequence_valid():
    r = result([frame(2, seq='one'), frame(0, seq='two'), frame(3, seq='one'), frame(1, seq='two')])
    assert r['sequences'] == 2
    assert r['frames'] == 4


def test_invalid_config_parameters_rejected():
    with pytest.raises(BenchmarkInputError):
        result([frame()], iou_threshold=0)
    with pytest.raises(BenchmarkInputError):
        result([frame()], confidence_threshold=math.inf)
    with pytest.raises(BenchmarkInputError):
        result([frame()], max_switch_gap=True)


def test_iou_exact_and_no_overlap():
    assert bbox_iou((0,0,2,2), (0,0,2,2)) == 1
    assert bbox_iou((0,0,2,2), (2,2,4,4)) == 0
    assert bbox_iou((0,0,2,2), (1,1,3,3)) == pytest.approx(1/7)


def test_assignment_prefers_global_optimum_over_greedy():
    # One gt has 2 valid candidates, the other only 1. Optimal assignment
    # chooses the two disjoint eligible pairs, rather than best local overlap.
    g = [{'bbox': (0,0,10,10)}, {'bbox': (5,0,15,10)}]
    p = [{'bbox': (4,0,14,10)}, {'bbox': (0,0,8,10)}]
    pairs = _optimal_match(g, p, .4)
    assert {(i, j) for i,j,_ in pairs} == {(0,1), (1,0)}


def test_assignment_handles_uneven_list_sizes_and_ties():
    g = [{'bbox': (0,0,10,10)}]
    p = [{'bbox': (0,0,10,10)}, {'bbox': (20,20,30,30)}]
    assert _optimal_match(g, p, .5) == [(0,0,1.)]
    assert _optimal_match([], p, .5) == []
    assert _optimal_match(g, [], .5) == []


def test_max_simultaneous_64_drone_tracks():
    gt_objects = []
    predictions = []
    for n in range(64):
        x = n * 10
        # Larger 700 px frame, avoids overlap.
        box = [x, 0, x+5, 5]
        gt_objects.append(gt('gt'+str(n), box=box))
        predictions.append(pred('tr'+str(n), box=box))
    item = frame(0, gt_objects, predictions)
    item['width'] = 700
    r = result([item])
    assert r['detection_classification']['tp'] == 64
    assert r['drone_tracking']['max_simultaneous_predicted_drones'] == 64


def test_bird_not_counted_as_drone_and_no_switch():
    rows = [frame(0, [gt('b1', 'bird')], [pred('t1', 'bird')]),
            frame(1, [gt('b1', 'bird')], [pred('t2', 'bird')])]
    r = result(rows)
    assert r['per_class']['bird']['tp'] == 2
    assert r['drone_tracking']['id_switches'] == 0


def test_read_jsonl_strict_utf8_and_nan(tmp_path):
    data = tmp_path / 'bad.jsonl'
    data.write_bytes(b'{"frame_index":NaN}\n')
    with pytest.raises(BenchmarkInputError):
        list(read_jsonl(data))
    data.write_bytes(b'\xff')
    with pytest.raises(BenchmarkInputError):
        list(read_jsonl(data))


def test_read_jsonl_blank_lines_and_cli(tmp_path, capsys):
    data = tmp_path / 'frames.jsonl'
    data.write_text('\n' + json.dumps(frame()) + '\n', encoding='utf-8')
    output = tmp_path / 'results.json'
    assert main(['--input', str(data), '--output', str(output)]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['detection_classification']['f1'] == 1
    assert main(['--input', str(data), '--output', str(output)]) == 2
    assert 'already exists' in capsys.readouterr().err


def test_cli_invalid_json_no_report(tmp_path):
    data = tmp_path / 'frames.jsonl'
    data.write_text('garbage', encoding='utf-8')
    output = tmp_path / 'results.json'
    assert main(['--input', str(data), '--output', str(output)]) == 2
    assert not output.exists()


def test_io_limit_rejects_large_line(tmp_path):
    data = tmp_path / 'big.jsonl'
    data.write_bytes(b'{' + b'a' * (4 * 1024 * 1024) + b'}')
    with pytest.raises(BenchmarkInputError, match='line 1 exceeds'):
        list(read_jsonl(data))


def test_per_sequence_disjoint_score():
    r = result([frame(0, seq='one'), frame(0, seq='two', predictions=[])])
    assert r['per_sequence']['s:one']['detection_classification']['tp'] == 1
    assert r['per_sequence']['s:two']['detection_classification']['fn'] == 1
    assert r['detection_classification']['recall'] == .5


def test_string_and_integer_sequence_ids_do_not_collide():
    r = result([frame(0, seq='5'), frame(0, seq=5)])
    assert r['sequences'] == 2
    assert set(r['per_sequence']) == {'s:5', 'i:5'}

def test_prediction_classification_can_include_smoke_without_drones():
    r = result([frame(0, [gt('bird1', 'bird')], [pred('smoke1', 'smoke')])])
    assert r['confusion']['bird']['smoke'] == 1
    assert r['per_class']['bird']['fn'] == 1
    assert r['per_class']['smoke']['fp'] == 1
