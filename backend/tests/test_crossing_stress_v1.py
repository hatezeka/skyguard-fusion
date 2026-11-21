"""42.3 offline synthetic stress tests, never real-world drone accuracy claims."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from app.evaluation.crossing_stress import (
    VERSION, StressInputError, analyze_frames, run_file, main,
)
from app.evaluation.mot_benchmark import evaluate_frames

SID = 'clip_01'


def f(index, objects, *, seq=SID, width=250, height=120):
    """Synthetic GT=box, predictions=box for each (gt ID, x, predicted ID, class, prediction class)."""
    gts, preds = [], []
    for obj in objects:
        gid, x, tid, *tail = obj
        cls = tail[0] if tail else 'drone'
        pred_cls = tail[1] if len(tail) > 1 else cls
        box = [x, 20, x + 10, 30]
        gts.append({'id': gid, 'class': cls, 'bbox_xyxy': box})
        if pred_cls is not None:
            pred = {'class': pred_cls, 'bbox_xyxy': box, 'confidence': 0.9}
            if tid is not None:
                pred['track_id'] = tid
            preds.append(pred)
    return {'sequence_id': seq, 'frame_index': index, 'width': width, 'height': height,
            'ground_truth': gts, 'predictions': preds}


def crossing(*, swapped=False):
    return [f(0, [('a', 10, 'id-a'), ('b', 110, 'id-b')]),
            f(1, [('a', 45, 'id-a'), ('b', 60, 'id-b')]),
            f(2, [('a', 110, 'id-b' if swapped else 'id-a'),
                  ('b', 10, 'id-a' if swapped else 'id-b')])]


def write_rows(tmp_path, rows, name='rows.jsonl'):
    path = tmp_path / name
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return path


def test_version_and_stress_metrics():
    out = analyze_frames(crossing())
    assert out['schema_version'] == VERSION
    assert out['stress_metrics']['frames_scored'] == 3
    assert out['stress_metrics']['frames_with_multiple_labeled_drones'] == 3
    assert out['stress_metrics']['frames_with_close_labeled_drone_pairs'] == 1
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 1
    assert out['stress_metrics']['id_switches'] == 0
    assert out['baseline']['detection_classification']['f1'] == 1.0
    assert out['density_frames_by_ground_truth_drone_count']['2'] == 3
    assert out['events']['projected_x_order_reversals'][0]['definition'] == '2d_x_order_reversal_with_close_approach'
    assert out['scope'].startswith('2d_image_space_')


def test_swapped_id_detects_both_switches_and_gt_fragmentation():
    out = analyze_frames(crossing(swapped=True))
    assert out['stress_metrics']['id_switches'] == 2
    assert out['stress_metrics']['id_switches_near_close_approach'] == 2
    assert out['stress_metrics']['ground_truth_drones_with_multiple_predicted_ids'] == 2
    assert out['stress_metrics']['predicted_track_ids_used_for_multiple_gt'] == 2
    assert len(out['events']['identity_switches']) == 2


def test_far_order_reversal_is_not_close_approach_candidate():
    rows = [f(0, [('a', 10, 1), ('b', 200, 2)]),
            f(1, [('a', 200, 1), ('b', 10, 2)])]
    out = analyze_frames(rows)
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 0
    assert out['stress_metrics']['frames_with_close_labeled_drone_pairs'] == 0


def test_order_reversal_with_near_frame_but_very_long_gap_is_not_candidate():
    rows = [f(0, [('a', 10, 1), ('b', 200, 2)]),
            f(1, [('a', 45, 1), ('b', 60, 2)]),
            f(20, [('a', 200, 1), ('b', 10, 2)])]
    out = analyze_frames(rows)
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 0
    assert out['stress_metrics']['labeled_gt_temporal_gaps'] == 2


def test_custom_crossing_max_gap_allows_reversal():
    rows = [f(0, [('a', 10, 1), ('b', 200, 2)]),
            f(1, [('a', 45, 1), ('b', 60, 2)]),
            f(20, [('a', 200, 1), ('b', 10, 2)])]
    assert analyze_frames(rows, crossing_max_gap=25)['stress_metrics']['projected_x_order_reversal_candidates'] == 1


def test_missing_track_id_is_counted_not_fabricated():
    out = analyze_frames([f(0, [('a', 10, None)])])
    assert out['stress_metrics']['matched_drone_observations_without_track_id'] == 1
    assert out['stress_metrics']['matched_drone_observations_with_track_id'] == 0
    assert out['stress_metrics']['id_switches'] == 0


def test_wrong_class_is_classification_error_not_successful_tracking():
    out = analyze_frames([f(0, [('a', 10, 4, 'drone', 'bird')])])
    assert out['stress_metrics']['matched_labeled_drone_wrong_class'] == 1
    assert out['stress_metrics']['matched_labeled_drone_observations'] == 0
    assert out['baseline']['per_class']['drone']['fn'] == 1


def test_predicted_dropout_and_recovery_transitions():
    rows = [f(0, [('a', 10, 1)]), f(1, [('a', 10, 1)]), f(2, [('a', 10, 1)]),
            f(3, [('a', 10, 1)]), f(4, [('a', 10, 1)])]
    rows[2]['predictions'] = []
    out = analyze_frames(rows)
    assert out['stress_metrics']['match_loss_transitions'] == 1
    assert out['stress_metrics']['match_recovery_transitions'] == 1
    assert out['stress_metrics']['unmatched_labeled_drone_observations'] == 1
    assert out['stress_metrics']['id_switches'] == 0


def test_unknown_GT_between_frames_does_not_invent_occlusion():
    out = analyze_frames([f(0, [('a', 10, 1)]), f(1, []), f(2, [('a', 10, 1)])])
    assert out['stress_metrics']['labeled_gt_temporal_gaps'] == 1
    assert out['stress_metrics']['match_loss_transitions'] == 0
    assert 'not measurable' in out['limitations'][2]


def test_id_reuse_across_different_ground_truth_is_disclosed():
    out = analyze_frames([f(0, [('a', 10, 5)]), f(1, [('b', 10, 5)])])
    assert out['stress_metrics']['predicted_track_ids_used_for_multiple_gt'] == 1
    assert out['stress_metrics']['id_switches'] == 0


def test_different_sessions_same_track_id_do_not_cross_contaminate():
    out = analyze_frames([f(0, [('a', 10, 5)], seq='clip-a'),
                          f(0, [('b', 10, 5)], seq='clip-b')])
    assert out['stress_metrics']['predicted_track_ids_used_for_multiple_gt'] == 0
    assert len(out['per_sequence']) == 2


def test_empty_gt_frame_and_false_positive_are_counted_baseline():
    rows = [f(0, []), f(1, [])]
    rows[0]['predictions'] = [{'class': 'drone', 'track_id': 9, 'confidence': 0.9,
                                'bbox_xyxy': [10, 20, 20, 30]}]
    out = analyze_frames(rows)
    assert out['stress_metrics']['frames_with_no_labeled_drone'] == 2
    assert out['baseline']['per_class']['drone']['fp'] == 1


def test_bird_and_drone_proximity_does_not_count_two_drones():
    out = analyze_frames([f(0, [('a', 45, 1, 'drone'), ('bird', 60, 2, 'bird')])])
    assert out['stress_metrics']['frames_with_multiple_labeled_drones'] == 0
    assert out['stress_metrics']['frames_with_close_labeled_drone_pairs'] == 0


def test_gt_density_four_plus():
    rows = [f(0, [(str(i), i*20, i) for i in range(5)])]
    out = analyze_frames(rows)
    assert out['density_frames_by_ground_truth_drone_count']['4_plus'] == 1
    assert out['stress_metrics']['frames_with_three_or_more_labeled_drones'] == 1
    assert out['baseline']['drone_tracking']['max_simultaneous_ground_truth_drones'] == 5


def test_densities_are_all_present_when_missing():
    out = analyze_frames([f(0, [('a', 10, 1)])])
    assert set(out['density_frames_by_ground_truth_drone_count']) == {'0','1','2','3','4_plus'}
    assert out['density_frames_by_ground_truth_drone_count']['3'] == 0


def test_swapped_boxes_but_same_predicted_identity_is_reported():
    rows = crossing()
    rows[2]['predictions'][0]['bbox_xyxy'], rows[2]['predictions'][1]['bbox_xyxy'] = (
        rows[2]['predictions'][1]['bbox_xyxy'], rows[2]['predictions'][0]['bbox_xyxy'])
    out = analyze_frames(rows)
    assert out['stress_metrics']['id_switches'] == 2


def test_confidence_drop_can_turn_detection_into_unmatched():
    rows = [f(0, [('a', 10, 1)])]
    rows[0]['predictions'][0]['confidence'] = .24
    out = analyze_frames(rows)
    assert out['stress_metrics']['unmatched_labeled_drone_observations'] == 1
    assert out['baseline']['retained_predictions'] == 0


def test_threshold_exact_inclusive():
    rows = [f(0, [('a', 10, 1)])]
    rows[0]['predictions'][0]['confidence'] = .25
    assert analyze_frames(rows)['stress_metrics']['matched_labeled_drone_observations'] == 1


def test_track_ids_with_different_types_remain_distinct():
    out = analyze_frames([f(0, [('a', 10, 7)]), f(1, [('a', 10, '7')])])
    assert out['stress_metrics']['id_switches'] == 1


def test_gt_reordered_per_frame_does_not_cause_false_id_switch():
    rows = crossing()
    rows[1]['ground_truth'].reverse()
    out = analyze_frames(rows)
    assert out['stress_metrics']['id_switches'] == 0
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 1


def test_repeated_frame_sequence_rejected():
    with pytest.raises(StressInputError, match='increase strictly'):
        analyze_frames([f(0, [('a', 10, 1)]), f(0, [('a', 10, 1)])])


def test_unordered_frame_sequence_rejected():
    with pytest.raises(StressInputError, match='increase strictly'):
        analyze_frames([f(2, [('a', 10, 1)]), f(1, [('a', 10, 1)])])


def test_duplicate_predicted_id_in_same_frame_rejected():
    with pytest.raises(StressInputError, match='duplicate predicted track_id'):
        analyze_frames([f(0, [('a', 10, 1), ('b', 80, 1)])])


def test_duplicate_gt_identity_same_frame_rejected():
    with pytest.raises(StressInputError, match='duplicate ground-truth ID'):
        analyze_frames([f(0, [('a', 10, 1), ('a', 80, 2)])])


@pytest.mark.parametrize('key,value', [
    ('iou_threshold', 0), ('iou_threshold', 1.001), ('confidence_threshold', -1),
    ('confidence_threshold', 1.1), ('max_switch_gap', 0), ('max_switch_gap', True),
    ('close_factor', -1), ('close_factor', float('inf')), ('close_factor', float('nan')),
    ('crossing_max_gap', 0), ('crossing_max_gap', 1001), ('crossing_max_gap', 1.5),
])
def test_bad_thresholds_rejected(key, value):
    with pytest.raises((StressInputError, ValueError)):
        analyze_frames([f(0, [('a', 10, 1)])], **{key: value})


def test_zero_frames_rejected():
    with pytest.raises(StressInputError, match='no frames'):
        analyze_frames([])


def test_output_deterministic_across_repeated_runs():
    a = analyze_frames(crossing(swapped=True))
    b = analyze_frames(crossing(swapped=True))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_event_list_is_bounded():
    rows = [f(i, [('a', 10, 'id1' if i % 2 == 0 else 'id2')]) for i in range(210)]
    out = analyze_frames(rows)
    assert out['stress_metrics']['id_switches'] == 209
    assert len(out['events']['identity_switches']) == 200
    assert out['events_truncated_to_first'] == 200


def test_cli_writes_only_new_file_and_rejects_overwrite(tmp_path):
    path = write_rows(tmp_path, crossing())
    out = tmp_path / 'report.json'
    assert main(['--input', str(path), '--output', str(out)]) == 0
    report = json.loads(out.read_text())
    assert report['stress_metrics']['id_switches'] == 0
    assert report['provenance']['origin'] == 'user_supplied_unverified'
    assert report['input_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert main(['--input', str(path), '--output', str(out)]) == 2


def test_nonexistent_source_rejected(tmp_path):
    assert main(['--input', str(tmp_path/'none.jsonl'), '--output', str(tmp_path/'out.json')]) == 2


def test_output_parent_nonexistent_rejected(tmp_path):
    path = write_rows(tmp_path, crossing())
    assert main(['--input', str(path), '--output', str(tmp_path/'nope'/'report.json')]) == 2


def test_malformed_json_rejected(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{broken\n', encoding='utf-8')
    assert main(['--input', str(path), '--output', str(tmp_path/'report.json')]) == 2


def test_reject_fabricated_gps_fields(tmp_path):
    rows = crossing()
    rows[0]['target_gps'] = [10, 10]
    path = write_rows(tmp_path, rows)
    with pytest.raises(StressInputError, match='frame keys'):
        run_file(path)


def test_reject_synthetic_claim_for_arbitrary_jsonl(tmp_path):
    path = write_rows(tmp_path, crossing())
    with pytest.raises(StressInputError, match='bundled verified synthetic fixture'):
        run_file(path, synthetic_self_test=True)


def test_accept_bundled_fixture_synthetic_label():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'phase42_1_synthetic_crossing.jsonl'
    out = run_file(path, synthetic_self_test=True)
    assert out['provenance']['origin'] == 'bundled_artificial_synthetic_fixture'
    assert out['provenance']['real_video_claim'] is False
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 1


def test_reject_synthetic_and_manifest_together(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'examples' / 'phase42_1_synthetic_crossing.jsonl'
    with pytest.raises(StressInputError, match='synthetic fixture'):
        run_file(path, synthetic_self_test=True, manifest=tmp_path/'whatever')


def valid_manifest(rows, sequence=SID):
    return {'schema_version': 'skyguard.phase42.real_video_adapter.v1',
            'baseline': 'ultralytics_yolo_track_not_skyguard_live_tracker',
            'sequence_id': sequence, 'performance': {'annotated_frames': len(rows),
                'annotation_coverage': 1.0, 'partial_annotation_coverage': False},
            **{x: 'a'*64 for x in (
                'video_sha256','checkpoint_sha256','independent_gt_sha256','class_map_sha256')}}


def test_manifest_from_42_2_metadata_declaration_not_attestation(tmp_path):
    rows = crossing()
    file = write_rows(tmp_path, rows)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(valid_manifest(rows)), encoding='utf-8')
    result = run_file(file, manifest=manifest)
    assert result['provenance']['origin'] == 'declared_42_2_ultralytics_real_video_baseline'
    assert result['provenance']['real_video_claim'] == 'declared_not_independently_verified'
    assert result['provenance']['annotation_coverage_declared'] == 1.0
    assert result['provenance']['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize('mutation,fragment', [
    ({'schema_version': 'other'}, 'not a 42.2'),
    ({'baseline': 'skyguard-live'}, 'not an Ultralytics'),
    ({'sequence_id': 'wrong'}, 'sequence ID differs'),
    ({'performance': {'annotated_frames': 99}}, 'frame count inconsistent'),
    ({'video_sha256': 'invalid'}, 'video_sha256'),
    ({'performance': []}, 'missing annotated'),
])
def test_invalid_manifest_does_not_produce_metrics(tmp_path, mutation, fragment):
    rows = crossing()
    input_file = write_rows(tmp_path, rows)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({**valid_manifest(rows), **mutation}), encoding='utf-8')
    with pytest.raises(StressInputError, match=fragment):
        run_file(input_file, manifest=manifest)


def test_manifest_wrong_shape_rejected(tmp_path):
    rows = crossing()
    path = write_rows(tmp_path, rows)
    manifest = tmp_path/'manifest.json'
    manifest.write_text('[]', encoding='utf-8')
    with pytest.raises(StressInputError):
        run_file(path, manifest=manifest)


def test_max_frames_guard():
    with pytest.raises(StressInputError, match='more than 50000'):
        analyze_frames((f(i, []) for i in range(50001)))


def test_many_drones_per_frame_must_obey_42_1_maximum():
    with pytest.raises(StressInputError, match='at most 128'):
        analyze_frames([f(0, [(f'a{i}', 1, i) for i in range(129)], width=500)])


def test_side_by_side_sequences_have_independent_switches():
    rows = [f(0, [('a', 10, 1)], seq='first'), f(0, [('a', 10, 2)], seq='second'),
            f(1, [('a', 10, 1)], seq='first'), f(1, [('a', 10, 2)], seq='second')]
    out = analyze_frames(rows)
    assert out['stress_metrics']['id_switches'] == 0
    assert out['baseline']['sequences'] == 2


def test_source_not_mutated():
    rows = crossing(swapped=True)
    before = copy.deepcopy(rows)
    analyze_frames(rows)
    assert rows == before


def test_empty_predictions_and_no_drone_creates_null_baseline_rates():
    out = analyze_frames([f(0, [])])
    assert out['stress_metrics']['id_switches'] == 0
    assert out['baseline']['detection_classification']['f1'] is None


def test_all_predicted_ids_dropped_by_threshold():
    rows = crossing()
    for row in rows:
        for p in row['predictions']:
            p['confidence'] = 0.1
    out = analyze_frames(rows)
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 1
    assert out['stress_metrics']['matched_labeled_drone_observations'] == 0
    assert out['stress_metrics']['id_switches'] == 0


def test_x_ties_do_not_create_spurious_projected_order_reversal():
    rows = [f(0, [('a', 20, 1), ('b', 100, 2)]),
            f(1, [('a', 45, 1), ('b', 45, 2)]),
            f(2, [('a', 100, 1), ('b', 20, 2)])]
    # Exact overlapping ground-truth boxes are valid; matching would be ambiguous;
    # the evaluator still requires a deterministic one-to-one assignment.
    out = analyze_frames(rows)
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 1


def test_bad_ground_truth_bbox_rejected():
    rows = [f(0, [('a', 10, 1)])]
    rows[0]['ground_truth'][0]['bbox_xyxy'] = [10, 20, 9, 30]
    with pytest.raises(StressInputError, match='positive area'):
        analyze_frames(rows)


def test_drop_across_long_gap_is_not_switch_with_narrow_gap_policy():
    rows = [f(0, [('a', 10, 1)]), f(15, [('a', 10, 2)])]
    assert analyze_frames(rows, max_switch_gap=10)['stress_metrics']['id_switches'] == 0
    assert analyze_frames(rows, max_switch_gap=20)['stress_metrics']['id_switches'] == 1


def test_synthetic_id_swap_fixture_is_recognized_and_exposes_two_switches():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'phase42_3_SYNTHETIC_identity_swap.jsonl'
    out = run_file(path, synthetic_self_test=True)
    assert out['provenance']['origin'] == 'bundled_artificial_synthetic_fixture'
    assert out['stress_metrics']['id_switches'] == 2
    assert out['stress_metrics']['projected_x_order_reversal_candidates'] == 1
    assert out['stress_metrics']['frames_with_multiple_labeled_drones'] == 3
