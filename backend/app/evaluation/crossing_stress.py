"""SkyGuard 42.3: offline image-space multi-object identity stress analysis.

Inputs are independent GT and predicted boxes/IDs in Phase 42.1 JSONL format.
"Crossing" is only a 2D projected ordering reversal with an observed close
approach. This module does not verify a physical crossing or drone identity.
It does NOT execute inference or evaluate the actual SkyGuard live tracker unless
its genuine predictions and independent GT are supplied in this exact format.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

from .mot_benchmark import (
    BenchmarkInputError, _identifier, _integer, _number, _optimal_match,
    _parse_frame, evaluate_frames, read_jsonl,
)

VERSION = 'skyguard.phase42.crossing_stress.v1'
_MAX_FRAMES = 50000
_MAX_EVENTS = 200
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_BASELINE = 'ultralytics_yolo_track_not_skyguard_live_tracker'
_BUNDLED_SYNTHETIC_SHA256 = frozenset({
    'a6f84c5853b4c2f0f51ed3fe9e44abd39246b33c38221ba5010210758ab7e291',
    '3ab4cb2be46223ca6c82579f9989562e630eda2a4bc12d3553babb5325076a34',
})


class StressInputError(BenchmarkInputError):
    """Unusable source or configuration; do not emit a benchmark report."""


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _squared_diagonal(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _density(n: int) -> str:
    return str(n) if n < 4 else '4_plus'


def _append_limited(collection: list[dict[str, Any]], event: dict[str, Any]) -> None:
    if len(collection) < _MAX_EVENTS:
        collection.append(event)


def analyze_frames(
    frames: Iterable[Mapping[str, Any]], *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
    max_switch_gap: int = 30,
    close_factor: float = 1.5,
    crossing_max_gap: int = 10,
) -> dict[str, Any]:
    """Compute comparable 42.1 totals and descriptive 2D identity stress counts.

    Does not infer physical crossing from projection alone. Ambiguous or missing
    predictions never turn into fabricated track IDs. Source is unverified.
    """
    iou_threshold = _number(iou_threshold, 'iou_threshold', 0.000001, 1)
    confidence_threshold = _number(confidence_threshold, 'confidence_threshold', 0, 1)
    max_switch_gap = _integer(max_switch_gap, 'max_switch_gap', 1, 1_000_000)
    close_factor = _number(close_factor, 'close_factor', 0.1, 100)
    crossing_max_gap = _integer(crossing_max_gap, 'crossing_max_gap', 1, 1000)

    rows = []
    for row in frames:
        if len(rows) >= _MAX_FRAMES:
            raise StressInputError(f'more than {_MAX_FRAMES} rows; split the benchmark by sequence')
        rows.append(row)
    try:
        baseline = evaluate_frames(
            rows, iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold, max_switch_gap=max_switch_gap,
        )
    except BenchmarkInputError as exc:
        raise StressInputError(str(exc)) from exc

    summary: Counter[str] = Counter()
    density = Counter()
    by_seq: dict[str, Counter[str]] = defaultdict(Counter)
    near_gt_ids: set[tuple[str, str]] = set()
    gt_to_pred_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    pred_to_gt_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    last_match: dict[tuple[str, str], tuple[str, int]] = {}
    last_seen_gt: dict[tuple[str, str], tuple[int, bool]] = {}
    last_near_gt: dict[tuple[str, str], int] = {}
    last_order: dict[tuple[str, str, str], tuple[int, int]] = {}
    last_pair_near: dict[tuple[str, str, str], int] = {}
    events: dict[str, list[dict[str, Any]]] = {
        'projected_x_order_reversals': [], 'identity_switches': [],
        'prediction_id_used_for_multiple_gt': [],
    }

    for raw in rows:
        frame = _parse_frame(raw)
        seq, idx = frame['sequence_id'], frame['frame_index']
        gt_all = frame['ground_truth']
        pred = [p for p in frame['predictions'] if p['confidence'] >= confidence_threshold]
        gt_drones = [g for g in gt_all if g['class'] == 'drone']
        n = len(gt_drones)
        summary['frames_scored'] += 1
        by_seq[seq]['frames_scored'] += 1
        density[_density(n)] += 1
        if n >= 2:
            summary['frames_with_multiple_labeled_drones'] += 1
            by_seq[seq]['frames_with_multiple_labeled_drones'] += 1
        if n >= 3:
            summary['frames_with_three_or_more_labeled_drones'] += 1
        if n == 0:
            summary['frames_with_no_labeled_drone'] += 1
        summary['labeled_drone_observations'] += n

        close_gt_this_frame: set[str] = set()
        close_pairs_here = 0
        for a_idx in range(n):
            for b_idx in range(a_idx + 1, n):
                a, b = gt_drones[a_idx], gt_drones[b_idx]
                ida, idb = _pair_key(a['id'], b['id'])
                pa, pb = _center(a['bbox']), _center(b['bbox'])
                d2 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                scale2 = max(_squared_diagonal(a['bbox']), _squared_diagonal(b['bbox']))
                near = d2 <= close_factor * close_factor * scale2
                pair = (seq, ida, idb)
                if near:
                    close_pairs_here += 1
                    close_gt_this_frame.update((a['id'], b['id']))
                    last_pair_near[pair] = idx
                ax = pa[0] if a['id'] == ida else pb[0]
                bx = pb[0] if b['id'] == idb else pa[0]
                sign = 1 if ax > bx else -1 if ax < bx else 0
                previous = last_order.get(pair)
                recent_near = last_pair_near.get(pair, -crossing_max_gap - 1)
                if (sign != 0 and previous is not None and previous[0] != sign
                    and 0 < idx - previous[1] <= crossing_max_gap
                    and previous[1] <= recent_near <= idx):
                    summary['projected_x_order_reversal_candidates'] += 1
                    by_seq[seq]['projected_x_order_reversal_candidates'] += 1
                    _append_limited(events['projected_x_order_reversals'], {
                        'sequence': seq, 'frame_index': idx,
                        'gt_pair': [ida, idb], 'previous_frame_index': previous[1],
                        'definition': '2d_x_order_reversal_with_close_approach',
                    })
                if sign != 0:
                    last_order[pair] = (sign, idx)
        if close_pairs_here:
            summary['frames_with_close_labeled_drone_pairs'] += 1
            by_seq[seq]['frames_with_close_labeled_drone_pairs'] += 1
        summary['close_labeled_drone_pair_observations'] += close_pairs_here
        summary['close_labeled_drone_observations'] += len(close_gt_this_frame)
        for gid in close_gt_this_frame:
            near_gt_ids.add((seq, gid))
            last_near_gt[(seq, gid)] = idx

        matched = _optimal_match(gt_all, pred, iou_threshold)
        matched_by_gt = {i: j for i, j, _ in matched}
        for gi, gt in enumerate(gt_all):
            if gt['class'] != 'drone':
                continue
            key = (seq, gt['id'])
            prev_observation = last_seen_gt.get(key)
            if prev_observation is not None and idx - prev_observation[0] > 1:
                summary['labeled_gt_temporal_gaps'] += 1
            pj = matched_by_gt.get(gi)
            accepted = pj is not None and pred[pj]['class'] == 'drone'
            if accepted:
                summary['matched_labeled_drone_observations'] += 1
                by_seq[seq]['matched_labeled_drone_observations'] += 1
                if gt['id'] in close_gt_this_frame:
                    summary['close_matched_labeled_drone_observations'] += 1
                tid = pred[pj]['track_id']
                if tid is not None:
                    summary['matched_drone_observations_with_track_id'] += 1
                    if gt['id'] in close_gt_this_frame:
                        summary['close_matched_drone_observations_with_track_id'] += 1
                    gt_to_pred_ids[key].add(tid)
                    pred_to_gt_ids[(seq, tid)].add(gt['id'])
                    prior = last_match.get(key)
                    if prior is not None and 0 < idx - prior[1] <= max_switch_gap and prior[0] != tid:
                        summary['id_switches'] += 1
                        by_seq[seq]['id_switches'] += 1
                        near_recent = (idx - last_near_gt.get(key, -max_switch_gap-1)) <= max_switch_gap
                        if near_recent:
                            summary['id_switches_near_close_approach'] += 1
                        _append_limited(events['identity_switches'], {
                            'sequence': seq, 'frame_index': idx, 'ground_truth_id': gt['id'],
                            'previous_predicted_id': prior[0], 'current_predicted_id': tid,
                            'previous_match_frame': prior[1], 'near_close_approach_recently': near_recent,
                        })
                    last_match[key] = (tid, idx)
                else:
                    summary['matched_drone_observations_without_track_id'] += 1
            elif pj is not None:
                summary['matched_labeled_drone_wrong_class'] += 1
            else:
                summary['unmatched_labeled_drone_observations'] += 1
            if prev_observation is not None and prev_observation[1] and not accepted:
                summary['match_loss_transitions'] += 1
            if prev_observation is not None and not prev_observation[1] and accepted:
                summary['match_recovery_transitions'] += 1
            last_seen_gt[key] = (idx, accepted)

    summary['ground_truth_drones_with_close_approach'] = len(near_gt_ids)
    summary['ground_truth_drones_with_multiple_predicted_ids'] = sum(len(s) > 1 for s in gt_to_pred_ids.values())
    summary['predicted_track_ids_used_for_multiple_gt'] = sum(len(s) > 1 for s in pred_to_gt_ids.values())
    for (seq, tid), ids in sorted(pred_to_gt_ids.items()):
        if len(ids) > 1:
            _append_limited(events['prediction_id_used_for_multiple_gt'], {
                'sequence': seq, 'predicted_track_id': tid,
                'distinct_gt_ids': sorted(ids),
                'note': 'Possible tracker ID reuse or identity mixing, not physical proof',
            })

    if summary['id_switches'] != baseline['drone_tracking']['id_switches']:
        raise StressInputError('internal identity switch result differs from Phase 42.1 evaluator')
    if summary['frames_scored'] != baseline['frames']:
        raise StressInputError('internal frame count differs from Phase 42.1 evaluator')
    if summary['matched_drone_observations_without_track_id'] != baseline['drone_tracking']['matched_drones_without_track_id']:
        raise StressInputError('internal missing track ID count differs from Phase 42.1 evaluator')

    keys = (
        'frames_scored','frames_with_multiple_labeled_drones','frames_with_three_or_more_labeled_drones',
        'frames_with_no_labeled_drone', 'labeled_drone_observations',
        'matched_labeled_drone_observations', 'matched_drone_observations_with_track_id',
        'matched_drone_observations_without_track_id', 'matched_labeled_drone_wrong_class',
        'unmatched_labeled_drone_observations','frames_with_close_labeled_drone_pairs',
        'close_labeled_drone_pair_observations','close_labeled_drone_observations',
        'close_matched_labeled_drone_observations','close_matched_drone_observations_with_track_id',
        'projected_x_order_reversal_candidates','id_switches','id_switches_near_close_approach',
        'match_loss_transitions','match_recovery_transitions','labeled_gt_temporal_gaps',
        'ground_truth_drones_with_close_approach','ground_truth_drones_with_multiple_predicted_ids',
        'predicted_track_ids_used_for_multiple_gt',
    )
    return {
        'schema_version': VERSION,
        'evidence_type': 'offline_annotated_predictions_with_source_provenance_unverified',
        'scope': '2d_image_space_not_physical_crossing_or_real_world_drone_verification',
        'baseline': baseline,
        'configuration': {'iou_threshold': iou_threshold,
                          'confidence_threshold': confidence_threshold,
                          'max_switch_gap': max_switch_gap,
                          'close_factor_bbox_diagonals': close_factor,
                          'crossing_max_gap': crossing_max_gap},
        'stress_metrics': {k: int(summary[k]) for k in keys},
        'density_frames_by_ground_truth_drone_count': {k: int(density[k]) for k in ('0','1','2','3','4_plus')},
        'per_sequence': {seq: dict(sorted(counts.items())) for seq, counts in sorted(by_seq.items())},
        'events': events,
        'events_truncated_to_first': _MAX_EVENTS,
        'limitations': [
            'A 2D x-order reversal with close approach is NOT independently verified physical crossing',
            'ID changes are relative to independently annotated video GT IDs, not drone serial identities',
            'Detection missing during unannotated frames is not measurable as occlusion from this input',
            'Observed simultaneous drone counts do not prove maximum tracker capacity',
            'Ultralytics baseline output does not assess the SkyGuard live tracker',
            'No target GPS, range, perimeter violation or autonomous response evidence',
        ],
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_info(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 1024*1024:
        raise StressInputError('manifest missing or exceeds 1 MiB')
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise StressInputError('manifest must contain valid JSON') from exc
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 'skyguard.phase42.real_video_adapter.v1':
        raise StressInputError('manifest is not a 42.2 runner manifest')
    if manifest.get('baseline') != _BASELINE:
        raise StressInputError('manifest is not an Ultralytics tracker baseline')
    seq_id = _identifier(manifest.get('sequence_id'), 'manifest.sequence_id')
    if list(report['baseline']['per_sequence']) != [seq_id]:
        raise StressInputError('manifest sequence ID differs from annotated predictions')
    perf = manifest.get('performance')
    if not isinstance(perf, dict) or type(perf.get('annotated_frames')) is not int:
        raise StressInputError('manifest missing annotated frame count')
    if perf['annotated_frames'] != report['baseline']['frames']:
        raise StressInputError('manifest annotated frame count inconsistent with predictions')
    hashes = {}
    for key in ('video_sha256', 'checkpoint_sha256', 'independent_gt_sha256', 'class_map_sha256'):
        value = manifest.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise StressInputError('manifest missing a valid ' + key)
        hashes[key] = value
    return {
        'manifest_sha256': _file_hash(path),
        'manifest_declared_baseline': _BASELINE,
        'declared_input_hashes_not_rechecked_against_source_files': hashes,
        'annotation_coverage_declared': perf.get('annotation_coverage'),
        'partial_annotation_coverage_declared': perf.get('partial_annotation_coverage'),
        'warning': 'Manifest is self-declared by the 42.2 runner; input/video/GT origin not independently attested',
    }


def run_file(path: str | Path, *, manifest: str | Path | None = None,
             synthetic_self_test: bool = False, **kwargs: Any) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise StressInputError('annotated predictions JSONL file missing')
    try:
        report = analyze_frames(read_jsonl(source), **kwargs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise StressInputError(str(exc)) from exc
    report['input_sha256'] = _file_hash(source)
    if synthetic_self_test and report['input_sha256'] not in _BUNDLED_SYNTHETIC_SHA256:
        raise StressInputError('synthetic-self-test is reserved for the bundled verified synthetic fixture')
    report['provenance'] = (
        {'origin': 'bundled_artificial_synthetic_fixture', 'real_video_claim': False}
        if synthetic_self_test else
        {'origin': 'user_supplied_unverified', 'real_video_claim': False}
    )
    if manifest is not None:
        if synthetic_self_test:
            raise StressInputError('synthetic fixture must not be supplied with a real-video manifest')
        report['provenance'] = {
            'origin': 'declared_42_2_ultralytics_real_video_baseline',
            'real_video_claim': 'declared_not_independently_verified',
            **_manifest_info(Path(manifest), report),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='SkyGuard 42.3 image-space crossing and identity stress diagnostics; no physical range/GPS')
    parser.add_argument('--input', required=True, help='42.1 format annotated_predictions.jsonl, or synthetic fixture')
    parser.add_argument('--output', required=True, help='new JSON report; existing files never overwritten')
    parser.add_argument('--manifest', help='optional 42.2 manifest.json for declared provenance consistency checks')
    parser.add_argument('--synthetic-self-test', action='store_true', help='label only bundled artificial fixture; NOT real model performance')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--confidence-threshold', type=float, default=0.25)
    parser.add_argument('--max-switch-gap', type=int, default=30)
    parser.add_argument('--close-factor', type=float, default=1.5)
    parser.add_argument('--crossing-max-gap', type=int, default=10)
    args = parser.parse_args(argv)
    output = Path(args.output)
    try:
        if output.exists():
            raise StressInputError('refuse to overwrite an existing report')
        if not output.parent.is_dir():
            raise StressInputError('output directory does not exist')
        report = run_file(
            args.input, manifest=args.manifest, synthetic_self_test=args.synthetic_self_test,
            iou_threshold=args.iou_threshold, confidence_threshold=args.confidence_threshold,
            max_switch_gap=args.max_switch_gap, close_factor=args.close_factor,
            crossing_max_gap=args.crossing_max_gap,
        )
        with output.open('x', encoding='utf-8') as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write('\n')
    except (BenchmarkInputError, OSError, ValueError) as exc:
        print('PHASE_42_3_REJECTED: ' + str(exc), file=sys.stderr)
        return 2
    print('PHASE_42_3_IMAGE_SPACE_STRESS_REPORT_WRITTEN: ' + str(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
