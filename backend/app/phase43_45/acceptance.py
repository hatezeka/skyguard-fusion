"""Evidence-based acceptance; no self-awarded green PASS for field trials."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
from .inventory import json_dump_atomic

_CRITERIA = {
 'classification':'43/45: 4-class precision-recall + negative DRONE FP',
 'open_set':'43: AIRPLANE/HELICOPTER negative gallery, calibrated rejection',
 'tracking':'43/45: multiobject MOT GT + IDF1/HOTA/IDSW',
 'performance':'43/45: FPS/latency P50/P95/P99 with hardware metadata',
 'maps':'44/45: licensed geographic and tactical map state, same IDs/timestamp',
 'range':'45: verified physical range and reference error bounds',
 'security':'44: API/RTSP authorization and adversarial tests',
 'robustness':'44: disconnect/stale/restart corruption tests',
 'cost':'44/45: priced and sourced component BOM',
 'usb_field':'45: physical USB camera proof',
 'drone_field':'45: real drone measured field proof',
}


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeDecodeError):
        return None


def build_matrix(reports_root: str | Path) -> dict:
    root=Path(reports_root)
    measured = _read_json(root/'43'/'detection_tracking_metrics.json')
    performance = _read_json(root/'43'/'performance.json')
    maps = _read_json(root/'44'/'maps_contract.json')
    security = _read_json(root/'44'/'security.json')
    robustness = _read_json(root/'44'/'robustness.json')
    cost = _read_json(root/'44'/'cost_bom.json')
    entries={}
    for key, description in _CRITERIA.items():
        entries[key]={'criterion':description, 'status':'BLOCKED',
                      'evidence':[], 'note':'No validated evidence supplied'}
    if measured:
        entries['classification'].update(status='REVIEW_REQUIRED',
             evidence=['43/detection_tracking_metrics.json'],
             note='Measured offline detection, thresholds and official acceptance need independent review')
        if measured.get('model_capability',{}).get('open_set_candidate'):
            entries['open_set'].update(status='REVIEW_REQUIRED',
               evidence=['43/detection_tracking_metrics.json'],
               note='Negative class exists; threshold calibration and independent negatives still required')
        if measured.get('idf1') is not None:
            entries['tracking'].update(status='REVIEW_REQUIRED',
               evidence=['43/detection_tracking_metrics.json'],
               note='Single-object source not a multiobject challenge; HOTA remains uncomputed')
    if performance:
        entries['performance'].update(status='REVIEW_REQUIRED',
             evidence=['43/performance.json'],
             note='Offline benchmark only; field camera FPS/latency pending')
    for name,item in [('maps',maps),('security',security),('robustness',robustness),('cost',cost)]:
        if item:
            entries[name].update(status='REVIEW_REQUIRED',evidence=[{'maps':'44/maps_contract.json',
                'security':'44/security.json','robustness':'44/robustness.json',
                'cost':'44/cost_bom.json'}[name]],
                note='Evidence exists; needs objective contract and independent verification')
    for key in ('range','usb_field','drone_field'):
        entries[key].update(status='FIELD_PENDING',
                            note='Real camera/drone + measured independent ground truth are mandatory')
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),
            'overall':'NOT_YET_CERTIFIED',
            'policy':'No PASS may be claimed from file presence or demo simulation',
            'criteria':entries}
    json_dump_atomic(result,root/'acceptance_matrix.json')
    return result


def field_template() -> dict:
    return {'status':'FIELD_PENDING',
            'warning':'Fill from PHYSICAL USB camera and real drone, never synthetic input',
            'trial_id':None, 'date_utc':None,
            'camera':{'make_model':None,'usb_identifier':None,'sensor_resolution':None,
                      'fov_calibration_reference':None},
            'drone':{'model':None,'safe_field_permission':None},
            'independent_reference':{'method':None,'position_source':None,
                                     'range_meter_measurement':None,'range_uncertainty_m':None},
            'measurements':{'input_fps':None,'output_fps':None,'p50_latency_ms':None,
                             'p95_latency_ms':None,'p99_latency_ms':None,
                             'detection_gt_tp':None,'detection_gt_fp':None,
                             'detection_gt_fn':None,'id_switches':None,
                             'range_mae_m':None,'range_p95_abs_error_m':None},
            'evidence':{'video_sha256':None,'annotation_sha256':None,
                        'camera_device_photo_sha256':None,'run_log_sha256':None},
            'operator_signature':None}
