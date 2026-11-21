"""Evidence-led software release gate. Never equates code presence or 200 OK with physical acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
from datetime import datetime,timezone
from pathlib import Path


def digest(p:Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def repo_path(path:Path,root:Path):
    try:return str(path.relative_to(root)).replace('\\','/')
    except ValueError:return '[external]'


def check_db(root:Path):
    db=root/'backend'/'.skyguard_data'/'skyguard.sqlite3'
    if not db.is_file():return {'status':'BLOCKED','reason':'Operational SQLite absent'}
    url=db.resolve().as_uri()+'?mode=ro'
    with sqlite3.connect(url,uri=True) as cx:
        integrity=cx.execute('PRAGMA integrity_check').fetchone()[0]
        names={r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required={'sg_cameras','sg_sessions','sg_frames','sg_detections','sg_tracks',
                  'sg_track_observations','sg_geo_evidence','sg_alerts','sg_performance_samples',
                  'sg_evidence_files','sg_benchmark_runs','sg_audit_events','alembic_version'}
        if integrity!='ok' or required-names:
            return {'status':'BLOCKED','integrity':integrity,'missing_tables':sorted(required-names)}
        unknown=cx.execute("SELECT COUNT(*) FROM sg_detections WHERE classification='UNKNOWN' AND track_id IS NOT NULL").fetchone()[0]
        frames=cx.execute('SELECT COUNT(*) FROM sg_frames').fetchone()[0]
        return {'status':'PASS' if unknown==0 else 'FAIL', 'integrity':integrity,
                'frames_in_operational_db':frames, 'unknown_with_track':unknown,
                'schema_count':len(required)}


def git_detail(root:Path):
    def git(*args):
        r=subprocess.run(['git','-C',str(root),*args],text=True,capture_output=True,timeout=10)
        if r.returncode!=0:raise RuntimeError('Git check failed: '+r.stderr[:200])
        return r.stdout.strip()
    status=git('status','--porcelain','--untracked-files=normal')
    head=git('rev-parse','HEAD')
    branch=git('branch','--show-current')
    tracked=git('ls-files').splitlines()
    leak=[f for f in tracked if f.startswith('backend/.skyguard_data/') or
          f.endswith(('.sqlite3','.db','.env'))]
    return {'branch':branch,'head':head,'status_line_count':len(status.splitlines()),
            'tracked_sensitive_files':leak,'ok':branch=='feature/skyguard-mvp' and not leak}


def evaluate(root:Path,e2e:Path|None,tests:dict):
    db=check_db(root)
    git=git_detail(root)
    video=None
    if e2e and e2e.is_file():video=json.loads(e2e.read_text(encoding='utf8'))
    good_video=bool(video and video.get('result')=='PASS_RECORDED_E2E_NO_FIELD_CLAIM')
    have_gt=bool(good_video and video.get('ground_truth_metrics') is not None)
    tri=bool(good_video and video.get('tri_class_model'))
    soft_ok=all(tests.get(k)==0 for k in ('backend','frontend','lint','build'))
    criteria=[
      {'criterion':'production_api_and_database','status':'PASS' if soft_ok and db['status']=='PASS' else 'BLOCKED',
       'evidence':'full regression/frontend tests + SQLite PRAGMA integrity'},
      {'criterion':'recorded_media_ai_to_db_reopen','status':'PASS' if good_video else 'BLOCKED',
       'evidence':'actual YOLO/production tracker/LiveRecorder isolated SQLite, never browser polling'},
      {'criterion':'tri_class_recognition','status':'PARTIAL' if tri and have_gt else 'BLOCKED',
       'evidence':'three-class model+independent reviewed GT needed; class list alone is not accuracy'},
      {'criterion':'tracking_idf1_hota_crossings','status':'PARTIAL' if have_gt else 'BLOCKED',
       'evidence':'IDF1 available only with track-ID GT; HOTA and crossings require dedicated benchmark'},
      {'criterion':'range_position_uncertainty','status':'BLOCKED',
       'evidence':'physical or separately surveyed independent GT absent; never fabricate coordinates'},
      {'criterion':'synchronized_two_maps','status':'PARTIAL',
       'evidence':'frontend geometry and contracts exist; same-frame geo/tactical rendered E2E not demonstrated'},
      {'criterion':'fps_latency_resource_benchmark','status':'PARTIAL' if good_video else 'BLOCKED',
       'evidence':'recorded elapsed FPS and latency measured; GPU/CPU/RAM/P95 system profile pending'},
      {'criterion':'security_and_recovery','status':'PARTIAL' if soft_ok else 'BLOCKED',
       'evidence':'test suite passes; independent auth/RTSP/abuse/restart black-box acceptance pending'},
      {'criterion':'hardware_cost_bill_of_materials','status':'BLOCKED',
       'evidence':'no dated purchase list and real hardware performance-to-cost result'},
      {'criterion':'field_usb_real_drone','status':'FIELD_PENDING',
       'evidence':'reserved Phase 45, not a software result'},
      {'criterion':'clean_jury_reproduction','status':'PARTIAL' if soft_ok and git['ok'] else 'BLOCKED',
       'evidence':'checklist and automated scripts exist; new clean checkout full acceptance pending'},
    ]
    return {'schema_version':'skyguard-release-report-v1',
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'source':'local_computer_validation_not_organizer_jury',
            'test_exit_codes':tests,'git':git,'operational_database':db,
            'recorded_e2e':video,'criteria':criteria,
            'software_release_acceptance':('PASS' if all(c['status']=='PASS' for c in criteria if c['criterion']!='field_usb_real_drone')
                                           else 'NOT_YET_PROVEN'),
            'final_challenge_acceptance':'FIELD_PENDING',
            'guarantees':'No guarantee of performance on unknown jury tests; no empirical claim without evidence'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--test-results',type=Path,required=True)
    p.add_argument('--e2e-report',type=Path)
    a=p.parse_args()
    tests=json.loads(a.test_results.read_text(encoding='utf8'))
    report=evaluate(a.repo.resolve(),a.e2e_report,tests)
    a.report_dir.mkdir(parents=True,exist_ok=True)
    jsonfile=a.report_dir/'skyguard_software_acceptance.json'
    jsonfile.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    md=['# SkyGuard Software Release Evidence','',
        f"Generated (UTC): {report['generated_utc']}",'',
        f"Software release: **{report['software_release_acceptance']}**",'',
        '| Criterion | Status | Evidence / outstanding work |','|---|---|---|']
    for item in report['criteria']:
        md.append(f"| {item['criterion']} | {item['status']} | {item['evidence']} |")
    md += ['','**Field validation: PENDING, Phase 45.**',
           '**UNKNOWN may not have track IDs or trajectories.**',
           '**No generated GPS/range without traceable calibrated reference data.**',
           '**Never claim official challenge acceptance based solely on this report.**']
    (a.report_dir/'skyguard_software_acceptance.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print(json.dumps({'software_release':report['software_release_acceptance'],
                      'report':str(jsonfile),'criteria':{x['criterion']:x['status'] for x in report['criteria']}}))

if __name__=='__main__': main()
