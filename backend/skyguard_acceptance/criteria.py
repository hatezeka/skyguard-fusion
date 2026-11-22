"""Fail-closed evidence matrix; no fabricated challenge compliance."""
from __future__ import annotations
import json
from pathlib import Path

REQUIRED = ('trained_checkpoint','held_out_classification','unknown_rejection',
            'multi_object_tracking','runtime_fps_latency','range_uncertainty',
            'dual_map_consistency','api_security','ui_end_to_end',
            'physical_usb_drone','cost_and_reproducibility')
VALID = {'PASS','FAIL','NOT_TESTED','BLOCKED'}

def acceptance_matrix(evidence: dict):
    matrix={}
    for item in REQUIRED:
        row=evidence.get(item,{})
        status=row.get('status','NOT_TESTED')
        if status not in VALID:
            raise ValueError(f'INVALID_STATUS_{item}')
        references=row.get('evidence',[])
        if status=='PASS' and (not isinstance(references,list) or not references):
            raise ValueError(f'UNSUPPORTED_PASS_{item}')
        matrix[item]={'status':status,'evidence':references,'notes':row.get('notes','')}
    ready=all(x['status']=='PASS' for x in matrix.values())
    return {'overall':'EVIDENCE_COMPLETE' if ready else 'EVIDENCE_INCOMPLETE',
            'criteria':matrix,'note':'PASS labels are user-provided claims with references; verify files and official thresholds independently.'}

def from_file(source, out):
    target=Path(out)
    if target.exists(): raise FileExistsError(target)
    data=json.loads(Path(source).read_text(encoding='utf-8'))
    result=acceptance_matrix(data)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
