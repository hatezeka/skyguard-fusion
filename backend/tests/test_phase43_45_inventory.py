from zipfile import ZipFile
from pathlib import Path
import pytest
from app.phase43_45.inventory import inspect_halmstad_zip, stratified_splits, inspect_dut_zip
from app.phase43_45.export_yolo import _bbox_to_yolo


def test_stratified_split_keeps_ir_visible_group_together():
    rows=[]
    for cls in ('DRONE','BIRD','AIRPLANE','HELICOPTER'):
        for i in range(12):
            for mod in ('IR','VISIBLE'):
                rows.append({'group_id':f'{cls}:{i:03d}', 'class':cls,'modality':mod})
    stratified_splits(rows)
    mapping={}
    for r in rows:
        if r['group_id'] in mapping:
            assert mapping[r['group_id']]==r['split']
        mapping[r['group_id']]=r['split']
    assert set(x['split'] for x in rows)=={'train','val','test'}


def test_archive_zip_slip_is_rejected(tmp_path):
    zpath=tmp_path/'bad.zip'
    with ZipFile(zpath,'w') as z:
        z.writestr('../foo.mp4','bad')
    with pytest.raises(ValueError):inspect_halmstad_zip(zpath)


def test_inventory_does_not_misreport_missing_gt(tmp_path):
    zpath=tmp_path/'partial.zip'
    with ZipFile(zpath,'w') as z:
        z.writestr('Drone-detection-dataset-master/Data/Video_V/V_DRONE_001.mp4','raw')
    r=inspect_halmstad_zip(zpath)
    assert r['counts']['unpaired_videos']==1
    assert r['inventory_integrity']=='FAIL'


def test_dut_readme_only_not_treated_as_dataset(tmp_path):
    zp=tmp_path/'dut.zip'
    with ZipFile(zp,'w') as z:
        z.writestr('DUT-Anti-UAV-master/README.md','download links')
    r=inspect_dut_zip(zp)
    assert r['status']=='DATASET_NOT_INCLUDED'


def test_yolo_bounds_check():
    assert _bbox_to_yolo([0,0,100,50],100,50)==pytest.approx((.5,.5,1,1))
    assert _bbox_to_yolo(None,100,50) is None
    with pytest.raises(ValueError):_bbox_to_yolo([-500,0,3,3],100,50)
