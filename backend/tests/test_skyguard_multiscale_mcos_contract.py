import sys, types, json, zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from skyguard_multiscale.mcos import inspect_record, propose_clean_manifest
import pytest


class FakeCapture:
    def __init__(self,path,frames):
        self.frames=frames;self.index=0;self.released=False
    def isOpened(self):return True
    def read(self):
        if self.index>=self.frames:return False,None
        self.index+=1
        return True,None
    def release(self):self.released=True


def test_mcos_counts_and_release_before_cleanup(tmp_path, monkeypatch):
    z=tmp_path/'dummy.zip'
    with zipfile.ZipFile(z,'w') as archive:
        archive.writestr('clip.mp4',b'fake');archive.writestr('clip.mat',b'fake')
    cap=FakeCapture('unused',3)
    monkeypatch.setitem(sys.modules,'cv2',types.SimpleNamespace(VideoCapture=lambda path:cap))
    monkeypatch.setitem(sys.modules,'mcos_decoder',types.SimpleNamespace(load_groundtruth=lambda path:[None]*4))
    row=dict(sequence_id='IR_BIRD_001',split='train',**{'class':'BIRD'},modality='IR',
             video_zip_entry='clip.mp4',label_zip_entry='clip.mat')
    with zipfile.ZipFile(z) as archive:
        result=inspect_record(archive,row)
    assert cap.released
    assert result['state']=='QUARANTINE_OFF_BY_ONE'
    assert result['difference']==-1
    assert result['decoded_frames']==3 and result['annotation_frames']==4


def test_major_difference_never_auto_fixed(tmp_path,monkeypatch):
    z=tmp_path/'dummy.zip'
    with zipfile.ZipFile(z,'w') as archive:
        archive.writestr('clip.mp4',b'fake');archive.writestr('clip.mat',b'fake')
    cap=FakeCapture('unused',316)
    monkeypatch.setitem(sys.modules,'cv2',types.SimpleNamespace(VideoCapture=lambda path:cap))
    monkeypatch.setitem(sys.modules,'mcos_decoder',types.SimpleNamespace(load_groundtruth=lambda path:[None]*97))
    row=dict(sequence_id='V_HELICOPTER_040',split='test',**{'class':'HELICOPTER'},modality='VISIBLE',
             video_zip_entry='clip.mp4',label_zip_entry='clip.mat')
    with zipfile.ZipFile(z) as archive:
        result=inspect_record(archive,row)
    assert result['state']=='QUARANTINE_MAJOR_MISMATCH'
    assert result['difference']==219


def test_manifest_only_count_matches_are_candidates(tmp_path):
    manifest=dict(source_sha256='abc',source_zip_path='unused',records=[
        dict(sequence_id='good',split='train',**{'class':'BIRD'},modality='IR'),
        dict(sequence_id='bad',split='train',**{'class':'BIRD'},modality='IR')])
    audit=dict(source_sha256='abc',total=2,outcomes=[
        dict(sequence_id='good',state='COUNT_MATCH_ALIGNMENT_UNVERIFIED'),
        dict(sequence_id='bad',state='QUARANTINE_OFF_BY_ONE')])
    (tmp_path/'m.json').write_text(json.dumps(manifest))
    (tmp_path/'a.json').write_text(json.dumps(audit))
    output=tmp_path/'clean.json'
    res=propose_clean_manifest(tmp_path/'m.json',tmp_path/'a.json',output)
    assert res['status']=='CANDIDATE_ONLY_NOT_TRAIN_READY'
    assert [r['sequence_id'] for r in json.loads(output.read_text())['records']]==['good']
