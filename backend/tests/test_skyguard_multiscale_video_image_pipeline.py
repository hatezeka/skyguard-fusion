"""End-to-end I/O tests on generated media, with fake model (not AI accuracy)."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from skyguard_multiscale import cli

class FakeAdapter:
    def __init__(self, weights): pass
    def predict(self, frame, imgsz, confidence):
        if imgsz==640 and frame.shape[1]==320:
            return [('DRONE',.9,(50,50,150,150)),('AIRPLANE',.95,(180,80,280,180))]
        return []


def test_still_image_output_has_class_and_source_hash(tmp_path,monkeypatch):
    image=tmp_path/'input.jpg'; weights=tmp_path/'best.pt'
    assert cv2.imwrite(str(image),np.zeros((240,320,3),dtype=np.uint8))
    weights.write_bytes(b'test stub checkpoint')
    monkeypatch.setattr(cli,'YOLOAdapter',FakeAdapter)
    args=type('Args',(),dict(input=str(image),weights=str(weights),out=str(tmp_path/'out'),
                             max_frames=0,max_tiles=128,accept_confidence=.6))
    cli.run_media(args)
    rows=[json.loads(line) for line in (tmp_path/'out'/'detections.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['frame_index']==0
    assert rows[0]['timestamp_source']=='still_image'
    assert {d['class_name'] for d in rows[0]['detections']}=={'DRONE','UNKNOWN'}
    assert all(d['track_id'] is None for d in rows[0]['detections'] if d['class_name']=='UNKNOWN')


def test_short_video_decodes_frames_and_monotonic_ids(tmp_path,monkeypatch):
    path=tmp_path/'clip.avi';weights=tmp_path/'best.pt';weights.write_bytes(b'stub')
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240))
    if not writer.isOpened(): pytest.skip('MJPG encoder unavailable')
    for i in range(3): writer.write(np.full((240,320,3),i*25,dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(cli,'YOLOAdapter',FakeAdapter)
    args=type('Args',(),dict(input=str(path),weights=str(weights),out=str(tmp_path/'out'),
                             max_frames=0,max_tiles=128,accept_confidence=.6))
    cli.run_media(args)
    rows=[json.loads(line) for line in (tmp_path/'out'/'detections.jsonl').read_text().splitlines()]
    assert [r['frame_index'] for r in rows]==[0,1,2]
    ids=[[d['track_id'] for d in r['detections'] if d['class_name']=='DRONE'] for r in rows]
    assert ids==[[1],[1],[1]]
    assert all(d['track_id'] is None for r in rows for d in r['detections'] if d['class_name']=='UNKNOWN')
