"""Run with: python -m pytest -q deploy/test_gateway.py"""
from __future__ import annotations
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import httpx
from fastapi.testclient import TestClient

HERE = Path(__file__).parent
import tempfile
static = Path(tempfile.mkdtemp(prefix='sg-deploy-gateway-'))
(static / 'operator.html').write_text('<html>Operator</html>')
salt = bytes.fromhex('102030405060708090a0b0c0d0e0f011')
password = 'test-local-operator-strong-password'
hash_value = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 450000).hex()
os.environ['SKYGUARD_OPERATOR_HASH'] = f'450000:{salt.hex()}:{hash_value}'
os.environ['SKYGUARD_DB_API_KEY'] = 'T'*64
os.environ['SKYGUARD_SESSION_SIGNING_KEY'] = 'S'*64
os.environ['SKYGUARD_STATIC_ROOT'] = str(static)
os.environ['SKYGUARD_PUBLIC_ORIGIN'] = 'http://testserver'
os.environ['SKYGUARD_SECURE_COOKIES'] = 'false'
spec = importlib.util.spec_from_file_location('sg_test_gateway', HERE / 'gateway.py')
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)


def login(client):
    r = client.post('/auth/login', json={'username':'operator', 'password':password},
                    headers={'Origin': 'http://testserver'})
    assert r.status_code == 200
    assert 'httponly' in r.headers['set-cookie'].lower()
    assert 'samesite=strict' in r.headers['set-cookie'].lower()


def test_anonymous_api_is_denied():
    with TestClient(g.app) as client:
        r=client.get('/api/persistence/health')
        assert r.status_code == 401
        assert client.get('/operator.html', follow_redirects=False).status_code == 303


def test_operator_login_and_logout():
    with TestClient(g.app) as client:
        assert client.post('/auth/login',json={'username':'operator','password':'bad'},headers={'Origin':'http://testserver'}).status_code==401
        login(client)
        assert client.get('/auth/session').json()['authenticated'] is True
        assert client.get('/operator.html').status_code==200
        assert client.post('/auth/logout',headers={'Origin':'http://testserver'}).status_code==200
        assert client.get('/api/persistence/health').status_code == 401


def test_scientific_registration_rejected_from_browser():
    with TestClient(g.app) as client:
        login(client)
        for item in ('frames', 'benchmarks', 'evidence', 'sessions'):
            r=client.post('/api/persistence/'+item, json={},headers={'Origin':'http://testserver'})
            assert r.status_code==403, (item,r.text)
        assert client.post('/api/persistence/cameras',json={},headers={'Origin':'https://evil.test'}).status_code==403
        assert client.get('/api/elsewhere').status_code==403


def test_server_key_forwarded_and_never_returned(monkeypatch):
    recorded={}
    class MockAsyncClient:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def request(self,method,url,**kwargs):
            recorded.update({'method':method,'url':url, 'headers':kwargs['headers']})
            return httpx.Response(200, json={'status':'ok','backend':'postgresql'})
    monkeypatch.setattr(g.httpx,'AsyncClient',MockAsyncClient)
    with TestClient(g.app) as client:
        login(client)
        r=client.get('/api/persistence/health')
        assert r.status_code==200 and r.json()['backend']=='postgresql'
        assert recorded['headers']['Authorization']=='Bearer '+'T'*64
        assert 'T'*64 not in r.text
        assert client.get('/api/persistence/../../etc').status_code in (400,404)


def test_media_only_authenticated_and_limited():
    with TestClient(g.app) as client:
        assert client.post('/api/media/upload',content=b'hello',headers={'X-Skyguard-Media-Type':'IMAGE'}).status_code==401
        login(client)
        assert client.post('/api/media/upload',content=b'hello',headers={'Origin':'https://evil.example','X-Skyguard-Media-Type':'IMAGE'}).status_code==403
        assert client.post('/api/media/upload',content=b'hello',headers={'Origin':'http://testserver','X-Skyguard-Media-Type':'ZIP'}).status_code==422
        assert client.post('/api/media/upload',content=b'x'*(12*1024*1024+1),headers={'Origin':'http://testserver','X-Skyguard-Media-Type':'VIDEO', 'Content-Length':str(41*1024*1024)}).status_code==413
        assert client.get('/api/media/jobs/../../etc').status_code in (403,404)


def test_live_camera_rtsp_ssrf_guard():
    assert g._validate_live_camera_json(b'{"source_type":"rtsp","source_location":"rtsp://192.168.1.70:554/stream"}')
    assert not g._validate_live_camera_json(b'{"source_type":"rtsp","source_location":"rtsp://169.254.169.254/metadata"}')
    assert not g._validate_live_camera_json(b'{"source_type":"rtsp","source_location":"rtsp://127.0.0.1/"}')
    assert not g._validate_live_camera_json(b'{"source_type":"rtsp","source_location":"rtsp://example.org/"}')
    assert g._validate_live_camera_json(b'{"source_type":"usb","camera_index":0}')
    assert not g._validate_live_camera_json(b'{"source_type":"usb","camera_index":999}')
