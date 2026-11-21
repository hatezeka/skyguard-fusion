"""SkyGuard FUSION session gateway, deployed beside existing FastAPI.

Browser never receives SKYGUARD_DB_API_KEY. Operator auth is single-user,
not multi-user RBAC. Scientific ingestion is backend-only.
"""
from __future__ import annotations

import base64
import ipaddress
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

ROOT = Path(os.getenv("SKYGUARD_STATIC_ROOT", "/opt/skyguard/static")).resolve()
BACKEND = os.getenv("SKYGUARD_BACKEND_ORIGIN", "http://backend:8000").rstrip("/")
SERVICE_KEY = os.getenv("SKYGUARD_DB_API_KEY", "")
OPERATOR_HASH = os.getenv("SKYGUARD_OPERATOR_HASH", "")
SIGN_KEY = os.getenv("SKYGUARD_SESSION_SIGNING_KEY", "")
ORIGIN = os.getenv("SKYGUARD_PUBLIC_ORIGIN", "http://127.0.0.1:8088").rstrip("/")
SECURE = os.getenv("SKYGUARD_SECURE_COOKIES", "false").lower() == "true"
COOKIE = "__Host-skyguard-session" if SECURE else "skyguard-session-local"
SESSION_SECONDS = 8 * 60 * 60
FAILED: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(title="SkyGuard authenticated operator gateway", docs_url=None, redoc_url=None, openapi_url=None)


def required_configuration() -> None:
    if not SERVICE_KEY or len(SERVICE_KEY) < 32 or not SIGN_KEY or len(SIGN_KEY) < 32:
        raise RuntimeError("Missing service key / gateway signing key")
    if not OPERATOR_HASH or len(OPERATOR_HASH.split(":")) != 3:
        raise RuntimeError("Invalid operator password PBKDF2 specification")


@app.on_event("startup")
def check_startup() -> None:
    required_configuration()
    if not (ROOT / "operator.html").is_file():
        raise RuntimeError("Operator frontend not built: operator.html missing")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _new_session() -> str:
    payload = json.dumps({"u": "operator", "e": int(time.time()) + SESSION_SECONDS,
                          "n": secrets.token_hex(16)}, separators=(",", ":")).encode()
    token = _b64(payload)
    sig = _b64(hmac.new(SIGN_KEY.encode(), token.encode(), hashlib.sha256).digest())
    return token + "." + sig


def _authorized(request: Request) -> bool:
    value = request.cookies.get(COOKIE, "")
    try:
        token, supplied_sig = value.split(".", 1)
        correct = _b64(hmac.new(SIGN_KEY.encode(), token.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied_sig, correct):
            return False
        claims = json.loads(_decode(token))
        return claims.get("u") == "operator" and int(claims["e"]) > time.time()
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _same_origin(request: Request) -> bool:
    source = request.headers.get("origin", "")
    return (not source or source.rstrip("/") == ORIGIN)


def _password_matches(password: str) -> bool:
    try:
        cycles, salt, good = OPERATOR_HASH.split(":")
        rounds = int(cycles)
        if rounds < 400000 or rounds > 2000000:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), rounds)
        return hmac.compare_digest(digest, bytes.fromhex(good))
    except (ValueError, TypeError):
        return False


def _check_rate_limit(request: Request) -> None:
    address = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = FAILED[address]
    while window and now - window[0] > 120:
        window.popleft()
    if len(window) >= 6:
        raise HTTPException(429, "Too many login attempts")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": message}, headers={"Cache-Control": "no-store"})


@app.get("/healthz")
def healthz():
    return {"gateway": "ready", "database": "not_checked"}


@app.get("/auth/session")
def current_session(request: Request):
    return {"authenticated": _authorized(request), "identity": "operator" if _authorized(request) else None}


@app.post("/auth/login")
async def login(request: Request):
    if not _same_origin(request):
        return _error(403, "Origin rejected")
    if not request.headers.get("content-type", "").startswith("application/json"):
        return _error(415, "JSON required")
    _check_rate_limit(request)
    raw = await request.body()
    if len(raw) > 4096:
        return _error(413, "Payload too large")
    try:
        data = json.loads(raw)
        username, password = data.get("username"), data.get("password")
    except (ValueError, TypeError, AttributeError):
        return _error(400, "Malformed login")
    if username != "operator" or not isinstance(password, str) or not _password_matches(password):
        address = request.client.host if request.client else "unknown"
        FAILED[address].append(time.monotonic())
        return _error(401, "Invalid credentials")
    answer = JSONResponse({"authenticated": True, "identity": "operator"})
    answer.set_cookie(COOKIE, _new_session(), max_age=SESSION_SECONDS,
                      secure=SECURE, httponly=True, samesite="strict", path="/")
    answer.headers["Cache-Control"] = "no-store"
    return answer


@app.post("/auth/logout")
def logout(request: Request):
    if not _same_origin(request):
        return _error(403, "Origin rejected")
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(COOKIE, path="/", secure=SECURE, httponly=True, samesite="strict")
    response.headers["Cache-Control"] = "no-store"
    return response


# Browser is not an evidence ingestion or scientific registration client.
# Interactive operator actions are limited; admin-only writes remain private to backend.
def _allowed_post(path: str) -> bool:
    return bool(
        path == "persistence/cameras" or
        path in ("persistence/live/start", "persistence/live/stop") or
        re.fullmatch(r"persistence/alerts/[A-Za-z0-9_-]{1,128}/acknowledge", path)
    )


# Public browser never sees service API key; media uploads are user inputs, NOT
# privileged scientific registrations. The backend alone writes evidence frames.
MEDIA_GET_RE = re.compile(r"(?:health|jobs/[0-9a-f-]{36})\Z")
MEDIA_POSTS = {"upload", "browser/start", "browser/frame", "browser/stop"}


@app.api_route("/api/media/{path:path}", methods=["GET", "POST"])
async def media_proxy(path: str, request: Request):
    if not _authorized(request):
        return _error(401, "Operator session required")
    if not ((request.method == 'GET' and MEDIA_GET_RE.fullmatch(path)) or
            (request.method == 'POST' and path in MEDIA_POSTS)):
        return _error(403, "Media endpoint not available to operator")
    if request.method == 'POST' and not _same_origin(request):
        return _error(403, "Origin rejected")
    if request.method == 'POST' and request.headers.get('sec-fetch-site','same-origin') not in ('same-origin','none'):
        return _error(403, "Cross-site request rejected")
    limit = 40*1024*1024 if path=='upload' else (2*1024*1024 if path=='browser/frame' else 4096)
    header = request.headers.get('content-length')
    if header:
        try:
            if int(header)>limit: return _error(413, "Payload too large")
        except ValueError: return _error(400, "Invalid content length")
    body = bytearray()
    if request.method == 'POST':
        async for part in request.stream():
            body.extend(part)
            if len(body)>limit: return _error(413, "Payload too large")
    headers = {"Authorization": f"Bearer {SERVICE_KEY}",
               "Accept": "application/json",
               "Content-Type": "application/octet-stream" if path=='upload' else
                               ("image/jpeg" if path=='browser/frame' else "application/json")}
    if path=='upload':
        kind=request.headers.get('x-skyguard-media-type','').upper()
        if kind not in ('IMAGE','VIDEO'):return _error(422,'Invalid media type')
        headers['X-Skyguard-Media-Type']=kind
    try:
        async with httpx.AsyncClient(timeout=120,follow_redirects=False) as client:
            upstream=await client.request(request.method,f'{BACKEND}/api/media/{path}',content=bytes(body),headers=headers)
        if len(upstream.content)>5_000_000:return _error(502,'Media result exceeds gateway limit')
    except httpx.HTTPError:
        return _error(502,'Backend unavailable')
    return Response(upstream.content,status_code=upstream.status_code,
                    media_type='application/json',headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})


def _validate_live_camera_json(body: bytes) -> bool:
    """Private LAN IP RTSP only; block localhost, cloud metadata and DNS rebinding."""
    try:
        payload=json.loads(body)
        kind=payload.get('source_type')
        if kind!='rtsp':
            return kind=='usb' and type(payload.get('camera_index')) is int and 0<=payload['camera_index']<=4
        loc=payload['source_location']
        parsed=urlsplit(loc)
        if parsed.scheme!='rtsp' or parsed.hostname is None or len(loc)>500:
            return False
        host=ipaddress.ip_address(parsed.hostname)
        if host.version!=4:return False
        networks=[ipaddress.ip_network(n) for n in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16')]
        return any(host in net for net in networks)
    except (KeyError,ValueError,TypeError):
        return False


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def backend_proxy(path: str, request: Request):
    if not _authorized(request):
        return _error(401, "Operator session required")
    if request.method not in ("GET", "POST"):
        return _error(405, "HTTP method not permitted by operator gateway")
    if not path.startswith("persistence/"):
        return _error(403, "Route is not exposed to operator gateway")
    if not re.fullmatch(r"[a-zA-Z0-9_/-]{1,255}", path) or ".." in path or "//" in path:
        return _error(400, "Invalid route")
    if request.method == "POST" and not _allowed_post(path):
        return _error(403, "Scientific registration/ingestion is backend-only")
    if request.method == "POST" and not _same_origin(request):
        return _error(403, "Origin rejected")
    body = await request.body() if request.method == "POST" else b""
    if len(body) > 65536:
        return _error(413, "Operator payload too large")
    if path=='persistence/live/start' and not _validate_live_camera_json(body):
        return _error(422, 'Live camera must be USB index 0..4 or a private IPv4 RTSP URL')
    url = f"{BACKEND}/api/{path}"
    if request.url.query:
        url += "?" + request.url.query
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=False) as client:
            upstream = await client.request(request.method, url, content=body,
                                            headers={"Authorization": f"Bearer {SERVICE_KEY}",
                                                     "Accept": "application/json",
                                                     "Content-Type": "application/json"})
        if len(upstream.content) > 10_000_000:
            return _error(502, "Upstream payload too large")
    except httpx.HTTPError:
        return _error(502, "Backend unavailable")
    media_type = upstream.headers.get("content-type", "application/json").split(";", 1)[0]
    return Response(content=upstream.content, status_code=upstream.status_code,
                    media_type=media_type, headers={"Cache-Control": "no-store",
                                                    "X-Content-Type-Options": "nosniff"})


LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SkyGuard Operator Sign In</title><style>body{background:#071321;color:#e9f1fc;font:16px system-ui;display:grid;place-items:center;min-height:100vh;margin:0}main{border:1px solid #2a4761;background:#101f30;padding:2rem;border-radius:1rem;max-width:400px;width:85vw}h1{font-size:1.4rem;margin-top:0}input,button{box-sizing:border-box;width:100%;font:inherit;border-radius:.5rem;padding:.8rem;margin:.5rem 0 1rem}input{background:#081624;color:white;border:1px solid #496078}button{background:#3a95c8;color:white;border:0;cursor:pointer}p{font-size:.9rem;color:#a5bacd}</style></head><body><main><h1>SKYGUARD FUSION</h1><p>Authenticated operator access · backend scientific registration only</p><form id="form"><label>Operator<input name="username" value="operator" autocomplete="username" required></label><label>Password<input name="password" type="password" autocomplete="current-password" required></label><button type="submit">Sign in</button></form><p id="result" role="status"></p></main><script>document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();let f=new FormData(e.target);let r=await fetch('/auth/login',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:f.get('username'),password:f.get('password')})});if(r.ok)location.replace('/operator.html');else document.getElementById('result').textContent='Authentication failed or temporarily limited';});</script></body></html>"""


@app.get("/login.html")
def login_page():
    return Response(LOGIN_HTML, media_type="text/html", headers={"Cache-Control": "no-store",
                 "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'"})


@app.get("/{asset_path:path}")
def site(asset_path: str, request: Request):
    if not _authorized(request):
        return RedirectResponse("/login.html", status_code=303)
    asset_path = asset_path or "operator.html"
    if asset_path == "login.html":
        return RedirectResponse("/operator.html", status_code=303)
    try:
        target = (ROOT / asset_path).resolve()
        target.relative_to(ROOT)
    except ValueError:
        raise HTTPException(404, "Not found")
    if not target.is_file():
        raise HTTPException(404, "Not found")
    response = FileResponse(target, media_type=mimetypes.guess_type(str(target))[0])
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store" if target.suffix == ".html" else "public, max-age=3600"
    return response
