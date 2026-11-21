"""Conservative, reversible SkyGuard frontend gateway-mode integration.

Only touches the existing API clients and Operator Console auth state.
Does not modify maps, styling, detection logic, science or database schema.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--backup')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
root = Path(args.repo).resolve()
if not (root / 'frontend/operator.vite.config.ts').is_file():
    raise SystemExit('OPERATOR_FRONTEND_MISSING')


def change_one(src: str, old: str, new: str, label: str) -> str:
    if new in src:
        if src.count(new) != 1:
            raise ValueError(f'DUPLICATE_PATCH: {label}')
        return src  # already patched
    count = src.count(old)
    if count != 1:
        raise ValueError(f'UNRECOGNIZED_SOURCE: {label}, occurrences={count}; source untouched')
    return src.replace(old, new, 1)


def patch_operator_api(src: str) -> str:
    src = change_one(
        src,
        '    const key=this.token();if(!key)throw new Error(\'Local operator key is required; never store a service key in production frontend\')',
        '    const deployed=import.meta.env.VITE_SKYGUARD_DEPLOYED===\'true\';const key=this.token();if(!deployed&&!key)throw new Error(\'Local operator key is required; never store a service key in production frontend\')',
        'operator/api.ts token gate'
    )
    src = change_one(
        src,
        "headers:{'Authorization':`Bearer ${key}`,...(init?.body?{'Content-Type':'application/json'}:{}),...init?.headers}",
        "headers:{...(deployed?{}:{'Authorization':`Bearer ${key}`}),...(init?.body?{'Content-Type':'application/json'}:{}),...init?.headers}",
        'operator/api.ts auth header'
    )
    return src


def patch_shared_api(src: str) -> str:
    src = change_one(
        src,
        "    if (!token) throw new Error('Authenticated persistence session required')",
        "    const deployed = import.meta.env.VITE_SKYGUARD_DEPLOYED === 'true'\n    if (!deployed && !token) throw new Error('Authenticated persistence session required')",
        'shared api gate'
    )
    src = change_one(
        src,
        '      headers: { Authorization: `Bearer ${token}` },',
        "      headers: deployed ? {} : { Authorization: `Bearer ${token}` },\n      credentials: 'same-origin',",
        'shared api header'
    )
    return src


def patch_operator_app(src: str) -> str:
    anchor = "import './operator.css'"
    if "const SKYGUARD_GATEWAY = import.meta.env.VITE_SKYGUARD_DEPLOYED === 'true'" not in src:
        src = change_one(src, anchor,
                         anchor + "\n// Public boolean only; production API credentials remain server-side.\nconst SKYGUARD_GATEWAY = import.meta.env.VITE_SKYGUARD_DEPLOYED === 'true'", 'app gateway flag')
    src = change_one(src, "[secret,setSecret]=useState('')",
                     "[secret,setSecret]=useState(SKYGUARD_GATEWAY?'SERVER_SESSION':'')", 'app auth state')
    src = change_one(src,
                     "if(!local){setError('Direct bearer key access blocked outside localhost. Deploy authenticated BFF with secure cookies.');return}",
                     "if(!local&&!SKYGUARD_GATEWAY){setError('Direct bearer key access blocked outside localhost. Deploy authenticated BFF with secure cookies.');return}",
                     'app deployment guard')
    anchor = "  useEffect(()=>{const t=window.setInterval(()=>setClock(new Date()),1000);return()=>clearInterval(t)},[]);"
    # Some versions do not preserve double leading spaces.
    if anchor not in src:
        anchor=anchor.lstrip()
    replacement = (
        "  const gatewayInitialConnect=useRef(false);\n"
        "  useEffect(()=>{if(SKYGUARD_GATEWAY&&!gatewayInitialConnect.current){gatewayInitialConnect.current=true;void connect()}},[connect]);\n"
        + anchor
    )
    src = change_one(src, anchor, replacement, 'app initial session connection')
    # Preserve development settings. In deployed builds the interface has only
    # a server-authenticated operator session and no service API-key input.
    old_start = '<div className="sg-settings-grid"><Panel title="Local API connection"'
    new_start = (
        '<div className="sg-settings-grid">{SKYGUARD_GATEWAY?'
        '<Panel title="Secure operator session" subtitle="Authenticated server gateway">'
        '<div className="sg-form"><p>Connected with an HttpOnly operator session. '
        'The PostgreSQL service key is held only by the backend gateway.</p>'
        '<p>Status: {connected?\'Database connected\':\'Not connected\'}</p>'
        '<div className="sg-inline-actions">'
        '<button className="sg-btn primary" onClick={()=>void connect()} disabled={busy}>Reconnect</button>'
        '<button className="sg-btn outline" onClick={()=>{void fetch(\'/auth/logout\',{method:\'POST\',credentials:\'same-origin\'}).finally(()=>window.location.assign(\'/login.html\'))}}>Sign out</button>'
        '</div></div></Panel>:'
        '<Panel title="Local API connection"'
    )
    src = change_one(src, old_start, new_start, 'app settings gateway start')
    src = change_one(src,
                     '</Panel><Panel title="Live telemetry adapter"',
                     '</Panel>}<Panel title="Live telemetry adapter"',
                     'app settings gateway end')
    # Remove an old fixed scientific test-count assertion on the validation page
    # if present; production must read actual benchmark reports from the backend.
    if '47 / 47 (previous PowerShell run)' in src:
        src = change_one(src, '47 / 47 (previous PowerShell run)',
                         'See backend reports', 'remove static science claim')
    src = change_one(src,
        "import './operator.css'\n// Public boolean only; production API credentials remain server-side.\nconst SKYGUARD_GATEWAY",
        "import './operator.css'\nimport MediaLab from './MediaLab'\n// Public boolean only; production API credentials remain server-side.\nconst SKYGUARD_GATEWAY",
        'media lab component import')
    # Preserve any pre-existing USB/live-camera controls, alerts and monitoring
    # UI. Different SkyGuard operator versions may insert additional panels
    # before sg-monitor-grid, so anchor to the unique monitor page opening
    # instead of relying on the old immediate first child.
    src = change_one(src,
        "{page==='monitor'&&<>",
        "{page==='monitor'&&<><MediaLab/>",
        'media upload/browser camera placement')
    return src

def patch_map_typing(src: str) -> str:
    # Compatible with MapLibre versions that require an attribution options object.
    for old in ('attributionControl: true', 'attributionControl:true'):
        if old in src:
            return change_one(src, old, old.replace('true','{}'), 'MapLibre attribution options')
    return src

edits = {
    'frontend/src/operator/api.ts': patch_operator_api,
    'frontend/src/operator/App.tsx': patch_operator_app,
    'frontend/src/api/skyguardPersistence.ts': patch_shared_api,
    'frontend/src/operator/GeoPanels.tsx': patch_map_typing,
}
plan = []
for rel, func in edits.items():
    file = root / rel
    if not file.is_file():
        raise RuntimeError(f'MISSING_EXPECTED_FILE: {rel}')
    raw = file.read_bytes()
    try:
        txt = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise RuntimeError(f'NOT_UTF8: {rel}') from exc
    normalized = txt.replace('\r\n', '\n')
    changed = func(normalized)
    newline = '\r\n' if '\r\n' in txt else '\n'
    candidate = changed.replace('\n', newline).encode('utf-8')
    plan.append((file, raw, candidate, rel))
for addon in ('MediaLab.tsx','media-lab.css'):
    file = root / 'frontend' / 'src' / 'operator' / addon
    packaged = Path(__file__).parent / addon
    wanted = packaged.read_bytes()
    current = file.read_bytes() if file.exists() else b''
    if current and current != wanted:
        raise RuntimeError(f'EXISTING_MEDIA_COMPONENT_CONFLICT: {file}')
    plan.append((file,current,wanted,'frontend/src/operator/'+addon))
for file, old, new, rel in plan:
    print(json.dumps({'path': rel, 'current_sha256': hashlib.sha256(old).hexdigest(),
                      'candidate_sha256': hashlib.sha256(new).hexdigest(),
                      'edit': old != new}, ensure_ascii=True))
if args.dry_run:
    print('DEPLOY_FRONTEND_DRY_RUN_OK: nothing modified')
else:
    if not args.backup:
        raise RuntimeError('--backup is required for applying patches')
    b = Path(args.backup).resolve()
    if any(old != new for _, old, new, _ in plan):
        b.mkdir(parents=True, exist_ok=False)
        for file, old, new, rel in plan:
            if old != new and old:
                prev = b / rel
                prev.parent.mkdir(parents=True, exist_ok=True)
                prev.write_bytes(old)
        for file, old, new, rel in plan:
            if old != new:
                # Reconfirm no asynchronous edits between validation and write.
                if (file.read_bytes() if file.exists() else b'') != old:
                    raise RuntimeError(f'CONCURRENT_EDIT: {rel}; stop before other changes')
                temp = file.with_suffix(file.suffix + '.sgtemp')
                temp.write_bytes(new)
                temp.replace(file)
        print('DEPLOY_FRONTEND_PATCHED_AND_BACKED_UP: ' + str(b))
    else:
        print('DEPLOY_FRONTEND_ALREADY_PATCHED: no files changed')
