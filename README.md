# SkyGuard FUSION — Track 3 MVP

Phase 3 foundation: FastAPI health endpoint, shared validated schemas, and tests.
**Not yet implemented:** drone detection, tracking, localization, camera integration, frontend.

## Setup (Python 3.11+)

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux/macOS: .venv/bin/python -m pip install -r requirements.txt
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --reload --port 8000
```

Visit http://127.0.0.1:8000/docs or GET http://127.0.0.1:8000/api/v1/health.

## Architecture

Camera input -> profile manager -> detection/classification -> tracking -> localization -> storage/API -> map.

Camera-model identification is optional. Unknown cameras must use the standard fallback.
Unknown distances must not be invented. Tests must distinguish synthetic benchmarks from real-camera validation.
