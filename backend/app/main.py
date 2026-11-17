from fastapi import FastAPI
from app.schemas import CameraProfile

app = FastAPI(title='SkyGuard FUSION API', version='0.1.0')

@app.get('/api/v1/health')
def health():
    return {'status': 'ok', 'service': 'skyguard-fusion', 'version': '0.1.0'}

@app.get('/api/v1/cameras', response_model=list[CameraProfile])
def cameras():
    # Camera adapters are implemented in a later phase.
    return []
