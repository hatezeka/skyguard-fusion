from fastapi import FastAPI, Depends
from app.schemas import CameraProfile
from app.security import require_api_key

app = FastAPI(title='SkyGuard FUSION API', version='0.1.0')

@app.get('/api/v1/health')
def health():
    return {'status': 'ok', 'service': 'skyguard-fusion', 'version': '0.1.0'}

@app.get('/api/v1/cameras', response_model=list[CameraProfile])
def cameras():
    # Camera adapters are implemented in a later phase.
    return []

from app.api.camera_management import router as camera_management_router
app.include_router(
    camera_management_router,
    dependencies=[Depends(require_api_key)],
)

from app.api.live_camera import router as live_camera_router
app.include_router(
    live_camera_router,
    dependencies=[Depends(require_api_key)],
)

from app.api.geo import router as geo_router
app.include_router(
    geo_router,
    dependencies=[Depends(require_api_key)],
)
