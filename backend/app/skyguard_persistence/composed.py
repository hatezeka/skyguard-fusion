"""Import an EXISTING SkyGuard FastAPI instance and add persistence without editing it.

Set SKYGUARD_MAIN_APP=app.main:app (default) or the actual module:attribute.
Run: python -m uvicorn app.skyguard_persistence.composed:app --host 127.0.0.1 --port 8000
"""
import importlib
import os

from fastapi import FastAPI
from .api import create_router
from .database import Database
from .live_bridge import mount_live_bridge
from .media_jobs import MediaJobs

main_spec = os.getenv("SKYGUARD_MAIN_APP", "app.main:app")
module_name, sep, attr_name = main_spec.partition(":")
if not sep or not module_name or not attr_name:
    raise RuntimeError("SKYGUARD_MAIN_APP must have form module:fastapi_instance")
try:
    module = importlib.import_module(module_name)
    app = getattr(module, attr_name)
except (ImportError, AttributeError) as exc:
    raise RuntimeError(f"Cannot load existing SkyGuard FastAPI app at {main_spec}: {exc}") from exc
if not isinstance(app, FastAPI):
    raise TypeError(f"{main_spec} is not a FastAPI instance")
if not getattr(app.state, "skyguard_persistence_mounted", False):
    database = Database()
    app.include_router(create_router(database))
    mount_live_bridge(app, database)
    app.state.skyguard_persistence_mounted = True

    @app.on_event("startup")
    def start_media_jobs():
        if main_spec == "app.main:app":
            jobs = MediaJobs(database)
            jobs.interrupt_pending()
            app.state.skyguard_media_jobs = jobs
