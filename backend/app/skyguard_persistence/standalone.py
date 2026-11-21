"""Runs the persistence API on its own when the existing app cannot yet be wired."""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import create_router

app = FastAPI(title="SkyGuard Persistence API", version="1.0.0")
allowed_origins = [x.strip() for x in os.getenv(
    "SKYGUARD_DB_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
app.include_router(create_router())
