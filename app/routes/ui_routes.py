from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
import redis
import json
import logging
from typing import Dict, Any, List, Optional

from app.services.fakemail_service import FakeMailService
from app.services.processor_service import EmailProcessor
from app.services.metrics_service import MetricsService
from app.services.config_service import ConfigService

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["ui"])

# Templates setup
templates_dir = Path(__file__).parent.parent / "templates"
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Redis client dependency
def get_redis_client():
    return redis.Redis(host="redis", port=6379, db=0)

# FakeMail service dependency
def get_fakemail_service(redis_client: redis.Redis = Depends(get_redis_client)):
    fakemail_url = os.environ.get("FAKEMAIL_API_URL", "http://localhost:8000")
    return FakeMailService(redis_client, fakemail_url)

# Metrics service dependency
def get_metrics_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return MetricsService(redis_client)

# Config service dependency
def get_config_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return ConfigService(redis_client)

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main UI for the email processor"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Email Processor Dashboard"}
    )

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """Dashboard UI for the email processor"""
    # Get the processing stats
    stats = metrics_service.get_stats()
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request, 
            "title": "Email Processor Dashboard",
            "stats": stats
        }
    )

@router.get("/setup", response_class=HTMLResponse)
async def setup_ui(request: Request):
    """Setup UI for the email processor"""
    return templates.TemplateResponse(
        "setup.html", 
        {"request": request, "title": "Email Processor Setup"}
    )

@router.get("/emails", response_class=HTMLResponse)
async def emails_ui(
    request: Request,
    fakemail_service: FakeMailService = Depends(get_fakemail_service)
):
    """UI for browsing processed emails"""
    return templates.TemplateResponse(
        "emails.html", 
        {"request": request, "title": "Processed Emails"}
    )