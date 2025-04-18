from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import redis
import json
import logging
import os

from app.services.email_service import EmailService
from app.services.processor_service import EmailProcessor
from app.services.metrics_service import MetricsService
from app.services.config_service import ConfigService
from app.services.fakemail_service import FakeMailService
from app.models.fakemail import EmailClassification

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])

# Redis client dependency
def get_redis_client():
    return redis.Redis(host="redis", port=6379, db=0)

# Service dependencies
def get_email_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return EmailService(redis_client)
    
def get_metrics_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return MetricsService(redis_client)
    
def get_config_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return ConfigService(redis_client)

def get_fakemail_service(redis_client: redis.Redis = Depends(get_redis_client)):
    fakemail_url = os.environ.get("FAKEMAIL_API_URL", "http://localhost:8000")
    return FakeMailService(redis_client, fakemail_url)

# API Endpoints
@router.get("/stats", response_model=Dict[str, Any])
async def get_stats(metrics_service: MetricsService = Depends(get_metrics_service)):
    """Get system statistics and performance metrics"""
    stats = metrics_service.get_stats()
    
    # Format stats to match what the UI expects
    return {
        "total_emails_processed": stats.total_emails_processed,
        "success_rate": stats.success_rate,
        "avg_processing_time_ms": stats.avg_processing_time_ms,
        "last_history_id": stats.last_history_id,
        "emails_by_classification": stats.emails_by_classification,
        "uptime_seconds": stats.uptime_seconds,
        "last_processed_at": stats.last_processed_at,
        
        # Additional details
        "emails": {
            "total": stats.total_emails_processed,
            "success": stats.successful_emails,
            "failed": stats.failed_emails,
        },
        "performance": {
            "avg_queue_time_ms": stats.avg_queue_time_ms,
            "emails_per_minute": stats.emails_per_minute
        },
        "system": {
            "cpu_usage": stats.cpu_usage,
            "memory_usage": stats.memory_usage
        },
        "timestamps": {
            "last_email_processed": stats.last_email_processed.isoformat() if stats.last_email_processed else None,
            "current_time": datetime.now().isoformat()
        }
    }

"""
This endpoint has been moved to webhook_routes.py for better organization.
The /results/{email_id} endpoint is now in the webhook router to make it more prominent.
"""

@router.get("/config")
async def get_configuration(config_service: ConfigService = Depends(get_config_service)):
    """Get current system configuration"""
    return config_service.get_system_config()

@router.put("/config/{key}")
async def update_configuration(
    key: str,
    value: Dict[str, Any],
    config_service: ConfigService = Depends(get_config_service)
):
    """Update a specific configuration value"""
    update_dict = {key: value}
    updated_config = config_service.update_system_config(update_dict)
    return {"status": "success", "key": key, "value": value, "config": updated_config}

@router.post("/process/trigger")
async def trigger_processing(background_tasks: BackgroundTasks):
    """Manually trigger email processing"""
    redis_client = get_redis_client()
    redis_client.publish("email:control", json.dumps({"command": "process_now"}))
    return {"status": "success", "message": "Processing triggered"}

@router.post("/system/reset-stats")
async def reset_statistics(metrics_service: MetricsService = Depends(get_metrics_service)):
    """Reset system statistics"""
    # Initialize new stats
    metrics_service._initialize_stats()
    return {"status": "success", "message": "Statistics reset successfully"}