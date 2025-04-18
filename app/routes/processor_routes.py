from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks, Query, Path
from typing import Dict, Any, List, Optional
import redis
import os
import logging
import time
import asyncio
from app.models.fakemail import (
    WebhookPayload, 
    EmailClassification, 
    ProcessingStats, 
    ProcessingBatch,
    SystemConfig,
    ProcessingRule,
    HealthStatus,
    ClassificationType
)
from app.services.fakemail_service import FakeMailService
from app.services.processor_service import EmailProcessor
from app.services.queue_service import EmailQueue
from app.services.metrics_service import MetricsService
from app.services.config_service import ConfigService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["email_processor"])

# Get the FakeMail API base URL from environment variable
FAKEMAIL_API_URL = os.getenv("FAKEMAIL_API_URL", "http://fakemail-api:8000")
WORKER_COUNT = int(os.getenv("PROCESSOR_WORKER_COUNT", "3"))
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
STARTUP_TIME = time.time()

# Redis client dependency
def get_redis_client():
    return redis.Redis(host="redis", port=6379, db=0)

# FakeMail service dependency
def get_fakemail_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return FakeMailService(redis_client, FAKEMAIL_API_URL)

# Config service dependency
def get_config_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return ConfigService(redis_client)

# Metrics service dependency
def get_metrics_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return MetricsService(redis_client)

# Email queue dependency
def get_email_queue(redis_client: redis.Redis = Depends(get_redis_client)):
    return EmailQueue(redis_client)

# Email processor dependency
def get_email_processor(
    fakemail_service: FakeMailService = Depends(get_fakemail_service),
    email_queue: EmailQueue = Depends(get_email_queue),
    metrics_service: MetricsService = Depends(get_metrics_service),
    config_service: ConfigService = Depends(get_config_service)
):
    processor = EmailProcessor(
        fakemail_service, 
        email_queue, 
        metrics_service, 
        config_service
    )
    return processor

async def process_emails(
    history_id: int,
    fakemail_service: FakeMailService,
    email_processor: EmailProcessor
):
    """
    Background task to process emails from FakeMail
    """
    start_time = time.time()
    logger.info(f"Starting email processing for history_id {history_id}")
    
    try:
        # Get the current history_id from our records
        current_history_id = fakemail_service.get_current_history_id()
        
        # Get all email IDs since our last recorded history_id
        email_ids = await fakemail_service.get_email_ids_since(current_history_id)
        
        if not email_ids:
            logger.info(f"No new emails to process since history_id {current_history_id}")
            return
            
        logger.info(f"Processing {len(email_ids)} new emails")
        
        # Start the processor if not already running
        if not email_processor.running:
            await email_processor.start()
        
        # Process the emails
        batch_id = await email_processor.process_emails(history_id, email_ids)
        
        processing_time = time.time() - start_time
        logger.info(f"Email processing queued in {processing_time:.2f} seconds, batch_id: {batch_id}")
        
    except Exception as e:
        logger.error(f"Error in email processing task: {str(e)}")

"""
The main webhook endpoint has been moved to webhook_routes.py for better organization.
This ensures that the primary endpoints required for the task are clearly visible in the FastAPI docs.
"""

"""
The get results endpoint has been moved to webhook_routes.py for better organization.
This ensures that the primary endpoints required for the task are clearly visible in the FastAPI docs.
"""

@router.post("/setup")
async def setup_processor(
    request: Request,
    fakemail_service: FakeMailService = Depends(get_fakemail_service),
    email_processor: EmailProcessor = Depends(get_email_processor)
):
    """
    Set up the email processor by:
    1. Calling the /watch endpoint to get the initial history_id
    2. Subscribing to webhook notifications
    """
    logger.info("Setting up email processor")
    
    try:
        # Parse the request body
        data = await request.json()
        webhook_url = data.get("webhook_url")
        
        if not webhook_url:
            logger.error("Missing webhook_url in request body")
            return {"status": "error", "message": "Missing webhook_url in request body"}
        
        # Get the initial history_id
        history_id = await fakemail_service.watch()
        logger.info(f"Initial history_id: {history_id}")
        
        # Subscribe to webhook notifications
        await fakemail_service.subscribe(webhook_url)
        logger.info(f"Subscribed to webhook notifications at {webhook_url}")
        
        # Start the processor
        if not email_processor.running:
            await email_processor.start()
            logger.info("Email processor started")
        
        return {
            "status": "ok",
            "message": "Email processor set up successfully",
            "initial_history_id": history_id
        }
    except Exception as e:
        logger.error(f"Error setting up processor: {str(e)}")
        return {"status": "error", "message": str(e)}

@router.get("/stats", response_model=ProcessingStats)
async def get_stats(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Get email processing statistics
    """
    stats = metrics_service.get_stats()
    return stats

@router.get("/health")
async def health_check(
    redis_client: redis.Redis = Depends(get_redis_client),
    fakemail_service: FakeMailService = Depends(get_fakemail_service),
    metrics_service: MetricsService = Depends(get_metrics_service),
    email_processor: EmailProcessor = Depends(get_email_processor)
):
    """
    Health check endpoint that verifies system status
    """
    # Check if Redis is available
    try:
        redis_status = redis_client.ping()
    except Exception:
        redis_status = False
        
    # Check processor status
    processor_status = email_processor.running
        
    # Get basic metrics
    stats = metrics_service.get_stats()
    
    return {
        "status": "healthy" if redis_status and processor_status else "unhealthy",
        "components": {
            "redis": "connected" if redis_status else "disconnected",
            "processor": "running" if processor_status else "stopped"
        },
        "metrics": {
            "total_emails_processed": stats.total_emails_processed,
            "success_rate": stats.success_rate,
            "avg_processing_time_ms": stats.avg_processing_time_ms,
            "uptime_seconds": int(time.time() - STARTUP_TIME)
        }
    }

@router.get("/batches/{batch_id}", response_model=ProcessingBatch)
async def get_batch(
    batch_id: str,
    fakemail_service: FakeMailService = Depends(get_fakemail_service)
):
    """
    Get information about a specific processing batch
    """
    batch = fakemail_service.get_processing_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
    return batch

@router.post("/process")
async def manual_process(
    from_history_id: int = Query(..., description="Process emails from this history_id"),
    to_history_id: Optional[int] = Query(None, description="Process emails up to this history_id (optional)"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    run_async: bool = Query(True, description="Run in background (true) or wait for completion (false)"),
    fakemail_service: FakeMailService = Depends(get_fakemail_service),
    email_processor: EmailProcessor = Depends(get_email_processor)
):
    """
    Manually trigger email processing from a specific history_id
    """
    logger.info(f"Manually processing emails from history_id {from_history_id}")
    
    # If to_history_id is provided, use that; otherwise use from_history_id
    history_id = to_history_id or from_history_id
    
    # Start the processor if not already running
    if not email_processor.running:
        await email_processor.start()
    
    # Schedule the email processing as a background task if requested
    if run_async:
        background_tasks.add_task(
            process_emails,
            history_id,
            fakemail_service,
            email_processor
        )
        return {"status": "ok", "message": "Manual email processing scheduled"}
    else:
        # Process immediately (synchronously)
        await process_emails(history_id, fakemail_service, email_processor)
        return {"status": "ok", "message": "Manual email processing completed"}

@router.get("/workers", response_model=Dict[str, Any])
async def get_workers(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Get information about worker processes
    """
    return metrics_service.get_worker_metrics()

@router.get("/metrics/recent", response_model=Dict[str, Any])
async def get_recent_metrics(
    hours: int = Query(24, description="Number of hours to include in the metrics"),
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Get detailed metrics for the past hours
    """
    return metrics_service.get_recent_metrics(hours)

@router.get("/config", response_model=SystemConfig)
async def get_system_config(
    config_service: ConfigService = Depends(get_config_service)
):
    """
    Get current system configuration
    """
    return config_service.get_system_config()

@router.patch("/config", response_model=SystemConfig)
async def update_system_config(
    request: Request,
    config_service: ConfigService = Depends(get_config_service)
):
    """
    Update system configuration
    """
    updates = await request.json()
    return config_service.update_system_config(updates)

@router.get("/rules", response_model=Dict[str, ProcessingRule])
async def get_processing_rules(
    active_only: bool = Query(False, description="Only return active rules"),
    config_service: ConfigService = Depends(get_config_service)
):
    """
    Get all processing rules
    """
    if active_only:
        return config_service.get_active_processing_rules()
    else:
        return config_service.get_all_processing_rules()

@router.post("/rules", response_model=Dict[str, Any])
async def add_processing_rule(
    rule: ProcessingRule,
    config_service: ConfigService = Depends(get_config_service)
):
    """
    Add a new processing rule
    """
    rule_id = config_service.add_processing_rule(rule)
    return {"status": "ok", "rule_id": rule_id}

@router.get("/rules/{rule_id}", response_model=ProcessingRule)
async def get_processing_rule(
    rule_id: str,
    config_service: ConfigService = Depends(get_config_service)
):
    """
    Get a specific processing rule by ID
    """
    rule = config_service.get_processing_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return rule

@router.patch("/rules/{rule_id}", response_model=ProcessingRule)
async def update_processing_rule(
    rule_id: str,
    request: Request,
    config_service: ConfigService = Depends(get_config_service)
):
    """
    Update a processing rule
    """
    updates = await request.json()
    rule = config_service.update_processing_rule(rule_id, updates)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return rule

@router.delete("/rules/{rule_id}", response_model=Dict[str, Any])
async def delete_processing_rule(
    rule_id: str,
    config_service: ConfigService = Depends(get_config_service)
):
    """
    Delete a processing rule
    """
    success = config_service.delete_processing_rule(rule_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return {"status": "ok", "message": f"Rule {rule_id} deleted"}