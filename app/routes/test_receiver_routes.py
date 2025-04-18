from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import redis
import httpx
import logging
import os
import json
from app.services.fakemail_service import FakeMailService
from app.models.fakemail import EmailPayload

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["test_receiver"])

# Redis client dependency
def get_redis_client():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    return redis.Redis(host=redis_host, port=redis_port, db=0)

# FakeMail service dependency
def get_fakemail_service(redis_client: redis.Redis = Depends(get_redis_client)):
    fakemail_api_url = os.getenv("FAKEMAIL_API_URL", "http://fakemail-api:8000")
    return FakeMailService(redis_client, fakemail_api_url)

@router.post("/test-webhook", response_model=Dict[str, Any])
async def test_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    fakemail_service: FakeMailService = Depends(get_fakemail_service)
):
    """
    Receiver webhook endpoint that simulates a FakeMail webhook.
    This webhook can be called by another service to test the email processing functionality.
    """
    logger.info("Received test webhook")
    
    try:
        # Parse the webhook payload
        data = await request.json()
        
        # Extract the history_id
        history_id = data.get("history_id")
        if not history_id:
            logger.error("Missing history_id in webhook payload")
            return {"status": "error", "message": "Missing history_id in webhook payload"}
        
        logger.info(f"Test webhook received with history_id: {history_id}")
        
        # Get the current history_id from our records
        current_history_id = fakemail_service.get_current_history_id()
        
        # Get all email IDs since our last recorded history_id
        email_ids = await fakemail_service.get_email_ids_since(current_history_id)
        
        if not email_ids:
            logger.info(f"No new emails to process since history_id {current_history_id}")
            return {"status": "ok", "message": "No new emails to process", "email_count": 0}
        
        logger.info(f"Processing {len(email_ids)} new emails")
        
        # Update current history_id
        fakemail_service.update_current_history_id(history_id)
        
        # Get and process emails
        results = []
        for email_id in email_ids:
            try:
                # Get the email from FakeMail API
                email = await fakemail_service.get_email(email_id)
                
                # Prepare the payload for classification
                classification_payload = {
                    "subject": email.subject,
                    "body": email.body
                }
                
                # Classify the email
                classification, confidence = await fakemail_service.classify_email(classification_payload)
                
                # Store the result
                fakemail_service.store_processed_email(
                    email_id, 
                    classification, 
                    {
                        "subject": email.subject,
                        "body": email.body
                    },
                    confidence=confidence
                )
                
                results.append({
                    "email_id": email_id,
                    "classification": classification,
                    "confidence": confidence
                })
                
                logger.info(f"Email {email_id} processed successfully: {classification}")
                
            except Exception as e:
                logger.error(f"Error processing email {email_id}: {str(e)}")
                results.append({
                    "email_id": email_id,
                    "error": str(e)
                })
        
        return {
            "status": "ok", 
            "message": f"Processed {len(email_ids)} emails", 
            "email_count": len(email_ids),
            "results": results
        }
            
    except Exception as e:
        logger.error(f"Error processing test webhook: {str(e)}")
        return {"status": "error", "message": str(e)}

@router.post("/send-test-email")
async def send_test_email(
    email: EmailPayload,
    fakemail_service: FakeMailService = Depends(get_fakemail_service)
):
    """
    Send a test email using the FakeMail API
    """
    try:
        email_id = await fakemail_service.send_email(email.subject, email.body)
        return {
            "status": "success",
            "message": "Test email sent successfully",
            "email_id": email_id
        }
    except Exception as e:
        logger.error(f"Error sending test email: {str(e)}")
        return {"status": "error", "message": str(e)}

@router.post("/trigger-webhook/{history_id}")
async def trigger_webhook(
    history_id: int,
    request: Request,
    fakemail_service: FakeMailService = Depends(get_fakemail_service)
):
    """
    Manually trigger a webhook to our own endpoint
    """
    try:
        # Get the server's base URL from the request
        base_url = str(request.base_url).rstrip('/')
        
        # Get the configured webhook URL from Redis
        webhook_url = fakemail_service.redis_client.get("fakemail:webhook_url")
        if webhook_url:
            webhook_url = webhook_url.decode('utf-8')
        else:
            # If not configured, use our own webhook endpoint
            webhook_url = f"{base_url}/webhook"
        
        logger.info(f"Triggering webhook to {webhook_url} with history_id: {history_id}")
        
        # Prepare payload
        payload = {"history_id": history_id}
        
        # Send webhook request
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            
            return {
                "status": "success",
                "message": f"Webhook triggered successfully to {webhook_url}",
                "response": response.json()
            }
            
    except Exception as e:
        logger.error(f"Error triggering webhook: {str(e)}")
        return {"status": "error", "message": str(e)}

@router.get("/results/{email_id}")
async def get_result(
    email_id: str,
    fakemail_service: FakeMailService = Depends(get_fakemail_service)
):
    """
    Get the processing result for a specific email
    """
    result = fakemail_service.get_processed_email(email_id)
    if not result:
        raise HTTPException(status_code=404, detail="No result found for this email ID")
    
    return {
        "email_id": result.email_id,
        "classification": result.classification,
        "processed_at": result.processed_at,
        "confidence": result.confidence,
        "subject": result.subject,
        "body": result.body
    }

@router.post("/setup")
async def setup_receiver(
    request: Request,
    fakemail_service: FakeMailService = Depends(get_fakemail_service)
):
    """
    Set up the receiver by:
    1. Calling the FakeMail /watch endpoint to get the initial history_id
    2. Subscribing our test-webhook to receive notifications
    """
    try:
        data = await request.json()
        webhook_url = data.get("webhook_url")
        
        if not webhook_url:
            # Use the base URL to construct our test webhook URL
            base_url = str(request.base_url).rstrip('/')
            webhook_url = f"{base_url}/test-webhook"
        
        # Call watch to get initial history_id
        history_id = await fakemail_service.watch()
        
        # Subscribe to FakeMail webhooks
        await fakemail_service.subscribe(webhook_url)
        
        return {
            "status": "success",
            "message": "Test receiver set up successfully",
            "initial_history_id": history_id,
            "webhook_url": webhook_url
        }
    
    except Exception as e:
        logger.error(f"Error setting up test receiver: {str(e)}")
        return {"status": "error", "message": str(e)}