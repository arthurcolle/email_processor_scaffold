from fastapi import APIRouter, Request, Depends, HTTPException, Body, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import redis
import json
import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import os
from pathlib import Path

from app.services.fakemail_service import FakeMailService
from app.models.fakemail import FakeMailEmail, ClassificationType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["simulator"])

# Templates setup
templates_dir = Path(__file__).parent.parent / "templates"
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Redis keys
FAKEMAIL_EMAILS_KEY = "fakemail:emails"
FAKEMAIL_HISTORY_ID_KEY = "fakemail:history_id"
FAKEMAIL_WEBHOOK_URLS_KEY = "fakemail:webhook_urls"
FAKEMAIL_SUBSCRIPTIONS_KEY = "fakemail:subscriptions"

# Redis client dependency
def get_redis_client():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    return redis.Redis(host=redis_host, port=redis_port, db=0)

# FakeMail service dependency (for real service integration)
def get_fakemail_service(redis_client: redis.Redis = Depends(get_redis_client)):
    fakemail_url = os.environ.get("FAKEMAIL_API_URL", "http://localhost:8000")
    return FakeMailService(redis_client, fakemail_url)

# Simulator functions
def get_current_history_id(redis_client: redis.Redis) -> int:
    """Get the current history_id from Redis"""
    history_id = redis_client.get(FAKEMAIL_HISTORY_ID_KEY)
    if history_id is None:
        # Initialize the history ID
        redis_client.set(FAKEMAIL_HISTORY_ID_KEY, 1)
        return 1
    return int(history_id)

def increment_history_id(redis_client: redis.Redis) -> int:
    """Increment and return the history_id"""
    return redis_client.incr(FAKEMAIL_HISTORY_ID_KEY)

def store_email(redis_client: redis.Redis, email: Dict[str, Any]) -> str:
    """Store an email in Redis and return its ID"""
    email_id = email.get("id", str(uuid.uuid4()))
    email["id"] = email_id
    email["history_id"] = get_current_history_id(redis_client)
    
    # Store the email
    redis_client.hset(FAKEMAIL_EMAILS_KEY, email_id, json.dumps(email))
    
    # Add to the history
    redis_client.sadd(f"fakemail:history:{email['history_id']}", email_id)
    
    return email_id

def get_email(redis_client: redis.Redis, email_id: str) -> Optional[Dict[str, Any]]:
    """Get an email by ID"""
    email_data = redis_client.hget(FAKEMAIL_EMAILS_KEY, email_id)
    if email_data is None:
        return None
    return json.loads(email_data)

def get_emails_since_history_id(redis_client: redis.Redis, history_id: int) -> List[str]:
    """Get all email IDs with history_id greater than the given value"""
    current_history_id = get_current_history_id(redis_client)
    
    email_ids = []
    for h_id in range(history_id + 1, current_history_id + 1):
        ids = redis_client.smembers(f"fakemail:history:{h_id}")
        email_ids.extend([id.decode() for id in ids])
    
    return email_ids

def notify_webhooks(redis_client: redis.Redis, history_id: int) -> None:
    """Notify all registered webhooks about the new history_id"""
    import httpx
    import asyncio
    
    async def send_notifications():
        logger.info(f"Notifying webhooks of new emails with history_id {history_id}")
        
        # Add a local webhook URL to ensure we always notify our own service
        local_webhook_url = "http://localhost:8005/webhook"
        
        # Get registered webhooks
        urls = redis_client.smembers(FAKEMAIL_WEBHOOK_URLS_KEY)
        if not urls:
            logger.warning("No webhook URLs registered")
            
        # Convert set of bytes to list of strings
        url_list = [url.decode() for url in urls]
        
        # Add local webhook if not already present
        if local_webhook_url not in url_list:
            url_list.append(local_webhook_url)
            
        logger.info(f"Sending webhook notifications to: {url_list}")
            
        async with httpx.AsyncClient() as client:
            tasks = []
            for url in url_list:
                payload = {"history_id": history_id}
                logger.info(f"Sending webhook to {url} with payload: {payload}")
                tasks.append(client.post(url, json=payload))
                
            if tasks:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        logger.error(f"Error notifying webhook {url_list[i]}: {str(response)}")
                    else:
                        logger.info(f"Webhook notification sent to {url_list[i]}: {response.status_code}")
    
    # Run in the background
    asyncio.create_task(send_notifications())

# Routes for FakeMail simulator
@router.get("/simulator", response_class=HTMLResponse)
async def simulator_ui(request: Request):
    """UI for the FakeMail simulator"""
    return templates.TemplateResponse(
        "simulator.html", 
        {"request": request, "title": "FakeMail Simulator"}
    )

@router.post("/simulator/watch")
async def simulator_watch(redis_client: redis.Redis = Depends(get_redis_client)):
    """FakeMail watch endpoint"""
    history_id = get_current_history_id(redis_client)
    return {"history_id": history_id}

@router.post("/simulator/subscribe")
async def simulator_subscribe(
    webhook_url: str = Body(..., embed=True),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """FakeMail subscribe endpoint"""
    redis_client.sadd(FAKEMAIL_WEBHOOK_URLS_KEY, webhook_url)
    return {"status": "ok"}

@router.get("/simulator/emails")
async def simulator_get_emails(
    from_history_id: int = Query(...),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """FakeMail get emails endpoint"""
    email_ids = get_emails_since_history_id(redis_client, from_history_id)
    logger.info(f"Returning {len(email_ids)} emails since history_id {from_history_id}")
    return email_ids

@router.get("/simulator/email/{email_id}")
async def simulator_get_email(
    email_id: str,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """FakeMail get email endpoint"""
    email = get_email(redis_client, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")
    return email

@router.post("/simulator/send_email")
async def simulator_send_email(
    subject: str = Body(...),
    body: str = Body(...),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """FakeMail send email endpoint"""
    # Create an email
    email_id = str(uuid.uuid4())
    email = {
        "id": email_id,
        "subject": subject,
        "body": body,
        "sender": "simulator@fakemail.com",
        "recipient": "user@example.com",
        "timestamp": datetime.now().isoformat()
    }
    
    # Store the email
    store_email(redis_client, email)
    
    # Increment the history ID
    new_history_id = increment_history_id(redis_client)
    
    # Notify webhooks
    notify_webhooks(redis_client, new_history_id)
    
    return {"email_id": email_id, "history_id": new_history_id}

@router.post("/simulator/classify")
async def simulator_classify(
    subject: str = Body(...),
    body: str = Body(...)
):
    """FakeMail classify endpoint"""
    # Simple classifier based on keywords
    text = (subject + " " + body).lower()
    
    classification = None
    confidence = 0.0
    
    # Add some randomness to make it look more realistic
    rand_factor = random.random() * 0.2
    
    if "meet" in text or "meeting" in text or "call" in text or "zoom" in text or "appointment" in text:
        classification = ClassificationType.MEETING.value
        confidence = 0.8 + rand_factor
    elif "sale" in text or "discount" in text or "offer" in text or "promo" in text or "buy" in text:
        classification = ClassificationType.PROMOTION.value
        confidence = 0.75 + rand_factor
    elif "hello" in text or "hi" in text or "introduce" in text or "meet" in text or "introduction" in text:
        classification = ClassificationType.INTRO.value
        confidence = 0.7 + rand_factor
    else:
        # Choose randomly if no keywords matched
        classifications = [c.value for c in ClassificationType]
        classification = random.choice(classifications[:-1])  # Exclude UNKNOWN
        confidence = 0.5 + rand_factor
    
    # Cap confidence at 0.95
    confidence = min(confidence, 0.95)
    
    return {"classification": classification, "confidence": confidence}

@router.get("/simulator/status")
async def simulator_status(redis_client: redis.Redis = Depends(get_redis_client)):
    """Get the status of the FakeMail simulator"""
    history_id = get_current_history_id(redis_client)
    
    # Get webhook URLs
    webhook_urls = redis_client.smembers(FAKEMAIL_WEBHOOK_URLS_KEY)
    webhook_urls = [url.decode() for url in webhook_urls]
    
    # Get email counts
    email_count = redis_client.hlen(FAKEMAIL_EMAILS_KEY)
    
    return {
        "history_id": history_id,
        "webhook_urls": webhook_urls,
        "email_count": email_count,
        "status": "running"
    }

@router.post("/simulator/reset")
async def simulator_reset(redis_client: redis.Redis = Depends(get_redis_client)):
    """Reset the FakeMail simulator"""
    # Delete all simulator keys
    keys = redis_client.keys("fakemail:*")
    if keys:
        redis_client.delete(*keys)
    
    # Reset the history ID
    redis_client.set(FAKEMAIL_HISTORY_ID_KEY, 1)
    
    return {"status": "ok", "message": "Simulator reset successfully"}

@router.get("/simulator/template-emails")
async def get_template_emails():
    """Get a list of template emails for testing"""
    templates = [
        {
            "subject": "Meeting Tomorrow at 2pm",
            "body": "Hi team,\n\nLet's meet tomorrow at 2pm to discuss the project progress. Zoom link will be sent separately.\n\nRegards,\nManager"
        },
        {
            "subject": "Introduction - New Team Member",
            "body": "Hello everyone,\n\nI'd like to introduce Alice, who is joining our team as a senior developer. Please welcome her!\n\nBest,\nHR"
        },
        {
            "subject": "50% Off Summer Sale!",
            "body": "Don't miss our HUGE summer sale! Everything is 50% off this weekend only. Visit our website to see all the amazing deals.\n\nYour Favorite Shop"
        },
        {
            "subject": "Weekly Report - Q2 Results",
            "body": "Team,\n\nAttached is the weekly report for Q2 results. We've exceeded our targets in most areas. Let's discuss in our next meeting.\n\nAnalytics Team"
        },
        {
            "subject": "Password Reset Request",
            "body": "You (or someone else) requested a password reset for your account. Click the link below to reset your password. If you didn't request this, please ignore this email.\n\nIT Support"
        }
    ]
    
    return {"templates": templates}
