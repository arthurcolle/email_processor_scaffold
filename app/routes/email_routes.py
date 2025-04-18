from fastapi import APIRouter, Request, Depends, HTTPException, Query
from typing import List, Optional
from app.models.email import Email, EmailFilter
from app.services.email_service import EmailService
import redis
from datetime import datetime

router = APIRouter(tags=["email"])

# Redis client dependency
def get_redis_client():
    return redis.Redis(host="redis", port=6379, db=0)

# Email service dependency
def get_email_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return EmailService(redis_client)

@router.post("/emails", response_model=str)
async def create_email(email: Email, email_service: EmailService = Depends(get_email_service)):
    """Create a new email in the system"""
    email_id = email_service.create_email(email)
    return email_id

@router.get("/emails", response_model=List[Email])
async def list_emails(
    sender: Optional[str] = None,
    recipient: Optional[str] = None,
    subject: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    has_attachments: Optional[bool] = None,
    label: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    email_service: EmailService = Depends(get_email_service)
):
    """List emails with optional filtering and pagination"""
    filter_params = None
    if any([sender, recipient, subject, date_from, date_to, has_attachments is not None, label]):
        filter_params = EmailFilter(
            sender=sender,
            recipient=recipient,
            subject=subject,
            date_from=date_from,
            date_to=date_to,
            has_attachments=has_attachments,
            label=label
        )
    
    emails = email_service.list_emails(filter_params, offset, limit)
    return emails

@router.get("/emails/{email_id}", response_model=Email)
async def get_email(
    email_id: str,
    email_service: EmailService = Depends(get_email_service)
):
    """Get an email by ID"""
    email = email_service.get_email(email_id)
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    return email

@router.delete("/emails/{email_id}", response_model=bool)
async def delete_email(
    email_id: str,
    email_service: EmailService = Depends(get_email_service)
):
    """Delete an email by ID"""
    success = email_service.delete_email(email_id)
    if not success:
        raise HTTPException(status_code=404, detail="Email not found")
    return success

@router.post("/emails/{email_id}/read", response_model=bool)
async def mark_as_read(
    email_id: str,
    email_service: EmailService = Depends(get_email_service)
):
    """Mark an email as read"""
    success = email_service.mark_as_read(email_id)
    if not success:
        raise HTTPException(status_code=404, detail="Email not found")
    return success

@router.post("/emails/{email_id}/labels/{label}", response_model=bool)
async def add_label(
    email_id: str,
    label: str,
    email_service: EmailService = Depends(get_email_service)
):
    """Add a label to an email"""
    success = email_service.add_label(email_id, label)
    if not success:
        raise HTTPException(status_code=404, detail="Email not found")
    return success

@router.delete("/emails/{email_id}/labels/{label}", response_model=bool)
async def remove_label(
    email_id: str,
    label: str,
    email_service: EmailService = Depends(get_email_service)
):
    """Remove a label from an email"""
    success = email_service.remove_label(email_id, label)
    if not success:
        raise HTTPException(status_code=404, detail="Email not found")
    return success

@router.post("/webhook")
async def webhook(request: Request):
    """Webhook endpoint for receiving emails from external sources"""
    data = await request.json()
    redis_client = get_redis_client()
    email_service = EmailService(redis_client)
    
    # Process the webhook data to create an email
    # This is a simplified example - in a real system you'd validate and transform the data
    try:
        email = Email(
            sender=data.get("from"),
            recipients=data.get("to", []),
            cc=data.get("cc", []),
            bcc=data.get("bcc", []),
            subject=data.get("subject", ""),
            body=data.get("body", ""),
            html_body=data.get("html", ""),
            attachments=data.get("attachments", [])
        )
        
        email_id = email_service.create_email(email)
        return {"status": "ok", "email_id": email_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}