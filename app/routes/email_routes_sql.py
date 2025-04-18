from fastapi import APIRouter, Request, Depends, HTTPException, Query, Path, Form, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import logging
from app.services.database_service import DatabaseService
from pydantic import BaseModel, EmailStr, Field
from fastapi_pagination import Page, add_pagination, paginate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emails", tags=["emails"])

# Models for requests and responses
class EmailAddressModel(BaseModel):
    email: EmailStr
    name: Optional[str] = None

class AttachmentModel(BaseModel):
    filename: str
    content_type: str
    size: int
    content_id: Optional[str] = None
    content: Optional[str] = None  # Base64 encoded content

class EmailCreateModel(BaseModel):
    thread_id: Optional[str] = None
    sender: EmailAddressModel
    recipients: List[EmailAddressModel]
    cc: List[EmailAddressModel] = []
    bcc: List[EmailAddressModel] = []
    reply_to: Optional[EmailAddressModel] = None
    in_reply_to: Optional[str] = None
    subject: str
    body: str
    html_body: Optional[str] = None
    labels: List[str] = []

class EmailFilterModel(BaseModel):
    sender: Optional[str] = None
    recipient: Optional[str] = None
    subject: Optional[str] = None
    content: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_attachments: Optional[bool] = None
    label: Optional[str] = None
    folder: Optional[str] = None
    is_read: Optional[bool] = None
    is_starred: Optional[bool] = None
    is_important: Optional[bool] = None
    thread_id: Optional[str] = None
    classification: Optional[str] = None
    min_confidence: Optional[float] = None
    domain: Optional[str] = None

class EmailResponseModel(BaseModel):
    id: str
    thread_id: Optional[str] = None
    sender: Dict[str, Any]
    recipients: List[Dict[str, Any]]
    cc: List[Dict[str, Any]] = []
    bcc: List[Dict[str, Any]] = []
    subject: str
    body: str
    html_body: Optional[str] = None
    received_at: str
    read: bool = False
    starred: bool = False
    important: bool = False
    spam: bool = False
    draft: bool = False
    sent: bool = False
    folder: str = "inbox"
    labels: List[str] = []
    attachments: List[Dict[str, Any]] = []
    classification: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str

# Database service dependency
def get_database_service():
    """Dependency for database service"""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./emails.db")
    return DatabaseService(db_url)

@router.post("/", response_model=str, status_code=201)
async def create_email(
    email: EmailCreateModel,
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Create a new email
    
    Creates a new email in the system and returns its ID
    """
    try:
        # Convert to dict
        email_dict = email.model_dump()
        
        # Create email
        email_id = db_service.create_email(email_dict)
        
        return email_id
    except Exception as e:
        logger.error(f"Error creating email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating email: {str(e)}")

@router.get("/", response_model=Page[EmailResponseModel])
async def list_emails(
    sender: Optional[str] = None,
    recipient: Optional[str] = None,
    subject: Optional[str] = None,
    content: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    has_attachments: Optional[bool] = None,
    label: Optional[str] = None,
    folder: Optional[str] = None,
    is_read: Optional[bool] = None,
    is_starred: Optional[bool] = None,
    is_important: Optional[bool] = None,
    thread_id: Optional[str] = None,
    classification: Optional[str] = None,
    min_confidence: Optional[float] = None,
    domain: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("received_at", description="Field to sort by"),
    sort_dir: str = Query("desc", description="Sort direction (asc or desc)"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    List emails with advanced filtering and sorting
    
    Returns a paginated list of emails that match the given filters
    """
    # Build filter params
    filters = {}
    if sender:
        filters["sender"] = sender
    if recipient:
        filters["recipient"] = recipient
    if subject:
        filters["subject"] = subject
    if content:
        filters["content"] = content
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    if has_attachments is not None:
        filters["has_attachments"] = has_attachments
    if label:
        filters["label"] = label
    if folder:
        filters["folder"] = folder
    if is_read is not None:
        filters["is_read"] = is_read
    if is_starred is not None:
        filters["is_starred"] = is_starred
    if is_important is not None:
        filters["is_important"] = is_important
    if thread_id:
        filters["thread_id"] = thread_id
    if classification:
        filters["classification"] = classification
    if min_confidence is not None:
        filters["min_confidence"] = min_confidence
    if domain:
        filters["domain"] = domain
    
    # Get emails
    emails, total_count = db_service.list_emails(filters, offset, limit, sort_by, sort_dir)
    
    # Create paginated response
    return paginate(emails, total_count, limit, offset)

@router.get("/{email_id}", response_model=EmailResponseModel)
async def get_email(
    email_id: str = Path(..., description="ID of the email to retrieve"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get an email by ID
    
    Returns detailed information about a specific email
    """
    email = db_service.get_email(email_id)
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
        
    return email

@router.delete("/{email_id}", response_model=bool)
async def delete_email(
    email_id: str = Path(..., description="ID of the email to delete"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Delete an email
    
    Deletes an email by ID
    """
    success = db_service.delete_email(email_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Email not found")
        
    return success

@router.post("/{email_id}/read", response_model=bool)
async def mark_as_read(
    email_id: str = Path(..., description="ID of the email to mark as read"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Mark an email as read
    
    Marks an email as read
    """
    success = db_service.mark_as_read(email_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Email not found")
        
    return success

@router.post("/{email_id}/labels/{label}", response_model=bool)
async def add_label(
    email_id: str = Path(..., description="ID of the email to label"),
    label: str = Path(..., description="Label to add"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Add a label to an email
    
    Adds a label to an email
    """
    success = db_service.add_label(email_id, label)
    
    if not success:
        raise HTTPException(status_code=404, detail="Email not found")
        
    return success

@router.delete("/{email_id}/labels/{label}", response_model=bool)
async def remove_label(
    email_id: str = Path(..., description="ID of the email to modify"),
    label: str = Path(..., description="Label to remove"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Remove a label from an email
    
    Removes a label from an email
    """
    success = db_service.remove_label(email_id, label)
    
    if not success:
        raise HTTPException(status_code=404, detail="Email not found or label doesn't exist")
        
    return success

@router.post("/{email_id}/classify", response_model=bool)
async def classify_email(
    email_id: str = Path(..., description="ID of the email to classify"),
    classification_type: str = Form(..., description="Classification type"),
    confidence: float = Form(1.0, description="Classification confidence"),
    model_version: Optional[str] = Form(None, description="Model version"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Classify an email
    
    Assigns a classification to an email
    """
    success = db_service.classify_email(email_id, classification_type, confidence, model_version)
    
    if not success:
        raise HTTPException(status_code=404, detail="Email not found or classification error")
        
    return success

@router.post("/search", response_model=Page[EmailResponseModel])
async def search_emails(
    filters: EmailFilterModel,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("received_at", description="Field to sort by"),
    sort_dir: str = Query("desc", description="Sort direction (asc or desc)"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Advanced email search
    
    Searches for emails using advanced filtering parameters
    """
    # Convert filter model to dict
    filter_dict = filters.model_dump(exclude_none=True)
    
    # Get emails
    emails, total_count = db_service.list_emails(filter_dict, offset, limit, sort_by, sort_dir)
    
    # Create paginated response
    return paginate(emails, total_count, limit, offset)

# Add pagination to router
add_pagination(router)