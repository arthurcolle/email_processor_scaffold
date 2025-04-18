from fastapi import APIRouter, Depends, Query, HTTPException, Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from app.services.database_service import DatabaseService
from app.models.database import ExtendedClassificationType
from fastapi_pagination import Page, paginate
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

def get_database_service():
    """Dependency for database service"""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./emails.db")
    return DatabaseService(db_url)

@router.get("/classifications", response_model=Dict[str, int])
async def get_classification_counts(
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get counts of emails by classification type
    
    Returns a dictionary mapping classification types to counts
    """
    return db_service.get_classification_counts()

@router.get("/classifications/{classification_type}", response_model=List[Dict[str, Any]])
async def get_emails_by_classification(
    classification_type: str = Path(..., description="Classification type to filter by"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of emails to return"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get emails with a specific classification
    
    Returns a list of emails that have been classified as the specified type
    """
    # Validate classification type
    try:
        ExtendedClassificationType(classification_type)
    except ValueError:
        valid_types = [c.value for c in ExtendedClassificationType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid classification type. Valid types are: {', '.join(valid_types)}"
        )
        
    return db_service.get_emails_by_classification(classification_type, limit)

@router.get("/top-senders", response_model=List[Dict[str, Any]])
async def get_top_senders(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of senders to return"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get top email senders by count
    
    Returns a list of top senders with their email counts
    """
    return db_service.get_top_senders(limit)

@router.get("/email-volume", response_model=List[Dict[str, Any]])
async def get_email_volume_by_date(
    days: int = Query(30, ge=1, le=365, description="Number of days to include in the analysis"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get email volume by date
    
    Returns a list of email counts by date for the specified number of days
    """
    return db_service.get_email_volume_by_date(days)

@router.get("/attachments-stats", response_model=Dict[str, Any])
async def get_attachments_stats(
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get statistics about email attachments
    
    Returns statistics about emails with attachments, total attachments, and file types
    """
    return db_service.get_attachments_stats()

@router.get("/domain-stats", response_model=List[Dict[str, Any]])
async def get_domain_stats(
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get statistics about email domains
    
    Returns counts of emails by sender domain
    """
    return db_service.get_domain_stats()

@router.get("/email-metrics", response_model=Dict[str, Any])
async def get_email_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days to include in the analysis"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get aggregate email metrics
    
    Returns aggregate metrics about emails, such as average word count, etc.
    """
    with db_service.get_db() as db:
        from sqlalchemy import func, desc
        from app.models.database import Email, EmailMetrics
        
        # Time range
        start_date = datetime.now() - timedelta(days=days)
        
        # Query for emails in the time range
        emails_count = db.query(func.count(Email.id)) \
                        .filter(Email.received_at >= start_date) \
                        .scalar() or 0
                        
        # Get metrics
        metrics = db.query(
            func.avg(EmailMetrics.word_count).label('avg_word_count'),
            func.avg(EmailMetrics.character_count).label('avg_character_count'),
            func.avg(EmailMetrics.recipient_count).label('avg_recipient_count'),
            func.avg(EmailMetrics.attachment_count).label('avg_attachment_count'),
            func.sum(EmailMetrics.total_attachment_size).label('total_attachment_size'),
        ) \
        .join(Email, Email.id == EmailMetrics.email_id) \
        .filter(Email.received_at >= start_date) \
        .first()
        
        result = {
            "emails_count": emails_count,
            "days_analyzed": days,
            "emails_per_day": emails_count / days if days > 0 else 0,
            "avg_word_count": float(metrics.avg_word_count or 0),
            "avg_character_count": float(metrics.avg_character_count or 0),
            "avg_recipient_count": float(metrics.avg_recipient_count or 0),
            "avg_attachment_count": float(metrics.avg_attachment_count or 0),
            "total_attachment_size": int(metrics.total_attachment_size or 0),
        }
        
        return result