import os
import logging
from sqlalchemy import create_engine, text, func, desc, and_, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import json

from app.models.database import Base, Email, EmailAddress, Attachment, Label, Classification, EmailMetrics, ExtendedClassificationType, EmailFolder

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, db_url: Optional[str] = None):
        """Initialize the database service"""
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./emails.db")
            
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
        
    @contextmanager
    def get_db(self):
        """Get a database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
            
    def get_or_create_email_address(self, db: Session, email: str, name: Optional[str] = None) -> EmailAddress:
        """Get or create an email address record"""
        email_address = db.query(EmailAddress).filter(EmailAddress.email == email).first()
        
        if email_address:
            return email_address
            
        # Extract domain from email
        domain = email.split('@')[-1] if '@' in email else None
        
        # Create new email address
        email_address = EmailAddress(email=email, name=name, domain=domain)
        db.add(email_address)
        db.flush()
        
        return email_address
        
    def get_or_create_label(self, db: Session, name: str, color: Optional[str] = None) -> Label:
        """Get or create a label"""
        label = db.query(Label).filter(Label.name == name).first()
        
        if label:
            return label
            
        # Create new label
        label = Label(name=name, color=color or "#000000")
        db.add(label)
        db.flush()
        
        return label
        
    def create_email(self, email_data: Dict[str, Any]) -> str:
        """Create a new email record in the database"""
        with self.get_db() as db:
            try:
                # Create or get sender email address
                sender_data = email_data.get("sender", {})
                if isinstance(sender_data, dict):
                    sender_email = sender_data.get("email")
                    sender_name = sender_data.get("name")
                else:
                    sender_email = sender_data
                    sender_name = None
                    
                sender = self.get_or_create_email_address(db, sender_email, sender_name)
                
                # Create a new email record
                email = Email(
                    thread_id=email_data.get("thread_id"),
                    sender_id=sender.id,
                    subject=email_data.get("subject", ""),
                    body=email_data.get("body", ""),
                    html_body=email_data.get("html_body"),
                    received_at=email_data.get("received_at", datetime.now()),
                    read=email_data.get("read", False),
                    starred=email_data.get("starred", False),
                    important=email_data.get("important", False),
                    spam=email_data.get("spam", False),
                    draft=email_data.get("draft", False),
                    sent=email_data.get("sent", False),
                    folder=email_data.get("folder", EmailFolder.INBOX)
                )
                
                db.add(email)
                db.flush()
                
                # Handle recipients - add one by one to avoid duplicates
                seen_recipients = set()
                for recipient_data in email_data.get("recipients", []):
                    if isinstance(recipient_data, dict):
                        recipient_email = recipient_data.get("email")
                        recipient_name = recipient_data.get("name")
                    else:
                        recipient_email = recipient_data
                        recipient_name = None
                    
                    # Skip if we've seen this recipient already
                    if recipient_email in seen_recipients:
                        continue
                    seen_recipients.add(recipient_email)
                        
                    recipient = self.get_or_create_email_address(db, recipient_email, recipient_name)
                    email.recipients.append(recipient)
                
                # Handle CC recipients - add one by one to avoid duplicates
                seen_cc = set()
                for cc_data in email_data.get("cc", []):
                    if isinstance(cc_data, dict):
                        cc_email = cc_data.get("email")
                        cc_name = cc_data.get("name")
                    else:
                        cc_email = cc_data
                        cc_name = None
                    
                    # Skip if we've seen this cc recipient already
                    if cc_email in seen_cc:
                        continue
                    seen_cc.add(cc_email)
                        
                    cc = self.get_or_create_email_address(db, cc_email, cc_name)
                    email.cc.append(cc)
                    
                # Handle BCC recipients - add one by one to avoid duplicates
                seen_bcc = set()
                for bcc_data in email_data.get("bcc", []):
                    if isinstance(bcc_data, dict):
                        bcc_email = bcc_data.get("email")
                        bcc_name = bcc_data.get("name")
                    else:
                        bcc_email = bcc_data
                        bcc_name = None
                    
                    # Skip if we've seen this bcc recipient already
                    if bcc_email in seen_bcc:
                        continue
                    seen_bcc.add(bcc_email)
                        
                    bcc = self.get_or_create_email_address(db, bcc_email, bcc_name)
                    email.bcc.append(bcc)
                    
                # Handle attachments
                for attachment_data in email_data.get("attachments", []):
                    attachment = Attachment(
                        email_id=email.id,
                        filename=attachment_data.get("filename", ""),
                        content_type=attachment_data.get("content_type", "application/octet-stream"),
                        size=attachment_data.get("size", 0),
                        content_id=attachment_data.get("content_id"),
                        content=attachment_data.get("content")
                    )
                    db.add(attachment)
                    
                # Handle labels - add one by one to avoid duplicates
                seen_labels = set()
                for label_name in email_data.get("labels", []):
                    # Skip if we've seen this label already
                    if label_name in seen_labels:
                        continue
                    seen_labels.add(label_name)
                    
                    label = self.get_or_create_label(db, label_name)
                    email.labels.append(label)
                    
                # Create email metrics
                metrics = EmailMetrics(
                    email_id=email.id,
                    word_count=len(email.body.split()) if email.body else 0,
                    character_count=len(email.body) if email.body else 0,
                    recipient_count=len(email_data.get("recipients", [])),
                    cc_count=len(email_data.get("cc", [])),
                    bcc_count=len(email_data.get("bcc", [])),
                    attachment_count=len(email_data.get("attachments", [])),
                    total_attachment_size=sum(att.get("size", 0) for att in email_data.get("attachments", [])),
                    links_count=email.body.count("http") if email.body else 0,
                    images_count=email.html_body.count("<img") if email.html_body else 0,
                )
                db.add(metrics)
                
                db.commit()
                return email.id
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Error creating email: {str(e)}")
                raise
                
    def get_email(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Get an email by ID"""
        with self.get_db() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            
            if not email:
                return None
                
            return self._serialize_email(email)
    
    def list_emails(self, 
                  filters: Optional[Dict[str, Any]] = None, 
                  offset: int = 0, 
                  limit: int = 50,
                  sort_by: str = "received_at",
                  sort_dir: str = "desc") -> Tuple[List[Dict[str, Any]], int]:
        """
        List emails with filtering, sorting and pagination
        Returns tuple of (emails, total_count)
        """
        with self.get_db() as db:
            # Base query
            query = db.query(Email).filter(Email.deleted == False)
            
            # Apply filters
            if filters:
                query = self._apply_filters(query, filters)
                
            # Get total count
            total_count = query.count()
            
            # Apply sorting
            if sort_by:
                column = getattr(Email, sort_by, Email.received_at)
                if sort_dir.lower() == "desc":
                    query = query.order_by(desc(column))
                else:
                    query = query.order_by(column)
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Execute query and serialize results
            emails = [self._serialize_email(email) for email in query.all()]
            
            return emails, total_count
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to query"""
        for key, value in filters.items():
            if value is None:
                continue
                
            if key == "sender":
                # Join with sender
                query = query.join(Email.sender_obj)
                query = query.filter(EmailAddress.email.ilike(f"%{value}%"))
            elif key == "recipient":
                # Filter by recipient (needs to join many-to-many table)
                query = query.join(Email.recipients)
                query = query.filter(EmailAddress.email.ilike(f"%{value}%"))
            elif key == "subject":
                query = query.filter(Email.subject.ilike(f"%{value}%"))
            elif key == "content" or key == "body":
                query = query.filter(or_(
                    Email.body.ilike(f"%{value}%"),
                    Email.html_body.ilike(f"%{value}%")
                ))
            elif key == "date_from":
                query = query.filter(Email.received_at >= value)
            elif key == "date_to":
                query = query.filter(Email.received_at <= value)
            elif key == "has_attachments":
                if value:
                    query = query.join(Email.attachments)
                    query = query.filter(Attachment.id.isnot(None))
                    query = query.group_by(Email.id)
                else:
                    # No attachments
                    attachment_subquery = db.query(Attachment.email_id).filter(Attachment.email_id == Email.id).exists()
                    query = query.filter(~attachment_subquery)
            elif key == "label":
                # Filter by label (needs to join many-to-many table)
                query = query.join(Email.labels)
                query = query.filter(Label.name == value)
            elif key == "folder":
                query = query.filter(Email.folder == value)
            elif key == "is_read":
                query = query.filter(Email.read == value)
            elif key == "is_starred":
                query = query.filter(Email.starred == value)
            elif key == "is_important":
                query = query.filter(Email.important == value)
            elif key == "thread_id":
                query = query.filter(Email.thread_id == value)
            elif key == "classification":
                query = query.join(Email.classification)
                query = query.filter(Classification.type == value)
            elif key == "min_confidence":
                query = query.join(Email.classification)
                query = query.filter(Classification.confidence >= value)
            elif key == "domain":
                query = query.join(Email.sender_obj)
                query = query.filter(EmailAddress.domain == value)
        
        return query
        
    def _serialize_email(self, email: Email) -> Dict[str, Any]:
        """Convert a database Email object to a dictionary"""
        result = {
            "id": email.id,
            "thread_id": email.thread_id,
            "subject": email.subject,
            "body": email.body,
            "html_body": email.html_body,
            "received_at": email.received_at.isoformat() if email.received_at else None,
            "read": email.read,
            "starred": email.starred,
            "important": email.important,
            "spam": email.spam,
            "draft": email.draft,
            "sent": email.sent,
            "folder": email.folder.value if email.folder else "INBOX",
            "created_at": email.created_at.isoformat() if email.created_at else None,
            "updated_at": email.updated_at.isoformat() if email.updated_at else None,
        }
        
        # Add sender
        if email.sender_obj:
            result["sender"] = {
                "email": email.sender_obj.email,
                "name": email.sender_obj.name
            }
            
        # Add recipients
        result["recipients"] = [{"email": r.email, "name": r.name} for r in email.recipients]
        result["cc"] = [{"email": r.email, "name": r.name} for r in email.cc]
        result["bcc"] = [{"email": r.email, "name": r.name} for r in email.bcc]
        
        # Add attachments
        result["attachments"] = [
            {
                "filename": a.filename,
                "content_type": a.content_type,
                "size": a.size,
                "content_id": a.content_id
            } for a in email.attachments
        ]
        
        # Add labels
        result["labels"] = [l.name for l in email.labels]
        
        # Add classification if exists
        if email.classification:
            result["classification"] = {
                "type": email.classification.type.value,
                "confidence": email.classification.confidence,
                "model_version": email.classification.model_version,
                "processed_at": email.classification.processed_at.isoformat() if email.classification.processed_at else None
            }
            
        # Add metrics if exists
        if email.metrics:
            result["metrics"] = {
                "word_count": email.metrics.word_count,
                "character_count": email.metrics.character_count,
                "recipient_count": email.metrics.recipient_count,
                "attachment_count": email.metrics.attachment_count,
                "total_attachment_size": email.metrics.total_attachment_size,
                "links_count": email.metrics.links_count,
                "images_count": email.metrics.images_count,
                "sentiment_score": email.metrics.sentiment_score,
                "priority_score": email.metrics.priority_score
            }
            
        return result
    
    def delete_email(self, email_id: str) -> bool:
        """Delete an email by ID (soft delete)"""
        with self.get_db() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            
            if not email:
                return False
                
            # Mark as deleted
            email.deleted = True
            db.commit()
            
            return True
            
    def mark_as_read(self, email_id: str) -> bool:
        """Mark an email as read"""
        with self.get_db() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            
            if not email:
                return False
                
            email.read = True
            db.commit()
            
            return True
            
    def add_label(self, email_id: str, label_name: str) -> bool:
        """Add a label to an email"""
        with self.get_db() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            
            if not email:
                return False
                
            # Get or create label
            label = self.get_or_create_label(db, label_name)
            
            # Check if label already exists on this email
            if label in email.labels:
                return True
                
            # Add label
            email.labels.append(label)
            db.commit()
            
            return True
    
    def remove_label(self, email_id: str, label_name: str) -> bool:
        """Remove a label from an email"""
        with self.get_db() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            
            if not email:
                return False
                
            # Find label
            label = db.query(Label).filter(Label.name == label_name).first()
            
            if not label or label not in email.labels:
                return False
                
            # Remove label
            email.labels.remove(label)
            db.commit()
            
            return True
    
    def get_classification_counts(self) -> Dict[str, int]:
        """Get counts of emails by classification type"""
        with self.get_db() as db:
            result = db.query(Classification.type, func.count(Classification.id)) \
                       .group_by(Classification.type) \
                       .all()
                       
            return {r[0].value: r[1] for r in result}
    
    def get_emails_by_classification(self, classification_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get emails with a specific classification"""
        with self.get_db() as db:
            query = db.query(Email) \
                     .join(Email.classification) \
                     .filter(Classification.type == classification_type) \
                     .order_by(desc(Email.received_at)) \
                     .limit(limit)
                     
            return [self._serialize_email(email) for email in query.all()]
    
    def get_top_senders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top email senders by count"""
        with self.get_db() as db:
            subq = db.query(Email.sender_id, func.count(Email.id).label('email_count')) \
                     .filter(Email.deleted == False) \
                     .group_by(Email.sender_id) \
                     .subquery()
                     
            query = db.query(EmailAddress, subq.c.email_count) \
                      .join(subq, EmailAddress.id == subq.c.sender_id) \
                      .order_by(desc(subq.c.email_count)) \
                      .limit(limit)
                      
            return [{"email": r[0].email, "name": r[0].name, "count": r[1]} for r in query.all()]
    
    def get_email_volume_by_date(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get email volume by date"""
        with self.get_db() as db:
            # SQLite specific date formatting
            date_format = "date(received_at)"
            
            query = db.query(
                text(date_format), 
                func.count(Email.id)
            ) \
            .filter(Email.received_at >= text(f"datetime('now', '-{days} days')")) \
            .group_by(text(date_format)) \
            .order_by(text(date_format))
            
            return [{"date": r[0], "count": r[1]} for r in query.all()]
    
    def classify_email(self, email_id: str, classification_type: str, confidence: float = 1.0, 
                      model_version: Optional[str] = None) -> bool:
        """Classify an email"""
        with self.get_db() as db:
            try:
                email = db.query(Email).filter(Email.id == email_id).first()
                
                if not email:
                    return False
                
                # Check if classification already exists
                if email.classification:
                    # Update existing classification
                    email.classification.type = ExtendedClassificationType(classification_type)
                    email.classification.confidence = confidence
                    email.classification.model_version = model_version
                    email.classification.processed_at = datetime.utcnow()
                else:
                    # Create new classification
                    classification = Classification(
                        email_id=email_id,
                        type=ExtendedClassificationType(classification_type),
                        confidence=confidence,
                        model_version=model_version,
                        processed_at=datetime.utcnow()
                    )
                    db.add(classification)
                
                db.commit()
                return True
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error classifying email: {str(e)}")
                return False
    
    def get_attachments_stats(self) -> Dict[str, Any]:
        """Get statistics about email attachments"""
        with self.get_db() as db:
            # Total emails with attachments
            emails_with_attachments = db.query(func.count(Email.id.distinct())) \
                                       .join(Email.attachments) \
                                       .scalar() or 0
            
            # Total attachments
            total_attachments = db.query(func.count(Attachment.id)).scalar() or 0
            
            # Total attachment size
            total_size = db.query(func.sum(Attachment.size)).scalar() or 0
            
            # Average attachments per email
            avg_attachments = total_attachments / emails_with_attachments if emails_with_attachments > 0 else 0
            
            # Top content types
            content_types = db.query(
                Attachment.content_type,
                func.count(Attachment.id).label('count')
            ) \
            .group_by(Attachment.content_type) \
            .order_by(desc('count')) \
            .limit(10) \
            .all()
            
            return {
                "emails_with_attachments": emails_with_attachments,
                "total_attachments": total_attachments,
                "total_size_bytes": total_size,
                "avg_attachments_per_email": avg_attachments,
                "top_content_types": [{"type": ct, "count": count} for ct, count in content_types]
            }
    
    def get_domain_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about email domains"""
        with self.get_db() as db:
            # Count emails by sender domain
            domain_stats = db.query(
                EmailAddress.domain,
                func.count(Email.id).label('count')
            ) \
            .join(Email, Email.sender_id == EmailAddress.id) \
            .filter(EmailAddress.domain != None) \
            .group_by(EmailAddress.domain) \
            .order_by(desc('count')) \
            .limit(20) \
            .all()
            
            return [{"domain": domain, "count": count} for domain, count in domain_stats]