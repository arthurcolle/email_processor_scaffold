from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import re

class Attachment(BaseModel):
    filename: str
    content_type: str
    size: int
    content_id: Optional[str] = None
    content: Optional[str] = None  # Base64 encoded content

class EmailAddress(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    
    def __str__(self):
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email

class Thread(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    participants: List[str] = []
    last_updated: datetime = Field(default_factory=datetime.now)
    email_count: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Email(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: Optional[str] = None
    sender: EmailAddress
    recipients: List[EmailAddress]
    cc: List[EmailAddress] = []
    bcc: List[EmailAddress] = []
    reply_to: Optional[EmailAddress] = None
    in_reply_to: Optional[str] = None  # ID of the email this is replying to
    references: List[str] = []  # Chain of message IDs this email references
    subject: str
    body: str
    html_body: Optional[str] = None
    attachments: List[Attachment] = []
    headers: Dict[str, str] = {}
    received_at: datetime = Field(default_factory=datetime.now)
    read: bool = False
    starred: bool = False
    important: bool = False
    spam: bool = False
    draft: bool = False
    sent: bool = False
    labels: List[str] = []
    folder: str = "INBOX"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('subject')
    def normalize_subject(cls, v):
        """Normalize subject by removing Re:, Fwd: prefixes for threading"""
        return re.sub(r'^(?:Re|Fwd):\s*', '', v, flags=re.IGNORECASE)
    
    def generate_preview(self, max_length: int = 100) -> str:
        """Generate a preview of the email body"""
        text = self.body or ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

class EmailFilter(BaseModel):
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

class SearchQuery(BaseModel):
    query: str
    fields: List[str] = ["subject", "body", "sender", "recipients"]

class Rule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    conditions: List[Dict[str, Any]]  # e.g. [{"field": "sender", "operator": "contains", "value": "example.com"}]
    actions: List[Dict[str, Any]]  # e.g. [{"type": "move", "folder": "work"}, {"type": "label", "label": "important"}]
    active: bool = True
    
    def applies_to(self, email: Email) -> bool:
        """Check if this rule applies to the given email"""
        for condition in self.conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if not all([field, operator, value]):
                continue
                
            field_value = getattr(email, field, None)
            if field_value is None:
                continue
                
            # Handle list fields like recipients, labels
            if isinstance(field_value, list):
                if operator == "contains":
                    if not any(value in str(item) for item in field_value):
                        return False
                continue
                
            # Handle string fields
            if operator == "equals" and str(field_value) != str(value):
                return False
            elif operator == "contains" and str(value) not in str(field_value):
                return False
            elif operator == "starts_with" and not str(field_value).startswith(str(value)):
                return False
            elif operator == "ends_with" and not str(field_value).endswith(str(value)):
                return False
                
        return True