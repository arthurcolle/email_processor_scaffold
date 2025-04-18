from sqlalchemy import Column, Integer, String, Text, Boolean, Float, ForeignKey, DateTime, Table, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid
import json

Base = declarative_base()

# Association tables for many-to-many relationships
email_labels = Table(
    'email_labels',
    Base.metadata,
    Column('email_id', String(36), ForeignKey('emails.id'), primary_key=True),
    Column('label_id', Integer, ForeignKey('labels.id'), primary_key=True)
)

email_recipients = Table(
    'email_recipients',
    Base.metadata,
    Column('email_id', String(36), ForeignKey('emails.id'), primary_key=True),
    Column('recipient_id', Integer, ForeignKey('email_addresses.id'), primary_key=True)
)

email_cc = Table(
    'email_cc',
    Base.metadata,
    Column('email_id', String(36), ForeignKey('emails.id'), primary_key=True),
    Column('cc_id', Integer, ForeignKey('email_addresses.id'), primary_key=True)
)

email_bcc = Table(
    'email_bcc',
    Base.metadata,
    Column('email_id', String(36), ForeignKey('emails.id'), primary_key=True),
    Column('bcc_id', Integer, ForeignKey('email_addresses.id'), primary_key=True)
)

class EmailFolder(enum.Enum):
    INBOX = "inbox"
    SENT = "sent"
    DRAFTS = "drafts"
    TRASH = "trash"
    ARCHIVE = "archive"
    SPAM = "spam"
    JUNK = "junk"
    IMPORTANT = "important"

class ExtendedClassificationType(enum.Enum):
    # Original classifications
    INTRO = "intro"
    PROMOTION = "promotion"
    MEETING = "meeting"
    UNKNOWN = "unknown"
    
    # Additional classifications - General
    NEWSLETTER = "newsletter"
    NOTIFICATION = "notification"
    TRANSACTION = "transaction"
    INVITATION = "invitation"
    ALERT = "alert"
    ANNOUNCEMENT = "announcement"
    RECEIPT = "receipt"
    REQUEST = "request"
    SOCIAL = "social"
    UPDATES = "updates"
    SECURITY = "security"
    PERSONAL = "personal"
    BILL = "bill"
    TRAVEL = "travel"
    SHIPPING = "shipping"
    SURVEY = "survey"
    
    # Business - Human Resources
    JOB_APPLICATION = "job_application"
    EMPLOYMENT_OFFER = "employment_offer"
    ONBOARDING = "onboarding"
    PERFORMANCE_REVIEW = "performance_review"
    BENEFITS = "benefits"
    PAYROLL = "payroll"
    TRAINING = "training"
    RECRUITMENT = "recruitment"
    EMPLOYEE_ENGAGEMENT = "employee_engagement"
    EMPLOYEE_OFFBOARDING = "employee_offboarding"
    
    # Business - Finance
    INVOICE = "invoice"
    EXPENSE_REPORT = "expense_report"
    BUDGET = "budget"
    FINANCIAL_STATEMENT = "financial_statement"
    TAX_DOCUMENT = "tax_document"
    PURCHASE_ORDER = "purchase_order"
    PAYMENT_CONFIRMATION = "payment_confirmation"
    CREDIT_MEMO = "credit_memo"
    AUDIT = "audit"
    FUNDING = "funding"
    
    # Business - Sales & Marketing
    LEAD = "lead"
    CUSTOMER_INQUIRY = "customer_inquiry"
    QUOTATION = "quotation"
    SALES_OPPORTUNITY = "sales_opportunity"
    MARKETING_CAMPAIGN = "marketing_campaign"
    CUSTOMER_FEEDBACK = "customer_feedback"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    MARKET_RESEARCH = "market_research"
    PRESS_RELEASE = "press_release"
    BRAND_ASSETS = "brand_assets"
    
    # Business - Legal & Compliance
    CONTRACT = "contract"
    LEGAL_NOTICE = "legal_notice"
    COMPLIANCE = "compliance"
    REGULATORY = "regulatory"
    NDA = "nda"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    LITIGATION = "litigation"
    GDPR = "gdpr"
    DATA_PRIVACY = "data_privacy"
    TERMS_CONDITIONS = "terms_conditions"
    
    # Business - IT & Operations
    IT_SUPPORT = "it_support"
    SYSTEM_ALERT = "system_alert"
    MAINTENANCE_NOTIFICATION = "maintenance_notification"
    SOFTWARE_LICENSE = "software_license"
    DATA_BACKUP = "data_backup"
    INCIDENT_REPORT = "incident_report"
    PROJECT_STATUS = "project_status"
    PRODUCTION_ISSUE = "production_issue"
    VENDOR_COMMUNICATION = "vendor_communication"
    SERVICE_LEVEL_AGREEMENT = "service_level_agreement"

class Label(Base):
    __tablename__ = 'labels'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    color = Column(String(20), default="#000000")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Label(name='{self.name}')>"

class EmailAddress(Base):
    __tablename__ = 'email_addresses'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    domain = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Emails where this is the sender
    sent_emails = relationship("Email", back_populates="sender_obj", foreign_keys="Email.sender_id")
    
    def __repr__(self):
        return f"<EmailAddress(email='{self.email}', name='{self.name}')>"

class Attachment(Base):
    __tablename__ = 'attachments'
    
    id = Column(Integer, primary_key=True)
    email_id = Column(String(36), ForeignKey('emails.id'))
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    size = Column(Integer, nullable=False)
    content_id = Column(String(255), nullable=True)
    content = Column(Text, nullable=True)  # Base64 encoded content
    created_at = Column(DateTime, default=datetime.utcnow)
    
    email = relationship("Email", back_populates="attachments")
    
    def __repr__(self):
        return f"<Attachment(filename='{self.filename}', size={self.size})>"

class Classification(Base):
    __tablename__ = 'classifications'
    
    id = Column(Integer, primary_key=True)
    email_id = Column(String(36), ForeignKey('emails.id'), unique=True)
    type = Column(Enum(ExtendedClassificationType), nullable=False)
    confidence = Column(Float, default=1.0)
    model_version = Column(String(50), nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    classification_metadata = Column(JSON, nullable=True)
    
    email = relationship("Email", back_populates="classification")
    
    def __repr__(self):
        return f"<Classification(type='{self.type}', confidence={self.confidence})>"

class EmailMetrics(Base):
    __tablename__ = 'email_metrics'
    
    id = Column(Integer, primary_key=True)
    email_id = Column(String(36), ForeignKey('emails.id'), unique=True)
    word_count = Column(Integer, default=0)
    character_count = Column(Integer, default=0)
    recipient_count = Column(Integer, default=0)
    cc_count = Column(Integer, default=0)
    bcc_count = Column(Integer, default=0)
    attachment_count = Column(Integer, default=0)
    total_attachment_size = Column(Integer, default=0)
    links_count = Column(Integer, default=0)
    images_count = Column(Integer, default=0)
    sentiment_score = Column(Float, nullable=True)
    priority_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    email = relationship("Email", back_populates="metrics")
    
    def __repr__(self):
        return f"<EmailMetrics(email_id='{self.email_id}')>"

class Email(Base):
    __tablename__ = 'emails'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String(36), nullable=True, index=True)
    sender_id = Column(Integer, ForeignKey('email_addresses.id'))
    reply_to_id = Column(Integer, ForeignKey('email_addresses.id'), nullable=True)
    in_reply_to = Column(String(36), nullable=True)
    subject = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    html_body = Column(Text, nullable=True)
    received_at = Column(DateTime, default=datetime.utcnow, index=True)
    read = Column(Boolean, default=False)
    starred = Column(Boolean, default=False)
    important = Column(Boolean, default=False)
    spam = Column(Boolean, default=False)
    deleted = Column(Boolean, default=False)
    draft = Column(Boolean, default=False)
    sent = Column(Boolean, default=False)
    folder = Column(Enum(EmailFolder), default=EmailFolder.INBOX)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sender_obj = relationship("EmailAddress", foreign_keys=[sender_id], back_populates="sent_emails")
    reply_to_obj = relationship("EmailAddress", foreign_keys=[reply_to_id])
    
    recipients = relationship("EmailAddress", secondary=email_recipients)
    cc = relationship("EmailAddress", secondary=email_cc)
    bcc = relationship("EmailAddress", secondary=email_bcc)
    
    attachments = relationship("Attachment", back_populates="email", cascade="all, delete-orphan")
    labels = relationship("Label", secondary=email_labels)
    classification = relationship("Classification", back_populates="email", uselist=False, cascade="all, delete-orphan")
    metrics = relationship("EmailMetrics", back_populates="email", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Email(id='{self.id}', subject='{self.subject}')>"