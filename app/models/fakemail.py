from pydantic import BaseModel, Field, validator, ConfigDict
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
import uuid
import enum

class EmailPayload(BaseModel):
    subject: str
    body: str

class WebhookPayload(BaseModel):
    history_id: int

class FakeMailEmail(BaseModel):
    id: str
    subject: str
    body: str
    # Add any other fields that might be in the response

class ClassificationType(str, enum.Enum):
    INTRO = "intro"
    PROMOTION = "promotion"
    MEETING = "meeting"
    UNKNOWN = "unknown"

class ProcessingStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    
# Structured JSON schema definition for the classification response
class EmailClassificationSchema(BaseModel):
    """
    Structured output schema for email classification results.
    This schema ensures consistent JSON responses that adhere to 
    the specified structure for all email classification results.
    """
    email_id: str = Field(..., description="Unique identifier of the processed email")
    classification: Literal["meeting", "promotion", "intro", "unknown"] = Field(
        ..., description="The classified category of the email"
    )
    confidence: float = Field(
        ..., description="Confidence score of the classification (0.0-1.0)",
        ge=0.0, le=1.0
    )
    processed_at: str = Field(..., description="ISO-formatted timestamp when the email was processed")
    subject: Optional[str] = Field(None, description="Email subject (only included when include_content=true)")
    body: Optional[str] = Field(None, description="Email body (only included when include_content=true)")
    processing_time_ms: Optional[float] = Field(None, description="Time taken to process the email in milliseconds")
    processor_id: Optional[str] = Field(None, description="Identifier of the processor that classified the email")

class ClassificationResult(BaseModel):
    classification: ClassificationType
    confidence: float = 1.0
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    
    model_config = ConfigDict(
        use_enum_values=True
    )

class EmailMetadata(BaseModel):
    received_at: str
    size_bytes: int = 0
    has_attachments: bool = False
    word_count: int = 0
    sender_domain: Optional[str] = None
    priority: int = 1  # 1-5, 1 is highest

class ProcessingMetrics(BaseModel):
    queue_time_ms: Optional[float] = None  # Time spent in queue
    fetch_time_ms: Optional[float] = None  # Time to fetch email
    classification_time_ms: Optional[float] = None  # Time to classify
    total_time_ms: Optional[float] = None  # Total processing time
    retry_count: int = 0
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None

class EmailClassification(BaseModel):
    email_id: str
    classification: ClassificationType
    processed_at: str
    subject: Optional[str] = None
    body: Optional[str] = None
    confidence: float = 1.0
    processed_by: str = "default-processor"
    worker_id: Optional[str] = None
    batch_id: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    metadata: Optional[EmailMetadata] = None
    metrics: Optional[ProcessingMetrics] = None
    version: str = "1.0.0"
    
    model_config = ConfigDict(
        use_enum_values=True
    )

class ProcessingStats(BaseModel):
    total_emails_processed: int = 0
    emails_by_classification: Dict[str, int] = Field(default_factory=lambda: {
        "intro": 0, "promotion": 0, "meeting": 0, "unknown": 0
    })
    avg_processing_time_ms: float = 0
    min_processing_time_ms: Optional[float] = None
    max_processing_time_ms: Optional[float] = None
    last_history_id: int = 0
    last_processed_at: Optional[str] = None
    success_rate: float = 100.0  # Percentage
    error_count: int = 0
    retry_count: int = 0
    throttled_count: int = 0
    busy_workers: int = 0
    available_workers: int = 0
    queue_depth: int = 0
    uptime_seconds: int = 0
    
class HealthStatus(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    details: Dict[str, Any] = {}
    uptime_seconds: int = 0
    version: str = "1.0.0"
    redis_connected: bool = False
    fakemail_api_connected: bool = False
    email_count_24h: int = 0
    error_count_24h: int = 0
    
class ProcessingBatch(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    history_id: int
    start_time: str
    end_time: Optional[str] = None
    email_ids: List[str] = []
    status: ProcessingStatus = ProcessingStatus.PROCESSING
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    worker_assignments: Dict[str, List[str]] = Field(default_factory=dict)
    priority: int = 3  # 1-5, 1 is highest
    
    model_config = ConfigDict(
        use_enum_values=True
    )
    
class ProcessingRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    active: bool = True
    conditions: List[Dict[str, Any]] = []  # [{field: "subject", operator: "contains", value: "meeting"}]
    actions: List[Dict[str, Any]] = []  # [{action: "prioritize", value: 1}]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
class SystemConfig(BaseModel):
    worker_count: int = 3
    max_retries: int = 3
    retry_delay_ms: int = 1000
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 10
    batch_size: int = 20
    worker_idle_timeout_seconds: int = 300
    throttle_threshold: int = 100  # Max emails per minute
    default_priority: int = 3  # 1-5, 1 is highest