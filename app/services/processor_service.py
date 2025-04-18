import asyncio
import time
import logging
import uuid
import os
import psutil
import json
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from app.services.fakemail_service import FakeMailService
from app.services.email_queue import EmailQueue
from app.services.metrics_service import MetricsService
from app.services.config_service import ConfigService
from app.models.fakemail import ProcessingStatus, ProcessingMetrics, EmailMetadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessorWorker:
    """
    A worker that processes emails from the queue
    """
    def __init__(
        self, 
        worker_id: str,
        fakemail_service: FakeMailService,
        email_queue: EmailQueue,
        metrics_service: MetricsService,
        config_service: ConfigService
    ):
        self.worker_id = worker_id
        self.fakemail_service = fakemail_service
        self.email_queue = email_queue
        self.metrics_service = metrics_service
        self.config_service = config_service
        self.running = False
        self.current_email_id: Optional[str] = None
        self.process = psutil.Process(os.getpid())
        
    async def start(self) -> None:
        """Start the worker"""
        if self.running:
            return
            
        self.running = True
        logger.info(f"[{self.worker_id}] Worker started")
        
        # Update worker status
        self.metrics_service.update_worker_status(
            self.worker_id, 
            busy=False,
            queue_size=await self.email_queue.get_queue_depth()
        )
        
        # Start processing loop
        asyncio.create_task(self.processing_loop())
        
    async def stop(self) -> None:
        """Stop the worker"""
        self.running = False
        logger.info(f"[{self.worker_id}] Worker stopping")
        
    async def processing_loop(self) -> None:
        """Main processing loop that consumes emails from Redis queue"""
        while self.running:
            try:
                # Get next email from queue
                result = await self.email_queue.dequeue()
                
                if result:
                    email_id, metadata = result
                    queue_time = time.time()
                    self.current_email_id = email_id
                    
                    # Update worker status
                    self.metrics_service.update_worker_status(
                        self.worker_id, 
                        busy=True,
                        queue_size=await self.email_queue.get_queue_depth()
                    )
                    
                    logger.info(f"[{self.worker_id}] Processing email {email_id} from Redis queue")
                    
                    # Process the email
                    try:
                        await self.process_email(email_id, queue_time, metadata)
                        await self.email_queue.complete(email_id)
                        logger.info(f"[{self.worker_id}] Email {email_id} processed successfully")
                    except Exception as e:
                        logger.error(f"[{self.worker_id}] Error processing email {email_id}: {str(e)}")
                        await self.email_queue.fail(email_id, str(e))
                        
                    # Clear current email
                    self.current_email_id = None
                    
                    # Update worker status
                    self.metrics_service.update_worker_status(
                        self.worker_id, 
                        busy=False,
                        queue_size=await self.email_queue.get_queue_depth()
                    )
                else:
                    # No emails to process, wait a bit
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[{self.worker_id}] Error in processing loop: {str(e)}")
                await asyncio.sleep(1)  # Wait a bit before trying again
                
    async def process_email(self, email_id: str, queue_time: float, metadata: Dict[str, Any]) -> None:
        """Process a single email"""
        start_time = time.time()
        queue_time_ms = (start_time - queue_time) * 1000
        retry_count = metadata.get("retry_count", 0)
        
        logger.info(f"[{self.worker_id}] Starting to process email {email_id} from Redis queue")
        
        # Initialize metrics
        metrics = ProcessingMetrics(
            queue_time_ms=queue_time_ms,
            retry_count=retry_count
        )
        
        # Get CPU & memory usage at start
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        metrics.cpu_usage_percent = cpu_percent
        metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
        
        # Get the email from FakeMail API
        fetch_start = time.time()
        email = await self.fakemail_service.get_email(email_id)
        fetch_end = time.time()
        metrics.fetch_time_ms = (fetch_end - fetch_start) * 1000
        
        # Create email metadata
        email_size = len(email.subject) + len(email.body)
        word_count = len(email.body.split())
        
        email_metadata = EmailMetadata(
            received_at=datetime.now().isoformat(),
            size_bytes=email_size,
            word_count=word_count,
            priority=metadata.get("priority", 3)
        )
        
        # Prepare the payload for classification
        classification_payload = {
            "subject": email.subject,
            "body": email.body
        }
        
        # Classify the email using improved classifier
        classify_start = time.time()
        
        # Simple rule-based classification logic
        subject = email.subject.lower()
        body = email.body.lower()
        
        # Define keywords for each category
        meeting_keywords = ["meeting", "appointment", "calendar", "schedule", "zoom", "teams", "join", "invite"]
        promotion_keywords = ["offer", "discount", "sale", "promotion", "deal", "limited time", "exclusive", "buy", "subscribe"]
        intro_keywords = ["introduction", "hello", "hi", "greetings", "welcome", "nice to meet", "introducing", "new", "connect"]
        
        # Check for keyword matches
        is_meeting = any(keyword in subject or keyword in body for keyword in meeting_keywords)
        is_promotion = any(keyword in subject or keyword in body for keyword in promotion_keywords)
        is_intro = any(keyword in subject or keyword in body for keyword in intro_keywords)
        
        # Determine classification and confidence
        confidence = 0.85  # Default confidence
        
        if is_meeting:
            classification = "meeting"
            confidence = 0.9 if any(keyword in subject for keyword in meeting_keywords) else 0.75
        elif is_promotion:
            classification = "promotion"
            confidence = 0.88 if any(keyword in subject for keyword in promotion_keywords) else 0.72
        elif is_intro:
            classification = "intro"
            confidence = 0.92 if any(keyword in subject for keyword in intro_keywords) else 0.78
        else:
            classification = "unknown"
            confidence = 0.60
            
        # Fallback to API call if confidence is low
        if confidence < 0.65:
            try:
                classification, api_confidence = await self.fakemail_service.classify_email(classification_payload)
                # Only use API result if confidence is higher
                if api_confidence > confidence:
                    confidence = api_confidence
            except Exception as e:
                logger.warning(f"API classification failed, using rule-based result: {str(e)}")
                
        classify_end = time.time()
        metrics.classification_time_ms = (classify_end - classify_start) * 1000
        
        logger.info(f"[{self.worker_id}] Classified email {email_id} as '{classification}' with confidence {confidence:.2f}")
        
        # Calculate total processing time
        processing_end = time.time()
        metrics.total_time_ms = (processing_end - start_time) * 1000
        
        # Store the result with extended metadata
        batch_id = metadata.get("batch_id")
        
        # Store the processed email result
        try:
            self.fakemail_service.store_processed_email(
                email_id, 
                classification, 
                {
                    "subject": email.subject,
                    "body": email.body
                },
                confidence=confidence,
                processing_time_ms=metrics.total_time_ms,
                processor_id=f"worker-{self.worker_id}",
                retry_count=retry_count
            )
            
            # Update metrics
            self.metrics_service.record_email_processed(
                classification,
                metrics.total_time_ms,
                success=True,
                retried=(retry_count > 0)
            )
            
            # Verify storage - double-check that the result is accessible
            result = self.fakemail_service.get_processed_email(email_id)
            if result:
                logger.info(f"[{self.worker_id}] Successfully stored classification for email {email_id} as '{classification}'")
                
                # Publish event for real-time updates
                try:
                    # Create event data
                    event_data = {
                        "email_id": email_id,
                        "classification": classification,
                        "confidence": confidence,
                        "processed_at": datetime.now().isoformat(),
                        "subject": email.subject,
                        "processor_id": f"worker-{self.worker_id}",
                        "processing_time_ms": metrics.total_time_ms,
                        "status": "completed"
                    }
                    
                    # Publish to Redis for WebSocket to pick up
                    redis_client = self.fakemail_service.redis_client
                    redis_client.publish("email:process", json.dumps(event_data))
                    logger.debug(f"Published email processing event for {email_id}")
                except Exception as pub_error:
                    logger.warning(f"Error publishing event for email {email_id}: {str(pub_error)}")
            else:
                logger.error(f"[{self.worker_id}] Failed to verify stored result for email {email_id}")
                
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error storing result for email {email_id}: {str(e)}")
            
            # Publish failure event for real-time updates
            try:
                # Create failure event data
                event_data = {
                    "email_id": email_id,
                    "processed_at": datetime.now().isoformat(),
                    "subject": email.subject,
                    "processor_id": f"worker-{self.worker_id}",
                    "status": "failed",
                    "error": str(e)
                }
                
                # Publish to Redis for WebSocket to pick up
                redis_client = self.fakemail_service.redis_client
                redis_client.publish("email:process", json.dumps(event_data))
            except Exception:
                pass  # Ignore errors in error handling
                
            raise
            
class EmailProcessor:
    """
    Manages a pool of workers for email processing
    """
    def __init__(
        self, 
        fakemail_service: FakeMailService,
        email_queue: EmailQueue,
        metrics_service: MetricsService,
        config_service: ConfigService,
        worker_count: int = 3
    ):
        self.fakemail_service = fakemail_service
        self.email_queue = email_queue
        self.metrics_service = metrics_service
        self.config_service = config_service
        
        # Read worker count from config
        config = config_service.get_system_config()
        self.worker_count = config.worker_count
        
        self.workers: List[ProcessorWorker] = []
        self.maintenance_task = None
        self.running = False
        
    async def start(self) -> None:
        """Start the processor and workers"""
        if self.running:
            logger.warning("Processor is already running")
            return
            
        self.running = True
        
        # Create and start workers
        self.workers = []
        for i in range(self.worker_count):
            worker_id = f"worker-{i+1}"
            worker = ProcessorWorker(
                worker_id,
                self.fakemail_service,
                self.email_queue,
                self.metrics_service,
                self.config_service
            )
            self.workers.append(worker)
            await worker.start()
            
        # Start maintenance task
        self.maintenance_task = asyncio.create_task(self.maintenance_loop())
            
        logger.info(f"Email processor started with {self.worker_count} workers")
        
    async def stop(self) -> None:
        """Stop the processor and all workers"""
        if not self.running:
            logger.warning("Processor is not running")
            return
            
        self.running = False
        
        # Stop maintenance task
        if self.maintenance_task:
            self.maintenance_task.cancel()
            
        # Stop all workers
        for worker in self.workers:
            await worker.stop()
            
        self.workers = []
        
        logger.info("Email processor stopped")
        
    async def maintenance_loop(self) -> None:
        """Background task for maintenance operations"""
        while self.running:
            try:
                # Clear stalled emails
                await self.email_queue.clear_stalled()
                
                # Retry some failed emails
                await self.email_queue.retry_failed(10)  # Retry up to 10 at a time
                
                # Sleep for a while
                await asyncio.sleep(60)  # Run maintenance every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a bit before trying again
        
    async def process_emails(self, history_id: int, email_ids: List[str]) -> str:
        """
        Process a batch of emails
        """
        if not email_ids:
            logger.warning("No emails to process")
            return "no-emails"
            
        # Create a batch ID
        batch_id = str(uuid.uuid4())
        logger.info(f"Starting batch {batch_id} with {len(email_ids)} emails")
        
        try:
            # Save batch info
            batch_info = {
                "batch_id": batch_id,
                "history_id": history_id,
                "email_count": len(email_ids),
                "start_time": datetime.now().isoformat(),
                "status": ProcessingStatus.QUEUED.value
            }
            
            self.fakemail_service.create_processing_batch(history_id)
            
            # Queue emails for processing
            await self.email_queue.enqueue_batch(email_ids, priority=3, batch_id=batch_id)
            
            # Update batch status
            self.fakemail_service.update_processing_batch(
                batch_id, 
                email_ids, 
                ProcessingStatus.PROCESSING.value
            )
            
            # Notify about new emails via Redis pub/sub for real-time updates
            try:
                # Get a sample of emails to include in the notification (first 5)
                sample_emails = []
                for email_id in email_ids[:5]:
                    try:
                        email = await self.fakemail_service.get_email(email_id)
                        sample_emails.append({
                            "email_id": email_id,
                            "subject": email.subject,
                            "status": "received"
                        })
                    except Exception:
                        sample_emails.append({"email_id": email_id, "status": "received"})
                
                # Publish event data
                event_data = {
                    "batch_id": batch_id,
                    "email_count": len(email_ids),
                    "sample_emails": sample_emails,
                    "status": "queued",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Publish to Redis for WebSocket to pick up
                redis_client = self.fakemail_service.redis_client
                redis_client.publish("email:new", json.dumps(event_data))
                logger.debug(f"Published new emails event for batch {batch_id}")
            except Exception as pub_error:
                logger.warning(f"Error publishing new emails event: {str(pub_error)}")
            
            # Update current history ID - we do this now, not after processing
            # This ensures we don't miss or reprocess emails if the process crashes
            self.fakemail_service.update_current_history_id(history_id)
            
            logger.info(f"Batch {batch_id} queued for processing")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error queueing batch {batch_id}: {str(e)}")
            
            # Update batch status
            self.fakemail_service.update_processing_batch(
                batch_id, 
                email_ids, 
                ProcessingStatus.FAILED.value,
                str(e)
            )
            
            # Re-raise the exception
            raise