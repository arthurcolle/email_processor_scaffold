import json
import redis
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailQueue:
    """
    Redis-backed priority queue for email processing
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.queue_key = "email_processor:queue"
        self.processing_set_key = "email_processor:processing"
        self.failed_queue_key = "email_processor:failed"
        self.email_metadata_prefix = "email_processor:queue:metadata:"
        
    async def enqueue(self, email_id: str, priority: int = 3, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an email to the processing queue with priority (1-5, 1 is highest)
        """
        # Validate priority
        if priority < 1 or priority > 5:
            priority = 3  # Default to medium priority
        
        # Store metadata if provided
        if metadata:
            metadata_key = f"{self.email_metadata_prefix}{email_id}"
            metadata_with_timestamp = metadata.copy()
            metadata_with_timestamp["enqueued_at"] = datetime.now().isoformat()
            metadata_with_timestamp["priority"] = priority
            self.redis_client.set(metadata_key, json.dumps(metadata_with_timestamp))
        
        # Calculate score for priority sorting (lower score = higher priority)
        # Use current timestamp as tiebreaker within same priority
        timestamp = time.time()
        score = priority * 1000000000 + timestamp  # This gives us 5 priority levels with timestamp-based ordering
        
        # Add to the queue
        self.redis_client.zadd(self.queue_key, {email_id: score})
        logger.info(f"Enqueued email {email_id} with priority {priority}")
        
        return True
        
    async def enqueue_batch(self, email_ids: List[str], priority: int = 3, batch_id: Optional[str] = None) -> int:
        """
        Add multiple emails to the queue in a single operation
        """
        if not email_ids:
            return 0
            
        # Validate priority
        if priority < 1 or priority > 5:
            priority = 3
            
        # Prepare batch for zadd
        timestamp = time.time()
        score = priority * 1000000000 + timestamp
        email_scores = {email_id: score for email_id in email_ids}
        
        # Add to queue
        added = self.redis_client.zadd(self.queue_key, email_scores)
        
        # Store basic metadata for each email
        pipeline = self.redis_client.pipeline()
        for email_id in email_ids:
            metadata = {
                "enqueued_at": datetime.now().isoformat(),
                "priority": priority,
                "batch_id": batch_id
            }
            metadata_key = f"{self.email_metadata_prefix}{email_id}"
            pipeline.set(metadata_key, json.dumps(metadata))
            
        pipeline.execute()
        
        logger.info(f"Enqueued batch of {len(email_ids)} emails with priority {priority}")
        return added
        
    async def dequeue(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the next email from the queue based on priority
        """
        # Use Lua script to atomically pop and move to processing set
        lua_script = """
        local item = redis.call('ZPOPMIN', KEYS[1], 1)
        if item[1] then
            redis.call('SADD', KEYS[2], item[1])
            return item[1]
        end
        return nil
        """
        
        # Execute the script
        result = self.redis_client.eval(
            lua_script, 
            2,  # number of keys
            self.queue_key, 
            self.processing_set_key
        )
        
        if not result:
            return None
            
        email_id = result.decode("utf-8")
        
        # Get metadata if available
        metadata = await self.get_email_metadata(email_id)
        
        return email_id, metadata
        
    async def complete(self, email_id: str) -> bool:
        """
        Mark an email as processed and remove from processing set
        """
        removed = self.redis_client.srem(self.processing_set_key, email_id)
        
        # Clean up metadata
        self.redis_client.delete(f"{self.email_metadata_prefix}{email_id}")
        
        return removed > 0
        
    async def fail(self, email_id: str, error: str, retry: bool = True) -> bool:
        """
        Mark an email as failed
        """
        # Get current metadata
        metadata = await self.get_email_metadata(email_id)
        if not metadata:
            metadata = {}
            
        # Update metadata with error info
        metadata["error"] = error
        metadata["failed_at"] = datetime.now().isoformat()
        metadata["retry_count"] = metadata.get("retry_count", 0) + 1
        
        # Store updated metadata
        self.redis_client.set(
            f"{self.email_metadata_prefix}{email_id}", 
            json.dumps(metadata)
        )
        
        # Remove from processing set
        self.redis_client.srem(self.processing_set_key, email_id)
        
        if retry and metadata.get("retry_count", 1) <= 3:
            # Re-queue with lower priority for retry
            priority = min(metadata.get("priority", 3) + 1, 5)
            await self.enqueue(email_id, priority, metadata)
            return True
        else:
            # Add to failed queue
            self.redis_client.zadd(
                self.failed_queue_key, 
                {email_id: time.time()}
            )
            return False
            
    async def get_email_metadata(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an email in the queue
        """
        metadata_key = f"{self.email_metadata_prefix}{email_id}"
        metadata_json = self.redis_client.get(metadata_key)
        
        if not metadata_json:
            return None
            
        return json.loads(metadata_json)
        
    async def get_queue_depth(self) -> int:
        """
        Get the number of emails in the queue
        """
        return self.redis_client.zcard(self.queue_key)
        
    async def get_processing_count(self) -> int:
        """
        Get the number of emails currently being processed
        """
        return self.redis_client.scard(self.processing_set_key)
        
    async def get_failed_count(self) -> int:
        """
        Get the number of failed emails
        """
        return self.redis_client.zcard(self.failed_queue_key)
        
    async def retry_failed(self, max_count: int = 50) -> int:
        """
        Retry a batch of failed emails
        """
        # Get failed emails, oldest first
        failed_emails = self.redis_client.zrange(
            self.failed_queue_key, 
            0, 
            max_count - 1
        )
        
        if not failed_emails:
            return 0
            
        # Re-queue each failed email
        count = 0
        for email_bytes in failed_emails:
            email_id = email_bytes.decode("utf-8")
            metadata = await self.get_email_metadata(email_id)
            
            # Use higher priority for retries to avoid starvation
            priority = 2  # High priority
            await self.enqueue(email_id, priority, metadata)
            
            # Remove from failed queue
            self.redis_client.zrem(self.failed_queue_key, email_id)
            count += 1
            
        logger.info(f"Requeued {count} failed emails")
        return count
        
    async def clear_stalled(self, max_age_seconds: int = 300) -> int:
        """
        Clear emails that have been in processing state for too long
        """
        # Get all emails in processing set
        processing_emails = self.redis_client.smembers(self.processing_set_key)
        
        if not processing_emails:
            return 0
            
        now = datetime.now()
        count = 0
        
        for email_bytes in processing_emails:
            email_id = email_bytes.decode("utf-8")
            metadata = await self.get_email_metadata(email_id)
            
            if not metadata or "enqueued_at" not in metadata:
                # No metadata or timestamp, assume stalled
                await self.fail(email_id, "Processing timed out", retry=True)
                count += 1
                continue
                
            try:
                # Check if processing for too long
                enqueued_time = datetime.fromisoformat(metadata["enqueued_at"])
                age_seconds = (now - enqueued_time).total_seconds()
                
                if age_seconds > max_age_seconds:
                    await self.fail(email_id, "Processing timed out", retry=True)
                    count += 1
            except (ValueError, TypeError):
                # Invalid timestamp, assume stalled
                await self.fail(email_id, "Invalid timestamp in metadata", retry=True)
                count += 1
                
        if count > 0:
            logger.info(f"Cleared {count} stalled emails from processing")
            
        return count