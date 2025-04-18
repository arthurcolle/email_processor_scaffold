import httpx
import json
import redis
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from app.models.fakemail import (
    EmailPayload, 
    FakeMailEmail, 
    EmailClassification,
    ClassificationResult,
    ProcessingStats,
    ProcessingBatch
)
from contextlib import asynccontextmanager
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get worker count from environment variables or use default
WORKER_COUNT = int(os.getenv("PROCESSOR_WORKER_COUNT", "3"))
MAX_RETRIES = int(os.getenv("PROCESSOR_MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("PROCESSOR_RETRY_DELAY_MS", "1000"))

class FakeMailService:
    def __init__(self, redis_client: redis.Redis, fakemail_base_url: str):
        self.redis_client = redis_client
        self.base_url = fakemail_base_url
        self.current_history_id_key = "fakemail:current_history_id"
        self.webhook_url_key = "fakemail:webhook_url"
        self.processed_email_prefix = "processed_email:"
        self.stats_key = "fakemail:stats"
        self.processing_batch_prefix = "processing_batch:"
        self.httpx_timeout = 10.0  # Default timeout in seconds
        
    @asynccontextmanager
    async def http_client(self):
        """
        Context manager for httpx client with timeout
        """
        async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
            yield client
        
    async def watch(self) -> int:
        """
        Call FakeMail's /watch endpoint to get the current history_id
        """
        logger.info("Calling FakeMail /watch endpoint")
        
        try:
            async with self.http_client() as client:
                response = await client.post(f"{self.base_url}/watch")
                response.raise_for_status()
                
                # Parse the response
                try:
                    data = response.json()
                    logger.info(f"Watch response: {data}")
                except json.JSONDecodeError:
                    # Handle non-JSON response
                    logger.info(f"Watch response is not JSON: {response.text}")
                    data = {"history_id": 0}
                    
                # Extract history_id - in some implementations it might be directly in the response
                if isinstance(data, dict):
                    history_id = data.get("history_id", 0)
                else:
                    try:
                        # Try to parse it directly
                        history_id = int(response.text.strip())
                        logger.info(f"Extracted history_id directly from response: {history_id}")
                    except (ValueError, TypeError):
                        logger.warning("Could not parse history_id from response, using 0")
                        history_id = 0
                
                # Store the current history_id in Redis
                self.redis_client.set(self.current_history_id_key, history_id)
                
                logger.info(f"Successfully watched FakeMail, current history_id: {history_id}")
                return history_id
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while watching FakeMail: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error while watching FakeMail: {str(e)}")
            raise
            
    async def subscribe(self, webhook_url: str) -> bool:
        """
        Subscribe to FakeMail's webhook notifications
        """
        logger.info(f"Subscribing to FakeMail webhooks with URL: {webhook_url}")
        payload = {"webhook_url": webhook_url}
        
        try:
            async with self.http_client() as client:
                response = await client.post(
                    f"{self.base_url}/subscribe",
                    json=payload
                )
                response.raise_for_status()
                
                # Store the webhook URL in Redis
                self.redis_client.set(self.webhook_url_key, webhook_url)
                
                logger.info(f"Successfully subscribed to FakeMail webhooks")
                return True
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while subscribing to FakeMail: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error while subscribing to FakeMail: {str(e)}")
            raise
            
    async def get_email_ids_since(self, history_id: int) -> List[str]:
        """
        Get all email IDs with history_id greater than the given history_id
        """
        logger.info(f"Getting email IDs since history_id: {history_id}")
        
        try:
            async with self.http_client() as client:
                response = await client.get(
                    f"{self.base_url}/emails",
                    params={"from_history_id": history_id}
                )
                response.raise_for_status()
                
                # Parse the response
                try:
                    data = response.json()
                    logger.info(f"Email IDs response type: {type(data)}")
                    
                    # Handle different response formats
                    if isinstance(data, list):
                        # If response is already a list of email IDs
                        email_ids = data
                    elif isinstance(data, dict) and "emails" in data:
                        # If response is a dict with an "emails" key
                        email_ids = data.get("emails", [])
                    elif isinstance(data, dict) and "email_ids" in data:
                        # If response is a dict with an "email_ids" key
                        email_ids = data.get("email_ids", [])
                    else:
                        # If we can't find emails in the expected structure,
                        # log the response and use an empty list
                        logger.warning(f"Unexpected response format: {data}")
                        email_ids = []
                        
                        # Try to extract email IDs from the response if it's a dict
                        if isinstance(data, dict):
                            # Look for any key that might contain a list of email IDs
                            for key, value in data.items():
                                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                                    logger.info(f"Found potential email IDs in key '{key}'")
                                    email_ids = value
                                    break
                
                except json.JSONDecodeError:
                    # Handle non-JSON response
                    logger.warning(f"Non-JSON response when getting email IDs: {response.text}")
                    email_ids = []
                    
                    # Try to parse response as a simple list of IDs, one per line
                    try:
                        lines = response.text.strip().split('\n')
                        if all(line.strip() for line in lines):
                            email_ids = [line.strip() for line in lines]
                            logger.info(f"Parsed {len(email_ids)} email IDs from text response")
                    except Exception as parse_e:
                        logger.error(f"Error parsing text response: {str(parse_e)}")
                
                logger.info(f"Found {len(email_ids)} new emails")
                return email_ids
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while getting email IDs: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error while getting email IDs: {str(e)}")
            raise
            
    async def get_email(self, email_id: str) -> FakeMailEmail:
        """
        Get a specific email by ID
        """
        logger.info(f"Getting email with ID: {email_id}")
        
        try:
            async with self.http_client() as client:
                response = await client.get(f"{self.base_url}/email/{email_id}")
                response.raise_for_status()
                
                try:
                    email_data = response.json()
                    logger.info(f"Email data response type: {type(email_data)}")
                    
                    # Handle different response formats
                    if isinstance(email_data, dict):
                        # Check if the required fields are present
                        missing_fields = []
                        for field in ['id', 'subject', 'body']:
                            if field not in email_data:
                                missing_fields.append(field)
                        
                        if missing_fields:
                            logger.warning(f"Missing required fields in email data: {missing_fields}")
                            
                            # Try to look for the data in a nested structure
                            if 'email' in email_data and isinstance(email_data['email'], dict):
                                logger.info("Found email data in nested 'email' field")
                                email_data = email_data['email']
                            
                            # If still missing fields, add defaults
                            for field in missing_fields:
                                if field == 'id' and 'id' not in email_data:
                                    email_data['id'] = email_id
                                elif field == 'subject' and 'subject' not in email_data:
                                    email_data['subject'] = "No Subject"
                                elif field == 'body' and 'body' not in email_data:
                                    email_data['body'] = ""
                    else:
                        logger.warning(f"Unexpected email data format: {email_data}")
                        # Create a default email object
                        email_data = {
                            'id': email_id,
                            'subject': "Unknown Subject",
                            'body': "Email body could not be retrieved"
                        }
                        
                    return FakeMailEmail(**email_data)
                    
                except json.JSONDecodeError:
                    # Handle non-JSON response
                    logger.warning(f"Non-JSON response when getting email: {response.text}")
                    
                    # Create a default email object
                    return FakeMailEmail(
                        id=email_id,
                        subject="Unknown Subject",
                        body=response.text if response.text else "Email body could not be retrieved"
                    )
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while getting email {email_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error while getting email {email_id}: {str(e)}")
            raise
            
    async def classify_email(self, email_payload: Dict[str, Any], retry_count: int = 0) -> Tuple[str, float]:
        """
        Call FakeMail's classify endpoint to get the classification for an email
        """
        logger.info(f"Classifying email: {email_payload.get('subject', '')[:30]}...")
        
        try:
            # Ensure the payload contains at least subject and body
            if 'subject' not in email_payload:
                email_payload['subject'] = "No Subject"
            if 'body' not in email_payload:
                email_payload['body'] = "No Content"
            
            async with self.http_client() as client:
                response = await client.post(
                    f"{self.base_url}/classify",
                    json=email_payload
                )
                response.raise_for_status()
                
                try:
                    result = response.json()
                    logger.info(f"Classification response: {result}")
                    
                    # Handle different response formats
                    if isinstance(result, dict):
                        classification = result.get("classification")
                        confidence = result.get("confidence", 1.0)
                    elif isinstance(result, str):
                        # If the response is just a string (the classification)
                        classification = result
                        confidence = 1.0
                    else:
                        logger.warning(f"Unexpected classification result format: {result}")
                        # Default values as fallback
                        classification = "unknown"
                        confidence = 0.5
                    
                    # Validate classification value
                    valid_classifications = ["meeting", "promotion", "intro", "unknown"]
                    if not classification or classification not in valid_classifications:
                        logger.warning(f"Invalid classification '{classification}', defaulting to 'unknown'")
                        classification = "unknown"
                        confidence = 0.5
                    
                    logger.info(f"Email classified as '{classification}' with confidence {confidence}")
                    return classification, confidence
                    
                except json.JSONDecodeError:
                    # Handle non-JSON response
                    logger.warning(f"Non-JSON response when classifying email: {response.text}")
                    
                    # Try to extract classification from text response
                    text = response.text.strip().lower()
                    if "meeting" in text:
                        return "meeting", 0.8
                    elif "promotion" in text:
                        return "promotion", 0.8
                    elif "intro" in text:
                        return "intro", 0.8
                    else:
                        return "unknown", 0.5
                        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while classifying email: {str(e)}")
            
            # Implement retry logic
            if retry_count < MAX_RETRIES:
                retry_count += 1
                logger.info(f"Retrying classification (attempt {retry_count}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY / 1000)  # Convert ms to seconds
                return await self.classify_email(email_payload, retry_count)
            else:
                # Fallback classification based on the email content
                subject = email_payload.get('subject', '').lower()
                body = email_payload.get('body', '').lower()
                
                # Simple rule-based classification as fallback
                if any(kw in subject or kw in body for kw in ["meeting", "appointment", "schedule"]):
                    return "meeting", 0.7
                elif any(kw in subject or kw in body for kw in ["offer", "discount", "sale", "promotion"]):
                    return "promotion", 0.7
                elif any(kw in subject or kw in body for kw in ["hello", "hi", "introduction", "introducing"]):
                    return "intro", 0.7
                else:
                    return "unknown", 0.5
        except Exception as e:
            logger.error(f"Error while classifying email: {str(e)}")
            
            # Fallback classification
            subject = email_payload.get('subject', '').lower()
            body = email_payload.get('body', '').lower()
            
            # Simple rule-based classification as fallback
            if any(kw in subject or kw in body for kw in ["meeting", "appointment", "schedule"]):
                return "meeting", 0.6
            elif any(kw in subject or kw in body for kw in ["offer", "discount", "sale", "promotion"]):
                return "promotion", 0.6
            elif any(kw in subject or kw in body for kw in ["hello", "hi", "introduction", "introducing"]):
                return "intro", 0.6
            else:
                return "unknown", 0.5
            
    async def send_email(self, subject: str, body: str) -> str:
        """
        Send an email using FakeMail's API
        """
        logger.info(f"Sending email: {subject[:30]}...")
        payload = EmailPayload(subject=subject, body=body)
        
        try:
            async with self.http_client() as client:
                response = await client.post(
                    f"{self.base_url}/send_email",
                    json=payload.model_dump()
                )
                response.raise_for_status()
                
                # Assuming the response contains the email ID
                result = response.json()
                email_id = result.get("email_id")
                
                logger.info(f"Email sent successfully with ID: {email_id}")
                return email_id
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while sending email: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error while sending email: {str(e)}")
            raise
            
    def get_current_history_id(self) -> int:
        """
        Get the current history_id from Redis
        """
        history_id = self.redis_client.get(self.current_history_id_key)
        if history_id is None:
            return 0
        return int(history_id)
        
    def update_current_history_id(self, history_id: int) -> None:
        """
        Update the current history_id in Redis
        """
        self.redis_client.set(self.current_history_id_key, history_id)
        
        # Also update the stats
        self._update_stats_history_id(history_id)
        
    def _update_stats_history_id(self, history_id: int) -> None:
        """
        Update the history_id in the stats
        """
        stats = self.get_processing_stats()
        stats.last_history_id = history_id
        stats.last_processed_at = datetime.now().isoformat()
        self._store_processing_stats(stats)
        
    def store_processed_email(
        self, 
        email_id: str, 
        classification: str, 
        email_data: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        processing_time_ms: Optional[float] = None,
        processor_id: str = "default-processor",
        retry_count: int = 0
    ) -> None:
        """
        Store a processed email classification in Redis
        """
        try:
            logger = logging.getLogger(__name__)
            logger.info(f"Storing processed email {email_id} with classification {classification}")
            
            now = datetime.now().isoformat()
            
            # Make sure we're using a valid classification enum
            if isinstance(classification, str) and not any(classification == e.value for e in ClassificationType):
                logger.warning(f"Converting non-enum classification '{classification}' to enum")
                try:
                    classification = ClassificationType(classification).value
                except ValueError:
                    logger.warning(f"Invalid classification '{classification}', using UNKNOWN")
                    classification = ClassificationType.UNKNOWN.value
            
            result = EmailClassification(
                email_id=email_id,
                classification=classification,
                processed_at=now,
                confidence=confidence,
                processed_by=processor_id,
                processing_time_ms=processing_time_ms,
                retry_count=retry_count
            )
            
            # Optionally store the email data
            if email_data:
                result.subject = email_data.get("subject")
                result.body = email_data.get("body")
            
            # Serialize to JSON
            json_data = result.model_dump_json()
            logger.info(f"Serialized data for email {email_id}: {json_data[:100]}...")
                
            # Store in Redis
            key = f"{self.processed_email_prefix}{email_id}"
            logger.info(f"Storing to Redis with key: {key}")
            
            # Make sure the Redis client is connected
            try:
                ping_result = self.redis_client.ping()
                logger.info(f"Redis ping before store: {ping_result}")
            except Exception as redis_err:
                logger.error(f"Redis ping error: {str(redis_err)}")
            
            # Store in Redis with explicit serialization
            store_result = self.redis_client.set(key, json_data)
            logger.info(f"Redis SET operation result: {store_result}")
            
            # Verify the data was stored
            stored_data = self.redis_client.get(key)
            logger.info(f"Verification - Redis GET returned: {stored_data is not None}")
            
            if stored_data:
                logger.info(f"Successfully stored classification for email {email_id}")
            else:
                logger.error(f"Failed to store or retrieve data for email {email_id}")
                # Try a different redis command to diagnose
                try:
                    keys = self.redis_client.keys(f"{self.processed_email_prefix}*")
                    logger.info(f"Redis keys with prefix '{self.processed_email_prefix}': {keys}")
                except Exception as e:
                    logger.error(f"Error listing Redis keys: {str(e)}")
            
            # Update processing stats
            self._update_stats_after_processing(classification, processing_time_ms)
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error storing processed email {email_id}: {str(e)}")
            raise
        
    def _update_stats_after_processing(self, classification: str, processing_time_ms: Optional[float]) -> None:
        """
        Update processing stats after processing an email
        """
        stats = self.get_processing_stats()
        
        # Update total count
        stats.total_emails_processed += 1
        
        # Update classification count
        if classification in stats.emails_by_classification:
            stats.emails_by_classification[classification] += 1
        else:
            stats.emails_by_classification[classification] = 1
            
        # Update average processing time
        if processing_time_ms is not None:
            if stats.avg_processing_time_ms == 0:
                stats.avg_processing_time_ms = processing_time_ms
            else:
                # Compute new average
                total_time = stats.avg_processing_time_ms * (stats.total_emails_processed - 1)
                new_avg = (total_time + processing_time_ms) / stats.total_emails_processed
                stats.avg_processing_time_ms = new_avg
                
        # Update last processed timestamp
        stats.last_processed_at = datetime.now().isoformat()
        
        # Store updated stats
        self._store_processing_stats(stats)
        
    def get_processed_email(self, email_id: str) -> Optional[EmailClassification]:
        """
        Retrieve a processed email classification from Redis
        """
        logger = logging.getLogger(__name__)
        key = f"{self.processed_email_prefix}{email_id}"
        logger.info(f"Retrieving processed email from Redis with key: {key}")
        
        try:
            # Check Redis connection
            try:
                ping_result = self.redis_client.ping()
                logger.info(f"Redis ping before retrieval: {ping_result}")
            except Exception as e:
                logger.error(f"Redis ping error before retrieval: {str(e)}")
            
            data = self.redis_client.get(key)
            
            if not data:
                logger.warning(f"No data found in Redis for key: {key}")
                
                # Debug: Check if any keys with this prefix exist
                try:
                    keys = self.redis_client.keys(f"{self.processed_email_prefix}*")
                    logger.info(f"Existing keys with prefix '{self.processed_email_prefix}': {keys}")
                except Exception as e:
                    logger.error(f"Error listing Redis keys: {str(e)}")
                
                return None
            
            logger.info(f"Data retrieved from Redis for email {email_id}: {data[:100] if data else None}...")
            
            try:
                result_dict = json.loads(data)
                return EmailClassification(**result_dict)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for email {email_id}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Error parsing result for email {email_id}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving processed email {email_id}: {str(e)}")
            return None
    
    def create_processing_batch(self, history_id: int) -> str:
        """
        Create a new processing batch record
        """
        batch = ProcessingBatch(
            history_id=history_id,
            start_time=datetime.now().isoformat()
        )
        
        self.redis_client.set(
            f"{self.processing_batch_prefix}{batch.batch_id}",
            batch.model_dump_json()
        )
        
        return batch.batch_id
        
    def update_processing_batch(self, batch_id: str, email_ids: List[str], status: str, error: Optional[str] = None) -> None:
        """
        Update a processing batch record
        """
        batch_data = self.redis_client.get(f"{self.processing_batch_prefix}{batch_id}")
        if not batch_data:
            logger.warning(f"Processing batch {batch_id} not found")
            return
            
        batch_dict = json.loads(batch_data)
        batch = ProcessingBatch(**batch_dict)
        
        batch.email_ids = email_ids
        batch.status = status
        batch.error = error
        batch.end_time = datetime.now().isoformat()
        
        self.redis_client.set(
            f"{self.processing_batch_prefix}{batch_id}",
            batch.model_dump_json()
        )
        
    def get_processing_batch(self, batch_id: str) -> Optional[ProcessingBatch]:
        """
        Get a processing batch record
        """
        batch_data = self.redis_client.get(f"{self.processing_batch_prefix}{batch_id}")
        if not batch_data:
            return None
            
        batch_dict = json.loads(batch_data)
        return ProcessingBatch(**batch_dict)
        
    def get_processing_stats(self) -> ProcessingStats:
        """
        Get the current processing stats
        """
        stats_data = self.redis_client.get(self.stats_key)
        if not stats_data:
            return ProcessingStats()
            
        stats_dict = json.loads(stats_data)
        return ProcessingStats(**stats_dict)
        
    def _store_processing_stats(self, stats: ProcessingStats) -> None:
        """
        Store the processing stats in Redis
        """
        self.redis_client.set(self.stats_key, stats.model_dump_json())