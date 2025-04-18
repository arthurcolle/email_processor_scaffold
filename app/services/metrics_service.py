import json
import redis
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.models.fakemail import ProcessingStats, ClassificationType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsService:
    """
    Service for tracking and aggregating email processing metrics
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.stats_key = "email_processor:stats"
        self.hourly_metrics_prefix = "email_processor:metrics:hourly:"
        self.daily_metrics_prefix = "email_processor:metrics:daily:"
        self.worker_metrics_prefix = "email_processor:metrics:worker:"
        self.startup_time = time.time()
        
        # Initialize stats if not exists
        if not self.redis_client.exists(self.stats_key):
            self._initialize_stats()
            
    def _initialize_stats(self) -> None:
        """
        Initialize empty stats
        """
        stats = ProcessingStats()
        self.redis_client.set(self.stats_key, stats.model_dump_json())
        
    def get_stats(self) -> ProcessingStats:
        """
        Get current processing stats
        """
        stats_data = self.redis_client.get(self.stats_key)
        if not stats_data:
            self._initialize_stats()
            stats_data = self.redis_client.get(self.stats_key)
            
        stats_dict = json.loads(stats_data)
        
        # Update uptime every time stats are requested
        stats = ProcessingStats(**stats_dict)
        stats.uptime_seconds = int(time.time() - self.startup_time)
        
        return stats
        
    def update_stats(self, updates: Dict[str, Any]) -> ProcessingStats:
        """
        Update specific fields in the stats
        """
        stats = self.get_stats()
        stats_dict = stats.model_dump()
        
        # Update only the provided fields
        for key, value in updates.items():
            if key in stats_dict:
                stats_dict[key] = value
                
        updated_stats = ProcessingStats(**stats_dict)
        self.redis_client.set(self.stats_key, updated_stats.model_dump_json())
        
        return updated_stats
        
    def record_email_processed(
        self, 
        classification: str, 
        processing_time_ms: float,
        success: bool = True,
        retried: bool = False,
        history_id: Optional[int] = None
    ) -> None:
        """
        Record metrics for a processed email
        """
        # Get current stats
        stats = self.get_stats()
        
        # Update total count
        stats.total_emails_processed += 1
        
        # Update classification count
        if classification in stats.emails_by_classification:
            stats.emails_by_classification[classification] += 1
        else:
            stats.emails_by_classification[classification] = 1
            
        # Update processing time stats
        if stats.min_processing_time_ms is None or processing_time_ms < stats.min_processing_time_ms:
            stats.min_processing_time_ms = processing_time_ms
            
        if stats.max_processing_time_ms is None or processing_time_ms > stats.max_processing_time_ms:
            stats.max_processing_time_ms = processing_time_ms
            
        # Update average processing time
        if stats.avg_processing_time_ms == 0:
            stats.avg_processing_time_ms = processing_time_ms
        else:
            total_time = stats.avg_processing_time_ms * (stats.total_emails_processed - 1)
            new_avg = (total_time + processing_time_ms) / stats.total_emails_processed
            stats.avg_processing_time_ms = new_avg
            
        # Update error and retry counts
        if not success:
            stats.error_count += 1
            # Update success rate
            stats.success_rate = ((stats.total_emails_processed - stats.error_count) / 
                                 stats.total_emails_processed) * 100
                                 
        if retried:
            stats.retry_count += 1
            
        # Update history_id if provided
        if history_id is not None and history_id > stats.last_history_id:
            stats.last_history_id = history_id
            
        # Update last processed timestamp
        stats.last_processed_at = datetime.now().isoformat()
        
        # Save updated stats
        self.redis_client.set(self.stats_key, stats.model_dump_json())
        
        # Update time-based metrics
        self._update_time_based_metrics(classification, processing_time_ms, success)
        
    def _update_time_based_metrics(self, classification: str, processing_time_ms: float, success: bool) -> None:
        """
        Update hourly and daily metrics
        """
        now = datetime.now()
        
        # Update hourly metrics
        hourly_key = f"{self.hourly_metrics_prefix}{now.strftime('%Y-%m-%d:%H')}"
        self._increment_metric(hourly_key, "total", 1)
        self._increment_metric(hourly_key, f"classification:{classification}", 1)
        self._increment_metric(hourly_key, "processing_time_ms", processing_time_ms)
        
        if not success:
            self._increment_metric(hourly_key, "errors", 1)
            
        # Set expiry for hourly metrics (48 hours)
        self.redis_client.expire(hourly_key, 48 * 60 * 60)
        
        # Update daily metrics
        daily_key = f"{self.daily_metrics_prefix}{now.strftime('%Y-%m-%d')}"
        self._increment_metric(daily_key, "total", 1)
        self._increment_metric(daily_key, f"classification:{classification}", 1)
        self._increment_metric(daily_key, "processing_time_ms", processing_time_ms)
        
        if not success:
            self._increment_metric(daily_key, "errors", 1)
            
        # Set expiry for daily metrics (30 days)
        self.redis_client.expire(daily_key, 30 * 24 * 60 * 60)
        
    def _increment_metric(self, key: str, field: str, value: float) -> None:
        """
        Increment a field in a hash by the given value
        """
        # If field doesn't exist, hincrbyfloat will create it
        self.redis_client.hincrbyfloat(key, field, value)
        
    def update_worker_status(self, worker_id: str, busy: bool, queue_size: int) -> None:
        """
        Update worker status metrics
        """
        stats = self.get_stats()
        
        # Set worker status
        worker_key = f"{self.worker_metrics_prefix}{worker_id}"
        self.redis_client.hset(worker_key, "busy", "1" if busy else "0")
        self.redis_client.hset(worker_key, "last_updated", datetime.now().isoformat())
        
        # Update global worker counts and queue depth
        busy_count = 0
        total_count = 0
        
        # Count busy and available workers
        worker_keys = self.redis_client.keys(f"{self.worker_metrics_prefix}*")
        for wkey in worker_keys:
            total_count += 1
            worker_busy = self.redis_client.hget(wkey, "busy")
            if worker_busy and worker_busy.decode("utf-8") == "1":
                busy_count += 1
        
        # Update stats
        updates = {
            "busy_workers": busy_count,
            "available_workers": total_count - busy_count,
            "queue_depth": queue_size
        }
        self.update_stats(updates)
        
    def get_recent_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics for the recent hours
        """
        result = {
            "hourly": [],
            "by_classification": {
                "intro": 0,
                "promotion": 0,
                "meeting": 0,
                "unknown": 0
            },
            "total": 0,
            "errors": 0,
            "avg_processing_time_ms": 0
        }
        
        now = datetime.now()
        total_time = 0
        total_count = 0
        
        # Collect hourly metrics for the past N hours
        for i in range(hours):
            time_point = now - timedelta(hours=i)
            hourly_key = f"{self.hourly_metrics_prefix}{time_point.strftime('%Y-%m-%d:%H')}"
            
            # Skip if no metrics for this hour
            if not self.redis_client.exists(hourly_key):
                continue
                
            hourly_data = self.redis_client.hgetall(hourly_key)
            hourly_metrics = {k.decode("utf-8"): float(v) for k, v in hourly_data.items()}
            
            # Format for the response
            hourly_entry = {
                "hour": time_point.strftime("%Y-%m-%d %H:00"),
                "total": int(hourly_metrics.get("total", 0)),
                "errors": int(hourly_metrics.get("errors", 0)),
                "processing_time_ms": hourly_metrics.get("processing_time_ms", 0),
                "classifications": {}
            }
            
            # Extract classification counts
            for k, v in hourly_metrics.items():
                if k.startswith("classification:"):
                    classification = k.split(":", 1)[1]
                    hourly_entry["classifications"][classification] = int(v)
                    result["by_classification"][classification] = result["by_classification"].get(classification, 0) + int(v)
            
            result["hourly"].append(hourly_entry)
            result["total"] += int(hourly_metrics.get("total", 0))
            result["errors"] += int(hourly_metrics.get("errors", 0))
            
            # Accumulate for average processing time
            if "total" in hourly_metrics and hourly_metrics["total"] > 0:
                total_time += hourly_metrics.get("processing_time_ms", 0)
                total_count += hourly_metrics["total"]
        
        # Calculate average processing time
        if total_count > 0:
            result["avg_processing_time_ms"] = total_time / total_count
                
        return result
        
    def get_worker_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for all workers
        """
        result = {
            "workers": [],
            "total_workers": 0,
            "busy_workers": 0,
            "available_workers": 0
        }
        
        # Get all worker keys
        worker_keys = self.redis_client.keys(f"{self.worker_metrics_prefix}*")
        result["total_workers"] = len(worker_keys)
        
        # Collect metrics for each worker
        for wkey in worker_keys:
            worker_id = wkey.decode("utf-8").replace(self.worker_metrics_prefix, "")
            worker_data = self.redis_client.hgetall(wkey)
            
            worker_metrics = {k.decode("utf-8"): v.decode("utf-8") for k, v in worker_data.items()}
            is_busy = worker_metrics.get("busy") == "1"
            
            if is_busy:
                result["busy_workers"] += 1
            else:
                result["available_workers"] += 1
                
            worker_entry = {
                "worker_id": worker_id,
                "busy": is_busy,
                "last_updated": worker_metrics.get("last_updated"),
                # Add any other worker-specific metrics here
            }
            
            result["workers"].append(worker_entry)
            
        return result