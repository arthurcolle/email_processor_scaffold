import asyncio
import logging
from typing import Dict, Any
import json
import redis.asyncio as aioredis
from app.routes.websocket_routes import broadcast_email_event

logger = logging.getLogger(__name__)

class EventService:
    """Service for managing real-time events and notifications"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.pubsub = None
        self.running = False
        self.listener_task = None
    
    async def start(self):
        """Start the event listener service"""
        if self.running:
            return
            
        try:
            self.pubsub = self.redis.pubsub()
            # Subscribe to email-related channels
            await self.pubsub.subscribe(
                "email:new",
                "email:update",
                "email:delete",
                "email:process"
            )
            
            self.running = True
            self.listener_task = asyncio.create_task(self._listener())
            logger.info("Event service started")
        except Exception as e:
            logger.error(f"Failed to start event service: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the event listener service"""
        if not self.running:
            return
            
        self.running = False
        
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
            self.listener_task = None
        
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
            self.pubsub = None
            
        logger.info("Event service stopped")
    
    async def _listener(self):
        """Background task to listen for Redis events and forward them to WebSockets"""
        try:
            while self.running:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    try:
                        channel = message["channel"].decode("utf-8")
                        data = message["data"]
                        
                        # Parse the message data
                        if isinstance(data, bytes):
                            data = json.loads(data.decode("utf-8"))
                        
                        # Handle different event types
                        if channel == "email:new":
                            await broadcast_email_event("new", data)
                        elif channel == "email:update":
                            await broadcast_email_event("update", data)
                        elif channel == "email:delete":
                            await broadcast_email_event("delete", data)
                        elif channel == "email:process":
                            # Send the process event
                            await broadcast_email_event("process", data)
                            
                            # Also send stats update after a short delay to ensure metrics are updated
                            asyncio.create_task(self._send_stats_update(0.5))
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {str(e)}")
                
                # Small sleep to prevent CPU hogging
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Event listener task cancelled")
        except Exception as e:
            logger.error(f"Event listener error: {str(e)}")
            self.running = False
    
    async def _send_stats_update(self, delay: float = 0.5):
        """Send a stats update event after a delay to ensure metrics are updated"""
        # Wait a moment for metrics to be updated
        await asyncio.sleep(delay)
        
        try:
            # Import MetricsService here to avoid circular imports
            from app.services.metrics_service import MetricsService
            metrics_service = MetricsService(self.redis)
            
            # Get updated stats
            stats = metrics_service.get_stats()
            
            # Convert stats object to dictionary
            stats_dict = stats.model_dump() if hasattr(stats, 'model_dump') else stats.__dict__
            
            # Remove any non-JSON serializable objects
            stats_dict = {k: v for k, v in stats_dict.items() if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
            
            # Create event data with stats
            event_data = {
                "type": "stats_update",
                "data": stats_dict,
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to all clients
            await broadcast_email_event("stats", event_data)
            logger.debug("Sent stats update event")
        except Exception as e:
            logger.error(f"Error sending stats update: {str(e)}")
    
    async def publish_event(self, channel: str, data: Dict[str, Any]):
        """Publish an event to a Redis channel"""
        try:
            await self.redis.publish(channel, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Error publishing event to {channel}: {str(e)}")
            return False