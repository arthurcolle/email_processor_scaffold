from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import List, Dict
import json
import asyncio
import logging
from datetime import datetime
from app.services.email_service import EmailService
import redis

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)

# Connection manager to handle WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        logger.info(f"Client {client_id} connected, total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            if websocket in self.active_connections[client_id]:
                self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected, remaining connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
    
    async def broadcast(self, message: dict, client_id: str = None):
        if client_id and client_id in self.active_connections:
            # Send to specific client only
            disconnected = []
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Clean up any disconnected clients
            for conn in disconnected:
                try:
                    await self.disconnect(conn, client_id)
                except Exception:
                    pass
        else:
            # Broadcast to all clients
            for client_id, connections in list(self.active_connections.items()):
                disconnected = []
                for connection in connections:
                    try:
                        await connection.send_json(message)
                    except Exception:
                        disconnected.append(connection)
                
                # Clean up any disconnected clients
                for conn in disconnected:
                    try:
                        await self.disconnect(conn, client_id)
                    except Exception:
                        pass

# Create a connection manager instance
manager = ConnectionManager()

# Redis client dependency
def get_redis_client():
    return redis.Redis(host="redis", port=6379, db=0)

# Email service dependency
def get_email_service(redis_client: redis.Redis = Depends(get_redis_client)):
    return EmailService(redis_client)

# WebSocket endpoint for real-time email updates
@router.websocket("/ws/emails/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            try:
                # Parse the message as JSON
                message = json.loads(data)
                command = message.get("command")
                
                # Handle different commands
                if command == "ping":
                    await manager.send_personal_message({"type": "pong", "timestamp": datetime.now().isoformat()}, websocket)
                elif command == "subscribe":
                    # Client is subscribing to a topic
                    topic = message.get("topic")
                    if topic:
                        await manager.send_personal_message(
                            {"type": "subscription", "status": "success", "topic": topic},
                            websocket
                        )
                        
                        # If client is subscribing to email events, send current stats
                        if topic == "email:all":
                            # Import the metrics service
                            from app.services.metrics_service import MetricsService
                            metrics_service = MetricsService(get_redis_client())
                            
                            # Get current stats
                            stats = metrics_service.get_stats()
                            
                            # Convert stats to dict
                            stats_dict = stats.model_dump() if hasattr(stats, 'model_dump') else stats.__dict__
                            
                            # Remove any non-JSON serializable objects
                            stats_dict = {k: v for k, v in stats_dict.items() 
                                        if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
                            
                            # Send stats to the client
                            await manager.send_personal_message({
                                "type": "email_event",
                                "event": "stats",
                                "data": stats_dict,
                                "timestamp": datetime.now().isoformat()
                            }, websocket)
                elif command == "fetch_stats":
                    # Client is requesting latest stats
                    from app.services.metrics_service import MetricsService
                    metrics_service = MetricsService(get_redis_client())
                    
                    # Get current stats
                    stats = metrics_service.get_stats()
                    
                    # Convert stats to dict
                    stats_dict = stats.model_dump() if hasattr(stats, 'model_dump') else stats.__dict__
                    
                    # Remove any non-JSON serializable objects
                    stats_dict = {k: v for k, v in stats_dict.items() 
                                if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
                    
                    # Send stats to the client
                    await manager.send_personal_message({
                        "type": "email_event",
                        "event": "stats",
                        "data": stats_dict,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            except json.JSONDecodeError:
                await manager.send_personal_message({"type": "error", "message": "Invalid JSON"}, websocket)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await manager.send_personal_message({"type": "error", "message": str(e)}, websocket)
    except WebSocketDisconnect:
        await manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await manager.disconnect(websocket, client_id)
        except Exception:
            pass

# Utility function to broadcast email events
async def broadcast_email_event(event_type: str, email_data: dict, client_id: str = None):
    """Broadcast an email event to connected clients"""
    message = {
        "type": "email_event",
        "event": event_type,
        "data": email_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message, client_id)

# Export the broadcast function so other services can use it
__all__ = ["broadcast_email_event"]