import redis
import redis.asyncio as aioredis
import os
import time
import logging
import asyncio
from typing import Any, Dict
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager
from app.router import register_routes
from app.services.email_queue import EmailQueue
from app.services.fakemail_service import FakeMailService
from app.services.processor_service import EmailProcessor
from app.services.metrics_service import MetricsService
from app.services.config_service import ConfigService
from app.services.database_service import DatabaseService
from app.database import init_db
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
FAKEMAIL_API_URL = os.getenv("FAKEMAIL_API_URL", "http://localhost:8005/simulator")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
STARTUP_TIME = time.time()

# Initialize Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total count of HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency in seconds", ["method", "endpoint"])
EMAILS_PROCESSED = Counter("emails_processed_total", "Total number of emails processed", ["classification", "status"])
PROCESSING_TIME = Histogram("email_processing_time_seconds", "Time to process an email in seconds")

# Create service instances
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
redis_async_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
email_queue = EmailQueue(redis_client)
fakemail_service = FakeMailService(redis_client, FAKEMAIL_API_URL)
metrics_service = MetricsService(redis_client)
config_service = ConfigService(redis_client)

# Initialize database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./emails.db")
db_service = DatabaseService(DATABASE_URL)

# Import event service
from app.services.event_service import EventService
event_service = EventService(redis_async_client)

email_processor = EmailProcessor(
    fakemail_service,
    email_queue,
    metrics_service,
    config_service
)

# Lifespan context manager for setup and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize services
    logger.info("Starting application services")
    
    # Initialize database tables
    try:
        db_service.create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
    
    # Start email processor
    try:
        await email_processor.start()
        logger.info("Email processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start email processor: {str(e)}")
        
    # Start event service for WebSockets
    try:
        await event_service.start()
        logger.info("Event service started successfully")
    except Exception as e:
        logger.error(f"Failed to start event service: {str(e)}")
        
    yield
    
    # Shutdown: cleanup resources
    logger.info("Shutting down application services")
    
    # Stop email processor
    try:
        await email_processor.stop()
        logger.info("Email processor stopped")
    except Exception as e:
        logger.error(f"Error stopping email processor: {str(e)}")
        
    # Stop event service
    try:
        await event_service.stop()
        logger.info("Event service stopped")
    except Exception as e:
        logger.error(f"Error stopping event service: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title="Enterprise Email Processor",
    description="""
    Advanced email processing system for FakeMail with structured output support.
    
    ## Structured Outputs
    
    The API uses structured output schemas to ensure consistent, well-formatted responses
    that follow predefined JSON schemas. The email classification endpoint returns data that 
    strictly adheres to the EmailClassificationSchema, providing a reliable contract for API consumers.
    
    Example classification response:
    ```json
    {
        "email_id": "abc123",
        "classification": "meeting",
        "confidence": 0.93,
        "processed_at": "2025-04-17T14:32:21.435Z",
        "processor_id": "processor-1",
        "processing_time_ms": 125.4
    }
    ```
    """,
    version=APP_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Request metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Start timer
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    endpoint = request.url.path
    method = request.method
    status = response.status_code
    
    # Skip metrics collection for the metrics endpoint itself
    if endpoint != "/metrics":
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
    
    return response

# Register all routes from the routes directory
register_routes(app)

# Direct import of critical routes to ensure they are registered
try:
    # Explicitly register results route
    from app.routes.results_routes import router as results_router
    app.include_router(results_router)
    logger.info("Explicitly registered routes from app.routes.results_routes")
    
    # Explicitly register webhook route
    from app.routes.webhook_routes import router as webhook_router
    app.include_router(webhook_router)
    logger.info("Explicitly registered routes from app.routes.webhook_routes")
    
    # Verify route registration
    webhook_endpoint = [route for route in app.routes if route.path == "/webhook" and "POST" in route.methods]
    results_endpoint = [route for route in app.routes if "/results/" in route.path and "GET" in route.methods]
    
    if webhook_endpoint:
        logger.info(f"Webhook route successfully registered: {webhook_endpoint[0]}")
    else:
        logger.error("Webhook route was not registered correctly")
        
    if results_endpoint:
        logger.info(f"Results route successfully registered: {results_endpoint[0]}")
    else:
        logger.error("Results route was not registered correctly")
        
except ImportError as e:
    logger.error(f"Failed to import critical routes: {str(e)}")

@app.get("/")
async def root():
    return {
        "name": "Enterprise Email Processor",
        "version": APP_VERSION,
        "status": "running",
        "uptime_seconds": int(time.time() - STARTUP_TIME),
        "fakemail_api_url": FAKEMAIL_API_URL
    }

@app.get("/health")
async def health_check():
    # Check if Redis is available
    try:
        redis_status = redis_client.ping()
    except Exception:
        redis_status = False
        
    # Check processor status
    processor_status = email_processor.running
        
    # Get basic metrics
    stats = metrics_service.get_stats()
    
    return {
        "status": "healthy" if redis_status and processor_status else "unhealthy",
        "components": {
            "redis": "connected" if redis_status else "disconnected",
            "processor": "running" if processor_status else "stopped"
        },
        "metrics": {
            "total_emails_processed": stats.total_emails_processed,
            "success_rate": stats.success_rate,
            "avg_processing_time_ms": stats.avg_processing_time_ms,
            "uptime_seconds": stats.uptime_seconds
        }
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8005, reload=True)