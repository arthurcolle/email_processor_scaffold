# Enterprise Email Processor for FakeMail

A production-grade email processing system that integrates with FakeMail's webhook system to process and classify emails with high throughput, resilience, and observability.

## Overview

The Enterprise Email Processor is designed to process and classify emails from the FakeMail service. It features:

- Real-time email classification and processing
- Redis-backed queue system for reliability
- Webhook integration for instant notifications
- RESTful API with structured JSON outputs
- Comprehensive metrics and analytics
- Containerized deployment with Docker

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+

### Running with Docker

1. Start the service:

```bash
docker-compose up -d
```

2. Initialize the application with FakeMail:

```bash
# Initialize with default settings
python initialize_fakemail.py

# With custom settings
python initialize_fakemail.py --fakemail "http://fakemail-api-url" --webhook "http://your-public-webhook-url"

# Test the endpoints after initialization
python initialize_fakemail.py --test
```

3. The service will be available at http://localhost:8005

4. To expose your webhook to the public internet (for FakeMail to call), you can use ngrok:

```bash
# Install ngrok if you haven't already
# Then run:
ngrok http 8005

# Use the generated URL for your webhook when initializing:
python initialize_fakemail.py --webhook "https://your-ngrok-url/webhook"
```

### Environment Variables

Configure the application using these environment variables:

- `REDIS_HOST` - Redis host (default: "redis")
- `REDIS_PORT` - Redis port (default: 6379)
- `FAKEMAIL_API_URL` - URL for the FakeMail API (default: "http://localhost:8005/simulator")
- `PROCESSOR_WORKER_COUNT` - Number of processing workers (default: 3)
- `PROCESSOR_MAX_RETRIES` - Maximum number of processing retries (default: 3)
- `PROCESSOR_RETRY_DELAY_MS` - Delay between retries in ms (default: 1000)
- `LOG_LEVEL` - Logging level (default: "INFO")

## Architecture

The system consists of several components:

1. **API Layer** - FastAPI-based RESTful API
2. **FakeMail Integration** - Webhook receiving and email fetching
3. **Email Processor** - Classification and processing workers
4. **Redis Queue** - Reliable message queue with priority support
5. **Storage** - SQLite database for persistent storage
6. **Analytics** - Metrics and performance tracking
7. **Real-time Updates** - WebSocket-based real-time dashboard updates

### Process Flow

1. Webhook notification received from FakeMail with a history_id
2. System fetches new emails since the last processed history_id
3. Emails are queued for processing and WebSocket clients are notified
4. Worker processes classify emails 
5. Results are stored and made available via API
6. WebSocket clients are notified of processed emails in real-time
7. Dashboard updates automatically to reflect the latest statistics

## API Endpoints

### Core Endpoints

- `POST /webhook` - Webhook receiver for FakeMail notifications
- `GET /results/{email_id}` - Get classification result for an email with structured JSON output

#### Structured Output Example:

```json
{
  "email_id": "e12345",
  "classification": "meeting",
  "confidence": 0.92,
  "processed_at": "2025-04-17T15:23:45.122Z",
  "processor_id": "processor-a",
  "processing_time_ms": 234.5
}
```

### System Status

- `GET /health` - Check system health
- `GET /metrics` - Get system metrics (Prometheus format)
- `GET /stats` - Get processing statistics
- `WS /ws/emails/{client_id}` - WebSocket endpoint for real-time updates

### Email Management

- `GET /emails` - List emails with filtering options
- `GET /emails/{email_id}` - Get email details
- `POST /simulator/send_email` - Send a test email (simulation mode)

### Real-time Updates

The system provides real-time updates to the dashboard using WebSockets:

- **Email Processing Events**: When emails are processed, the dashboard updates automatically
- **Stats Updates**: System statistics are updated in real-time as emails are processed
- **Queue Updates**: New emails being added to the queue trigger dashboard notifications
- **Connection Management**: Automatic reconnection if the WebSocket connection is lost

Example WebSocket message:
```json
{
  "type": "email_event",
  "event": "process",
  "data": {
    "email_id": "abc123",
    "classification": "meeting",
    "confidence": 0.92,
    "processed_at": "2025-04-17T15:23:45.122Z"
  },
  "timestamp": "2025-04-17T15:23:45.125Z"
}
```

## Testing

### Running Tests

```bash
# Run the end-to-end test flow
python test_flow.py

# Test with a large batch of emails
python test_large_batch.py

# Test the webhook functionality
python test_webhook.py
```

### Test Flow

The `test_flow.py` script performs an end-to-end test of the email processing system:

1. Sets up the system by initializing the webhook
2. Sends test emails to the simulator
3. Waits for email processing to complete
4. Checks if all emails were processed successfully
5. Provides a summary of results

## Development Guide

### Project Structure

```
/
├── app/                        # Main application code
│   ├── models/                 # Data models
│   │   ├── database.py         # Database model definitions
│   │   ├── email.py            # Email data models
│   │   └── fakemail.py         # FakeMail integration models
│   ├── routes/                 # API route definitions
│   │   ├── api_routes.py       # Generic API endpoints
│   │   ├── email_routes.py     # Email management endpoints
│   │   ├── results_routes.py   # Classification results
│   │   ├── simulator_routes.py # FakeMail simulator
│   │   └── webhook_routes.py   # Webhook handler
│   ├── services/               # Core business logic
│   │   ├── config_service.py   # Configuration management
│   │   ├── email_queue.py      # Email queuing system
│   │   ├── email_service.py    # Email operations
│   │   ├── event_service.py    # Event handling 
│   │   ├── fakemail_service.py # FakeMail integration
│   │   ├── metrics_service.py  # Metrics collection
│   │   └── processor_service.py # Email processor
│   ├── templates/              # HTML templates for UI
│   ├── database.py             # Database configuration
│   ├── main.py                 # Application entry point
│   └── router.py               # Route registration
├── scripts/                    # Utility scripts
├── test_flow.py                # End-to-end test script
├── test_large_batch.py         # Batch processing test
├── test_webhook.py             # Webhook testing
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
└── requirements.txt            # Python dependencies
```

### Email Classification Types

The system supports these classification types:

- `meeting` - Meeting invitations, calendar events
- `promotion` - Marketing, promotions, offers
- `intro` - Introductions, greetings, welcome emails
- `unknown` - Emails that don't match other categories

## Troubleshooting

### Common Issues

#### Redis Connection Problems

If you're experiencing Redis connection issues:

1. Verify Redis is running:
   ```bash
   docker ps | grep redis
   ```

2. Check Redis connection inside the container:
   ```bash
   docker exec email_processor_scaffold-app-1 python -c "import redis; r = redis.Redis(host='redis', port=6379); print(f'Redis ping: {r.ping()}')"
   ```

3. Verify Redis host in the app:
   ```bash
   docker exec email_processor_scaffold-app-1 env | grep REDIS
   ```

#### Email Processing Issues

If emails aren't being processed correctly:

1. Check app logs:
   ```bash
   docker logs email_processor_scaffold-app-1
   ```

2. Check system health:
   ```bash
   curl http://localhost:8005/health
   ```

3. Verify that webhooks are being received:
   - Check the logs for "Received webhook" messages
   - Ensure webhook URLs are correctly configured

4. Test the results endpoint directly:
   ```bash
   curl http://localhost:8005/results/test-email-id
   ```

### Diagnostic Commands

Check if application routes are registered:
```bash
curl http://localhost:8005/
```

Get Redis connection status:
```bash
curl http://localhost:8005/health | grep redis
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Acknowledgments

- FastAPI for the robust API framework
- Redis for reliable message queuing
- Docker for containerization