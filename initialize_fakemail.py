#!/usr/bin/env python3
"""
Script to initialize the application with FakeMail API.
This script:
1. Calls the /watch endpoint to get the current history_id
2. Sets up a webhook subscription to our application
3. Ensures Redis connections are working
"""
import asyncio
import httpx
import argparse
import logging
import os
import sys
import json
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("initialize")

class FakeMailInitializer:
    def __init__(self, fakemail_url, webhook_url, redis_host="localhost", redis_port=6379):
        self.fakemail_url = fakemail_url
        self.webhook_url = webhook_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.httpx_timeout = 30.0

    async def initialize(self):
        """Main initialization sequence"""
        # Step 1: Connect to Redis
        if not await self.connect_redis():
            logger.error("Failed to connect to Redis")
            return False

        # Step 2: Call /watch endpoint
        history_id = await self.watch()
        if history_id is None:
            logger.error("Failed to get history_id from /watch endpoint")
            return False

        # Step 3: Subscribe to FakeMail webhooks
        if not await self.subscribe():
            logger.error("Failed to subscribe to FakeMail webhooks")
            return False

        logger.info("Initialization completed successfully!")
        return True

    async def connect_redis(self):
        """Establish Redis connection"""
        logger.info(f"Connecting to Redis at {self.redis_host}:{self.redis_port}")
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=0)
            ping_result = self.redis_client.ping()
            logger.info(f"Redis ping result: {ping_result}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Redis: {str(e)}")
            return False

    async def watch(self):
        """Call FakeMail's /watch endpoint"""
        logger.info(f"Calling {self.fakemail_url}/watch")
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(f"{self.fakemail_url}/watch")
                response.raise_for_status()
                
                # Parse response
                try:
                    data = response.json()
                    logger.info(f"Watch response: {data}")
                    
                    # Extract history_id
                    if isinstance(data, dict):
                        history_id = data.get("history_id", 0)
                    else:
                        # Try to parse directly
                        history_id = int(response.text.strip())
                    
                    logger.info(f"Current history_id: {history_id}")
                    
                    # Store in Redis
                    try:
                        self.redis_client.set("fakemail:current_history_id", history_id)
                        logger.info(f"Stored history_id {history_id} in Redis")
                    except Exception as e:
                        logger.error(f"Error storing history_id in Redis: {str(e)}")
                    
                    return history_id
                except (json.JSONDecodeError, ValueError):
                    logger.error(f"Invalid response format: {response.text}")
                    return None
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error calling /watch: {str(e)}")
            return None

    async def subscribe(self):
        """Subscribe to FakeMail's webhook notifications"""
        logger.info(f"Subscribing to FakeMail webhooks with URL: {self.webhook_url}")
        
        try:
            payload = {"webhook_url": self.webhook_url}
            
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(
                    f"{self.fakemail_url}/subscribe",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Subscribe response: {response.status_code} - {response.text}")
                
                # Store webhook URL in Redis
                try:
                    self.redis_client.set("fakemail:webhook_url", self.webhook_url)
                    logger.info(f"Stored webhook URL {self.webhook_url} in Redis")
                except Exception as e:
                    logger.error(f"Error storing webhook URL in Redis: {str(e)}")
                
                return True
        except httpx.HTTPError as e:
            logger.error(f"HTTP error subscribing to webhooks: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error subscribing to webhooks: {str(e)}")
            return False

async def test_webhook_endpoint(webhook_url):
    """Test the webhook endpoint with a sample payload"""
    logger.info(f"Testing webhook endpoint: {webhook_url}")
    
    try:
        payload = {"history_id": 1, "email_ids": ["test-email-1", "test-email-2"]}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload
            )
            
            logger.info(f"Webhook test response: {response.status_code}")
            logger.info(f"Response body: {response.text}")
            
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing webhook: {str(e)}")
        return False

async def test_results_endpoint(webhook_url, email_id="test-email-1"):
    """Test the results endpoint"""
    base_url = webhook_url.split("/webhook")[0]
    results_url = f"{base_url}/results/{email_id}"
    
    logger.info(f"Testing results endpoint: {results_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First with no special params
            response = await client.get(results_url)
            logger.info(f"Results response (basic): {response.status_code}")
            
            # Then with force_generation=true
            response_force = await client.get(f"{results_url}?force_generation=true")
            logger.info(f"Results response (force): {response_force.status_code}")
            
            # Then with content included
            response_content = await client.get(f"{results_url}?include_content=true&force_generation=true")
            logger.info(f"Results response (content): {response_content.status_code}")
            
            if response_content.status_code == 200:
                result = response_content.json()
                logger.info(f"Results content: {json.dumps(result, indent=2)}")
            
            return response_force.status_code == 200
    except Exception as e:
        logger.error(f"Error testing results endpoint: {str(e)}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Initialize the application with FakeMail API")
    parser.add_argument("--fakemail", default="http://localhost:8005/simulator", help="FakeMail API URL")
    parser.add_argument("--webhook", default="http://localhost:8005/webhook", help="Webhook URL for FakeMail to call")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--test", action="store_true", help="Test the webhook and results endpoints")
    
    args = parser.parse_args()
    
    # Initialize FakeMail
    initializer = FakeMailInitializer(
        args.fakemail,
        args.webhook,
        args.redis_host,
        args.redis_port
    )
    
    success = await initializer.initialize()
    
    # Test endpoints if requested
    if args.test and success:
        logger.info("Testing webhook and results endpoints...")
        webhook_test = await test_webhook_endpoint(args.webhook)
        results_test = await test_results_endpoint(args.webhook)
        
        if webhook_test and results_test:
            logger.info("All tests passed!")
            return 0
        else:
            logger.error("Some tests failed")
            return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))