#!/usr/bin/env python3
"""
Simple test script for the /results endpoint.
"""
import asyncio
import httpx
import argparse
import logging
import uuid
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_results")

async def test_results_endpoint(base_url, email_id=None, force=False, include_content=False):
    """Test the results endpoint with various options"""
    if not email_id:
        # Generate test email IDs with different first characters
        email_ids = [
            f"a-test-{uuid.uuid4()}", # Should be classified as "meeting"
            f"b-test-{uuid.uuid4()}", # Should be classified as "promotion"
            f"c-test-{uuid.uuid4()}", # Should be classified as "intro"
            f"test-{uuid.uuid4()}",   # Generic
        ]
    else:
        email_ids = [email_id]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for email_id in email_ids:
            logger.info(f"Testing results for email ID: {email_id}")
            
            # Build URL with parameters
            params = {}
            if force:
                params["force_generation"] = "true"
            if include_content:
                params["include_content"] = "true"
                
            url = f"{base_url}/results/{email_id}"
            
            # Make the request
            try:
                response = await client.get(url, params=params)
                
                logger.info(f"Response status: {response.status_code}")
                if response.status_code == 200:
                    try:
                        result = response.json()
                        logger.info(f"Classification result: {result.get('classification', 'unknown')}")
                        logger.info(f"Full response: {json.dumps(result, indent=2)}")
                    except Exception as e:
                        logger.error(f"Error parsing response: {str(e)}")
                        logger.info(f"Raw response: {response.text}")
                else:
                    logger.error(f"Error response: {response.text}")
            except Exception as e:
                logger.error(f"Request error: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description="Test the results endpoint")
    parser.add_argument("--base-url", default="http://localhost:8005", help="Base URL of the API")
    parser.add_argument("--email-id", help="Specific email ID to test")
    parser.add_argument("--force", action="store_true", help="Use force_generation=true")
    parser.add_argument("--content", action="store_true", help="Include content in response")
    
    args = parser.parse_args()
    await test_results_endpoint(args.base_url, args.email_id, args.force, args.content)

if __name__ == "__main__":
    asyncio.run(main())