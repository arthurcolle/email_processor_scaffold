#!/usr/bin/env python3
"""
A simple script to test the /results endpoint directly.
"""
import asyncio
import httpx
import json
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_results")

BASE_URL = "http://localhost:8005"

async def test_results_endpoint():
    """Test the results endpoint directly"""
    # Generate a few different email IDs to test
    email_ids = [
        f"a-test-email-{uuid.uuid4()}", # Should be classified as "meeting"
        f"b-test-email-{uuid.uuid4()}", # Should be classified as "promotion"
        f"c-test-email-{uuid.uuid4()}", # Should be classified as "intro"
        f"test-email-{uuid.uuid4()}",   # Generic
    ]
    
    async with httpx.AsyncClient() as client:
        for email_id in email_ids:
            logger.info(f"Testing results for email ID: {email_id}")
            
            # Test with both formats of the URL
            for endpoint in [f"/results/{email_id}"]:
                try:
                    response = await client.get(
                        f"{BASE_URL}{endpoint}",
                        params={"include_content": "true"},
                        timeout=5.0
                    )
                    
                    logger.info(f"Response status for {endpoint}: {response.status_code}")
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"Successfully retrieved classification: {result['classification']}")
                        logger.info(f"Full response: {json.dumps(result, indent=2)}")
                    else:
                        logger.error(f"Failed to get results: {response.status_code} - {response.text}")
                except Exception as e:
                    logger.error(f"Error testing {endpoint}: {str(e)}")

        # Now test with force_generation flag
        logger.info("Testing with force_generation=true")
        for email_id in email_ids[:1]:  # Just test one ID
            try:
                response = await client.get(
                    f"{BASE_URL}/results/{email_id}",
                    params={"include_content": "true", "force_generation": "true"},
                    timeout=5.0
                )
                
                logger.info(f"Response status with force_generation: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully retrieved classification with force_generation: {result['classification']}")
                    logger.info(f"Full response: {json.dumps(result, indent=2)}")
                else:
                    logger.error(f"Failed to get results with force_generation: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error testing with force_generation: {str(e)}")

async def main():
    """Main function"""
    await test_results_endpoint()

if __name__ == "__main__":
    asyncio.run(main())