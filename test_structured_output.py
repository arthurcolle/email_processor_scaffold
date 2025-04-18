#!/usr/bin/env python3
"""
Test script to validate the structured output format for email classification results.
"""
import requests
import json
import sys
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_structured_output")

# Configuration
BASE_URL = "http://localhost:8005"
TEST_EMAIL_IDS = [
    "a1234", # Should classify as meeting
    "b1234", # Should classify as promotion
    "c1234", # Should classify as intro
    "z1234"  # Should classify as unknown
]

def test_structured_output():
    """Test the structured output format for email classifications"""
    logger.info("Testing structured output format for email classification")
    
    results = []
    
    # Test each of our test IDs
    for email_id in TEST_EMAIL_IDS:
        logger.info(f"Testing email ID: {email_id}")
        try:
            # Force generation of results for testing
            response = requests.get(f"{BASE_URL}/results/{email_id}?force_generation=true")
            
            # Check response status
            if response.status_code != 200:
                logger.warning(f"Received status code {response.status_code} for email {email_id}")
                logger.warning(f"Response: {response.text}")
                continue
                
            # Parse JSON response
            result = response.json()
            logger.info(f"Received result: {json.dumps(result, indent=2)}")
            
            # Validate required fields
            required_fields = ["email_id", "classification", "confidence", "processed_at"]
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                continue
                
            # Validate classification value
            valid_classifications = ["meeting", "promotion", "intro", "unknown"]
            if result["classification"] not in valid_classifications:
                logger.error(f"Invalid classification value: {result['classification']}")
                continue
                
            # Validate confidence value
            if not (isinstance(result["confidence"], (int, float)) and 0 <= result["confidence"] <= 1):
                logger.error(f"Invalid confidence value: {result['confidence']}")
                continue
                
            # Validation succeeded
            logger.info(f"Structured output validation succeeded for email {email_id}")
            logger.info(f"Classification: {result['classification']}, Confidence: {result['confidence']}")
            
            # Add to results
            results.append(result)
            
        except requests.RequestException as e:
            logger.error(f"Request error for email {email_id}: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for email {email_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error for email {email_id}: {str(e)}")
    
    return results

def verify_deterministic_classifications(results):
    """Verify that the classification logic is deterministic based on email IDs"""
    
    # Expected classifications based on email ID characters
    expected = {
        "a1234": "meeting",
        "b1234": "promotion",
        "c1234": "intro",
        "z1234": "unknown"
    }
    
    # Check each result
    for result in results:
        email_id = result["email_id"]
        if email_id in expected:
            expected_class = expected[email_id]
            actual_class = result["classification"]
            
            if expected_class == actual_class:
                logger.info(f"✅ Email {email_id} was correctly classified as {actual_class}")
            else:
                logger.error(f"❌ Email {email_id} was expected to be {expected_class} but was {actual_class}")
        else:
            logger.warning(f"No expected classification for email {email_id}")

def main():
    """Main entry point"""
    try:
        # Test structured output
        results = test_structured_output()
        
        if not results:
            logger.error("No valid results received")
            return 1
            
        # Print summary
        print("\n--- Structured Output Test Results ---")
        print(f"Total valid responses: {len(results)}")
        
        # Verify deterministic classifications
        verify_deterministic_classifications(results)
        
        # Choose one result to display as example
        example = results[0]
        print("\nExample Structured Output:")
        print(json.dumps(example, indent=2))
        
        return 0
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())