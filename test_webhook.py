#!/usr/bin/env python3
"""
Webhook testing script for FakeMail email processing system.
This script tests the webhook functionality by sending emails and verifying
that the webhook notification and processing flow works correctly.
"""
import asyncio
import httpx
import json
import time
import sys
import logging
import random
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add scripts to path for importing
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from email_generator import generate_email
except ImportError:
    def generate_email(category=None):
        # Fallback if the generator module is not available
        subject = f"Test email {random.randint(1, 1000)}"
        body = f"This is a test email generated at {datetime.now().isoformat()}"
        return {"subject": subject, "body": body, "category": category or "unknown"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_webhook")

# Configuration
BASE_URL = "http://localhost:8005"  # The base URL of the email processor service
WEBHOOK_URL = f"{BASE_URL}/webhook"  # The webhook URL
SIMULATOR_URL = f"{BASE_URL}/simulator"  # The simulator URL
DEFAULT_TEST_SIZE = 10  # Default number of emails to test

class WebhookTest:
    """Tests webhook functionality for the email processor"""
    
    def __init__(self, base_url=BASE_URL, webhook_url=WEBHOOK_URL, simulator_url=SIMULATOR_URL):
        self.base_url = base_url
        self.webhook_url = webhook_url
        self.simulator_url = simulator_url
        self.start_time = None
        self.http_client = None
        self.initial_history_id = None
        
    async def __aenter__(self):
        """Setup when entering context manager"""
        self.start_time = time.time()
        self.http_client = httpx.AsyncClient()
        logger.info("WebhookTest initialized")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager"""
        if self.http_client:
            await self.http_client.aclose()
        
        duration = time.time() - self.start_time
        logger.info(f"WebhookTest completed in {duration:.2f} seconds")
    
    async def setup_webhook(self):
        """Setup the webhook with the simulator"""
        logger.info("Setting up webhook...")
        
        try:
            # Reset the simulator first
            response = await self.http_client.post(f"{self.simulator_url}/reset")
            response.raise_for_status()
            logger.info("Simulator reset successfully")
            
            # Use the simulator watch endpoint to get the current history_id
            response = await self.http_client.post(f"{self.simulator_url}/watch")
            response.raise_for_status()
            initial_data = response.json()
            self.initial_history_id = initial_data.get("history_id", 0)
            
            logger.info(f"Initial history_id: {self.initial_history_id}")
            
            # Subscribe to webhooks
            response = await self.http_client.post(
                f"{self.simulator_url}/subscribe",
                json={"webhook_url": self.webhook_url}
            )
            response.raise_for_status()
            
            logger.info(f"Successfully subscribed to webhooks at {self.webhook_url}")
            return self.initial_history_id
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during webhook setup: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during webhook setup: {str(e)}")
            raise
    
    async def send_emails(self, count=5, categories=None):
        """Send test emails to trigger the webhook"""
        logger.info(f"Sending {count} test emails...")
        
        email_ids = []
        start_time = time.time()
        
        try:
            for i in range(count):
                # Generate a random email, optionally from specific categories
                category = random.choice(categories) if categories else None
                email = generate_email(category)
                
                # Send the email
                response = await self.http_client.post(
                    f"{self.simulator_url}/send_email",
                    json={"subject": email["subject"], "body": email["body"]}
                )
                response.raise_for_status()
                result = response.json()
                
                email_id = result.get("email_id")
                history_id = result.get("history_id")
                
                if email_id:
                    email_ids.append({
                        "id": email_id,
                        "subject": email["subject"],
                        "category": email.get("category", "unknown"),
                        "history_id": history_id
                    })
                    
                    if (i+1) % 10 == 0 or (i+1) == count:
                        logger.info(f"Sent {i+1}/{count} emails")
                else:
                    logger.warning(f"Failed to get email ID for email: {email['subject']}")
                
                # Small delay between emails
                if i < count - 1:
                    await asyncio.sleep(0.1)
                
            duration = time.time() - start_time
            rate = count / duration
            logger.info(f"Sent {len(email_ids)} emails in {duration:.2f} seconds ({rate:.2f} emails/sec)")
            
            return email_ids
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while sending emails: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while sending emails: {str(e)}")
            raise
    
    async def verify_processing(self, email_ids, timeout=60, poll_interval=1.0):
        """Verify that emails were processed via webhook"""
        logger.info(f"Verifying processing for {len(email_ids)} emails (timeout: {timeout}s)...")
        
        start_time = time.time()
        end_time = start_time + timeout
        
        processed_ids = set()
        results = {}
        
        while time.time() < end_time and len(processed_ids) < len(email_ids):
            # Check remaining emails
            remaining_ids = [e["id"] for e in email_ids if e["id"] not in processed_ids]
            
            # Check each remaining email
            for email_data in [e for e in email_ids if e["id"] in remaining_ids]:
                email_id = email_data["id"]
                
                try:
                    # Validate structured output format
                    response = await self.http_client.get(f"{self.base_url}/results/{email_id}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Verify our structured output schema
                        required_fields = ["email_id", "classification", "confidence", "processed_at"]
                        valid_classifications = ["meeting", "promotion", "intro", "unknown"]
                        
                        # Check required fields
                        if not all(field in result for field in required_fields):
                            logger.warning(f"Response for email {email_id} missing required fields: {[f for f in required_fields if f not in result]}")
                            continue
                            
                        # Validate classification value
                        if result["classification"] not in valid_classifications:
                            logger.warning(f"Invalid classification value: {result['classification']}")
                            continue
                            
                        # Validate confidence is a float between 0 and 1
                        if not (isinstance(result["confidence"], (int, float)) and 0 <= result["confidence"] <= 1):
                            logger.warning(f"Invalid confidence value: {result['confidence']}")
                            continue
                        
                        logger.debug(f"Email {email_id} processed: {result['classification']} (structured output validated)")
                        processed_ids.add(email_id)
                        results[email_id] = {
                            **result,
                            "subject": email_data["subject"],
                            "original_category": email_data.get("category", "unknown")
                        }
                except Exception as e:
                    logger.debug(f"Error checking email {email_id}: {str(e)}")
            
            # Log progress
            if remaining_ids:
                logger.info(f"Processed {len(processed_ids)}/{len(email_ids)} emails, waiting for {len(remaining_ids)} more...")
                await asyncio.sleep(poll_interval)
            else:
                break
        
        # Final status
        duration = time.time() - start_time
        processed_count = len(processed_ids)
        
        if processed_count == len(email_ids):
            logger.info(f"All {processed_count} emails processed successfully in {duration:.2f} seconds")
        else:
            logger.warning(f"Only {processed_count}/{len(email_ids)} emails were processed after {duration:.2f} seconds")
        
        return results
    
    async def check_webhook_status(self):
        """Check the status of the webhook configuration"""
        try:
            response = await self.http_client.get(f"{self.simulator_url}/status")
            response.raise_for_status()
            status_data = response.json()
            
            logger.info(f"Webhook Status:")
            logger.info(f"  Current history_id: {status_data.get('history_id', 'N/A')}")
            logger.info(f"  Registered webhook URLs: {status_data.get('webhook_urls', [])}")
            logger.info(f"  Total emails: {status_data.get('email_count', 0)}")
            logger.info(f"  Status: {status_data.get('status', 'unknown')}")
            
            return status_data
        except Exception as e:
            logger.error(f"Error checking webhook status: {str(e)}")
            return None
    
    def analyze_results(self, results):
        """Analyze the webhook processing results"""
        if not results:
            logger.warning("No results to analyze")
            return {}
        
        # Extract classifications
        classifications = {}
        for email_id, result in results.items():
            classification = result.get("classification")
            original_category = result.get("original_category", "unknown")
            confidence = result.get("confidence", 0)
            
            if classification not in classifications:
                classifications[classification] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "by_original": {},
                    "matched": 0
                }
            
            # Update counts
            classifications[classification]["count"] += 1
            
            # Update confidence
            old_avg = classifications[classification]["avg_confidence"]
            old_count = classifications[classification]["count"] - 1
            new_avg = (old_avg * old_count + confidence) / classifications[classification]["count"]
            classifications[classification]["avg_confidence"] = new_avg
            
            # Update original category stats
            if original_category not in classifications[classification]["by_original"]:
                classifications[classification]["by_original"][original_category] = 0
            classifications[classification]["by_original"][original_category] += 1
            
            # Check if classification matches original category
            if classification == original_category:
                classifications[classification]["matched"] += 1
        
        # Calculate overall statistics
        total_emails = len(results)
        categories = list(classifications.keys())
        total_matched = sum(c["matched"] for c in classifications.values())
        accuracy = total_matched / total_emails if total_emails > 0 else 0
        
        # Calculate average processing time if available
        processing_times = [r.get("processed_at") for r in results.values() if "processed_at" in r]
        if processing_times:
            try:
                # Try to parse timestamps if available
                timestamps = [datetime.fromisoformat(pt) for pt in processing_times]
                min_time = min(timestamps)
                max_time = max(timestamps)
                processing_duration = (max_time - min_time).total_seconds()
            except (ValueError, TypeError):
                processing_duration = None
        else:
            processing_duration = None
        
        return {
            "total_emails": total_emails,
            "classifications": classifications,
            "categories": categories,
            "accuracy": accuracy,
            "processing_duration": processing_duration,
            "emails_per_second": total_emails / processing_duration if processing_duration else None
        }
    
    def print_results_summary(self, results, analysis):
        """Print a summary of the webhook test results"""
        if not results or not analysis:
            logger.warning("No results to display")
            return
        
        print("\n--- Webhook Test Results Summary ---")
        print(f"Total emails processed: {analysis['total_emails']}")
        
        if analysis['processing_duration']:
            print(f"Processing duration: {analysis['processing_duration']:.2f} seconds")
            print(f"Processing rate: {analysis['emails_per_second']:.2f} emails/second")
        
        print(f"Overall accuracy: {analysis['accuracy']*100:.2f}%")
        
        print("\nClassification Breakdown:")
        for category in sorted(analysis['categories']):
            cat_data = analysis['classifications'][category]
            print(f"  {category}: {cat_data['count']} emails ({cat_data['count']/analysis['total_emails']*100:.1f}%)")
            print(f"    Average confidence: {cat_data['avg_confidence']:.4f}")
            print(f"    Accuracy: {cat_data['matched']/cat_data['count']*100:.1f}% ({cat_data['matched']}/{cat_data['count']})")
            print(f"    Original categories: {dict(sorted(cat_data['by_original'].items(), key=lambda x: x[1], reverse=True))}")
        
        print("\nSample Results (Structured Output):")
        for i, (email_id, result) in enumerate(list(results.items())[:5]):  # Show first 5 results
            print(f"  Email {i+1}:")
            print(f"    Subject: {result.get('subject', 'N/A')}")
            print(f"    Original category: {result.get('original_category', 'unknown')}")
            print(f"    Classified as: {result.get('classification', 'N/A')} (confidence: {result.get('confidence', 0):.4f})")
            print(f"    Processor ID: {result.get('processor_id', 'N/A')}")
            print(f"    Processing time: {result.get('processing_time_ms', 0):.2f} ms")
            print(f"    Processed at: {result.get('processed_at', 'N/A')}")
            
        # Display JSON example of first result
        if results:
            first_email_id = list(results.keys())[0]
            first_result = results[first_email_id]
            print("\nStructured Output Example (JSON):")
            structured_output = {
                "email_id": first_result.get("email_id", ""),
                "classification": first_result.get("classification", ""),
                "confidence": first_result.get("confidence", 0),
                "processed_at": first_result.get("processed_at", ""),
                "processor_id": first_result.get("processor_id", ""),
                "processing_time_ms": first_result.get("processing_time_ms", 0)
            }
            # Remove None values
            structured_output = {k: v for k, v in structured_output.items() if v is not None}
            print(json.dumps(structured_output, indent=2))

async def main():
    """Run webhook tests"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test webhook functionality")
    parser.add_argument("--count", "-c", type=int, default=DEFAULT_TEST_SIZE, 
                      help=f"Number of emails to test (default: {DEFAULT_TEST_SIZE})")
    parser.add_argument("--categories", type=str, 
                      help="Comma-separated list of categories to test (e.g. meeting,intro,promotion)")
    parser.add_argument("--timeout", "-t", type=int, default=60,
                      help="Timeout in seconds for webhook processing verification (default: 60)")
    args = parser.parse_args()
    
    # Parse categories if provided
    categories = args.categories.split(",") if args.categories else None
    
    try:
        # Run tests
        logger.info("Starting webhook tests...")
        
        async with WebhookTest() as test:
            # Setup webhook
            await test.setup_webhook()
            
            # Check initial webhook status
            await test.check_webhook_status()
            
            # Send test emails
            email_ids = await test.send_emails(args.count, categories)
            
            if not email_ids:
                logger.error("No emails were sent successfully")
                return 1
            
            # Wait a moment for webhook processing to start
            logger.info("Waiting for webhook processing to begin...")
            await asyncio.sleep(2)
            
            # Verify processing
            results = await test.verify_processing(email_ids, timeout=args.timeout)
            
            # Analyze results
            analysis = test.analyze_results(results)
            
            # Print summary
            test.print_results_summary(results, analysis)
            
            # Check final webhook status
            await test.check_webhook_status()
            
            # Return success if all emails were processed
            return 0 if len(results) == len(email_ids) else 1
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during webhook test: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)