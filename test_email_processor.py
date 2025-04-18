#!/usr/bin/env python3
"""
End-to-end test tool for the Email Processor application.
This script:
1. Initializes the connection with FakeMail
2. Sends test emails
3. Simulates webhooks
4. Verifies classification results
"""
import asyncio
import httpx
import argparse
import logging
import os
import sys
import json
import time
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_tool")

# Test email templates
TEST_EMAILS = [
    {
        "subject": "Meeting tomorrow at 2pm",
        "body": "Hi team,\n\nLet's meet tomorrow at 2pm in the conference room to discuss the project status.\n\nBest regards,\nJohn",
        "expected_classification": "meeting"
    },
    {
        "subject": "50% Off Summer Sale!",
        "body": "Don't miss our HUGE summer sale! Everything is 50% off this weekend only. Shop now!",
        "expected_classification": "promotion"
    },
    {
        "subject": "Introduction: New Team Member",
        "body": "Hello everyone, I'd like to introduce Alice, who is joining our team as a Senior Developer. Let's welcome her!",
        "expected_classification": "intro"
    },
    {
        "subject": "Project updates",
        "body": "This is a general update about our project. Things are going well. No need for action.",
        "expected_classification": "unknown"
    },
    {
        "subject": "Your calendar invitation: Strategy Session",
        "body": "You've been invited to a meeting on Zoom. Topic: Q3 Strategy Session. Time: Tomorrow 3-4pm.",
        "expected_classification": "meeting"
    },
]

class EmailProcessorTester:
    def __init__(self, base_url: str, fakemail_url: str, webhook_url: str):
        self.base_url = base_url
        self.fakemail_url = fakemail_url
        self.webhook_url = webhook_url
        self.httpx_timeout = 10.0
        self.email_ids = []
        self.current_history_id = 0
        self.results = {}

    async def run_tests(self, email_count: int = 5, delay: float = 0.5):
        """Run the complete test suite"""
        logger.info("Starting email processor test suite")
        
        # Step 1: Initialize with FakeMail
        if not await self.initialize():
            return False
            
        # Step 2: Send test emails
        if not await self.send_test_emails(email_count):
            return False
            
        # Step 3: Trigger webhook processing
        if not await self.trigger_webhook():
            return False
            
        # Step 4: Wait for processing and verify results
        await asyncio.sleep(delay * email_count)  # Allow time for processing
        if not await self.verify_results():
            return False
            
        # Display summary
        self.display_summary()
        
        return True

    async def initialize(self) -> bool:
        """Initialize connection with FakeMail"""
        logger.info("Initializing connection with FakeMail")
        
        try:
            # Call /watch endpoint
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(f"{self.fakemail_url}/watch")
                response.raise_for_status()
                
                try:
                    data = response.json()
                    logger.info(f"Watch response: {data}")
                    if isinstance(data, dict):
                        self.current_history_id = data.get("history_id", 0)
                    else:
                        self.current_history_id = 0
                except:
                    logger.info(f"Watch response is not JSON: {response.text}")
                    try:
                        self.current_history_id = int(response.text.strip())
                    except:
                        self.current_history_id = 0
                
                logger.info(f"Current history_id: {self.current_history_id}")
                
            # Subscribe to webhooks
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(
                    f"{self.fakemail_url}/subscribe",
                    json={"webhook_url": self.webhook_url}
                )
                response.raise_for_status()
                logger.info(f"Successfully subscribed to webhooks: {response.status_code}")
                
            return True
        except Exception as e:
            logger.error(f"Error initializing: {str(e)}")
            return False

    async def send_test_emails(self, count: int) -> bool:
        """Send test emails to FakeMail"""
        logger.info(f"Sending {count} test emails")
        self.email_ids = []
        
        try:
            for i in range(count):
                # Select a template (cycle through or random)
                template = TEST_EMAILS[i % len(TEST_EMAILS)]
                
                # Create unique subject line with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                subject = f"Test {i+1}: {template['subject']} ({timestamp})"
                
                # Send the email
                async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                    response = await client.post(
                        f"{self.fakemail_url}/send_email", 
                        json={
                            "subject": subject,
                            "body": template["body"]
                        }
                    )
                    response.raise_for_status()
                    
                    try:
                        data = response.json()
                        logger.info(f"Send email response: {data}")
                        email_id = data.get("email_id", f"test-email-{i+1}")
                    except:
                        logger.info(f"Send email response is not JSON: {response.text}")
                        email_id = f"test-email-{i+1}"
                    
                    logger.info(f"Sent email {i+1}: {subject} with ID: {email_id}")
                    
                    # Store the email ID and expected classification
                    self.email_ids.append({
                        "id": email_id,
                        "subject": subject,
                        "expected_classification": template["expected_classification"]
                    })
                    
                    # Small delay between sending emails
                    await asyncio.sleep(0.2)
                
            logger.info(f"Successfully sent {count} test emails")
            return True
        except Exception as e:
            logger.error(f"Error sending test emails: {str(e)}")
            return False

    async def trigger_webhook(self) -> bool:
        """Trigger webhook processing"""
        logger.info("Triggering webhook processing")
        
        try:
            # New history ID should be higher than the current one
            new_history_id = self.current_history_id + 1
            
            # Send a webhook notification directly to our app
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(
                    self.webhook_url,
                    json={
                        "history_id": new_history_id,
                        "email_ids": [email["id"] for email in self.email_ids]
                    }
                )
                
                logger.info(f"Webhook response: {response.status_code}")
                try:
                    data = response.json()
                    logger.info(f"Webhook response data: {data}")
                except:
                    logger.info(f"Webhook response is not JSON: {response.text}")
                
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Error triggering webhook: {str(e)}")
            return False

    async def verify_results(self) -> bool:
        """Verify classification results"""
        logger.info("Verifying classification results")
        
        success_count = 0
        correct_classifications = 0
        
        for email in self.email_ids:
            email_id = email["id"]
            expected = email["expected_classification"]
            
            try:
                # Try to get results
                async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                    response = await client.get(
                        f"{self.base_url}/results/{email_id}?include_content=true&force_generation=true"
                    )
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            actual = result.get("classification", "unknown")
                            confidence = result.get("confidence", 0.0)
                            
                            success = True
                            classification_correct = (actual == expected)
                            
                            if classification_correct:
                                correct_classifications += 1
                                logger.info(f"Email {email_id}: CORRECT classification '{actual}' (confidence: {confidence:.2f})")
                            else:
                                logger.warning(f"Email {email_id}: INCORRECT classification '{actual}', expected '{expected}' (confidence: {confidence:.2f})")
                                
                            # Store result for summary
                            self.results[email_id] = {
                                "subject": email["subject"],
                                "expected": expected,
                                "actual": actual,
                                "confidence": confidence,
                                "correct": classification_correct,
                                "success": success
                            }
                            
                            success_count += 1
                        except Exception as e:
                            logger.error(f"Error parsing results for {email_id}: {str(e)}")
                            self.results[email_id] = {
                                "subject": email["subject"],
                                "expected": expected,
                                "actual": "error",
                                "confidence": 0.0,
                                "correct": False,
                                "success": False,
                                "error": str(e)
                            }
                    else:
                        logger.error(f"Failed to get results for {email_id}: {response.status_code} - {response.text}")
                        self.results[email_id] = {
                            "subject": email["subject"],
                            "expected": expected,
                            "actual": "not found",
                            "confidence": 0.0,
                            "correct": False,
                            "success": False,
                            "error": f"HTTP {response.status_code}"
                        }
            except Exception as e:
                logger.error(f"Error verifying results for {email_id}: {str(e)}")
                self.results[email_id] = {
                    "subject": email["subject"],
                    "expected": expected,
                    "actual": "error",
                    "confidence": 0.0,
                    "correct": False,
                    "success": False,
                    "error": str(e)
                }
        
        success_rate = success_count / len(self.email_ids) if self.email_ids else 0
        accuracy = correct_classifications / len(self.email_ids) if self.email_ids else 0
        
        logger.info(f"Results verification complete: {success_count}/{len(self.email_ids)} successful ({success_rate:.0%})")
        logger.info(f"Classification accuracy: {correct_classifications}/{len(self.email_ids)} correct ({accuracy:.0%})")
        
        return success_count > 0

    def display_summary(self):
        """Display test results summary"""
        if not self.results:
            logger.info("No results to display")
            return
            
        success_count = sum(1 for result in self.results.values() if result["success"])
        correct_count = sum(1 for result in self.results.values() if result["correct"])
        total_count = len(self.results)
        
        success_rate = success_count / total_count if total_count else 0
        accuracy = correct_count / total_count if total_count else 0
        
        print("\n" + "="*80)
        print(f"EMAIL PROCESSOR TEST RESULTS SUMMARY")
        print("="*80)
        print(f"Total emails:      {total_count}")
        print(f"Successful:        {success_count}/{total_count} ({success_rate:.0%})")
        print(f"Correct:           {correct_count}/{total_count} ({accuracy:.0%})")
        print("-"*80)
        print(f"{'EMAIL ID':<36} {'EXPECTED':<10} {'ACTUAL':<10} {'CONFIDENCE':<10} {'RESULT'}")
        print("-"*80)
        
        for email_id, result in self.results.items():
            confidence = f"{result['confidence']:.2f}" if isinstance(result['confidence'], (int, float)) else "N/A"
            status = "✓" if result["correct"] else "✗"
            print(f"{email_id:<36} {result['expected']:<10} {result['actual']:<10} {confidence:<10} {status}")
            
        print("="*80)
        print(f"Classification accuracy: {accuracy:.0%}\n")

async def main():
    parser = argparse.ArgumentParser(description="Test the Email Processor application")
    parser.add_argument("--base-url", default="http://localhost:8005", help="Base URL of the Email Processor API")
    parser.add_argument("--fakemail", default="http://localhost:8005/simulator", help="FakeMail API URL")
    parser.add_argument("--webhook", default="http://localhost:8005/webhook", help="Webhook URL for FakeMail to call")
    parser.add_argument("--count", type=int, default=5, help="Number of test emails to send")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay for processing in seconds per email")
    
    args = parser.parse_args()
    
    tester = EmailProcessorTester(args.base_url, args.fakemail, args.webhook)
    success = await tester.run_tests(args.count, args.delay)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))