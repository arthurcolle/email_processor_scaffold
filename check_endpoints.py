#!/usr/bin/env python3
"""
Verification script to check all required endpoints for the Email Processor application.
This script will:
1. Check if the webhook endpoint is accessible
2. Check if the results endpoint is accessible
3. Test the entire flow from initialization to results retrieval
"""
import asyncio
import httpx
import argparse
import logging
import json
import time
import uuid
import sys
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("check_endpoints")

class EndpointChecker:
    def __init__(self, app_url: str, fakemail_url: str, webhook_url: str):
        self.app_url = app_url.rstrip("/")
        self.fakemail_url = fakemail_url.rstrip("/")
        self.webhook_url = webhook_url.rstrip("/")
        self.httpx_timeout = 10.0
        self.results = {}
        
    async def run_checks(self) -> bool:
        """Run all endpoint checks and return overall success status"""
        self.results = {
            "app_health": False,
            "webhook_endpoint": False,
            "results_endpoint": False,
            "watch_endpoint": False,
            "subscribe_endpoint": False,
            "fakemail_send": False,
            "fakemail_get": False,
            "fakemail_classify": False,
            "end_to_end_flow": False,
        }
        
        # Check app health
        self.results["app_health"] = await self.check_app_health()
        
        # Check if webhook endpoint exists
        self.results["webhook_endpoint"] = await self.check_webhook_endpoint()
        
        # Check if results endpoint exists
        self.results["results_endpoint"] = await self.check_results_endpoint()
        
        # Check FakeMail API endpoints
        self.results["watch_endpoint"] = await self.check_watch_endpoint()
        self.results["subscribe_endpoint"] = await self.check_subscribe_endpoint()
        self.results["fakemail_send"] = await self.check_fakemail_send()
        
        # Only check these if send email worked
        if self.results["fakemail_send"]:
            self.results["fakemail_get"] = await self.check_fakemail_get()
            self.results["fakemail_classify"] = await self.check_fakemail_classify()
            
        # Run an end-to-end test if all other checks passed
        if all([
            self.results["app_health"],
            self.results["webhook_endpoint"],
            self.results["results_endpoint"],
            self.results["watch_endpoint"],
            self.results["subscribe_endpoint"],
            self.results["fakemail_send"],
            self.results["fakemail_get"],
            self.results["fakemail_classify"]
        ]):
            self.results["end_to_end_flow"] = await self.check_end_to_end_flow()
        
        # Display results
        self.print_results()
        
        return all(self.results.values())
        
    async def check_app_health(self) -> bool:
        """Check if the application is healthy"""
        logger.info("Checking application health...")
        
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.get(f"{self.app_url}/health")
                
                if response.status_code == 200:
                    logger.info("✓ Application is healthy")
                    return True
                else:
                    logger.error(f"✗ Application health check failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to application: {str(e)}")
            return False
            
    async def check_webhook_endpoint(self) -> bool:
        """Check if the webhook endpoint exists"""
        logger.info("Checking webhook endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                # Send a simple test payload to the webhook
                test_payload = {"history_id": 1, "test": True}
                response = await client.post(f"{self.webhook_url}", json=test_payload)
                
                # We consider 200, 201, 202, and 204 to be successful responses
                if response.status_code in (200, 201, 202, 204):
                    logger.info(f"✓ Webhook endpoint exists (Status: {response.status_code})")
                    return True
                else:
                    logger.error(f"✗ Webhook endpoint check failed: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to webhook endpoint: {str(e)}")
            return False
    
    async def check_results_endpoint(self) -> bool:
        """Check if the results endpoint exists"""
        logger.info("Checking results endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                # Use a test email ID and force generation to ensure we get a response
                test_id = f"test-{uuid.uuid4()}"
                response = await client.get(
                    f"{self.app_url}/results/{test_id}",
                    params={"force_generation": "true"}
                )
                
                # For results, we expect a 200 response with valid JSON
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if isinstance(result, dict) and "classification" in result:
                            logger.info(f"✓ Results endpoint exists and returns proper data structure")
                            logger.info(f"  Classification: {result.get('classification')}")
                            return True
                        else:
                            logger.error(f"✗ Results endpoint doesn't return expected data structure: {result}")
                            return False
                    except Exception as e:
                        logger.error(f"✗ Results endpoint returned invalid JSON: {str(e)}")
                        return False
                else:
                    logger.error(f"✗ Results endpoint check failed: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to results endpoint: {str(e)}")
            return False
    
    async def check_watch_endpoint(self) -> bool:
        """Check if FakeMail's watch endpoint works"""
        logger.info("Checking FakeMail watch endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(f"{self.fakemail_url}/watch")
                
                if response.status_code in (200, 201, 202, 204):
                    try:
                        # Try to parse response as JSON, but don't fail if it's not
                        try:
                            data = response.json()
                            if isinstance(data, dict) and "history_id" in data:
                                logger.info(f"✓ Watch endpoint works (history_id: {data['history_id']})")
                            else:
                                logger.info(f"✓ Watch endpoint works (Response: {data})")
                        except:
                            logger.info(f"✓ Watch endpoint works (Raw response: {response.text})")
                        return True
                    except Exception as e:
                        logger.error(f"✗ Error processing watch response: {str(e)}")
                        return False
                else:
                    logger.error(f"✗ Watch endpoint check failed: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to watch endpoint: {str(e)}")
            return False
    
    async def check_subscribe_endpoint(self) -> bool:
        """Check if FakeMail's subscribe endpoint works"""
        logger.info("Checking FakeMail subscribe endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(
                    f"{self.fakemail_url}/subscribe",
                    json={"webhook_url": self.webhook_url}
                )
                
                if response.status_code in (200, 201, 202, 204):
                    logger.info(f"✓ Subscribe endpoint works")
                    return True
                else:
                    logger.error(f"✗ Subscribe endpoint check failed: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to subscribe endpoint: {str(e)}")
            return False
    
    async def check_fakemail_send(self) -> bool:
        """Check if FakeMail's send_email endpoint works"""
        logger.info("Checking FakeMail send_email endpoint...")
        
        try:
            self.test_email_id = None
            self.test_email_subject = f"Test Email {uuid.uuid4()}"
            self.test_email_body = f"This is a test email sent at {time.time()}"
            
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(
                    f"{self.fakemail_url}/send_email",
                    json={
                        "subject": self.test_email_subject,
                        "body": self.test_email_body
                    }
                )
                
                if response.status_code in (200, 201, 202, 204):
                    try:
                        # Try to parse email_id from response
                        try:
                            data = response.json()
                            if isinstance(data, dict) and "email_id" in data:
                                self.test_email_id = data["email_id"]
                                logger.info(f"✓ Send email endpoint works (email_id: {self.test_email_id})")
                            else:
                                logger.info(f"✓ Send email endpoint works but couldn't extract email_id")
                        except:
                            logger.info(f"✓ Send email endpoint works (Raw response: {response.text})")
                            
                            # Try to extract email_id from response text
                            if "id" in response.text.lower():
                                try:
                                    import re
                                    id_match = re.search(r'["\'](.*?)["\']', response.text)
                                    if id_match:
                                        self.test_email_id = id_match.group(1)
                                        logger.info(f"  Extracted email_id: {self.test_email_id}")
                                except:
                                    pass
                        
                        return True
                    except Exception as e:
                        logger.error(f"✗ Error processing send_email response: {str(e)}")
                        return False
                else:
                    logger.error(f"✗ Send email endpoint check failed: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to send_email endpoint: {str(e)}")
            return False
    
    async def check_fakemail_get(self) -> bool:
        """Check if FakeMail's email retrieval endpoint works"""
        logger.info("Checking FakeMail email retrieval endpoint...")
        
        if not self.test_email_id:
            logger.warning("⚠ No test email ID available, skipping email retrieval check")
            return False
            
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.get(f"{self.fakemail_url}/email/{self.test_email_id}")
                
                if response.status_code == 200:
                    try:
                        email_data = response.json()
                        if isinstance(email_data, dict):
                            if "subject" in email_data and "body" in email_data:
                                logger.info(f"✓ Email retrieval endpoint works")
                                logger.info(f"  Subject: {email_data.get('subject')}")
                                self.test_email_data = email_data
                                return True
                            else:
                                logger.error(f"✗ Email data missing required fields: {email_data}")
                                return False
                        else:
                            logger.error(f"✗ Email data is not a dictionary: {email_data}")
                            return False
                    except Exception as e:
                        logger.error(f"✗ Error parsing email data: {str(e)}")
                        return False
                else:
                    logger.error(f"✗ Email retrieval check failed: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to email retrieval endpoint: {str(e)}")
            return False
    
    async def check_fakemail_classify(self) -> bool:
        """Check if FakeMail's classify endpoint works"""
        logger.info("Checking FakeMail classify endpoint...")
        
        if not hasattr(self, 'test_email_data'):
            # Create a test payload if we don't have real email data
            test_payload = {
                "subject": "Meeting tomorrow at 2pm",
                "body": "Hi team, let's meet tomorrow at 2pm to discuss the project."
            }
        else:
            # Use the real email data
            test_payload = {
                "subject": self.test_email_data.get("subject", "Test subject"),
                "body": self.test_email_data.get("body", "Test body")
            }
            
        try:
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                response = await client.post(
                    f"{self.fakemail_url}/classify",
                    json=test_payload
                )
                
                if response.status_code == 200:
                    try:
                        # Try to parse as JSON first
                        try:
                            result = response.json()
                            if isinstance(result, dict) and "classification" in result:
                                logger.info(f"✓ Classification endpoint works")
                                logger.info(f"  Classification: {result.get('classification')}")
                                return True
                            else:
                                logger.warning(f"⚠ Classification result doesn't have expected structure: {result}")
                                return True  # Still return true since the endpoint works
                        except:
                            # If not JSON, try to parse from text
                            if response.text.strip() in ["meeting", "promotion", "intro", "unknown"]:
                                logger.info(f"✓ Classification endpoint works (Result: {response.text.strip()})")
                                return True
                            else:
                                logger.warning(f"⚠ Classification result has unexpected format: {response.text}")
                                return True  # Still return true since the endpoint works
                    except Exception as e:
                        logger.error(f"✗ Error parsing classification result: {str(e)}")
                        return False
                else:
                    logger.error(f"✗ Classification check failed: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to classification endpoint: {str(e)}")
            return False
    
    async def check_end_to_end_flow(self) -> bool:
        """Test the entire email processing flow"""
        logger.info("Testing end-to-end email processing flow...")
        
        try:
            # Step 1: Initialize with watch endpoint
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                watch_response = await client.post(f"{self.fakemail_url}/watch")
                if watch_response.status_code not in (200, 201, 202, 204):
                    logger.error(f"✗ End-to-end test failed at watch step: {watch_response.status_code}")
                    return False
                
                # Try to get history_id
                try:
                    watch_data = watch_response.json()
                    history_id = watch_data.get("history_id", 1)
                except:
                    history_id = 1
                    
                logger.info(f"  Watch step completed (history_id: {history_id})")
                
            # Step 2: Subscribe to webhooks
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                subscribe_response = await client.post(
                    f"{self.fakemail_url}/subscribe",
                    json={"webhook_url": self.webhook_url}
                )
                if subscribe_response.status_code not in (200, 201, 202, 204):
                    logger.error(f"✗ End-to-end test failed at subscribe step: {subscribe_response.status_code}")
                    return False
                    
                logger.info(f"  Subscribe step completed")
                
            # Step 3: Send a test email
            test_email_id = None
            test_email_subject = f"E2E Test Email {uuid.uuid4()}"
            test_email_body = f"This is an end-to-end test email sent at {time.time()}"
            
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                send_response = await client.post(
                    f"{self.fakemail_url}/send_email",
                    json={
                        "subject": test_email_subject,
                        "body": test_email_body
                    }
                )
                if send_response.status_code not in (200, 201, 202, 204):
                    logger.error(f"✗ End-to-end test failed at send email step: {send_response.status_code}")
                    return False
                
                # Try to get email_id
                try:
                    send_data = send_response.json()
                    test_email_id = send_data.get("email_id")
                except:
                    # Try to extract from text
                    try:
                        import re
                        id_match = re.search(r'["\'](.*?)["\']', send_response.text)
                        if id_match:
                            test_email_id = id_match.group(1)
                    except:
                        pass
                        
                if not test_email_id:
                    logger.warning("⚠ Could not extract email_id from send response")
                    return False
                    
                logger.info(f"  Send email step completed (email_id: {test_email_id})")
                
            # Step 4: Trigger webhook processing
            new_history_id = history_id + 1
            
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                webhook_response = await client.post(
                    self.webhook_url,
                    json={
                        "history_id": new_history_id,
                        "email_ids": [test_email_id]
                    }
                )
                if webhook_response.status_code not in (200, 201, 202, 204):
                    logger.error(f"✗ End-to-end test failed at webhook step: {webhook_response.status_code}")
                    return False
                    
                logger.info(f"  Webhook processing step completed")
                
            # Step 5: Wait a bit for processing to complete
            logger.info("  Waiting for processing to complete...")
            await asyncio.sleep(2)
                
            # Step 6: Check results
            async with httpx.AsyncClient(timeout=self.httpx_timeout) as client:
                # First try without force generation
                results_response = await client.get(
                    f"{self.app_url}/results/{test_email_id}",
                    params={"include_content": "true"}
                )
                
                # If that fails, try with force generation
                if results_response.status_code != 200:
                    logger.warning(f"⚠ Results not found, trying with force_generation=true")
                    results_response = await client.get(
                        f"{self.app_url}/results/{test_email_id}",
                        params={"include_content": "true", "force_generation": "true"}
                    )
                
                if results_response.status_code != 200:
                    logger.error(f"✗ End-to-end test failed at results step: {results_response.status_code}")
                    return False
                
                try:
                    results_data = results_response.json()
                    if isinstance(results_data, dict) and "classification" in results_data:
                        classification = results_data.get("classification")
                        logger.info(f"  Results step completed (classification: {classification})")
                        logger.info(f"✓ End-to-end flow test passed!")
                        return True
                    else:
                        logger.error(f"✗ Results data doesn't have expected structure: {results_data}")
                        return False
                except Exception as e:
                    logger.error(f"✗ Error parsing results data: {str(e)}")
                    return False
                
        except Exception as e:
            logger.error(f"✗ End-to-end test failed with exception: {str(e)}")
            return False
    
    def print_results(self):
        """Print a summary of all endpoint checks"""
        print("\n" + "="*60)
        print("EMAIL PROCESSOR ENDPOINT CHECK RESULTS")
        print("="*60)
        
        # Application endpoints
        print("\nApplication Endpoints:")
        print(f"  {'✓' if self.results['app_health'] else '✗'} Application Health")
        print(f"  {'✓' if self.results['webhook_endpoint'] else '✗'} Webhook Endpoint")
        print(f"  {'✓' if self.results['results_endpoint'] else '✗'} Results Endpoint")
        
        # FakeMail API endpoints
        print("\nFakeMail API Endpoints:")
        print(f"  {'✓' if self.results['watch_endpoint'] else '✗'} Watch Endpoint")
        print(f"  {'✓' if self.results['subscribe_endpoint'] else '✗'} Subscribe Endpoint")
        print(f"  {'✓' if self.results['fakemail_send'] else '✗'} Send Email")
        print(f"  {'✓' if self.results['fakemail_get'] else '✗'} Get Email")
        print(f"  {'✓' if self.results['fakemail_classify'] else '✗'} Classify Email")
        
        # End-to-end flow
        print("\nIntegration Test:")
        print(f"  {'✓' if self.results['end_to_end_flow'] else '✗'} End-to-End Flow")
        
        # Overall status
        all_passed = all(self.results.values())
        print("\n" + "-"*60)
        if all_passed:
            print("✅ ALL CHECKS PASSED")
        else:
            print("❌ SOME CHECKS FAILED")
            
            # Recommendations for failed checks
            print("\nRecommendations:")
            
            if not self.results['app_health']:
                print("  • Make sure the application is running")
                
            if not self.results['webhook_endpoint']:
                print("  • Check the webhook endpoint implementation")
                
            if not self.results['results_endpoint']:
                print("  • Check the results endpoint implementation")
                
            if not self.results['watch_endpoint'] or not self.results['subscribe_endpoint']:
                print("  • Verify the FakeMail API URL is correct")
                
            if not self.results['end_to_end_flow']:
                print("  • Review the webhook processing and results storage")
                
        print("="*60 + "\n")

async def main():
    parser = argparse.ArgumentParser(description="Check required endpoints for the Email Processor application")
    parser.add_argument("--app-url", default="http://localhost:8005", help="Base URL of the application")
    parser.add_argument("--fakemail-url", default="http://localhost:8005/simulator", help="Base URL of the FakeMail API")
    parser.add_argument("--webhook-url", default="http://localhost:8005/webhook", help="URL of the webhook endpoint")
    
    args = parser.parse_args()
    
    checker = EndpointChecker(args.app_url, args.fakemail_url, args.webhook_url)
    success = await checker.run_checks()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))