#!/usr/bin/env python3
"""
Email Processor Test Script

This script demonstrates how to:
1. Run the email processor server locally
2. Create and classify an email
3. Retrieve classification results
"""

import requests
import json
import time
import sys
import subprocess
import signal
import os
from datetime import datetime
import uuid

# Configuration
BASE_URL = "http://localhost:8005"
SERVER_PROCESS = None

def start_server():
    """Start the email processor server"""
    global SERVER_PROCESS
    print("Starting email processor server...")
    SERVER_PROCESS = subprocess.Popen(
        ["python", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    # Give the server time to start
    time.sleep(3)
    # Check if server is responsive
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("Server started successfully!")
            print(f"Health check response: {response.json()}")
        else:
            print(f"Server responded with status code {response.status_code}")
            shutdown()
            sys.exit(1)
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        shutdown()
        sys.exit(1)

def shutdown():
    """Shutdown the server process"""
    global SERVER_PROCESS
    if SERVER_PROCESS:
        print("Shutting down server...")
        SERVER_PROCESS.send_signal(signal.SIGINT)
        SERVER_PROCESS.wait()
        print("Server shut down")

def create_email():
    """Create a new email and return its ID"""
    print("\nCreating a new email...")
    
    # Generate a unique email ID
    email_id = str(uuid.uuid4())
    
    email_data = {
        "thread_id": None,
        "sender": {
            "email": "alice@example.com",
            "name": "Alice Smith"
        },
        "recipients": [
            {
                "email": "bob@example.com",
                "name": "Bob Johnson"
            }
        ],
        "cc": [],
        "bcc": [],
        "reply_to": None,
        "in_reply_to": None,
        "subject": "Meeting Tomorrow at 2pm",
        "body": "Hi team, let's meet tomorrow at 2pm to discuss the new project.",
        "html_body": "<p>Hi team, let's meet tomorrow at 2pm to discuss the new project.</p>",
        "labels": []
    }
    
    try:
        response = requests.post(f"{BASE_URL}/emails/", json=email_data)
        
        if response.status_code == 201:
            email_id = response.text.strip('"')  # Strip quotes from the response
            print(f"Email created successfully with ID: {email_id}")
            return email_id
        else:
            print(f"Failed to create email: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Error creating email: {e}")
        return None

def classify_email(email_id):
    """Classify an email"""
    print(f"\nClassifying email {email_id}...")
    
    classification_data = {
        "classification_type": "meeting",
        "confidence": 0.95,
        "model_version": "v1.0"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/emails/{email_id}/classify",
            data=classification_data
        )
        
        if response.status_code == 200:
            print(f"Email classified successfully: {response.json()}")
            return True
        else:
            print(f"Failed to classify email: {response.status_code} - {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Error classifying email: {e}")
        return False

def get_classification_results(email_id):
    """Get classification results for an email"""
    print(f"\nRetrieving classification results for email {email_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/results/{email_id}?include_content=true")
        
        if response.status_code == 200:
            results = response.json()
            print("Classification results:")
            print(f"  Email ID: {results['email_id']}")
            print(f"  Classification: {results['classification']}")
            print(f"  Confidence: {results['confidence']}")
            print(f"  Processed at: {results['processed_at']}")
            print(f"  Processor ID: {results['processor_id']}")
            print(f"  Processing time: {results['processing_time_ms']} ms")
            if results.get('subject'):
                print(f"  Subject: {results['subject']}")
            if results.get('body'):
                print(f"  Body: {results['body']}")
            return results
        else:
            print(f"Failed to get classification results: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Error getting classification results: {e}")
        return None

def main():
    """Main function to run the email processor test"""
    try:
        # Start the server
        start_server()
        
        # Create an email
        email_id = create_email()
        if not email_id:
            print("Failed to create email. Exiting.")
            shutdown()
            return
        
        # Classify the email
        if not classify_email(email_id):
            print("Failed to classify email. Continuing to retrieve results anyway...")
        
        # Wait a moment for processing
        time.sleep(1)
        
        # Get classification results
        results = get_classification_results(email_id)
        
        # Wrap up
        if results:
            print("\nEmail processing test completed successfully!")
        else:
            print("\nEmail processing test completed with errors.")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Always shut down the server
        shutdown()

if __name__ == "__main__":
    main()