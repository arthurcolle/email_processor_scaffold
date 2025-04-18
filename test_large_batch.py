#!/usr/bin/env python3
"""
Large-scale test script for FakeMail email processing system.
This script tests the system by sending 10,000 randomly generated emails 
and analyzing their classification results.
"""
import asyncio
import httpx
import json
import time
import sys
import logging
import random
import string
import csv
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import Counter
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_large_batch")

# Configuration
BASE_URL = "http://localhost:8005"  # The base URL of the email processor service
WEBHOOK_URL = f"{BASE_URL}/webhook"  # The webhook URL
SIMULATOR_URL = f"{BASE_URL}/simulator"  # The simulator URL
BATCH_SIZE = 100  # Number of emails to send in a batch
TOTAL_EMAILS = 10000  # Total number of emails to generate and process
PROCESSING_TIMEOUT = 300  # Maximum seconds to wait for all emails to process
RESULT_FILE = "email_classification_results.csv"

# Templates for random email generation
SUBJECTS = [
    "Meeting: {project} update on {date}",
    "Introduction: {person} from {company}",
    "{percent}% Off {product} Sale!",
    "Weekly Report: {project} Progress",
    "Password Reset for {service}",
    "Invitation to {event} on {date}",
    "Question about {topic}",
    "Feedback on {project} proposal",
    "Alert: {issue} detected in {system}",
    "Newsletter: {company} Updates for {month}",
    "Reminder: {task} due on {date}",
    "Confirmation: Your {service} account",
    "Invoice #{number} from {company}",
    "Important update about your {service}",
    "New message from {person}",
]

BODY_TEMPLATES = [
    # Meeting template
    """
Hi team,

Let's meet to discuss {project} on {date} at {time}. 
We need to review the latest updates and plan next steps.

Agenda:
1. Progress update
2. Challenges and blockers
3. Next sprint planning

{virtual_or_location}

Regards,
{sender}
    """,
    
    # Introduction template
    """
Hello everyone,

I'd like to introduce {person}, who is joining our team as {role}.
{person} has {years} years of experience in {field} and will be working on {project}.

Please welcome {person} to the team!

Best,
{sender}
    """,
    
    # Promotion template
    """
Don't miss our HUGE {season} sale! 

Everything is {percent}% off this weekend only. Visit our website to see all the amazing deals on {product_category}.

Limited time offer! Sale ends {date}.

{company} Team
    """,
    
    # Report template
    """
Team,

Attached is the weekly report for {project}. Here are the key highlights:

- {metric1}: {value1}
- {metric2}: {value2}
- {metric3}: {value3}

{assessment}

Let's discuss in our next meeting.

{department} Team
    """,
    
    # Generic template
    """
Hello {recipient},

I wanted to reach out regarding {topic}. {content}

{question_or_action}

Thanks,
{sender}
    """
]

# Data for random email generation
COMPANIES = ["Acme Inc", "TechCorp", "GlobalSoft", "DataSystems", "InnovateTech", "PeakPerformance", "FutureTech", "OptimaSolutions"]
PEOPLE = ["Alex", "Taylor", "Jordan", "Morgan", "Casey", "Riley", "Jamie", "Quinn", "Avery", "Pat"]
PROJECTS = ["Alpha", "Beta", "Phoenix", "Horizon", "Quantum", "Nexus", "Velocity", "Fusion", "Zenith", "Catalyst"]
PRODUCTS = ["Laptop", "Smartphone", "Headphones", "Software License", "Smart Watch", "Camera", "Tablet", "Monitor"]
SERVICES = ["Cloud Storage", "Email", "VPN", "Subscription", "Membership", "Account", "Premium Plan"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ROLES = ["Developer", "Manager", "Designer", "Analyst", "Engineer", "Specialist", "Coordinator", "Consultant"]
FIELDS = ["Software Development", "Data Science", "UX Design", "Project Management", "Marketing", "Sales", "Research"]
LOCATIONS = ["Conference Room A", "Meeting Room 3", "Office", "Headquarters", "Branch Office"]
VIRTUAL = ["Zoom link: https://zoom.us/j/123456789", "Teams meeting: Join with code XYZ-123", "Google Meet: meet.google.com/abc-defg-hij"]
DEPARTMENTS = ["Engineering", "Marketing", "Sales", "Product", "Design", "Research", "Finance", "Customer Support"]
METRICS = ["Revenue", "User Growth", "Conversion Rate", "Engagement", "Retention", "Churn Rate", "Satisfaction Score", "Performance"]
ASSESSMENTS = [
    "Overall, we're on track to meet our quarterly goals.",
    "We need to address some challenges in the next sprint.",
    "Performance exceeded expectations this week.",
    "There are some areas that need immediate attention."
]
QUESTIONS = [
    "Could you please review this and provide feedback?",
    "What are your thoughts on this approach?",
    "Would you be available to discuss this further?",
    "Is there anything else you need from my side?"
]
ACTIONS = [
    "Please complete this by end of the week.",
    "Let me know if you need any clarification.",
    "I'll follow up with more details soon.",
    "We should schedule a call to discuss this in detail."
]
TOPICS = ["the recent proposal", "our upcoming deadline", "the project requirements", "our collaboration", "the new feature request"]
CONTENT_SNIPPETS = [
    "I've been reviewing our progress and have some suggestions.",
    "We've made significant progress, but there are still some challenges to address.",
    "The client provided some feedback that we should incorporate.",
    "I think we should consider a different approach to solve this problem.",
    "Based on recent data, we may need to adjust our strategy."
]

async def setup_system():
    """Setup the system by initializing the webhook"""
    logger.info("Setting up the system...")
    try:
        async with httpx.AsyncClient() as client:
            # Reset the simulator first
            response = await client.post(f"{SIMULATOR_URL}/reset")
            response.raise_for_status()
            logger.info("Simulator reset successfully")
            
            # Use the simulator watch endpoint to get the current history_id
            response = await client.post(f"{SIMULATOR_URL}/watch")
            response.raise_for_status()
            initial_data = response.json()
            initial_history_id = initial_data.get("history_id", 0)
            
            logger.info(f"Initial history_id: {initial_history_id}")
            
            # Subscribe to webhooks
            response = await client.post(
                f"{SIMULATOR_URL}/subscribe",
                json={"webhook_url": WEBHOOK_URL}
            )
            response.raise_for_status()
            
            logger.info(f"Successfully subscribed to webhooks at {WEBHOOK_URL}")
            return initial_history_id
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during setup: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during setup: {str(e)}")
        raise

def generate_random_email():
    """Generate a random email with realistic content"""
    # Pick a template type with weighted distribution
    template_type = random.choices(
        ["meeting", "intro", "promotion", "report", "generic"],
        weights=[0.25, 0.15, 0.2, 0.15, 0.25]
    )[0]
    
    # Generate common data
    date = f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023"
    time = f"{random.randint(8, 17)}:{random.choice(['00', '15', '30', '45'])}"
    project = random.choice(PROJECTS)
    person = random.choice(PEOPLE)
    company = random.choice(COMPANIES)
    sender = random.choice(PEOPLE)
    recipient = random.choice(PEOPLE)
    month = random.choice(MONTHS)
    percent = random.choice([10, 15, 20, 25, 30, 40, 50, 60, 70])
    product = random.choice(PRODUCTS)
    service = random.choice(SERVICES)
    role = random.choice(ROLES)
    years = random.randint(1, 15)
    field = random.choice(FIELDS)
    
    # Special cases for each template type
    if template_type == "meeting":
        subject = f"Meeting: {project} update on {date}"
        virtual_or_location = random.choice(VIRTUAL) if random.random() > 0.3 else f"Location: {random.choice(LOCATIONS)}"
        body = BODY_TEMPLATES[0].format(
            project=project,
            date=date,
            time=time,
            virtual_or_location=virtual_or_location,
            sender=sender
        )
    
    elif template_type == "intro":
        subject = f"Introduction: {person} from {company}"
        body = BODY_TEMPLATES[1].format(
            person=person,
            role=role,
            years=years,
            field=field,
            project=project,
            sender=sender
        )
    
    elif template_type == "promotion":
        subject = f"{percent}% Off {product} Sale!"
        season = random.choice(["Summer", "Winter", "Spring", "Fall", "Holiday", "Black Friday"])
        product_category = random.choice(["electronics", "apparel", "home goods", "services", "software"])
        body = BODY_TEMPLATES[2].format(
            season=season,
            percent=percent,
            product_category=product_category,
            date=date,
            company=company
        )
    
    elif template_type == "report":
        subject = f"Weekly Report: {project} Progress"
        department = random.choice(DEPARTMENTS)
        metric1, metric2, metric3 = random.sample(METRICS, 3)
        value1 = f"{random.randint(1, 100)}%" if "Rate" in metric1 else str(random.randint(100, 10000))
        value2 = f"{random.randint(1, 100)}%" if "Rate" in metric2 else str(random.randint(100, 10000))
        value3 = f"{random.randint(1, 100)}%" if "Rate" in metric3 else str(random.randint(100, 10000))
        assessment = random.choice(ASSESSMENTS)
        body = BODY_TEMPLATES[3].format(
            project=project,
            metric1=metric1,
            value1=value1,
            metric2=metric2,
            value2=value2,
            metric3=metric3,
            value3=value3,
            assessment=assessment,
            department=department
        )
    
    else:  # generic
        topics = [
            f"the {project} project", 
            f"our {service} subscription", 
            f"the {random.choice(ROLES)} position", 
            f"our upcoming {random.choice(['meeting', 'deadline', 'launch', 'event'])}"
        ]
        topic = random.choice(topics)
        content = random.choice(CONTENT_SNIPPETS)
        question_or_action = random.choice(QUESTIONS) if random.random() > 0.5 else random.choice(ACTIONS)
        
        # More varied subject lines for generic emails
        subject_templates = [
            f"Question about {topic}",
            f"Update on {topic}",
            f"Information about {topic}",
            f"Follow-up: {topic}",
            f"Need your input on {topic}"
        ]
        subject = random.choice(subject_templates)
        
        body = BODY_TEMPLATES[4].format(
            recipient=recipient,
            topic=topic,
            content=content,
            question_or_action=question_or_action,
            sender=sender
        )
    
    return {"subject": subject, "body": body, "type": template_type}

async def send_email_batch(batch_number, emails):
    """Send a batch of emails using the simulator"""
    logger.info(f"Sending batch {batch_number} ({len(emails)} emails)...")
    email_ids = []
    
    try:
        async with httpx.AsyncClient() as client:
            for i, email in enumerate(emails):
                response = await client.post(
                    f"{SIMULATOR_URL}/send_email",
                    json={"subject": email["subject"], "body": email["body"]}
                )
                response.raise_for_status()
                result = response.json()
                email_id = result.get("email_id")
                if email_id:
                    email_ids.append((email_id, email["type"]))
                    if (i + 1) % 10 == 0:
                        logger.info(f"Batch {batch_number}: Sent {i + 1}/{len(emails)} emails")
                else:
                    logger.warning(f"Failed to get email ID for email: {email['subject']}")
                
                # Small delay between emails to avoid overwhelming the server
                await asyncio.sleep(0.05)
                
            logger.info(f"Successfully sent batch {batch_number} ({len(email_ids)} emails)")
            return email_ids
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error while sending emails: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while sending emails: {str(e)}")
        raise

async def send_all_emails():
    """Generate and send all emails in batches"""
    logger.info(f"Preparing to send {TOTAL_EMAILS} emails in batches of {BATCH_SIZE}...")
    all_email_ids = []
    start_time = time.time()
    
    num_batches = (TOTAL_EMAILS + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch in range(num_batches):
        # Generate batch of random emails
        batch_size = min(BATCH_SIZE, TOTAL_EMAILS - batch * BATCH_SIZE)
        emails = [generate_random_email() for _ in range(batch_size)]
        
        # Send the batch
        email_ids = await send_email_batch(batch + 1, emails)
        all_email_ids.extend(email_ids)
        
        # Log progress
        emails_sent = len(all_email_ids)
        elapsed = time.time() - start_time
        emails_per_second = emails_sent / elapsed
        estimated_remaining = (TOTAL_EMAILS - emails_sent) / emails_per_second if emails_per_second > 0 else 0
        
        logger.info(f"Progress: {emails_sent}/{TOTAL_EMAILS} emails sent ({emails_per_second:.1f} emails/sec, est. {estimated_remaining:.1f}s remaining)")
        
        # Small delay between batches
        if batch < num_batches - 1:
            await asyncio.sleep(1)
    
    total_time = time.time() - start_time
    logger.info(f"Sent {len(all_email_ids)} emails in {total_time:.2f} seconds ({len(all_email_ids)/total_time:.1f} emails/sec)")
    
    return all_email_ids

async def check_email_processing(email_ids_with_types, concurrency=20):
    """Check if the emails were processed correctly"""
    logger.info(f"Checking processing status for {len(email_ids_with_types)} emails...")
    results = {}
    start_time = time.time()
    email_ids = [id_type[0] for id_type in email_ids_with_types]
    email_types = {id_type[0]: id_type[1] for id_type in email_ids_with_types}
    
    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)
    
    async def check_email(email_id):
        async with semaphore:
            max_retries = 5
            retry_delay = 1.0
            
            for retry in range(max_retries):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{BASE_URL}/results/{email_id}")
                        
                        if response.status_code == 200:
                            return email_id, response.json(), None
                        elif response.status_code == 404:
                            if retry == max_retries - 1:
                                return email_id, None, f"Email not processed after {max_retries} retries"
                            await asyncio.sleep(retry_delay)
                        else:
                            return email_id, None, f"Unexpected status code {response.status_code}"
                except Exception as e:
                    return email_id, None, str(e)
            
            return email_id, None, "Max retries exceeded"
    
    # Schedule all tasks to run concurrently
    tasks = [check_email(email_id) for email_id in email_ids]
    
    # Process results as they complete
    completed = 0
    for future in asyncio.as_completed(tasks):
        email_id, result, error = await future
        if result:
            # Record the result with the original email type
            results[email_id] = {
                "classification": result["classification"],
                "confidence": result["confidence"],
                "original_type": email_types[email_id]
            }
        else:
            logger.warning(f"Failed to get result for email {email_id}: {error}")
        
        # Log progress periodically
        completed += 1
        if completed % 100 == 0 or completed == len(email_ids):
            success_rate = (len(results) / completed) * 100
            logger.info(f"Checked {completed}/{len(email_ids)} emails ({success_rate:.1f}% success rate)")
    
    total_time = time.time() - start_time
    logger.info(f"Retrieved results for {len(results)}/{len(email_ids)} emails in {total_time:.2f} seconds")
    
    return results

def analyze_results(results):
    """Analyze the processing results"""
    if not results:
        logger.warning("No results to analyze")
        return {}
    
    # Count classifications
    classifications = Counter([r["classification"] for r in results.values()])
    
    # Calculate accuracy (how well did the classifier match our expected categories)
    accurate_matches = 0
    type_classification_matrix = {
        "meeting": Counter(),
        "intro": Counter(),
        "promotion": Counter(),
        "report": Counter(),
        "generic": Counter()
    }
    
    for email_id, result in results.items():
        original_type = result["original_type"]
        classification = result["classification"]
        type_classification_matrix[original_type][classification] += 1
        
        # Check if the classification matches what we'd expect
        if (original_type == "meeting" and classification == "meeting") or \
           (original_type == "intro" and classification == "intro") or \
           (original_type == "promotion" and classification == "promotion"):
            accurate_matches += 1
    
    # Accuracy only counting the types that have direct matches (meeting, intro, promotion)
    matching_types = ["meeting", "intro", "promotion"]
    matching_type_count = sum(1 for r in results.values() if r["original_type"] in matching_types)
    accuracy = (accurate_matches / matching_type_count) * 100 if matching_type_count > 0 else 0
    
    # Calculate average confidence
    avg_confidence = sum(r["confidence"] for r in results.values()) / len(results) if results else 0
    
    # Confidence by classification
    confidence_by_class = {}
    for classification in set(r["classification"] for r in results.values()):
        class_results = [r for r in results.values() if r["classification"] == classification]
        confidence_by_class[classification] = sum(r["confidence"] for r in class_results) / len(class_results)
    
    analysis = {
        "total_processed": len(results),
        "classifications": dict(classifications),
        "accuracy": accuracy,
        "average_confidence": avg_confidence,
        "confidence_by_classification": confidence_by_class,
        "classification_matrix": {k: dict(v) for k, v in type_classification_matrix.items()}
    }
    
    return analysis

def export_results_to_csv(results, filename):
    """Export the results to a CSV file"""
    if not results:
        logger.warning("No results to export")
        return
    
    fieldnames = ["email_id", "original_type", "classification", "confidence", "accuracy"]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for email_id, result in results.items():
            original_type = result["original_type"]
            classification = result["classification"]
            accuracy = 1 if original_type == classification else 0
            
            writer.writerow({
                "email_id": email_id,
                "original_type": original_type,
                "classification": classification,
                "confidence": result["confidence"],
                "accuracy": accuracy
            })
    
    logger.info(f"Results exported to {filename}")

async def main():
    """Run the complete end-to-end test flow"""
    start_time = time.time()
    logger.info(f"Starting large-scale test with {TOTAL_EMAILS} emails...")
    
    try:
        # Setup the system
        initial_history_id = await setup_system()
        logger.info(f"System setup complete. Initial history_id: {initial_history_id}")
        
        # Send all test emails
        email_ids_with_types = await send_all_emails()
        if not email_ids_with_types:
            logger.error("No emails were sent successfully")
            return 1
            
        logger.info(f"Sent {len(email_ids_with_types)} test emails")
        
        # Wait a bit for the webhook to start processing the emails
        logger.info("Waiting for email processing to begin...")
        await asyncio.sleep(5)
        
        # Check email processing status with timeout
        processing_start = time.time()
        logger.info(f"Beginning to check processing status (timeout: {PROCESSING_TIMEOUT}s)...")
        
        results = await check_email_processing(email_ids_with_types)
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Export results to CSV
        if results:
            export_results_to_csv(results, RESULT_FILE)
        
        # Print summary
        logger.info("\n--- Test Flow Summary ---")
        logger.info(f"Total emails sent: {len(email_ids_with_types)}")
        logger.info(f"Successfully processed: {len(results)}")
        logger.info(f"Processing success rate: {len(results)/len(email_ids_with_types)*100:.1f}%")
        
        if analysis:
            logger.info("\n--- Classification Analysis ---")
            logger.info(f"Classification distribution: {analysis['classifications']}")
            logger.info(f"Classification accuracy: {analysis['accuracy']:.1f}%")
            logger.info(f"Average confidence: {analysis['average_confidence']:.2f}")
            
            # Print classification matrix
            logger.info("\n--- Classification Matrix (original_type → classification) ---")
            for original_type, counts in analysis['classification_matrix'].items():
                if counts:
                    logger.info(f"{original_type}: {counts}")
        
        total_time = time.time() - start_time
        logger.info(f"\nTotal test duration: {total_time:.2f} seconds")
        
        return 0
            
    except Exception as e:
        logger.error(f"Error during test flow: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)