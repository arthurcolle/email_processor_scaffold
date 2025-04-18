#!/usr/bin/env python3
"""
Quick test for the email classifier with custom sample emails.
"""
import os
import sys
import logging

# Add parent directory to path for importing our classifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_email import predict_email_category

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("quick_test")

# Test emails representing various categories from the Enron dataset
test_emails = [
    {
        "category": "meeting",
        "subject": "Team meeting - Tomorrow at 2pm",
        "body": "Let's meet tomorrow at 2pm in the conference room to discuss the project status. Please bring your updates."
    },
    {
        "category": "security",
        "subject": "Password expiration notice",
        "body": "Your network password will expire in 3 days. Please change it by visiting the IT portal."
    },
    {
        "category": "invitation",
        "subject": "Annual Company Party - Dec 15",
        "body": "You're invited to our annual holiday party on December 15 at 7pm at the Hyatt Downtown. Please RSVP by next Friday."
    },
    {
        "category": "report",
        "subject": "Q3 Financial Results",
        "body": "Attached is the quarterly financial report. Key metrics: Revenue: $2.3M, Expenses: $1.7M, Net Profit: $600K"
    },
    {
        "category": "contract",
        "subject": "Johnson Contract for Review",
        "body": "I've attached the Johnson contract for your review. Please focus on the liability section and let me know your thoughts by Friday."
    },
    {
        "category": "expense_report",
        "subject": "Expense Report Approval Needed",
        "body": "Please approve my expense report for the Chicago trip. Total: $1,542.68. Main expenses: Airfare $425, Hotel $684."
    },
    {
        "category": "newsletter",
        "subject": "May Company Newsletter",
        "body": "In this month's edition: New product launch, Employee spotlight, Upcoming events, Industry news, and more!"
    },
    {
        "category": "job_application",
        "subject": "Application for Trading Assistant Position",
        "body": "I'm writing to apply for the Trading Assistant position. I have 3 years of experience in energy trading operations. My resume is attached."
    },
    {
        "category": "transaction",
        "subject": "Order #12345 Confirmation",
        "body": "Thank you for your order! Your transaction has been completed. Order details: Item: Laptop, Amount: $1,299, Delivery: Express"
    },
    {
        "category": "bill",
        "subject": "Invoice #INV-29384 for April Services",
        "body": "Your invoice for April services is attached. Amount due: $2,450. Due date: May 15, 2001. Payment methods: Check, wire transfer."
    }
]

def main():
    """Run quick classifier tests"""
    # Try to classify each email
    logger.info("Testing email classifier with 10 custom emails...")
    
    correct = 0
    results = []
    
    for i, email in enumerate(test_emails):
        # Get prediction from our classifier
        predicted_category, confidence = predict_email_category(
            email["subject"], 
            email["body"]
        )
        
        # Record if prediction matches expected category
        is_correct = predicted_category == email["category"]
        if is_correct:
            correct += 1
        
        # Save result
        results.append({
            "email": i+1,
            "subject": email["subject"],
            "expected": email["category"],
            "predicted": predicted_category,
            "confidence": confidence,
            "correct": is_correct
        })
        
        # Log result
        logger.info(f"Email {i+1}: \"{email['subject']}\"")
        logger.info(f"  Expected: {email['category']}")
        logger.info(f"  Predicted: {predicted_category} (Confidence: {confidence:.2f})")
        logger.info(f"  Correct: {is_correct}")
    
    # Print summary
    accuracy = correct / len(test_emails) * 100
    logger.info(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{len(test_emails)})")
    
    return results

if __name__ == "__main__":
    main()