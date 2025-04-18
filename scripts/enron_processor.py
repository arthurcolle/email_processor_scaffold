#!/usr/bin/env python3
"""
Enron Email Dataset processor for email classification.
This script downloads, preprocesses, and prepares the Enron dataset for use with our classifier.
"""
import os
import sys
import logging
import tarfile
import tempfile
import urllib.request
import random
import json
import time
import argparse
import email
from email.header import decode_header
from email import policy
from email.parser import BytesParser

# Add parent directory to path for importing our classifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_email import predict_email_category, create_training_data, train_traditional_models, train_transformer_model

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enron_processor")

# Enron dataset URL
ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
ENRON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/enron")
ENRON_PROCESSED_DIR = os.path.join(ENRON_DIR, "processed")
ENRON_RESULTS_FILE = os.path.join(ENRON_DIR, "classification_results.json")

def download_enron_dataset():
    """Download the Enron dataset if not already downloaded"""
    tar_file_path = os.path.join(ENRON_DIR, "enron_mail.tar.gz")
    
    # Check if the dataset is already downloaded and extracted
    if os.path.exists(os.path.join(ENRON_DIR, "maildir")):
        logger.info("Enron dataset already downloaded and extracted.")
        return
    
    # Download the dataset if the tar file doesn't exist
    if not os.path.exists(tar_file_path):
        logger.info(f"Downloading Enron dataset from {ENRON_URL}...")
        urllib.request.urlretrieve(ENRON_URL, tar_file_path)
        logger.info(f"Download completed. File saved to {tar_file_path}")
    else:
        logger.info(f"Enron dataset archive already downloaded at {tar_file_path}")
    
    # Extract the dataset
    logger.info("Extracting the Enron dataset...")
    with tarfile.open(tar_file_path, "r:gz") as tar:
        tar.extractall(path=ENRON_DIR)
    logger.info("Extraction completed.")

def parse_email_file(file_path):
    """Parse an email file and extract relevant information"""
    try:
        with open(file_path, 'rb') as fp:
            msg = BytesParser(policy=policy.default).parse(fp)
        
        # Extract subject
        subject = msg.get('Subject', '')
        if subject:
            subject = decode_email_header(subject)
        
        # Extract body
        body = get_email_body(msg)
        
        # Get other metadata
        sender = decode_email_header(msg.get('From', ''))
        recipients = decode_email_header(msg.get('To', ''))
        date = msg.get('Date', '')
        
        return {
            'subject': subject,
            'body': body,
            'sender': sender,
            'recipients': recipients,
            'date': date,
            'file_path': file_path
        }
    except Exception as e:
        logger.warning(f"Error parsing email file {file_path}: {str(e)}")
        return None

def decode_email_header(header):
    """Decode encoded email headers"""
    if not header:
        return ""
    
    try:
        decoded_parts = []
        for text, encoding in decode_header(header):
            if isinstance(text, bytes):
                if encoding:
                    decoded_parts.append(text.decode(encoding, errors='replace'))
                else:
                    decoded_parts.append(text.decode('utf-8', errors='replace'))
            else:
                decoded_parts.append(str(text))
        return ''.join(decoded_parts)
    except Exception as e:
        logger.warning(f"Error decoding header: {str(e)}")
        # Return the original header if decoding fails
        return header

def get_email_body(msg):
    """Extract the text body from an email message"""
    # Check if the message is multipart
    if msg.is_multipart():
        body_parts = []
        for part in msg.iter_parts():
            if part.get_content_type() == "text/plain":
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    # If we can't decode a part, just skip it
                    pass
        return "\n".join(body_parts)
    else:
        # Not multipart, just return the content if it's text
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            try:
                return msg.get_content()
            except Exception:
                return ""
        return ""

def process_enron_dataset(max_emails=10000, save_processed=True):
    """Process the Enron dataset, extracting emails and relevant information"""
    maildir = os.path.join(ENRON_DIR, "maildir")
    if not os.path.exists(maildir):
        logger.error(f"Maildir not found at {maildir}. Please download the dataset first.")
        return None
    
    # Create processed directory if needed
    if save_processed and not os.path.exists(ENRON_PROCESSED_DIR):
        os.makedirs(ENRON_PROCESSED_DIR)
    
    # Find all email files
    logger.info("Finding email files...")
    email_files = []
    for root, _, files in os.walk(maildir):
        for file in files:
            # Skip directories and non-email files
            if os.path.isdir(os.path.join(root, file)) or file.startswith('.'):
                continue
            email_files.append(os.path.join(root, file))
    
    # Sample if there are too many emails
    if len(email_files) > max_emails:
        logger.info(f"Sampling {max_emails} emails from {len(email_files)} total...")
        email_files = random.sample(email_files, max_emails)
    
    # Process each email file
    processed_emails = []
    logger.info(f"Processing {len(email_files)} email files...")
    
    for i, file_path in enumerate(email_files):
        if i % 1000 == 0:
            logger.info(f"Processed {i}/{len(email_files)} emails...")
        
        email_data = parse_email_file(file_path)
        if email_data and email_data['subject'] and email_data['body']:
            processed_emails.append(email_data)
            
            # Optionally save each processed email
            if save_processed:
                file_name = os.path.basename(file_path) + ".json"
                output_path = os.path.join(ENRON_PROCESSED_DIR, file_name)
                with open(output_path, 'w') as f:
                    json.dump(email_data, f, indent=2)
    
    logger.info(f"Successfully processed {len(processed_emails)} emails.")
    return processed_emails

def classify_enron_emails(emails, output_file=ENRON_RESULTS_FILE):
    """Classify the processed Enron emails using our classifier"""
    logger.info(f"Classifying {len(emails)} Enron emails...")
    
    results = []
    start_time = time.time()
    
    for i, email_data in enumerate(emails):
        if i % 100 == 0:
            logger.info(f"Classified {i}/{len(emails)} emails...")
        
        # Get classification
        category, confidence = predict_email_category(
            email_data['subject'], 
            email_data['body']
        )
        
        # Add classification to results
        result = {
            'subject': email_data['subject'],
            'category': category,
            'confidence': confidence,
            'file_path': email_data.get('file_path', ''),
            'sender': email_data.get('sender', ''),
            'date': email_data.get('date', '')
        }
        results.append(result)
    
    # Calculate statistics
    categories = {}
    total_confidence = 0.0
    
    for result in results:
        category = result['category']
        confidence = result['confidence']
        
        if category not in categories:
            categories[category] = {
                'count': 0,
                'total_confidence': 0.0
            }
        
        categories[category]['count'] += 1
        categories[category]['total_confidence'] += confidence
        total_confidence += confidence
    
    # Calculate averages and percentages
    avg_confidence = total_confidence / len(results) if results else 0
    
    for category in categories:
        count = categories[category]['count']
        categories[category]['percentage'] = (count / len(results)) * 100
        categories[category]['avg_confidence'] = categories[category]['total_confidence'] / count
    
    # Sort categories by count
    sorted_categories = sorted(categories.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Create statistics summary
    stats = {
        'total_emails': len(results),
        'average_confidence': avg_confidence,
        'execution_time_seconds': time.time() - start_time,
        'category_distribution': {
            cat: {
                'count': data['count'],
                'percentage': data['percentage'],
                'avg_confidence': data['avg_confidence']
            } for cat, data in sorted_categories
        }
    }
    
    # Save the results
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'statistics': stats
        }, f, indent=2)
    
    logger.info(f"Classification complete. Results saved to {output_file}")
    
    # Print summary statistics
    logger.info("\nClassification Summary:")
    logger.info(f"Total emails classified: {len(results)}")
    logger.info(f"Average confidence: {avg_confidence:.2f}")
    logger.info(f"Execution time: {stats['execution_time_seconds']:.2f} seconds")
    logger.info("\nCategory Distribution:")
    
    for category, data in sorted_categories:
        logger.info(f"  {category}: {data['count']} emails ({data['percentage']:.1f}%), " +
                   f"avg confidence: {data['avg_confidence']:.2f}")
    
    return results, stats

def train_on_enron(num_synthetic=100000, use_transformer=True):
    """Train the classifier models using both synthetic and Enron data"""
    # First, process Enron emails if they haven't been processed yet
    enron_emails = process_enron_dataset(max_emails=50000, save_processed=True)
    
    if not enron_emails:
        logger.error("Failed to process Enron emails. Training aborted.")
        return False
    
    # Generate synthetic training data
    logger.info(f"Generating {num_synthetic} synthetic training emails...")
    synthetic_texts, synthetic_labels = create_training_data(
        num_samples=num_synthetic,
        augment=True,
        save=True,
        balanced=True
    )
    
    # Convert Enron emails to training format
    logger.info("Preparing Enron emails for training...")
    enron_texts = []
    
    for email_data in enron_emails:
        combined_text = f"Subject: {email_data['subject']}\n\nBody: {email_data['body']}"
        enron_texts.append(combined_text)
    
    # Combine synthetic and Enron data for training
    logger.info("Combining synthetic and Enron data for training...")
    all_texts = synthetic_texts + enron_texts
    
    # We only have labels for synthetic data, so we'll use "unknown" for Enron emails
    # This could be improved with manual labeling or semi-supervised learning
    all_labels = synthetic_labels + ["unknown"] * len(enron_texts)
    
    # Train traditional models
    logger.info("Training traditional models on combined dataset...")
    train_traditional_models(all_texts, all_labels, save_model=True)
    
    # Train transformer model if requested
    if use_transformer:
        try:
            import torch
            from transformers import AutoTokenizer
            
            logger.info("Training transformer model on combined dataset...")
            train_transformer_model(
                all_texts, all_labels, 
                save_model=True,
                batch_size=16,
                epochs=3
            )
        except ImportError:
            logger.warning("Transformer libraries not available. Skipping transformer training.")
    
    logger.info("Training completed successfully.")
    return True

def main():
    """Main function to process and classify Enron emails"""
    parser = argparse.ArgumentParser(description="Process and classify Enron emails")
    parser.add_argument("--download", action="store_true", help="Download the Enron dataset")
    parser.add_argument("--process", action="store_true", help="Process the Enron dataset")
    parser.add_argument("--classify", action="store_true", help="Classify the processed Enron emails")
    parser.add_argument("--train", action="store_true", help="Train models using Enron and synthetic data")
    parser.add_argument("--max-emails", type=int, default=10000, help="Maximum number of emails to process")
    parser.add_argument("--synthetic", type=int, default=100000, help="Number of synthetic emails to generate for training")
    
    args = parser.parse_args()
    
    # If no arguments provided, print help
    if not (args.download or args.process or args.classify or args.train):
        parser.print_help()
        return
    
    # Create the Enron directory if it doesn't exist
    if not os.path.exists(ENRON_DIR):
        os.makedirs(ENRON_DIR)
    
    # Download the dataset if requested
    if args.download:
        download_enron_dataset()
    
    # Process emails if requested
    processed_emails = None
    if args.process:
        processed_emails = process_enron_dataset(max_emails=args.max_emails)
    
    # Classify emails if requested
    if args.classify:
        # If we haven't processed the emails yet, do so now
        if processed_emails is None:
            # Check if we have processed emails already
            if os.path.exists(ENRON_PROCESSED_DIR) and os.listdir(ENRON_PROCESSED_DIR):
                logger.info("Loading pre-processed Enron emails...")
                processed_emails = []
                for filename in os.listdir(ENRON_PROCESSED_DIR):
                    if filename.endswith(".json"):
                        with open(os.path.join(ENRON_PROCESSED_DIR, filename), 'r') as f:
                            processed_emails.append(json.load(f))
                logger.info(f"Loaded {len(processed_emails)} pre-processed emails.")
            else:
                # Process the emails
                processed_emails = process_enron_dataset(max_emails=args.max_emails)
        
        if processed_emails:
            classify_enron_emails(processed_emails)
        else:
            logger.error("No processed emails available for classification.")
    
    # Train models if requested
    if args.train:
        train_on_enron(num_synthetic=args.synthetic)

if __name__ == "__main__":
    main()