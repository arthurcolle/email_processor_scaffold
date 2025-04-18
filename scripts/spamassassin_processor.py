#!/usr/bin/env python3
"""
SpamAssassin Corpus Processor for Email Classification System

This script processes the Apache SpamAssassin public corpus, which is a well-known
email dataset for spam detection and classification. The processed dataset is
integrated with the enhanced email classification system.

The SpamAssassin corpus contains both spam and legitimate emails (ham) and is
commonly used for benchmarking anti-spam systems.

Usage:
    python spamassassin_processor.py [--download] [--extract] [--process] [--sample-size NUMBER]

Options:
    --download    Download the SpamAssassin public corpus from the Apache website
    --extract     Extract the downloaded archive
    --process     Process the extracted emails into the compatible format
    --sample-size Set the number of emails to process (default: all available)
"""

import os
import sys
import re
import argparse
import json
import random
import logging
import tarfile
import urllib.request
import email
import email.parser
import email.policy
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('spamassassin_processor')

# Constants
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAM_ARCHIVES = ["20021010_spam.tar.bz2", "20030228_spam.tar.bz2", "20030228_spam_2.tar.bz2"]
HAM_ARCHIVES = ["20021010_easy_ham.tar.bz2", "20021010_hard_ham.tar.bz2", "20030228_easy_ham.tar.bz2", "20030228_easy_ham_2.tar.bz2"]
DATA_DIR = Path("/Users/agent/email_processor_scaffold/data/spamassassin")
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "samples"
EMAIL_PARSER = email.parser.BytesParser(policy=email.policy.default)


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    SAMPLE_DIR.mkdir(exist_ok=True, parents=True)
    logger.info(f"Created directories: {DATA_DIR}, {PROCESSED_DIR}, {SAMPLE_DIR}")


def download_corpus(force: bool = False) -> None:
    """Download the SpamAssassin corpus from Apache website."""
    setup_directories()
    archives = SPAM_ARCHIVES + HAM_ARCHIVES
    
    for archive in archives:
        target_path = DATA_DIR / archive
        if target_path.exists() and not force:
            logger.info(f"Archive {archive} already exists. Skipping download.")
            continue
        
        url = f"{SPAMASSASSIN_BASE_URL}{archive}"
        logger.info(f"Downloading {url} to {target_path}")
        
        try:
            urllib.request.urlretrieve(url, target_path)
            logger.info(f"Successfully downloaded {archive}")
        except Exception as e:
            logger.error(f"Failed to download {archive}: {str(e)}")


def extract_archives(force: bool = False) -> None:
    """Extract the downloaded archives."""
    archives = SPAM_ARCHIVES + HAM_ARCHIVES
    
    for archive in archives:
        archive_path = DATA_DIR / archive
        if not archive_path.exists():
            logger.warning(f"Archive {archive} not found. Skipping extraction.")
            continue
        
        extract_dir = DATA_DIR / archive.replace(".tar.bz2", "")
        if extract_dir.exists() and not force:
            logger.info(f"Directory {extract_dir} already exists. Skipping extraction.")
            continue
        
        logger.info(f"Extracting {archive} to {extract_dir}")
        try:
            extract_dir.mkdir(exist_ok=True)
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(path=extract_dir)
            logger.info(f"Successfully extracted {archive}")
        except Exception as e:
            logger.error(f"Failed to extract {archive}: {str(e)}")


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\s]', '', text)
    return text.strip()


def parse_email_file(file_path: Path) -> Dict[str, Any]:
    """Parse an email file into a structured format."""
    try:
        with open(file_path, 'rb') as f:
            email_content = f.read()
            
        msg = EMAIL_PARSER.parse(email_content)
        
        # Extract basic email metadata
        email_data = {
            'message_id': msg.get('Message-ID', ''),
            'subject': clean_text(msg.get('Subject', '')),
            'from': clean_text(msg.get('From', '')),
            'to': clean_text(msg.get('To', '')),
            'date': msg.get('Date', ''),
            'content_type': msg.get_content_type(),
            'is_spam': 'spam' in str(file_path).lower() and 'ham' not in str(file_path).lower(),
            'source': 'spamassassin'
        }
        
        # Extract body content
        if msg.is_multipart():
            body_parts = []
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    content = part.get_payload(decode=True)
                    try:
                        decoded_content = content.decode('utf-8', errors='replace')
                        body_parts.append(clean_text(decoded_content))
                    except Exception:
                        # Fall back if there are decoding issues
                        body_parts.append(clean_text(str(content)))
            email_data['body'] = '\n'.join(body_parts)
        else:
            content = msg.get_payload(decode=True)
            try:
                email_data['body'] = clean_text(content.decode('utf-8', errors='replace'))
            except Exception:
                email_data['body'] = clean_text(str(content))
        
        # Add additional headers
        for header, value in msg.items():
            if header.lower() not in ['message-id', 'subject', 'from', 'to', 'date', 'content-type']:
                email_data[f"header_{header.lower().replace('-', '_')}"] = clean_text(value)
        
        # Add file metadata
        email_data['file_path'] = str(file_path)
        email_data['file_name'] = file_path.name
        email_data['processed_date'] = datetime.now().isoformat()
        
        return email_data
    
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {str(e)}")
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'error': str(e),
            'is_spam': 'spam' in str(file_path).lower() and 'ham' not in str(file_path).lower(),
            'source': 'spamassassin',
            'processed_date': datetime.now().isoformat()
        }


def find_email_files() -> List[Path]:
    """Find all email files in the extracted directories."""
    email_files = []
    for dir_name in [d.replace(".tar.bz2", "") for d in SPAM_ARCHIVES + HAM_ARCHIVES]:
        dir_path = DATA_DIR / dir_name
        if not dir_path.exists():
            continue
        
        # Walk through the directory to find all files
        for root, _, files in os.walk(dir_path):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                # Skip directories or hidden files
                if file_path.is_dir() or file.startswith('.'):
                    continue
                email_files.append(file_path)
    
    logger.info(f"Found {len(email_files)} email files")
    return email_files


def process_emails(sample_size: Optional[int] = None) -> None:
    """Process emails from the SpamAssassin corpus into JSON format."""
    email_files = find_email_files()
    
    # Sample if requested
    if sample_size and sample_size < len(email_files):
        random.shuffle(email_files)
        email_files = email_files[:sample_size]
        logger.info(f"Randomly sampled {sample_size} emails for processing")
    
    # Process emails
    for i, file_path in enumerate(tqdm(email_files, desc="Processing emails")):
        try:
            email_data = parse_email_file(file_path)
            
            # Generate a unique ID for the file
            import hashlib
            file_id = hashlib.md5(str(file_path).encode()).hexdigest()
            output_path = PROCESSED_DIR / f"{file_id}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(email_data, f, ensure_ascii=False, indent=2)
                
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} emails...")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Successfully processed {len(email_files)} emails")


def create_dataset_samples(spam_ham_ratio: float = 0.3, sample_size: int = 5000) -> None:
    """Create balanced dataset samples from the processed emails."""
    processed_files = list(PROCESSED_DIR.glob('*.json'))
    if not processed_files:
        logger.error("No processed files found. Run the processing step first.")
        return
    
    spam_files = []
    ham_files = []
    
    # Categorize files
    for file_path in processed_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                email_data = json.load(f)
                if email_data.get('is_spam', False):
                    spam_files.append(file_path)
                else:
                    ham_files.append(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
    
    logger.info(f"Found {len(spam_files)} spam and {len(ham_files)} ham emails")
    
    # Calculate sample sizes
    spam_sample_size = int(sample_size * spam_ham_ratio)
    ham_sample_size = sample_size - spam_sample_size
    
    # Adjust if we don't have enough files
    spam_sample_size = min(spam_sample_size, len(spam_files))
    ham_sample_size = min(ham_sample_size, len(ham_files))
    
    # Random sampling
    random.shuffle(spam_files)
    random.shuffle(ham_files)
    
    selected_spam = spam_files[:spam_sample_size]
    selected_ham = ham_files[:ham_sample_size]
    
    # Create the dataset
    dataset = []
    for file_path in selected_spam + selected_ham:
        with open(file_path, 'r', encoding='utf-8') as f:
            email_data = json.load(f)
            dataset.append(email_data)
    
    # Save the dataset
    output_path = SAMPLE_DIR / f"spamassassin_dataset_{len(dataset)}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)
    
    logger.info(f"Created dataset with {spam_sample_size} spam and {ham_sample_size} ham emails")
    logger.info(f"Dataset saved to {output_path}")


def analyze_corpus() -> None:
    """Analyze the processed corpus and generate statistics."""
    processed_files = list(PROCESSED_DIR.glob('*.json'))
    if not processed_files:
        logger.error("No processed files found. Run the processing step first.")
        return
    
    stats = {
        "total_emails": len(processed_files),
        "spam_count": 0,
        "ham_count": 0,
        "has_subject": 0,
        "has_body": 0,
        "avg_subject_length": 0,
        "avg_body_length": 0,
        "by_content_type": {},
    }
    
    total_subject_length = 0
    total_body_length = 0
    
    for file_path in tqdm(processed_files, desc="Analyzing corpus"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                email_data = json.load(f)
                
                if email_data.get('is_spam', False):
                    stats["spam_count"] += 1
                else:
                    stats["ham_count"] += 1
                
                if email_data.get('subject', ''):
                    stats["has_subject"] += 1
                    total_subject_length += len(email_data.get('subject', ''))
                
                if email_data.get('body', ''):
                    stats["has_body"] += 1
                    total_body_length += len(email_data.get('body', ''))
                
                content_type = email_data.get('content_type', 'unknown')
                stats["by_content_type"][content_type] = stats["by_content_type"].get(content_type, 0) + 1
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
    
    # Calculate averages
    if stats["has_subject"] > 0:
        stats["avg_subject_length"] = total_subject_length / stats["has_subject"]
    
    if stats["has_body"] > 0:
        stats["avg_body_length"] = total_body_length / stats["has_body"]
    
    # Print statistics
    logger.info("SpamAssassin Corpus Statistics:")
    logger.info(f"Total emails: {stats['total_emails']}")
    logger.info(f"Spam emails: {stats['spam_count']} ({stats['spam_count']/stats['total_emails']*100:.1f}%)")
    logger.info(f"Ham emails: {stats['ham_count']} ({stats['ham_count']/stats['total_emails']*100:.1f}%)")
    logger.info(f"Emails with subject: {stats['has_subject']} ({stats['has_subject']/stats['total_emails']*100:.1f}%)")
    logger.info(f"Emails with body: {stats['has_body']} ({stats['has_body']/stats['total_emails']*100:.1f}%)")
    logger.info(f"Average subject length: {stats['avg_subject_length']:.1f} characters")
    logger.info(f"Average body length: {stats['avg_body_length']:.1f} characters")
    logger.info("Content types:")
    for content_type, count in sorted(stats["by_content_type"].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {content_type}: {count} ({count/stats['total_emails']*100:.1f}%)")
    
    # Save statistics to file
    stats_path = DATA_DIR / "corpus_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Statistics saved to {stats_path}")


def main() -> None:
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Process the SpamAssassin corpus for email classification")
    parser.add_argument('--download', action='store_true', help='Download the SpamAssassin corpus')
    parser.add_argument('--extract', action='store_true', help='Extract the downloaded archives')
    parser.add_argument('--process', action='store_true', help='Process the extracted emails')
    parser.add_argument('--analyze', action='store_true', help='Analyze the processed corpus')
    parser.add_argument('--create-samples', action='store_true', help='Create balanced dataset samples')
    parser.add_argument('--sample-size', type=int, help='Number of emails to process', default=None)
    parser.add_argument('--spam-ratio', type=float, help='Ratio of spam to ham in the samples', default=0.3)
    parser.add_argument('--force', action='store_true', help='Force redownload or reextraction')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Execute requested operations
    if args.download:
        download_corpus(force=args.force)
    
    if args.extract:
        extract_archives(force=args.force)
    
    if args.process:
        process_emails(sample_size=args.sample_size)
    
    if args.analyze:
        analyze_corpus()
    
    if args.create_samples:
        create_dataset_samples(spam_ham_ratio=args.spam_ratio, 
                              sample_size=args.sample_size or 5000)
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()