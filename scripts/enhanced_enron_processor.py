#!/usr/bin/env python3
"""
Enhanced Enron Email Processor - Processes the Enron email dataset for fine-grained classification.

This script extends the original enron_processor.py with advanced features:
1. Improved email parsing with metadata extraction
2. Multi-label classification using the enhanced taxonomy
3. Advanced content analysis for semantic features
4. Better handling of attachments and email threads
5. Support for the 60-180 category taxonomy
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
import re
import argparse
import email
import hashlib
import numpy as np
from email.header import decode_header
from email import policy
from email.parser import BytesParser
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from datetime import datetime, timedelta

# Add parent directory to path for importing our classifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import from the module in the same directory
from enhanced_taxonomy import build_enhanced_taxonomy, get_leaf_categories

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enhanced_enron_processor")

# Enron dataset URL
ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
ENRON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/enron")
ENRON_PROCESSED_DIR = os.path.join(ENRON_DIR, "enhanced_processed")
ENRON_METADATA_FILE = os.path.join(ENRON_DIR, "email_metadata.json")
ENRON_RESULTS_FILE = os.path.join(ENRON_DIR, "enhanced_classification_results.json")
TAXONOMY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/taxonomy")
TAXONOMY_FILE = os.path.join(TAXONOMY_DIR, "enhanced_taxonomy.json")

# Create the directory structure
os.makedirs(ENRON_PROCESSED_DIR, exist_ok=True)
os.makedirs(TAXONOMY_DIR, exist_ok=True)

# Load the enhanced taxonomy
if os.path.exists(TAXONOMY_FILE):
    with open(TAXONOMY_FILE, 'r') as f:
        ENHANCED_TAXONOMY = json.load(f)
else:
    ENHANCED_TAXONOMY = build_enhanced_taxonomy()
    # Save the taxonomy
    with open(TAXONOMY_FILE, 'w') as f:
        json.dump(ENHANCED_TAXONOMY, f, indent=2)

# Get leaf categories for classification
LEAF_CATEGORIES = get_leaf_categories(ENHANCED_TAXONOMY)

# Regular expressions for email analysis
RE_FORWARDED = re.compile(r"^-{3,}[\s\w]*Forwarded", re.MULTILINE)
RE_REPLIED = re.compile(r"^-{3,}[\s\w]*Original Message|^On .* wrote:", re.MULTILINE)
RE_MEETING = re.compile(r"\b(meeting|conference|discussion|call|agenda|attendees|dial-in|invite|forum|webinar)\b", re.IGNORECASE)
RE_LEGAL = re.compile(r"\b(confidential|attorney|legal|contract|agreement|terms|clause|compliance|regulation|lawsuit|liability|patent|trademark|copyright)\b", re.IGNORECASE)
RE_FINANCIAL = re.compile(r"\b(invoice|payment|budget|cost|price|financial|profit|revenue|expense|forecast|investment|funding|capital|roi|margin|projection)\b", re.IGNORECASE)
RE_PROJECT = re.compile(r"\b(project|deliverable|milestone|deadline|timeline|schedule|status|progress|backlog|sprint|scope|requirement)\b", re.IGNORECASE)
RE_TRADING = re.compile(r"\b(trade|trading|position|market|price|volume|megawatt|mmbtu|gas|power|futures|commodity|option|spread|hedge|volatility|liquidity)\b", re.IGNORECASE)
RE_HR = re.compile(r"\b(employee|hiring|interview|resume|performance|review|salary|compensation|benefits|hr|recruiter|applicant|candidate|onboarding|401k|pension)\b", re.IGNORECASE)
RE_IT = re.compile(r"\b(system|software|hardware|network|database|server|password|login|access|firewall|vpn|encryption|backup|cloud|interface|api|endpoint)\b", re.IGNORECASE)
RE_MARKETING = re.compile(r"\b(marketing|campaign|promotion|brand|advert|social media|seo|audience|lead|conversion|engagement|content|website|traffic|analytics)\b", re.IGNORECASE)
RE_CUSTOMER = re.compile(r"\b(customer|client|account|support|service|feedback|complaint|satisfaction|loyalty|retention|onboarding|churn|ticket|case|resolution)\b", re.IGNORECASE)
RE_OPERATIONS = re.compile(r"\b(operations|logistics|supply chain|inventory|warehouse|shipping|distribution|production|capacity|workflow|process|quality|inspection)\b", re.IGNORECASE)
RE_STRATEGIC = re.compile(r"\b(strategy|vision|mission|goal|objective|kpi|swot|pestle|competitive|advantage|expansion|growth|innovation|disruption|pivot)\b", re.IGNORECASE)
RE_NUMBERS = re.compile(r"[\$£€]?\s*\d+[.,]?\d*\s*(?:million|billion|k|M|B|MM)?")
RE_URLS = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL_ADDRESSES = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_DATES = re.compile(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2,4})\b", re.IGNORECASE)
RE_TIMES = re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b")
RE_BULLET_POINTS = re.compile(r"(?:^|\n)\s*(?:\*|\-|\•|\d+\.)\s+\w+", re.MULTILINE)
RE_SENSITIVE_INFORMATION = re.compile(r"\b(?:confidential|private|sensitive|internal( use)?( only)?|do not (forward|distribute|share))\b", re.IGNORECASE)

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

def extract_email_thread(body: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Extract email thread information, parsing forwarded messages and replies.
    Returns the clean body and a list of thread entries.
    """
    # Look for forwarded message markers
    forwarded_markers = [
        "-------- Forwarded Message --------",
        "--------------------------- Forwarded",
        "----- Forwarded by",
        "-----Original Message-----",
        "--------- Inline attachment follows ---------"
    ]
    
    # Look for reply markers
    reply_markers = [
        "On .* wrote:",
        "From: .*\nSent: .*\nTo: .*\nSubject:",
        "----- Original Message -----"
    ]
    
    # Track all thread parts
    thread_parts = []
    
    # Clean body will have thread parts removed
    clean_body = body
    
    # This is a simple implementation - a more robust one would use
    # a proper email thread parsing library
    
    # Extract forwarded messages
    for marker in forwarded_markers:
        if marker in body:
            parts = body.split(marker)
            clean_body = parts[0].strip()
            
            if len(parts) > 1:
                forwarded_content = marker + parts[1]
                thread_parts.append({
                    "type": "forwarded",
                    "content": forwarded_content
                })
    
    # Extract replies
    for marker in reply_markers:
        match = re.search(f"({marker}.*)", body, re.DOTALL)
        if match:
            reply_content = match.group(1)
            clean_body = body.replace(reply_content, "").strip()
            thread_parts.append({
                "type": "reply",
                "content": reply_content
            })
    
    return clean_body, thread_parts

def extract_email_features(email_data: Dict[str, str]) -> Dict[str, Any]:
    """Extract features from email content for classification"""
    subject = email_data.get('subject', '')
    body = email_data.get('body', '')
    sender = email_data.get('sender', '')
    recipients = email_data.get('recipients', '')
    cc = email_data.get('cc', '')
    bcc = email_data.get('bcc', '')
    day_of_week = email_data.get('day_of_week', '')
    hour_of_day = email_data.get('hour_of_day')
    
    # Extract email thread information
    clean_body, thread_parts = extract_email_thread(body)
    
    # Merged text for pattern analysis
    full_text = subject + " " + clean_body
    
    # Get counts of different patterns
    features = {
        "has_forwarded": bool(RE_FORWARDED.search(body)),
        "has_replied": bool(RE_REPLIED.search(body)),
        "is_long": len(body) > 1000,
        "is_very_long": len(body) > 3000,
        "is_short": len(body) < 300,
        "num_numbers": len(RE_NUMBERS.findall(body)),
        "num_urls": len(RE_URLS.findall(body)),
        "num_email_addresses": len(RE_EMAIL_ADDRESSES.findall(body)),
        "num_dates": len(RE_DATES.findall(body)),
        "num_times": len(RE_TIMES.findall(body)),
        "num_bullet_points": len(RE_BULLET_POINTS.findall(body)),
        "has_sensitive_info": bool(RE_SENSITIVE_INFORMATION.search(full_text)),
        "thread_depth": len(thread_parts),
        "has_recipient_list": ',' in recipients or ';' in recipients,
        "num_recipients": len(recipients.split(',')) if ',' in recipients else (1 if recipients else 0),
        "has_cc": bool(cc),
        "has_bcc": bool(bcc),
        "is_weekend": day_of_week in ['Saturday', 'Sunday'] if day_of_week else None,
        "is_business_hours": 9 <= hour_of_day <= 17 if hour_of_day is not None else None,
        "is_early_morning": 5 <= hour_of_day < 9 if hour_of_day is not None else None,
        "is_evening": 17 < hour_of_day < 22 if hour_of_day is not None else None,
        "is_late_night": (22 <= hour_of_day or hour_of_day < 5) if hour_of_day is not None else None,
        
        # Check different content categories
        "meeting_patterns": bool(RE_MEETING.search(full_text)),
        "legal_patterns": bool(RE_LEGAL.search(full_text)),
        "financial_patterns": bool(RE_FINANCIAL.search(full_text)),
        "project_patterns": bool(RE_PROJECT.search(full_text)),
        "trading_patterns": bool(RE_TRADING.search(full_text)),
        "hr_patterns": bool(RE_HR.search(full_text)),
        "it_patterns": bool(RE_IT.search(full_text)),
        "marketing_patterns": bool(RE_MARKETING.search(full_text)),
        "customer_patterns": bool(RE_CUSTOMER.search(full_text)),
        "operations_patterns": bool(RE_OPERATIONS.search(full_text)),
        "strategic_patterns": bool(RE_STRATEGIC.search(full_text)),
        
        # Content structure indicators
        "has_structured_format": bool(RE_BULLET_POINTS.search(body)) or ":" in body.split("\n")[0] if body else False,
        "has_greeting": bool(re.search(r"^(Dear|Hello|Hi|Good (morning|afternoon|evening))", body, re.MULTILINE | re.IGNORECASE)) if body else False,
        "has_signature": bool(re.search(r"(Regards|Sincerely|Best|Thanks|Thank you|Cheers)[,.]?\s+\w+", body, re.IGNORECASE)) if body else False,
        
        # Check subject line features
        "subject_length": len(subject),
        "subject_has_re": subject.lower().startswith('re:'),
        "subject_has_fwd": subject.lower().startswith('fw:') or subject.lower().startswith('fwd:'),
        "subject_has_question": '?' in subject,
        "subject_has_exclamation": '!' in subject,
        "subject_has_urgent": bool(re.search(r"\b(urgent|immediate|asap|priority|important)\b", subject, re.IGNORECASE)),
        "subject_has_brackets": '[' in subject and ']' in subject,
        "subject_has_parentheses": '(' in subject and ')' in subject,
        "subject_has_numbers": bool(RE_NUMBERS.search(subject)),
        "subject_all_caps": subject.isupper() and len(subject) > 3,
    }
    
    # Text analysis metrics
    if clean_body:
        lines = clean_body.split('\n')
        words = re.findall(r'\b\w+\b', clean_body.lower())
        
        features.update({
            "avg_line_length": sum(len(line) for line in lines) / max(1, len(lines)),
            "max_line_length": max((len(line) for line in lines), default=0),
            "paragraph_count": sum(1 for line in lines if line.strip() == ''),
            "unique_word_ratio": len(set(words)) / max(1, len(words)),
            "avg_word_length": sum(len(word) for word in words) / max(1, len(words)),
        })
    
    # Add domain-specific features based on sender/recipients
    if sender and '@' in sender:
        domain = sender.split('@')[1].lower()
        features["sender_domain"] = domain
        features["is_internal_email"] = "enron.com" in domain
    else:
        features["is_internal_email"] = False
    
    return features

def parse_email_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Parse an email file and extract relevant information with enhanced metadata"""
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
        cc = decode_email_header(msg.get('Cc', ''))
        bcc = decode_email_header(msg.get('Bcc', ''))
        date_str = msg.get('Date', '')
        message_id = msg.get('Message-ID', '')
        references = msg.get('References', '')
        in_reply_to = msg.get('In-Reply-To', '')
        
        # Parse the date if available
        try:
            date_obj = email.utils.parsedate_to_datetime(date_str)
            day_of_week = date_obj.strftime('%A')
            hour_of_day = date_obj.hour
            is_weekend = date_obj.weekday() >= 5  # 5=Saturday, 6=Sunday
        except:
            day_of_week = None
            hour_of_day = None
            is_weekend = None
        
        # Extract user folder path (helpful for Enron dataset)
        user_folder = None
        maildir_pos = file_path.find('maildir')
        if maildir_pos != -1:
            path_parts = file_path[maildir_pos+8:].split('/')
            if len(path_parts) > 0:
                user_folder = path_parts[0]  # First folder is usually the user
        
        # Look for attachments
        has_attachments = False
        attachment_names = []
        attachment_types = []
        
        if msg.is_multipart():
            for part in msg.iter_parts():
                content_disposition = part.get_content_disposition()
                if content_disposition and content_disposition.lower() == 'attachment':
                    has_attachments = True
                    filename = part.get_filename()
                    content_type = part.get_content_type()
                    if filename:
                        attachment_names.append(filename)
                    if content_type:
                        attachment_types.append(content_type)
        
        # Get clean body and thread information
        clean_body, thread_parts = extract_email_thread(body)
        
        # Extract features for classification
        features = extract_email_features({
            'subject': subject,
            'body': body,
            'sender': sender,
            'recipients': recipients
        })
        
        # Create a unique identifier for this email
        email_id = hashlib.md5(f"{file_path}:{sender}:{subject}:{date_str}".encode()).hexdigest()
        
        # Create the enhanced email object
        email_data = {
            'email_id': email_id,
            'subject': subject,
            'body': body,
            'clean_body': clean_body,
            'sender': sender,
            'recipients': recipients,
            'cc': cc,
            'bcc': bcc,
            'date': date_str,
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day,
            'is_weekend': is_weekend,
            'message_id': message_id,
            'references': references,
            'in_reply_to': in_reply_to,
            'file_path': file_path,
            'user_folder': user_folder,
            'has_attachments': has_attachments,
            'attachment_names': attachment_names,
            'attachment_types': attachment_types,
            'thread_parts': thread_parts,
            'features': features
        }
        
        return email_data
    except Exception as e:
        logger.warning(f"Error parsing email file {file_path}: {str(e)}")
        return None

def decode_email_header(header: str) -> str:
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

def get_email_body(msg: email.message.Message) -> str:
    """Extract the text body from an email message with enhanced handling"""
    text_content = []
    html_content = []
    
    # Check if the message is multipart
    if msg.is_multipart():
        for part in msg.iter_parts():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition')).lower()
            
            # Skip attachments
            if content_disposition and ('attachment' in content_disposition or 'inline' in content_disposition):
                continue
            
            if content_type == "text/plain":
                try:
                    text_content.append(part.get_content())
                except Exception:
                    # If we can't decode a part, just skip it
                    pass
            elif content_type == "text/html":
                try:
                    # Store HTML separately - could use HTML parsing if needed
                    html_content.append(part.get_content())
                except Exception:
                    pass
    else:
        # Not multipart, just return the content if it's text
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            try:
                text_content.append(msg.get_content())
            except Exception:
                pass
        elif content_type == "text/html":
            try:
                html_content.append(msg.get_content())
            except Exception:
                pass
    
    # Prefer plain text over HTML
    if text_content:
        return "\n".join(text_content)
    
    # If no plain text is found, do a simple conversion of HTML to text
    # In a production system, use a proper HTML to text converter
    if html_content:
        # Very basic HTML tag removal
        html_text = "\n".join(html_content)
        text = re.sub(r'<[^>]+>', ' ', html_text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    return ""

def process_enron_dataset(max_emails=10000, save_processed=True):
    """Process the Enron dataset, extracting emails with enhanced metadata"""
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
    logger.info(f"Processing {len(email_files)} email files with enhanced metadata...")
    
    for i, file_path in enumerate(email_files):
        if i % 1000 == 0:
            logger.info(f"Processed {i}/{len(email_files)} emails...")
        
        email_data = parse_email_file(file_path)
        if email_data and email_data['subject'] and email_data['body']:
            processed_emails.append(email_data)
            
            # Optionally save each processed email
            if save_processed:
                email_id = email_data['email_id']
                output_path = os.path.join(ENRON_PROCESSED_DIR, f"{email_id}.json")
                with open(output_path, 'w') as f:
                    json.dump(email_data, f, indent=2)
    
    logger.info(f"Successfully processed {len(processed_emails)} emails with enhanced metadata.")
    
    # Save email metadata summary
    if save_processed:
        logger.info("Generating metadata summary...")
        metadata = generate_metadata_summary(processed_emails)
        with open(ENRON_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata summary saved to {ENRON_METADATA_FILE}")
    
    return processed_emails

def generate_metadata_summary(emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from the processed emails"""
    metadata = {
        "total_emails": len(emails),
        "users": set(),
        "domains": defaultdict(int),
        "top_senders": Counter(),
        "daily_distribution": defaultdict(int),
        "hourly_distribution": defaultdict(int),
        "subject_length_distribution": defaultdict(int),
        "body_length_distribution": defaultdict(int),
        "feature_counts": defaultdict(int),
        "attachment_types": Counter(),
    }
    
    for email in emails:
        # Extract user information
        sender = email.get('sender', '')
        if '@' in sender:
            user, domain = sender.split('@', 1)
            metadata["users"].add(user)
            metadata["domains"][domain] += 1
            metadata["top_senders"][sender] += 1
        
        # Extract time information
        day_of_week = email.get('day_of_week')
        if day_of_week:
            metadata["daily_distribution"][day_of_week] += 1
        
        hour_of_day = email.get('hour_of_day')
        if hour_of_day is not None:
            metadata["hourly_distribution"][str(hour_of_day)] += 1
        
        # Extract length information
        subject_length = len(email.get('subject', ''))
        body_length = len(email.get('body', ''))
        
        # Bin the lengths
        if subject_length <= 20:
            metadata["subject_length_distribution"]["0-20"] += 1
        elif subject_length <= 50:
            metadata["subject_length_distribution"]["21-50"] += 1
        elif subject_length <= 100:
            metadata["subject_length_distribution"]["51-100"] += 1
        else:
            metadata["subject_length_distribution"]["100+"] += 1
        
        if body_length <= 100:
            metadata["body_length_distribution"]["0-100"] += 1
        elif body_length <= 500:
            metadata["body_length_distribution"]["101-500"] += 1
        elif body_length <= 1000:
            metadata["body_length_distribution"]["501-1000"] += 1
        elif body_length <= 5000:
            metadata["body_length_distribution"]["1001-5000"] += 1
        else:
            metadata["body_length_distribution"]["5000+"] += 1
        
        # Feature counts
        features = email.get('features', {})
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, bool) and feature_value:
                metadata["feature_counts"][feature_name] += 1
        
        # Count attachment types
        attachment_types = email.get('attachment_types', [])
        for attachment_type in attachment_types:
            metadata["attachment_types"][attachment_type] += 1
    
    # Convert sets to lists for JSON serialization
    metadata["users"] = list(metadata["users"])
    metadata["unique_users"] = len(metadata["users"])
    metadata["unique_domains"] = len(metadata["domains"])
    
    # Get top senders
    metadata["top_senders"] = [{sender: count} for sender, count in metadata["top_senders"].most_common(20)]
    
    # Sort distributions by keys
    metadata["daily_distribution"] = dict(sorted(metadata["daily_distribution"].items()))
    metadata["hourly_distribution"] = dict(sorted(metadata["hourly_distribution"].items()))
    
    # Get top attachment types
    metadata["attachment_types"] = [{atype: count} for atype, count in metadata["attachment_types"].most_common(10)]
    
    return metadata

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from the text using a basic TF-IDF approach"""
    # In a real implementation, use a proper NLP library
    # This is a simplified version for demonstration
    
    # Lowercase and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove stop words (a small set for demonstration)
    stop_words = {
        'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by',
        'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from',
        'at', 'as', 'your', 'have', 'has', 'was', 'were', 'will', 'would',
        'there', 'their', 'they', 'them', 'then', 'than', 'but', 'if', 'my',
        'his', 'her', 'our', 'we', 'us', 'an', 'am', 'me', 'him', 'she'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count occurrences
    word_counts = Counter(filtered_words)
    
    # Extract most common words as keywords
    keywords = [word for word, _ in word_counts.most_common(max_keywords)]
    
    return keywords

def classify_email_with_taxonomy(email: Dict[str, Any], confidence_threshold: float = 0.5) -> Dict[str, float]:
    """
    Classify an email using the enhanced taxonomy.
    Returns a dictionary of category IDs with confidence scores.
    
    This is a rule-based classifier as a placeholder. In a real implementation,
    you would use a machine learning model trained on labeled data.
    """
    # Get email components
    subject = email.get('subject', '')
    body = email.get('clean_body', '') or email.get('body', '')
    full_text = f"{subject}\n\n{body}"
    sender = email.get('sender', '')
    recipients = email.get('recipients', '')
    features = email.get('features', {})
    
    # Initialize confidence scores for all leaf categories
    scores = {category_id: 0.0 for category_id in LEAF_CATEGORIES}
    
    # Extract keywords
    keywords = extract_keywords(full_text)
    
    # Simple rule-based classification based on patterns and keywords
    
    # Meeting detection
    if features.get('meeting_patterns') or RE_MEETING.search(full_text):
        if "project" in full_text.lower() and "review" in full_text.lower():
            scores["project_review_meeting"] = 0.92
        elif "project" in full_text.lower():
            scores["project_meeting"] = 0.9
        elif "team" in full_text.lower() and "daily" in full_text.lower():
            scores["daily_standup"] = 0.95
        elif "team" in full_text.lower():
            scores["team_meeting"] = 0.88
        elif "client" in full_text.lower() or "customer" in full_text.lower():
            scores["client_meeting"] = 0.9
        elif "board" in full_text.lower() or "director" in full_text.lower():
            scores["board_meeting"] = 0.95
        elif "vendor" in full_text.lower() or "supplier" in full_text.lower():
            scores["vendor_meeting"] = 0.9
        elif "investor" in full_text.lower() or "shareholder" in full_text.lower():
            scores["investor_meeting"] = 0.93
        elif "executive" in full_text.lower() or "committee" in full_text.lower():
            scores["executive_committee"] = 0.92
        elif "urgent" in full_text.lower() or "emergency" in full_text.lower():
            scores["emergency_meeting"] = 0.95
        elif "quarter" in full_text.lower() and "review" in full_text.lower():
            scores["quarterly_review"] = 0.93
        else:
            scores["meeting_coordination"] = 0.85
    
    # Project-related emails
    if features.get('project_patterns'):
        if "status" in full_text.lower() or "update" in full_text.lower():
            scores["project_status"] = 0.88
        elif "kick" in full_text.lower() and "off" in full_text.lower():
            scores["project_kickoff"] = 0.92
        elif "issue" in full_text.lower() or "problem" in full_text.lower():
            scores["project_issue"] = 0.87
        elif "complet" in full_text.lower() or "done" in full_text.lower() or "finish" in full_text.lower():
            scores["project_closure"] = 0.89
        elif "budget" in full_text.lower() or "cost" in full_text.lower():
            scores["project_budget"] = 0.93
        elif "risk" in full_text.lower() or "mitigation" in full_text.lower():
            scores["project_risk"] = 0.91
        elif "scope" in full_text.lower() and "change" in full_text.lower():
            scores["project_scope_change"] = 0.94
        elif "scope" in full_text.lower():
            scores["project_scope"] = 0.9
        elif "milestone" in full_text.lower() or "deliverable" in full_text.lower():
            scores["project_milestone"] = 0.92
        elif "resource" in full_text.lower() and "allocation" in full_text.lower():
            scores["project_resource_allocation"] = 0.93
        elif "timeline" in full_text.lower() or "schedule" in full_text.lower():
            scores["project_timeline_update"] = 0.91
        elif "stakeholder" in full_text.lower() and "report" in full_text.lower():
            scores["project_stakeholder_report"] = 0.9
        elif "requirement" in full_text.lower() or "spec" in full_text.lower():
            scores["project_requirements"] = 0.89
        elif "quality" in full_text.lower() or "qa" in full_text.lower():
            scores["project_quality_assurance"] = 0.88
        else:
            scores["project"] = 0.8
    
    # Financial emails
    if features.get('financial_patterns'):
        if "report" in full_text.lower() and "quarter" in full_text.lower():
            scores["quarterly_financial"] = 0.94
        elif "report" in full_text.lower() and "annual" in full_text.lower():
            scores["annual_financial"] = 0.94
        elif "report" in full_text.lower():
            scores["financial_report"] = 0.87
        elif "budget" in full_text.lower() and "plan" in full_text.lower():
            scores["budget_planning"] = 0.92
        elif "forecast" in full_text.lower() or "projection" in full_text.lower():
            scores["financial_forecast"] = 0.9
        elif "analy" in full_text.lower() and "revenue" in full_text.lower():
            scores["revenue_analysis"] = 0.93
        elif "analy" in full_text.lower() and "expense" in full_text.lower():
            scores["expense_analysis"] = 0.92
        elif "analy" in full_text.lower():
            scores["financial_analysis"] = 0.89
        elif "invoice" in full_text.lower() or "bill" in full_text.lower():
            scores["invoice"] = 0.93
        elif "profit" in full_text.lower() and "loss" in full_text.lower():
            scores["profit_loss"] = 0.94
        elif "balance" in full_text.lower() and "sheet" in full_text.lower():
            scores["balance_sheet"] = 0.93
        elif "cash" in full_text.lower() and "flow" in full_text.lower():
            scores["cash_flow_statement"] = 0.93
        elif "variance" in full_text.lower() and ("budget" in full_text.lower() or "forecast" in full_text.lower()):
            scores["financial_variance"] = 0.9
        elif "cost" in full_text.lower() and "allocation" in full_text.lower():
            scores["cost_allocation"] = 0.91
        elif "investment" in full_text.lower() and "analysis" in full_text.lower():
            scores["investment_analysis"] = 0.92
        elif "capital" in full_text.lower() and "expenditure" in full_text.lower():
            scores["capital_expenditure"] = 0.93
        elif "debt" in full_text.lower() and "financ" in full_text.lower():
            scores["debt_financing"] = 0.9
        elif "equity" in full_text.lower() and "financ" in full_text.lower():
            scores["equity_financing"] = 0.9
        elif "tax" in full_text.lower() and "plan" in full_text.lower():
            scores["tax_planning"] = 0.88
        else:
            scores["finance"] = 0.8
    
    # Legal emails
    if features.get('legal_patterns'):
        if "contract" in full_text.lower() and "review" in full_text.lower():
            scores["contract_review"] = 0.93
        elif "litigation" in full_text.lower() or "lawsuit" in full_text.lower():
            scores["litigation"] = 0.92
        elif "dispute" in full_text.lower() and "legal" in full_text.lower():
            scores["legal_dispute"] = 0.93
        elif "settlement" in full_text.lower() and "negotiation" in full_text.lower():
            scores["settlement_negotiation"] = 0.95
        elif "risk" in full_text.lower() and "assessment" in full_text.lower() and "legal" in full_text.lower():
            scores["legal_risk_assessment"] = 0.91
        elif "confidentiality" in full_text.lower() or "nda" in full_text.lower():
            scores["confidentiality_agreement"] = 0.94
        elif "employment" in full_text.lower() and "law" in full_text.lower():
            scores["employment_law"] = 0.92
        elif "corporate" in full_text.lower() and "governance" in full_text.lower():
            scores["corporate_governance"] = 0.93
        elif "trademark" in full_text.lower() and "application" in full_text.lower():
            scores["trademark_application"] = 0.95
        elif "patent" in full_text.lower() and "filing" in full_text.lower():
            scores["patent_filing"] = 0.95
        elif "licensing" in full_text.lower() and "agreement" in full_text.lower():
            scores["licensing_agreement"] = 0.94
        elif "cease" in full_text.lower() and "desist" in full_text.lower():
            scores["cease_desist"] = 0.96
        elif "intellectual" in full_text.lower() or ("patent" in full_text.lower() and "trademark" in full_text.lower()):
            scores["intellectual_property"] = 0.89
        else:
            scores["legal"] = 0.8
    
    # Trading-related emails (Enron specific)
    if features.get('trading_patterns'):
        if "position" in full_text.lower():
            scores["trading_position"] = 0.88
        elif "market" in full_text.lower() and "analysis" in full_text.lower():
            scores["market_analysis"] = 0.88
        elif "gas" in full_text.lower() and "market" in full_text.lower():
            scores["gas_market"] = 0.9
        elif "power" in full_text.lower() and "market" in full_text.lower():
            scores["power_market"] = 0.9
        elif "day" in full_text.lower() and "ahead" in full_text.lower():
            scores["day_ahead_trading"] = 0.94
        elif "spot" in full_text.lower() and "trading" in full_text.lower():
            scores["spot_trading"] = 0.94
        elif "futures" in full_text.lower() and "trading" in full_text.lower():
            scores["futures_trading"] = 0.94
        elif "options" in full_text.lower() and "strategy" in full_text.lower():
            scores["options_strategy"] = 0.93
        elif "hedging" in full_text.lower() and "strategy" in full_text.lower():
            scores["hedging_strategy"] = 0.93
        elif "algorithm" in full_text.lower() and "trading" in full_text.lower():
            scores["trading_algorithm"] = 0.92
        elif "liquidity" in full_text.lower() and "analysis" in full_text.lower():
            scores["market_liquidity_analysis"] = 0.91
        elif "cross" in full_text.lower() and "commodity" in full_text.lower():
            scores["cross_commodity_spread"] = 0.95
        elif "volatility" in full_text.lower() and "analysis" in full_text.lower():
            scores["price_volatility_analysis"] = 0.92
        else:
            scores["energy_trading"] = 0.85
    
    # HR-related emails
    if features.get('hr_patterns'):
        if "interview" in full_text.lower() and "candidate" in full_text.lower():
            scores["candidate_evaluation"] = 0.93
        elif "review" in full_text.lower() and "performance" in full_text.lower():
            scores["performance_evaluation"] = 0.92
        elif "review" in full_text.lower() and "process" in full_text.lower():
            scores["performance_review_process"] = 0.91
        elif "leave" in full_text.lower() and "request" in full_text.lower():
            scores["leave_request"] = 0.94
        elif "benefit" in full_text.lower():
            scores["benefits"] = 0.89
        elif "salary" in full_text.lower() or "compensation" in full_text.lower():
            scores["compensation"] = 0.91
        elif "compensation" in full_text.lower() and "adjustment" in full_text.lower():
            scores["compensation_adjustment"] = 0.95
        elif "policy" in full_text.lower() and "workplace" in full_text.lower():
            scores["workplace_policy"] = 0.91
        elif "grievance" in full_text.lower() and "employee" in full_text.lower():
            scores["employee_grievance"] = 0.93
        elif "remote" in full_text.lower() and "policy" in full_text.lower():
            scores["remote_work_policy"] = 0.94
        elif "leadership" in full_text.lower() and "development" in full_text.lower():
            scores["leadership_development"] = 0.92
        elif "team" in full_text.lower() and "building" in full_text.lower():
            scores["team_building"] = 0.9
        elif "diversity" in full_text.lower() and "inclusion" in full_text.lower():
            scores["diversity_inclusion"] = 0.95
        elif "wellness" in full_text.lower() and "employee" in full_text.lower():
            scores["employee_wellness"] = 0.92
        else:
            scores["hr"] = 0.8
    
    # IT-related emails
    if features.get('it_patterns'):
        if "issue" in full_text.lower() and "software" in full_text.lower():
            scores["software_issue"] = 0.93
        elif "issue" in full_text.lower() and "hardware" in full_text.lower():
            scores["hardware_issue"] = 0.93
        elif "issue" in full_text.lower() and "network" in full_text.lower():
            scores["network_issue"] = 0.93
        elif "security" in full_text.lower() and "data" in full_text.lower():
            scores["data_security"] = 0.94
        elif "performance" in full_text.lower() and "database" in full_text.lower():
            scores["database_performance"] = 0.92
        elif "cloud" in full_text.lower() and "infrastructure" in full_text.lower():
            scores["cloud_infrastructure"] = 0.93
        elif "access" in full_text.lower() and "user" in full_text.lower():
            scores["user_access_management"] = 0.91
        elif "backup" in full_text.lower() and ("recovery" in full_text.lower() or "restore" in full_text.lower()):
            scores["backup_recovery"] = 0.94
        elif "deployment" in full_text.lower() and "software" in full_text.lower():
            scores["software_deployment"] = 0.92
        elif "cybersecurity" in full_text.lower() or "security incident" in full_text.lower():
            scores["cybersecurity_incident"] = 0.96
        elif "security" in full_text.lower():
            scores["it_security"] = 0.91
        elif "server" in full_text.lower() or "system update" in full_text.lower():
            scores["system_update"] = 0.88
        elif "password" in full_text.lower() or "login" in full_text.lower():
            scores["system_access"] = 0.89
        else:
            scores["it"] = 0.8
    
    # Marketing-related emails
    if features.get('marketing_patterns'):
        if "campaign" in full_text.lower() and "digital" in full_text.lower():
            scores["digital_marketing"] = 0.93
        elif "campaign" in full_text.lower() and "content" in full_text.lower():
            scores["content_marketing"] = 0.93
        elif "campaign" in full_text.lower() and "social media" in full_text.lower():
            scores["social_media_marketing"] = 0.95
        elif "campaign" in full_text.lower() and "result" in full_text.lower():
            scores["campaign_results"] = 0.92
        elif "campaign" in full_text.lower() and "email" in full_text.lower():
            scores["email_marketing"] = 0.94
        elif "seo" in full_text.lower() or "search engine optimization" in full_text.lower():
            scores["seo_strategy"] = 0.92
        elif "brand" in full_text.lower() and "position" in full_text.lower():
            scores["brand_positioning"] = 0.91
        elif "segment" in full_text.lower() and "market" in full_text.lower():
            scores["market_segmentation"] = 0.9
        elif "competitor" in full_text.lower() and "brand" in full_text.lower():
            scores["competitor_brand_analysis"] = 0.92
        elif "roi" in full_text.lower() and "marketing" in full_text.lower():
            scores["marketing_roi_analysis"] = 0.93
        else:
            scores["marketing"] = 0.8
    
    # Customer-related emails
    if features.get('customer_patterns'):
        if "complaint" in full_text.lower() and "customer" in full_text.lower():
            scores["customer_complaint"] = 0.94
        elif "request" in full_text.lower() and "service" in full_text.lower():
            scores["service_request"] = 0.93
        elif "inquiry" in full_text.lower() and "account" in full_text.lower():
            scores["account_inquiry"] = 0.92
        elif "appreciation" in full_text.lower() and "customer" in full_text.lower():
            scores["customer_appreciation"] = 0.91
        elif "support" in full_text.lower() and "product" in full_text.lower():
            scores["product_support"] = 0.9
        elif "outage" in full_text.lower() and "service" in full_text.lower():
            scores["service_outage"] = 0.95
        elif "feedback" in full_text.lower() and "analysis" in full_text.lower():
            scores["customer_feedback_analysis"] = 0.91
        elif "sla" in full_text.lower() or "service level agreement" in full_text.lower():
            scores["service_level_agreement"] = 0.94
        elif "satisfaction" in full_text.lower() and "survey" in full_text.lower():
            scores["customer_satisfaction_survey"] = 0.93
        elif "escalation" in full_text.lower() and "case" in full_text.lower():
            scores["case_escalation"] = 0.92
        else:
            scores["customer"] = 0.8
    
    # Operations-related emails
    if features.get('operations_patterns'):
        if "supply chain" in full_text.lower() and "disruption" in full_text.lower():
            scores["supply_chain_disruption"] = 0.94
        elif "warehouse" in full_text.lower() and "operations" in full_text.lower():
            scores["warehouse_operations"] = 0.93
        elif "production" in full_text.lower() and "schedule" in full_text.lower():
            scores["production_scheduling"] = 0.92
        elif "quality" in full_text.lower() and "inspection" in full_text.lower():
            scores["quality_inspection"] = 0.93
        elif "capacity" in full_text.lower() and "planning" in full_text.lower():
            scores["capacity_planning"] = 0.91
        else:
            scores["operations"] = 0.8
    
    # Regulatory compliance emails
    if "compliance" in full_text.lower():
        if "ferc" in full_text.lower() and "filing" in full_text.lower():
            scores["ferc_filing"] = 0.95
        elif "sec" in full_text.lower() and "filing" in full_text.lower():
            scores["sec_filing"] = 0.95
        elif "audit" in full_text.lower() and "compliance" in full_text.lower():
            scores["compliance_audit"] = 0.93
        elif "training" in full_text.lower() and "compliance" in full_text.lower():
            scores["compliance_training"] = 0.92
        elif "data privacy" in full_text.lower() and "compliance" in full_text.lower():
            scores["data_privacy_compliance"] = 0.94
        elif "environmental" in full_text.lower() and "compliance" in full_text.lower():
            scores["environmental_compliance"] = 0.94
        elif "trade" in full_text.lower() and "compliance" in full_text.lower():
            scores["trade_compliance"] = 0.92
        elif "aml" in full_text.lower() or "anti-money laundering" in full_text.lower():
            scores["aml_compliance"] = 0.95
        elif "investigation" in full_text.lower() and "regulatory" in full_text.lower():
            scores["regulatory_investigation"] = 0.93
        elif "attestation" in full_text.lower() and "compliance" in full_text.lower():
            scores["compliance_attestation"] = 0.92
        else:
            scores["compliance"] = 0.8
    
    # Detect communication types (orthogonal dimension)
    if features.get('subject_has_re'):
        scores["comm_response"] = 0.95
    elif features.get('subject_has_fwd'):
        scores["comm_discussion"] = 0.9
    elif features.get('subject_has_question') or "?" in body:
        scores["comm_inquiry"] = 0.85
    elif "invite" in full_text.lower() or "join" in full_text.lower():
        scores["comm_invitation"] = 0.85
    elif "update" in full_text.lower() or "status" in full_text.lower():
        scores["comm_update"] = 0.85
    elif "remind" in full_text.lower():
        scores["comm_reminder"] = 0.9
    elif "request" in full_text.lower() or "please" in full_text.lower():
        scores["comm_request"] = 0.85
    elif "introduc" in full_text.lower() or "new" in full_text.lower():
        scores["comm_introduction"] = 0.8
    elif "confirm" in full_text.lower():
        scores["comm_confirmation"] = 0.85
    elif features.get('subject_has_urgent') or "alert" in full_text.lower():
        scores["comm_alert"] = 0.92
    elif features.get('has_sensitive_info') or "announcement" in full_text.lower():
        scores["comm_announcement"] = 0.85
    
    # Detect urgency levels (orthogonal dimension)
    if features.get('subject_has_urgent') or "urgent" in full_text.lower() or "asap" in full_text.lower() or "immediately" in full_text.lower():
        scores["urgency_urgent"] = 0.95
    elif "important" in full_text.lower() or "critical" in full_text.lower() or "priority" in full_text.lower():
        scores["urgency_high"] = 0.9
    elif "fyi" in full_text.lower() or "for your information" in full_text.lower():
        scores["urgency_low"] = 0.9
    elif features.get('is_business_hours') and not features.get('is_weekend'):
        scores["urgency_medium"] = 0.7
    else:
        scores["urgency_routine"] = 0.7
    
    # Filter by confidence threshold
    confident_scores = {k: v for k, v in scores.items() if v >= confidence_threshold}
    
    return confident_scores

def multi_label_classification(emails: List[Dict[str, Any]], confidence_threshold: float = 0.5, batch_size: int = 1000) -> Dict[str, Any]:
    """
    Perform multi-label classification on a set of emails.
    
    Args:
        emails: List of processed email objects
        confidence_threshold: Minimum confidence threshold for a category to be assigned
        batch_size: Number of emails to process in each batch (for memory management)
    
    Returns:
        Dictionary with classification results and statistics
    """
    logger.info(f"Performing multi-label classification on {len(emails)} emails...")
    
    results = []
    all_category_counts = Counter()
    all_category_confidences = defaultdict(list)
    start_time = time.time()
    
    # Process in batches to manage memory
    num_batches = (len(emails) + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(emails))
        batch_emails = emails[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} ({batch_start+1}-{batch_end} of {len(emails)})...")
        
        for i, email_data in enumerate(batch_emails):
            if (i + batch_start) % 100 == 0:
                logger.info(f"Classified {i + batch_start}/{len(emails)} emails...")
            
            # Get classification
            category_scores = classify_email_with_taxonomy(
                email_data,
                confidence_threshold=confidence_threshold
            )
            
            # Update statistics
            for category, confidence in category_scores.items():
                all_category_counts[category] += 1
                all_category_confidences[category].append(confidence)
            
            # Add classification to results
            result = {
                'email_id': email_data.get('email_id'),
                'subject': email_data.get('subject'),
                'categories': category_scores,
                'num_categories': len(category_scores),
                'sender': email_data.get('sender', ''),
                'date': email_data.get('date', '')
            }
            results.append(result)
    
    # Calculate statistics
    total_confidence = 0.0
    total_categories = 0
    avg_categories_per_email = 0
    
    for category, confidences in all_category_confidences.items():
        all_category_confidences[category] = np.mean(confidences)
        total_confidence += sum(confidences)
        total_categories += len(confidences)
    
    if results:
        avg_categories_per_email = sum(r['num_categories'] for r in results) / len(results)
    
    # Calculate averages and percentages
    avg_confidence = total_confidence / total_categories if total_categories else 0
    
    # Create category statistics
    category_stats = []
    for category, count in all_category_counts.most_common():
        category_name = ENHANCED_TAXONOMY.get(category, {}).get("name", category)
        category_desc = ENHANCED_TAXONOMY.get(category, {}).get("description", "")
        avg_conf = all_category_confidences.get(category, 0)
        percentage = (count / len(emails)) * 100
        
        category_stats.append({
            "category_id": category,
            "name": category_name,
            "description": category_desc,
            "count": count,
            "percentage": percentage,
            "avg_confidence": avg_conf
        })
    
    # Create statistics summary
    stats = {
        'total_emails': len(emails),
        'total_categories_assigned': total_categories,
        'unique_categories_used': len(all_category_counts),
        'avg_categories_per_email': avg_categories_per_email,
        'average_confidence': avg_confidence,
        'execution_time_seconds': time.time() - start_time,
        'category_distribution': category_stats
    }
    
    # Return results and statistics
    return {
        'results': results,
        'statistics': stats
    }

def train_custom_classifier(emails: List[Dict[str, Any]], model_type: str = "ensemble") -> None:
    """
    Train a custom email classifier using the enhanced taxonomy.
    
    Args:
        emails: List of processed email objects to use for training
        model_type: Type of model to train ('traditional', 'transformer', or 'ensemble')
    
    Note: This is a placeholder for where you would implement training logic.
    In a real implementation, you would use proper machine learning libraries.
    """
    logger.info(f"Training {model_type} classifier with {len(emails)} emails...")
    
    # In a real implementation, this would:
    # 1. Preprocess the emails and extract features
    # 2. Generate or use labeled data
    # 3. Train the appropriate model type
    # 4. Save the model for later use
    
    # For now, log steps that would be performed
    logger.info("1. Preprocessing emails and extracting features")
    logger.info("2. Generating training data with enhanced taxonomy categories")
    logger.info("3. Training the model")
    logger.info("4. Evaluating model performance")
    logger.info("5. Saving the model")
    
    # This would be where the actual training code would go
    # For demonstration purposes, we're just logging the steps
    
    logger.info("Training completed. (Note: This is a placeholder implementation)")

def run_enron_analysis(sample_size: int = 3200, confidence_threshold: float = 0.5) -> None:
    """
    Run a complete analysis of the Enron dataset with enhanced taxonomy.
    
    Args:
        sample_size: Number of emails to sample from the dataset
        confidence_threshold: Minimum confidence for assigning categories
    """
    # Step 1: Process the Enron dataset
    logger.info("Step 1: Processing Enron dataset...")
    processed_emails = process_enron_dataset(max_emails=sample_size, save_processed=True)
    
    if not processed_emails:
        logger.error("Failed to process Enron emails. Analysis aborted.")
        return
    
    # Step 2: Classify the emails using enhanced taxonomy
    logger.info("Step 2: Classifying emails with enhanced taxonomy...")
    classification_results = multi_label_classification(
        processed_emails,
        confidence_threshold=confidence_threshold
    )
    
    # Save the classification results
    with open(ENRON_RESULTS_FILE, 'w') as f:
        json.dump(classification_results, f, indent=2)
    
    # Step 3: Analyze results
    logger.info("Step 3: Analyzing classification results...")
    stats = classification_results['statistics']
    
    logger.info("\nClassification Summary:")
    logger.info(f"Total emails classified: {stats['total_emails']}")
    logger.info(f"Total category assignments: {stats['total_categories_assigned']}")
    logger.info(f"Unique categories used: {stats['unique_categories_used']}")
    logger.info(f"Average categories per email: {stats['avg_categories_per_email']:.2f}")
    logger.info(f"Average confidence: {stats['average_confidence']:.2f}")
    logger.info(f"Execution time: {stats['execution_time_seconds']:.2f} seconds")
    
    logger.info("\nTop 10 Category Distribution:")
    for i, category in enumerate(stats['category_distribution'][:10]):
        logger.info(f"  {i+1}. {category['name']} ({category['category_id']}): {category['count']} emails " +
                  f"({category['percentage']:.1f}%), avg confidence: {category['avg_confidence']:.2f}")
    
    # Step 4: Create a summary report
    logger.info("Step 4: Creating summary report...")
    report = {
        "dataset_size": stats['total_emails'],
        "taxonomy_size": len(ENHANCED_TAXONOMY),
        "categories_used": stats['unique_categories_used'],
        "categories_coverage": (stats['unique_categories_used'] / len(ENHANCED_TAXONOMY)) * 100,
        "avg_categories_per_email": stats['avg_categories_per_email'],
        "avg_confidence": stats['average_confidence'],
        "top_categories": [
            {
                "id": cat['category_id'],
                "name": cat['name'],
                "count": cat['count'],
                "percentage": cat['percentage']
            } for cat in stats['category_distribution'][:20]
        ],
        "least_used_categories": [
            {
                "id": cat['category_id'],
                "name": cat['name'],
                "count": cat['count'],
                "percentage": cat['percentage']
            } for cat in sorted(stats['category_distribution'], key=lambda x: x['count'])[:10]
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    report_file = os.path.join(ENRON_DIR, "enhanced_analysis_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Summary report saved to {report_file}")
    logger.info("Analysis complete!")

def main():
    """Main function to process and classify Enron emails with enhanced taxonomy"""
    parser = argparse.ArgumentParser(description="Process and classify Enron emails with enhanced taxonomy")
    parser.add_argument("--download", action="store_true", help="Download the Enron dataset")
    parser.add_argument("--process", action="store_true", help="Process the Enron dataset with enhanced metadata")
    parser.add_argument("--classify", action="store_true", help="Classify the processed emails using enhanced taxonomy")
    parser.add_argument("--analyze", action="store_true", help="Run full analysis pipeline")
    parser.add_argument("--sample", type=int, default=3200, help="Number of emails to sample for analysis")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for classification")
    
    args = parser.parse_args()
    
    # If no arguments provided, print help
    if not (args.download or args.process or args.classify or args.analyze):
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
        processed_emails = process_enron_dataset(max_emails=args.sample)
    
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
                processed_emails = process_enron_dataset(max_emails=args.sample)
        
        if processed_emails:
            classification_results = multi_label_classification(
                processed_emails,
                confidence_threshold=args.confidence
            )
            
            # Save the results
            with open(ENRON_RESULTS_FILE, 'w') as f:
                json.dump(classification_results, f, indent=2)
            
            logger.info(f"Classification results saved to {ENRON_RESULTS_FILE}")
        else:
            logger.error("No processed emails available for classification.")
    
    # Run full analysis if requested
    if args.analyze:
        run_enron_analysis(sample_size=args.sample, confidence_threshold=args.confidence)

if __name__ == "__main__":
    main()