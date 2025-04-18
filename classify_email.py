#!/usr/bin/env python3
"""
Advanced email classifier using state-of-the-art NLP and transformer models.
This script creates training data, trains models, and provides a prediction API.
"""
import asyncio
import httpx
import json
import time
import sys
import logging
import subprocess
import signal
import uuid
import os
import numpy as np
import random
from typing import Dict, Any, Optional, List, Tuple, Union
import joblib

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("classify_email")

# Import the email generator to create synthetic training data
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from email_generator import (
    generate_email_batch, 
    EMAIL_CATEGORIES
)

# Configuration
BASE_URL = "http://localhost:8005"
SERVER_PROCESS = None
MODEL_DIR = "data/models"
DATASET_DIR = "data/datasets"
ENSEMBLE_MODEL_FILE = f"{MODEL_DIR}/email_ensemble_model.joblib"
VECTORIZER_FILE = f"{MODEL_DIR}/email_vectorizer.joblib"
LABEL_ENCODER_FILE = f"{MODEL_DIR}/email_label_encoder.joblib"
TRANSFORMER_TOKENIZER_DIR = f"{MODEL_DIR}/tokenizer"
TRANSFORMER_MODEL_DIR = f"{MODEL_DIR}/transformer"
USE_TRANSFORMER = True  # Enable the transformer model for better accuracy
MODEL_CACHE = {}  # Cache for loaded models to avoid reloading
MODEL_CACHE_EXPIRY = 3600  # Cache expiry in seconds (1 hour)
LAST_MODEL_LOAD_TIME = {}  # Track when models were last loaded

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_TOKENIZER_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_MODEL_DIR, exist_ok=True)

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info("Received interrupt signal, cleaning up...")
    stop_server()
    sys.exit(1)

def start_server():
    """Start the server if it's not already running"""
    global SERVER_PROCESS
    
    # Check if server is already running
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=2.0)
        if response.status_code == 200:
            logger.info("Server is already running")
            return True
    except:
        logger.info("Server is not running, starting it now...")
    
    # Start the server
    try:
        SERVER_PROCESS = subprocess.Popen(
            ["python", "main.py"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = httpx.get(f"{BASE_URL}/health", timeout=1.0)
                if response.status_code == 200:
                    logger.info("Server started successfully")
                    return True
            except:
                pass
            time.sleep(1)
            
        logger.error("Server failed to start within timeout period")
        stop_server()
        return False
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        if SERVER_PROCESS:
            stop_server()
        return False

def stop_server():
    """Stop the server if we started it"""
    global SERVER_PROCESS
    if SERVER_PROCESS:
        logger.info("Stopping server...")
        SERVER_PROCESS.terminate()
        try:
            SERVER_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not terminate, forcing...")
            SERVER_PROCESS.kill()
        SERVER_PROCESS = None
        logger.info("Server stopped")

def create_augmented_email(email: Dict[str, str]) -> Dict[str, str]:
    """Create an augmented version of an email for data augmentation"""
    import re
    from random import choice, random, randint, shuffle, sample
    
    # Get email components
    subject = email["subject"]
    body = email["body"]
    category = email["category"]
    
    # Text augmentation techniques
    techniques = [
        "synonym_replacement",
        "word_deletion",
        "word_order_swap",
        "sentence_deletion",
        "punctuation_modification"
    ]
    
    # Pick a technique based on the category (ensure we don't over-modify key signals)
    if category in ["meeting", "invitation"]:
        # For meeting emails, preserve time/date patterns but change other details
        technique = choice(["synonym_replacement", "sentence_deletion", "word_order_swap"])
    elif category in ["promotion", "billing"]:
        # For promotions, preserve numbers/prices but change wording
        technique = choice(["synonym_replacement", "word_order_swap", "punctuation_modification"])
    else:
        # For others, use any technique
        technique = choice(techniques)
    
    # Apply the chosen technique
    if technique == "synonym_replacement":
        # Simple synonym replacement (would use a proper synonym dictionary in production)
        replacements = {
            "meeting": ["conference", "discussion", "gathering", "session"],
            "update": ["refresh", "revision", "change", "modification"],
            "team": ["group", "crew", "staff", "department"],
            "important": ["crucial", "critical", "essential", "vital"],
            "hello": ["hi", "greetings", "hey", "good day"],
            "project": ["initiative", "endeavor", "undertaking", "venture"],
            "welcome": ["greet", "receive", "accommodate", "accept"],
            "discount": ["reduction", "markdown", "savings", "sale"],
            "offer": ["proposal", "deal", "promotion", "arrangement"],
            "send": ["deliver", "transmit", "forward", "convey"],
            "customer": ["client", "patron", "consumer", "user"],
            "security": ["protection", "safety", "defense", "safeguard"],
            "support": ["assistance", "help", "aid", "backing"],
            "issue": ["problem", "concern", "matter", "difficulty"],
            "message": ["communication", "note", "notification", "memo"]
        }
        
        for word, synonyms in replacements.items():
            if word in subject.lower():
                subject = re.sub(r'\b' + word + r'\b', choice(synonyms), subject, flags=re.IGNORECASE)
            if word in body.lower():
                body = re.sub(r'\b' + word + r'\b', choice(synonyms), body, flags=re.IGNORECASE)
    
    elif technique == "word_deletion":
        # Randomly delete some words (not too many)
        subject_words = subject.split()
        if len(subject_words) > 3:
            to_delete = randint(1, min(2, len(subject_words) // 3))
            for _ in range(to_delete):
                if len(subject_words) > 3:  # Make sure we still have words left
                    del subject_words[randint(0, len(subject_words) - 1)]
            subject = " ".join(subject_words)
            
        new_body_lines = []
        body_lines = body.split("\n")
        for line in body_lines:
            if line.strip():
                words = line.split()
                if len(words) > 5:
                    to_delete = randint(1, min(3, len(words) // 4))
                    for _ in range(to_delete):
                        if len(words) > 3:
                            del words[randint(1, len(words) - 2)]  # Don't delete first or last word
                    new_body_lines.append(" ".join(words))
                else:
                    new_body_lines.append(line)
            else:
                new_body_lines.append(line)
        body = "\n".join(new_body_lines)
    
    elif technique == "word_order_swap":
        # Swap the order of some words (within reasonable bounds)
        subject_words = subject.split()
        if len(subject_words) > 3:
            for _ in range(min(2, len(subject_words) // 3)):
                i, j = sorted(sample(range(len(subject_words)), 2))
                subject_words[i], subject_words[j] = subject_words[j], subject_words[i]
            subject = " ".join(subject_words)
            
        new_body_lines = []
        body_lines = body.split("\n")
        for line in body_lines:
            if line.strip():
                words = line.split()
                if len(words) > 5:
                    for _ in range(min(2, len(words) // 4)):
                        i, j = sorted(sample(range(len(words)), 2))
                        words[i], words[j] = words[j], words[i]
                    new_body_lines.append(" ".join(words))
                else:
                    new_body_lines.append(line)
            else:
                new_body_lines.append(line)
        body = "\n".join(new_body_lines)
    
    elif technique == "sentence_deletion":
        # Delete a sentence from the body
        body_sentences = re.split(r'(?<=[.!?])\s+', body)
        if len(body_sentences) > 2:
            to_delete = randint(1, min(2, len(body_sentences) // 3))
            for _ in range(to_delete):
                if len(body_sentences) > 2:
                    del_idx = randint(1, len(body_sentences) - 2)  # Don't delete first or last
                    del body_sentences[del_idx]
            body = " ".join(body_sentences)
    
    elif technique == "punctuation_modification":
        # Modify or add punctuation
        punctuation_changes = [
            (r'\.', '!'),
            (r'!', '.'),
            (r'\?', '??'),
            (r',', ';'),
            (r' - ', ' – '),
        ]
        
        change = choice(punctuation_changes)
        if random() < 0.7:  # 70% chance to apply to body
            body = re.sub(change[0], change[1], body)
        else:  # 30% chance to apply to subject
            subject = re.sub(change[0], change[1], subject)
    
    return {
        "subject": subject,
        "body": body,
        "category": category
    }

def create_training_data(num_samples=1000, augment=True, save=True, 
                     balanced=True, batch_size=10000, augment_ratio=0.3) -> Tuple[List[str], List[str]]:
    """Create a training dataset with synthetic emails
    
    Args:
        num_samples: Total number of samples to generate
        augment: Whether to augment the data with variations
        save: Whether to save the dataset to disk
        balanced: Whether to create a balanced dataset across categories
        batch_size: Process this many samples at a time to manage memory
        augment_ratio: Ratio of samples to augment (0.0-1.0)
    
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Generating {num_samples} synthetic emails for training...")
    
    # Get all available categories
    categories = list(EMAIL_CATEGORIES.keys())
    logger.info(f"Using {len(categories)} different email categories")
    
    # Calculate how many batches we need
    num_batches = max(1, num_samples // batch_size)
    samples_per_batch = num_samples // num_batches
    
    all_texts = []
    all_labels = []
    batches_saved = 0
    
    # Create samples in batches to manage memory usage for large datasets
    for batch in range(num_batches):
        batch_start_time = time.time()
        logger.info(f"Processing batch {batch+1}/{num_batches} with {samples_per_batch} samples")
        
        emails = []
        
        if balanced:
            # Generate balanced samples across categories
            samples_per_category = samples_per_batch // len(categories)
            
            for category in categories:
                # Generate emails for this category
                category_emails = generate_email_batch(samples_per_category, [category])
                emails.extend(category_emails)
                
                # Augment data if requested
                if augment:
                    num_to_augment = int(len(category_emails) * augment_ratio)
                    if num_to_augment > 0:
                        augmented_emails = []
                        for email in category_emails[:num_to_augment]:
                            augmented_emails.append(create_augmented_email(email))
                        emails.extend(augmented_emails)
        else:
            # Generate samples using natural distribution based on weights
            batch_emails = generate_email_batch(samples_per_batch)
            emails.extend(batch_emails)
            
            # Augment data if requested
            if augment:
                num_to_augment = int(len(batch_emails) * augment_ratio)
                if num_to_augment > 0:
                    indices_to_augment = random.sample(range(len(batch_emails)), num_to_augment)
                    augmented_emails = []
                    for idx in indices_to_augment:
                        augmented_emails.append(create_augmented_email(batch_emails[idx]))
                    emails.extend(augmented_emails)
        
        # Shuffle the batch
        random.shuffle(emails)
        
        # Extract text and labels
        batch_texts = []
        batch_labels = []
        for email in emails:
            # Combine subject and body for better classification
            combined_text = f"Subject: {email['subject']}\n\nBody: {email['body']}"
            batch_texts.append(combined_text)
            batch_labels.append(email['category'])
        
        # Add to overall collection
        all_texts.extend(batch_texts)
        all_labels.extend(batch_labels)
        
        # Save intermediate results for very large datasets
        if save and num_batches > 1:
            batch_dataset_path = f"{DATASET_DIR}/email_dataset_batch_{batch+1}_{len(batch_texts)}.json"
            with open(batch_dataset_path, 'w') as f:
                json.dump({
                    "texts": batch_texts,
                    "labels": batch_labels,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "num_samples": len(batch_texts),
                    "categories": categories,
                    "batch": batch+1,
                    "total_batches": num_batches
                }, f)
            batches_saved += 1
            
        batch_duration = time.time() - batch_start_time
        logger.info(f"Batch {batch+1} completed in {batch_duration:.2f} seconds")
    
    # Save the complete dataset if requested
    if save:
        dataset_path = f"{DATASET_DIR}/email_dataset_{len(all_texts)}.json"
        
        # For very large datasets, save metadata only
        if len(all_texts) > 100000:
            # Save only metadata for the full dataset
            with open(dataset_path.replace('.json', '_meta.json'), 'w') as f:
                json.dump({
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "num_samples": len(all_texts),
                    "categories": categories,
                    "batches": num_batches,
                    "category_distribution": {cat: all_labels.count(cat) for cat in set(all_labels)}
                }, f)
            logger.info(f"Saved dataset metadata to {dataset_path.replace('.json', '_meta.json')}")
        else:
            # Save full dataset for smaller datasets
            with open(dataset_path, 'w') as f:
                json.dump({
                    "texts": all_texts,
                    "labels": all_labels,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "num_samples": len(all_texts),
                    "categories": categories
                }, f)
            logger.info(f"Saved complete dataset to {dataset_path}")
    
    # Calculate category distribution for logging
    category_counts = {}
    for label in all_labels:
        category_counts[label] = category_counts.get(label, 0) + 1
    
    # Log the distribution
    logger.info(f"Created training dataset with {len(all_texts)} samples across {len(categories)} categories")
    logger.info("Category distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_labels)) * 100
        logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    return all_texts, all_labels

def train_traditional_models(texts: List[str], labels: List[str], save_model=True, 
                          use_hyperparameter_tuning=False):
    """Train traditional ML models for email classification
    
    Args:
        texts: List of email texts
        labels: List of corresponding labels
        save_model: Whether to save the trained model
        use_hyperparameter_tuning: Whether to perform hyperparameter optimization
    
    Returns:
        Tuple of (ensemble, vectorizer, label_encoder, trained_models)
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, accuracy_score, f1_score
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import HashingVectorizer
    except ImportError:
        logger.error("Required packages not installed. Please run: pip install scikit-learn")
        sys.exit(1)
    
    logger.info("Training traditional ML models...")
    start_time = time.time()
    
    # Convert labels to numeric form
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    
    # For very large datasets, use HashingVectorizer instead of TfidfVectorizer
    use_hashing = len(texts) > 500000
    
    if use_hashing:
        logger.info("Using HashingVectorizer for large dataset")
        vectorizer = HashingVectorizer(
            n_features=2**18,  # ~260K features
            alternate_sign=False,  # Use only positive values (like CountVectorizer)
            ngram_range=(1, 2),  # Include bigrams
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
    else:
        # Create a TF-IDF vectorizer with optimized settings
        logger.info("Using TfidfVectorizer")
        vectorizer = TfidfVectorizer(
            max_features=10000,  # Reduced feature count for faster processing
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
            ngram_range=(1, 2),  # Reduced to bigrams only
            sublinear_tf=True,  # Apply sublinear tf scaling (log)
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
    
    # Train vectorizer on training data only
    logger.info("Vectorizing training data...")
    vectorizer_start = time.time()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    logger.info(f"Vectorization completed in {time.time() - vectorizer_start:.2f} seconds")
    logger.info(f"Training data shape: {X_train_vect.shape}")
    
    # Define multiple models for ensemble with optimized parameters for large datasets
    models = {
        'rf': RandomForestClassifier(
            n_estimators=100,  # Reduced number of trees for faster training
            max_depth=20,      # Limit depth for faster training
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            class_weight='balanced',
            verbose=0
        ),
        'svm': LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=5000,     # Reduced max iterations for faster convergence
            dual=False if X_train_vect.shape[0] > X_train_vect.shape[1] else True  # Optimize for many samples
        ),
        'nb': MultinomialNB(alpha=0.1),
        'gb': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        ),
        'lr': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='saga',  # Efficient for large datasets
            n_jobs=-1,
            verbose=0
        )
    }
    
    # Hyperparameter tuning if requested
    if use_hyperparameter_tuning:
        logger.info("Performing hyperparameter optimization...")
        
        # Define parameter grids for each model
        param_grids = {
            'nb': {
                'alpha': [0.01, 0.1, 0.5, 1.0]
            },
            'lr': {
                'C': [0.1, 0.5, 1.0, 5.0],
                'solver': ['saga', 'liblinear']
            }
        }
        
        # Only tune a subset of models to save time
        for model_name in ['nb', 'lr']:
            if model_name in models and model_name in param_grids:
                logger.info(f"Tuning {model_name} model...")
                grid = GridSearchCV(
                    estimator=models[model_name],
                    param_grid=param_grids[model_name],
                    cv=3,  # 3-fold cross-validation
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
                grid.fit(X_train_vect, y_train)
                models[model_name] = grid.best_estimator_
                logger.info(f"Best parameters for {model_name}: {grid.best_params_}")
    
    # Train each model and evaluate
    trained_models = {}
    
    for name, model in models.items():
        model_start = time.time()
        logger.info(f"Training {name} model...")
        
        # For very large datasets, use a smaller subset for SVM and GB
        if name in ['svm', 'gb'] and len(X_train) > 100000:
            # Sample 100k examples randomly but stratified
            from sklearn.model_selection import train_test_split
            X_subset, _, y_subset, _ = train_test_split(
                X_train_vect, y_train, 
                train_size=100000, 
                random_state=42, 
                stratify=y_train
            )
            logger.info(f"Using {X_subset.shape[0]} samples for {name} model training due to large dataset")
            model.fit(X_subset, y_subset)
        else:
            model.fit(X_train_vect, y_train)
            
        training_time = time.time() - model_start
        
        # Evaluate
        eval_start = time.time()
        y_pred = model.predict(X_test_vect)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"{name.upper()} model: Accuracy={accuracy:.4f}, F1={f1:.4f}, " 
                   f"Training time={training_time:.2f}s, Prediction time={time.time()-eval_start:.2f}s")
        
        trained_models[name] = model
    
    # Create voting ensemble with all models that support predict_proba
    # Note: LinearSVC doesn't have predict_proba, so we can't use it in VotingClassifier with voting='soft'
    voting_models = []
    for name, model in trained_models.items():
        if name != 'svm':  # Exclude LinearSVC
            voting_models.append((name, model))
    
    ensemble = VotingClassifier(
        estimators=voting_models,
        voting='soft',  # Use probability weighted voting
        n_jobs=-1,
        verbose=0
    )
    
    logger.info("Training ensemble model...")
    ensemble_start = time.time()
    ensemble.fit(X_train_vect, y_train)
    logger.info(f"Ensemble training completed in {time.time() - ensemble_start:.2f} seconds")
    
    # Evaluate ensemble
    y_pred = ensemble.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"ENSEMBLE model: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    # Detailed report
    logger.info("Generating detailed classification report...")
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print(report)
    
    # Save models and vectorizer if requested
    if save_model:
        logger.info("Saving models to disk...")
        joblib.dump(ensemble, ENSEMBLE_MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        joblib.dump(label_encoder, LABEL_ENCODER_FILE)
        logger.info(f"Traditional models saved to {MODEL_DIR}")
    
    total_time = time.time() - start_time
    logger.info(f"Traditional model training completed in {total_time:.2f} seconds")
    
    return ensemble, vectorizer, label_encoder, trained_models

def train_transformer_model(texts: List[str], labels: List[str], save_model=True,
                         batch_size=16, learning_rate=2e-5, epochs=3, 
                         model_name="distilbert-base-uncased", max_length=512):
    """Train a transformer-based model for email classification with advanced techniques
    
    Args:
        texts: List of email texts
        labels: List of corresponding labels
        save_model: Whether to save the trained model
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        model_name: Name of the pretrained model to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (model, tokenizer, label_encoder)
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification, 
            TrainingArguments, Trainer, DataCollatorWithPadding,
            get_linear_schedule_with_warmup
        )
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split, StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
        import numpy as np
        import datasets
        from datasets import Dataset
        
        # Check if transformers version supports callbacks
        try:
            from transformers import EarlyStoppingCallback
            early_stopping_available = True
        except ImportError:
            early_stopping_available = False
    except ImportError:
        logger.error("Required packages not installed. Please run: pip install torch transformers datasets")
        logger.warning("Falling back to traditional models only")
        return None, None, None
    
    start_time = time.time()
    logger.info("Training advanced transformer-based model for email classification...")
    logger.info(f"Using {model_name} as base model")
    logger.info(f"Dataset size: {len(texts)} samples")
    
    # Select optimal model based on dataset size
    if len(texts) < 5000:
        # For very small datasets, a lightweight model might work better to avoid overfitting
        if model_name == "distilbert-base-uncased":
            logger.info(f"Small dataset ({len(texts)} samples), using distilbert-base-uncased which works well with limited data")
            # Increase epochs for small datasets
            epochs = min(epochs + 2, 5)  # Max 5 epochs for small datasets
    elif len(texts) < 20000:
        # For medium datasets, standard BERT models work well
        if model_name == "distilbert-base-uncased":
            logger.info("Medium dataset, using distilbert-base-uncased with optimized parameters")
            # Use default epochs
    elif len(texts) > 50000:
        # For large datasets, we could use a more efficient model
        if model_name == "distilbert-base-uncased":
            logger.info("Large dataset, using distilbert-base-uncased with adaptive batch size")
            # Increase batch size for larger datasets if memory allows
            batch_size = min(batch_size * 2, 32)  # Max batch size 32
            
    # Check for very large datasets - use stratified sampling
    if len(texts) > 100000:
        logger.info("Very large dataset detected, using a stratified sample for transformer training")
        # For large datasets, we'll use a subset of the data
        sample_size = min(100000, len(texts))
        texts_sample, _, labels_sample, _ = train_test_split(
            texts, labels, train_size=sample_size, random_state=42, stratify=labels
        )
        texts = texts_sample
        labels = labels_sample
        logger.info(f"Using subset of {len(texts)} samples for transformer training")
    
    # Convert labels to numeric form
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)
    logger.info(f"Number of classes: {num_labels}")
    
    # Check for class imbalance
    label_counts = np.bincount(encoded_labels)
    min_count = label_counts.min()
    max_count = label_counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 10:
        logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f}). Using class weights.")
        # Calculate class weights for addressing imbalance
        class_weights = len(encoded_labels) / (num_labels * label_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        use_class_weights = True
    else:
        logger.info(f"Class balance is acceptable (ratio: {imbalance_ratio:.1f})")
        use_class_weights = False
    
    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=42
    )
    
    logger.info(f"Training on {len(train_texts)} examples, validating on {len(val_texts)} examples")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens for email-specific processing
    special_tokens = {
        "additional_special_tokens": [
            "[EMAIL_SUBJECT]", 
            "[EMAIL_BODY]", 
            "[EMAIL_SIGNATURE]",
            "[EMAIL_DATE]",
            "[EMAIL_TIME]",
            "[EMAIL_RECIPIENT]",
            "[MEETING_CONTEXT]",
            "[FINANCIAL_CONTEXT]",
            "[URGENT_CONTEXT]"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Advanced tokenization function that extracts email structure
    def preprocess_email_text(text):
        """Extract structure from email text and add special tokens"""
        import re
        
        # Extract subject and body if in the standard format
        subject_match = re.search(r'Subject: (.*?)(?:\n\nBody:|$)', text, re.DOTALL)
        body_match = re.search(r'Body: (.*?)$', text, re.DOTALL)
        
        subject = subject_match.group(1).strip() if subject_match else ""
        body = body_match.group(1).strip() if body_match else text  # Default to full text if no match
        
        # Look for email signature patterns
        signature_patterns = [
            r'(?:\n|\r\n)--+(?:\n|\r\n).*$',  # Standard email signature separator
            r'(?:\n|\r\n)(?:regards|sincerely|best|cheers|thank you),?(?:\n|\r\n).*$',  # Common closing words
            r'(?:\n|\r\n)(?:[^\n\r]+(?:\n|\r\n)){1,3}(?:phone|email|web|www|\+\d).*$'  # Contact info block
        ]
        
        signature = ""
        for pattern in signature_patterns:
            sig_match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
            if sig_match:
                signature = sig_match.group(0)
                body = body[:sig_match.start()]
                break
        
        # Extract dates and times
        date_matches = re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}\b', 
                                body, re.IGNORECASE)
        
        time_matches = re.findall(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b', body, re.IGNORECASE)
        
        # Format with special tokens for enhanced understanding
        processed_text = f"[EMAIL_SUBJECT] {subject} "
        
        if date_matches:
            processed_text += f"[EMAIL_DATE] {' '.join(date_matches[:2])} "
            
        if time_matches:
            processed_text += f"[EMAIL_TIME] {' '.join(time_matches[:2])} "
            
        processed_text += f"[EMAIL_BODY] {body} "
        
        if signature:
            processed_text += f"[EMAIL_SIGNATURE] {signature}"
            
        # Detect special contexts
        if re.search(r'\b(meet|meeting|calendar|schedule|appointment)\b', text, re.IGNORECASE):
            processed_text += " [MEETING_CONTEXT]"
            
        if re.search(r'\b(payment|invoice|receipt|transaction|money|cost|price|\$)\b', text, re.IGNORECASE):
            processed_text += " [FINANCIAL_CONTEXT]"
            
        if re.search(r'\b(urgent|asap|immediately|quickly|deadline)\b', text, re.IGNORECASE):
            processed_text += " [URGENT_CONTEXT]"
            
        return processed_text
    
    # Define advanced tokenization function for emails
    def tokenize_function(examples):
        # Preprocess email text to highlight structural elements
        preprocessed_texts = [preprocess_email_text(text) for text in examples["text"]]
        
        # Tokenize with enhanced handling
        return tokenizer(
            preprocessed_texts, 
            padding="max_length", 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )
    
    logger.info(f"Creating datasets with enhanced email preprocessing and max_length={max_length}")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels
    })
    
    val_dataset = Dataset.from_dict({
        "text": val_texts,
        "label": val_labels
    })
    
    # Tokenize datasets with batching for efficiency
    logger.info("Tokenizing datasets with email-specific preprocessing...")
    tokenize_start = time.time()
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1000)
    logger.info(f"Enhanced tokenization completed in {time.time() - tokenize_start:.2f} seconds")
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Load pretrained model with email-specific optimizations
    logger.info(f"Loading model: {model_name} with {num_labels} output classes and email-specific modifications")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Custom loss function to handle class imbalance if needed
    if use_class_weights:
        logger.info("Using weighted CrossEntropyLoss for handling class imbalance")
        
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        trainer_class = CustomTrainer
    else:
        trainer_class = Trainer
    
    # Define detailed compute_metrics function with comprehensive metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Calculate overall metrics
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0, labels=range(num_labels)
        )
        
        # Compile results
        results = {
            "accuracy": accuracy, 
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }
        
        # Add per-class metrics for ALL classes
        for i in range(num_labels):
            class_name = label_encoder.classes_[i]
            short_name = class_name[:10]  # Truncate long class names
            results[f"{short_name}_prec"] = per_class_precision[i]
            results[f"{short_name}_rec"] = per_class_recall[i]
            results[f"{short_name}_f1"] = per_class_f1[i]
            results[f"{short_name}_supp"] = support[i]
        
        return results
    
    # Calculate learning rate schedule and warmup steps
    num_train_examples = len(train_dataset)
    steps_per_epoch = num_train_examples // batch_size + (1 if num_train_examples % batch_size > 0 else 0)
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(int(total_steps * 0.1), 1000)  # 10% of total steps or max 1000
    
    # Set up advanced training arguments with learning rate scheduling
    logger.info(f"Setting up training with email-optimized parameters: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")
    logger.info(f"Using warmup_steps={warmup_steps}, total_steps={total_steps}")
    
    training_args = TrainingArguments(
        output_dir=f"{MODEL_DIR}/results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",  # Use weighted F1 for imbalanced classes
        greater_is_better=True,
        save_total_limit=2,  # Only keep the 2 best checkpoints
        report_to="none",  # Disable wandb, tensorboard, etc.
        disable_tqdm=False,  # Show progress bars
        logging_steps=max(10, steps_per_epoch // 10),  # Log ~10 times per epoch
        logging_dir=f"{MODEL_DIR}/logs",
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU is available
        dataloader_num_workers=4,  # Faster data loading
        # Advanced parameters
        warmup_steps=warmup_steps,  # Gradual warmup
        gradient_accumulation_steps=max(1, 16 // batch_size) if batch_size < 16 else 1,  # For larger effective batch sizes
        # Early stopping
        early_stopping_patience=2 if early_stopping_available else None,
    )
    
    # Set up callbacks
    callbacks = []
    if early_stopping_available:
        logger.info("Using early stopping with patience=2")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    
    # Initialize trainer with all optimizations
    logger.info("Initializing advanced email-optimized Trainer...")
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=callbacks if callbacks else None,
    )
    
    # Train the model
    logger.info("Starting transformer model training with optimized configuration...")
    train_start = time.time()
    train_result = trainer.train()
    train_duration = time.time() - train_start
    
    # Log training metrics
    for key, value in train_result.metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"Transformer training completed in {train_duration:.2f} seconds")
    
    # Evaluate
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info("Transformer model evaluation results:")
    for metric_name, value in eval_results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # Generate and analyze confusion matrix
    try:
        predictions = trainer.predict(val_dataset)
        preds = predictions.predictions.argmax(-1)
        
        # Get the most confused pairs of classes
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(val_labels, preds, labels=range(num_labels))
        
        # Calculate per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Print per-class accuracy
        logger.info("\nPer-class accuracy:")
        for i, accuracy in enumerate(per_class_acc):
            class_name = label_encoder.classes_[i]
            logger.info(f"  {class_name}: {accuracy:.4f} (support: {cm[i].sum()})")
        
        # Get confusion rates for non-diagonal elements
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_cm = cm / cm.sum(axis=1, keepdims=True)
            norm_cm = np.nan_to_num(norm_cm, nan=0)  # Replace NaN with 0
        
        np.fill_diagonal(norm_cm, 0)  # Zero out the diagonal
        
        # Find the top 5 most confused pairs
        if len(label_encoder.classes_) > 2:  # Only for multiclass
            n_show = min(10, len(label_encoder.classes_))
            logger.info(f"\nTop {n_show} most confused class pairs:")
            confusion_flat = [(i, j, norm_cm[i, j]) 
                             for i in range(norm_cm.shape[0]) 
                             for j in range(norm_cm.shape[1]) if i != j and norm_cm[i, j] > 0]
            most_confused = sorted(confusion_flat, key=lambda x: x[2], reverse=True)[:n_show]
            
            for i, j, rate in most_confused:
                true_class = label_encoder.classes_[i]
                pred_class = label_encoder.classes_[j]
                logger.info(f"  {true_class} → {pred_class}: {rate:.4f} confusion rate ({cm[i, j]} examples)")
    except Exception as e:
        logger.warning(f"Could not compute detailed confusion statistics: {e}")
    
    # Save the model if requested
    if save_model:
        logger.info("Saving enhanced transformer model and tokenizer...")
        model.save_pretrained(TRANSFORMER_MODEL_DIR)
        tokenizer.save_pretrained(TRANSFORMER_TOKENIZER_DIR)
        joblib.dump(label_encoder, f"{MODEL_DIR}/transformer_label_encoder.joblib")
        
        # Save detailed training configuration
        with open(f"{TRANSFORMER_MODEL_DIR}/training_config.json", 'w') as f:
            config = {
                "model_name": model_name,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "actual_epochs": train_result.metrics.get("epoch", epochs),
                "dataset_size": len(texts),
                "num_classes": num_labels,
                "class_distribution": {label_encoder.classes_[i]: int(count) for i, count in enumerate(label_counts)},
                "training_duration": train_duration,
                "email_specific_processing": True,
                "special_tokens": list(special_tokens["additional_special_tokens"]),
                "class_weighted_loss": use_class_weights,
                "warmup_steps": warmup_steps,
                "evaluation_results": {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                     for k, v in eval_results.items() if not k.startswith("eval_")}
            }
            
            # Add training config
            json.dump(config, f, indent=2)
            
        logger.info(f"Transformer model and configuration saved to {TRANSFORMER_MODEL_DIR}")
    
    total_time = time.time() - start_time
    logger.info(f"Total transformer model processing completed in {total_time:.2f} seconds")
    
    return model, tokenizer, label_encoder

def load_traditional_models():
    """Load the trained traditional ML classifier models with caching"""
    # Check if models are in cache and not expired
    current_time = time.time()
    cache_key = "traditional_models"
    
    if (cache_key in MODEL_CACHE and 
        cache_key in LAST_MODEL_LOAD_TIME and
        current_time - LAST_MODEL_LOAD_TIME[cache_key] < MODEL_CACHE_EXPIRY):
        logger.debug("Using cached traditional ML models")
        return MODEL_CACHE[cache_key]
    
    try:
        # Load models from disk
        logger.info("Loading trained traditional ML models from disk")
        ensemble = joblib.load(ENSEMBLE_MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        
        # Update cache
        MODEL_CACHE[cache_key] = (ensemble, vectorizer, label_encoder)
        LAST_MODEL_LOAD_TIME[cache_key] = current_time
        
        logger.info("Loaded trained traditional ML models")
        return ensemble, vectorizer, label_encoder
    except (FileNotFoundError, ModuleNotFoundError) as e:
        logger.warning(f"Could not load traditional models: {e}")
        return None, None, None

def load_transformer_model():
    """Load the trained transformer model with caching"""
    # Check if models are in cache and not expired
    current_time = time.time()
    cache_key = "transformer_model"
    
    if (cache_key in MODEL_CACHE and 
        cache_key in LAST_MODEL_LOAD_TIME and
        current_time - LAST_MODEL_LOAD_TIME[cache_key] < MODEL_CACHE_EXPIRY):
        logger.debug("Using cached transformer model")
        return MODEL_CACHE[cache_key]
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        logger.info("Loading transformer model from disk")
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_TOKENIZER_DIR)
        label_encoder = joblib.load(f"{MODEL_DIR}/transformer_label_encoder.joblib")
        
        # Update cache
        MODEL_CACHE[cache_key] = (model, tokenizer, label_encoder)
        LAST_MODEL_LOAD_TIME[cache_key] = current_time
        
        logger.info("Loaded trained transformer model")
        return model, tokenizer, label_encoder
    except (ImportError, FileNotFoundError, ModuleNotFoundError) as e:
        logger.warning(f"Could not load transformer model: {e}")
        if 'FileNotFoundError' in str(e):
            logger.info("Transformer model not found. Run with --train and --transformer=true to train one.")
        return None, None, None

def predict_with_traditional_models(subject: str, body: str, 
                                   ensemble=None, vectorizer=None, label_encoder=None) -> Tuple[str, float]:
    """Predict using traditional ML models"""
    if ensemble is None or vectorizer is None or label_encoder is None:
        ensemble, vectorizer, label_encoder = load_traditional_models()
    
    if ensemble is None:
        logger.warning("No trained traditional models available. Training new models...")
        texts, labels = create_training_data()
        ensemble, vectorizer, label_encoder, _ = train_traditional_models(texts, labels)
    
    # Combine subject and body
    combined_text = f"Subject: {subject}\n\nBody: {body}"
    
    # Transform the text
    features = vectorizer.transform([combined_text])
    
    # Get prediction and probabilities
    prediction = ensemble.predict(features)[0]
    probabilities = ensemble.predict_proba(features)[0]
    
    # Get the class name and confidence
    class_name = label_encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]
    
    return class_name, confidence

def predict_with_transformer(subject: str, body: str, 
                           model=None, tokenizer=None, label_encoder=None) -> Tuple[str, float, Dict[str, float]]:
    """Predict using transformer model with improved context handling and email-specific optimizations"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        if model is None or tokenizer is None or label_encoder is None:
            model, tokenizer, label_encoder = load_transformer_model()
        
        if model is None:
            raise FileNotFoundError("No transformer model available")
        
        # Format for better context understanding with special tokens for email parts
        # This helps the model distinguish between subject and body better
        formatted_text = f"Subject: {subject.strip()} [SEP] Body: {body.strip()}"
        
        # Extract key email entities for additional context
        # Look for dates, times, and email-specific patterns
        import re
        email_entities = []
        
        # Check for dates (MM/DD/YYYY, Month DD, YYYY, etc.)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',              # MM/DD/YY(YY)
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}\b'  # Month DD, YYYY
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, body, re.IGNORECASE)
            if matches:
                email_entities.extend(matches[:2])  # Limit to first 2 matches
        
        # Check for times (HH:MM AM/PM, etc.)
        time_matches = re.findall(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b', body, re.IGNORECASE)
        if time_matches:
            email_entities.extend(time_matches[:2])
        
        # Add special tag for meeting detection
        if re.search(r'\b(meet|meeting|calendar|schedule|appointment)\b', formatted_text, re.IGNORECASE):
            email_entities.append("[MEETING_CONTEXT]")
            
        # Add special tag for financial/transaction context
        if re.search(r'\b(pay|payment|transaction|money|credit|debit|$|€|£)\b', formatted_text, re.IGNORECASE):
            email_entities.append("[FINANCIAL_CONTEXT]")
            
        # Add special tag for urgency
        if re.search(r'\b(urgent|asap|immediately|quickly|deadline)\b', formatted_text, re.IGNORECASE):
            email_entities.append("[URGENT_CONTEXT]")
        
        # Add extracted entities to the formatted text if found
        if email_entities:
            entity_text = " | ".join(email_entities)
            formatted_text += f" [SEP] Entities: {entity_text}"
            
        # Tokenize text with proper handling of longer emails
        # This ensures we don't lose important information from truncation
        max_length = 512
        if len(body) > 1000:  # For long emails
            # Process the subject and first/last parts of the body to capture key info
            subject_part = f"Subject: {subject.strip()}"
            body_start = body[:800].strip()  # First part
            body_end = body[-200:].strip()  # Last part - often contains signatures/important conclusions
            formatted_text = f"{subject_part} [SEP] Body_Start: {body_start} [SEP] Body_End: {body_end}"
            if email_entities:
                formatted_text += f" [SEP] Entities: {entity_text}"
        
        # Tokenize with attention to special tokens
        inputs = tokenizer(formatted_text, 
                         return_tensors="pt", 
                         padding=True, 
                         truncation=True, 
                         max_length=max_length)
        
        # Run prediction with temperature scaling for better calibration
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply temperature scaling for better calibrated probabilities
            temperature = 1.5  # Higher values make probabilities less extreme/more spread out
            scaled_logits = outputs.logits / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
        
        # Get top 3 predictions and their confidences for ensemble voting
        top_probs, top_indices = torch.topk(probabilities[0], min(3, probabilities.shape[1]))
        
        # Convert to class names and confidence dictionary
        all_predictions = {}
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = label_encoder.inverse_transform([idx.item()])[0]
            all_predictions[class_name] = prob.item()
        
        # Get the predicted class and confidence
        prediction = top_indices[0].item()
        confidence = top_probs[0].item()
        
        # Get the class name
        class_name = label_encoder.inverse_transform([prediction])[0]
        
        return class_name, confidence, all_predictions
    
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Error using transformer model: {e}")
        return None, 0.0, {}

def predict_email_category(subject: str, body: str) -> Tuple[str, float]:
    """
    Predict email category using intelligent ensemble of models (transformer + traditional)
    with advanced weighting and email-specific heuristics
    """
    # Define email-specific heuristics that can help with classification
    def apply_email_heuristics(subject: str, body: str) -> Dict[str, float]:
        import re
        heuristic_scores = {}
        
        # Meeting detection - check for dates, times, and meeting-related words
        meeting_keywords = ['meet', 'meeting', 'calendar', 'schedule', 'invite', 'appointment']
        meeting_score = 0.0
        
        # Check subject for meeting keywords (subject has higher weight)
        if any(word in subject.lower() for word in meeting_keywords):
            meeting_score += 0.3
            
        # Check for date and time patterns
        date_pattern = re.compile(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b', re.IGNORECASE)
        time_pattern = re.compile(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b', re.IGNORECASE)
        
        if date_pattern.search(subject) or date_pattern.search(body[:500]):
            meeting_score += 0.2
        if time_pattern.search(subject) or time_pattern.search(body[:500]):
            meeting_score += 0.2
            
        if meeting_score > 0:
            heuristic_scores['meeting'] = min(meeting_score, 0.6)  # Cap at 0.6
            
        # Transaction/bill detection
        transaction_keywords = ['payment', 'invoice', 'receipt', 'transaction', 'order', 'purchase']
        financial_pattern = re.compile(r'\$\d+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP)\b')
        
        transaction_score = 0.0
        if any(word in subject.lower() for word in transaction_keywords):
            transaction_score += 0.3
        if financial_pattern.search(subject) or financial_pattern.search(body[:500]):
            transaction_score += 0.25
            
        if transaction_score > 0:
            heuristic_scores['transaction'] = min(transaction_score, 0.6)
            heuristic_scores['bill'] = min(transaction_score * 0.8, 0.5)  # Bills are related to transactions
            
        # Security notifications
        security_keywords = ['security', 'password', 'login', 'access', 'unauthorized', 'verify', 'verification']
        if any(word in subject.lower() for word in security_keywords):
            heuristic_scores['security'] = 0.4
            
        # Employment related
        if 'job application' in subject.lower() or 'resume' in subject.lower():
            heuristic_scores['job_application'] = 0.4
            
        if 'offer' in subject.lower() and any(word in body.lower() for word in ['salary', 'position', 'job']):
            heuristic_scores['employment_offer'] = 0.4
            
        return heuristic_scores
    
    # Get predictions from both models
    transformer_prediction, transformer_confidence, transformer_top_predictions = None, 0.0, {}
    traditional_prediction, traditional_confidence = None, 0.0
    
    # Try transformer model first
    if USE_TRANSFORMER:
        try:
            transformer_prediction, transformer_confidence, transformer_top_predictions = predict_with_transformer(subject, body)
            logger.debug(f"Transformer prediction: {transformer_prediction} ({transformer_confidence:.2f})")
        except Exception as e:
            logger.warning(f"Error with transformer prediction: {e}")
    
    # Always get traditional model prediction for ensemble
    traditional_prediction, traditional_confidence = predict_with_traditional_models(subject, body)
    logger.debug(f"Traditional prediction: {traditional_prediction} ({traditional_confidence:.2f})")
    
    # Apply email-specific heuristics
    heuristic_scores = apply_email_heuristics(subject, body)
    logger.debug(f"Heuristic insights: {heuristic_scores}")
    
    # Perform ensemble decision with intelligent weighting
    if USE_TRANSFORMER and transformer_prediction is not None:
        # Create final prediction scores dictionary
        final_scores = {}
        
        # Add traditional model scores
        final_scores[traditional_prediction] = traditional_confidence * 1.0  # Base weight for traditional model
        
        # Add transformer top predictions
        for pred, conf in transformer_top_predictions.items():
            if pred in final_scores:
                final_scores[pred] += conf * 1.2  # Higher weight for transformer model
            else:
                final_scores[pred] = conf * 1.2
                
        # Add heuristic boosts
        for category, score in heuristic_scores.items():
            if category in final_scores:
                final_scores[category] += score * 0.8  # Heuristics have moderate weight
            else:
                final_scores[category] = score * 0.8
        
        # Select the highest scoring category
        best_category = max(final_scores.items(), key=lambda x: x[1])
        
        # Calculate a calibrated confidence score
        # If multiple models agree, confidence should be higher
        base_confidence = best_category[1]
        agreement_boost = 0.0
        
        # Check agreement between models
        if transformer_prediction == traditional_prediction == best_category[0]:
            agreement_boost = 0.15  # Strong agreement
        elif transformer_prediction == best_category[0] or traditional_prediction == best_category[0]:
            agreement_boost = 0.05  # Partial agreement
            
        # If heuristics also support this prediction, add confidence
        if best_category[0] in heuristic_scores:
            agreement_boost += 0.05
            
        # Calculate final confidence (capped at 0.95)
        final_confidence = min(0.95, base_confidence + agreement_boost)
        
        logger.debug(f"Ensemble decision: {best_category[0]} with confidence {final_confidence:.2f}")
        return best_category[0], final_confidence
    
    # If transformer is not available or failed, use traditional with heuristic boost
    else:
        final_confidence = traditional_confidence
        
        # Apply heuristic boost if available
        if traditional_prediction in heuristic_scores:
            final_confidence = min(0.95, final_confidence + heuristic_scores[traditional_prediction] * 0.1)
            
        return traditional_prediction, final_confidence

async def create_email() -> Optional[str]:
    """Create a test email and return its ID"""
    # Create an email through the SQL endpoint which we know works
    logger.info("Creating test email...")
    
    # Format for sql router (confirmed working)
    email_data = {
        "thread_id": None,
        "sender": {"email": "john@example.com", "name": "John Smith"},
        "recipients": [{"email": "team@example.com", "name": "Team"}],
        "cc": [],
        "bcc": [],
        "subject": "Important meeting tomorrow at 2pm",
        "body": "Hi team,\n\nLet's meet tomorrow at 2pm in the conference room to discuss the project status.\n\nBest regards,\nJohn",
        "html_body": None,
        "labels": []
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Use the endpoint we know works
            response = await client.post(
                f"{BASE_URL}/emails/", 
                json=email_data,
                timeout=10.0
            )
            
            if response.status_code in (200, 201):
                # Parse the response to get the email ID
                try:
                    email_id = response.text.strip('"')
                    logger.info(f"Email created successfully with ID: {email_id}")
                    return email_id
                except Exception as e:
                    logger.error(f"Error parsing email ID from response: {str(e)}")
                    return None
            else:
                logger.error(f"Failed to create email: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error creating email: {str(e)}")
        return None

async def classify_email_api(email_id: str, subject: str, body: str) -> bool:
    """Classify an email using our ML model and send results to the API"""
    logger.info(f"Classifying email with ID: {email_id}")
    
    # Use our ML model to predict the category
    category, confidence = predict_email_category(subject, body)
    
    # Convert to one of the four supported categories if needed
    supported_categories = ["meeting", "promotion", "intro", "unknown"]
    
    # Map categories to supported ones based on similarity
    category_mapping = {
        "report": "meeting",      # Reports are similar to meetings (business context)
        "news": "intro",          # News announcements are like introductions
        "support": "meeting",     # Support requests involve communication like meetings
        "billing": "promotion",   # Billing is promotional/transactional
        "security": "meeting",    # Security alerts are important communications
        "invitation": "meeting",  # Invitations are like meeting requests
        "feedback": "intro",      # Feedback is introductory in nature
        "notification": "promotion", # Notifications are promotional
        "inquiry": "meeting"      # Inquiries involve communication like meetings
    }
    
    # Map to supported category if needed
    if category not in supported_categories:
        mapped_category = category_mapping.get(category, "unknown")
        logger.info(f"Mapped category '{category}' to supported category '{mapped_category}'")
        category = mapped_category
    
    # Format for the API
    classification_data = {
        "classification_type": category,
        "confidence": round(float(confidence), 2),
        "model_version": "advanced-transformer-1.0"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Use the endpoint we know works with form data
            response = await client.post(
                f"{BASE_URL}/emails/{email_id}/classify", 
                data=classification_data,
                timeout=10.0
            )
            
            if response.status_code in (200, 201, 202):
                logger.info(f"Email classified successfully as '{category}' with confidence {confidence:.2f}")
                return True
            else:
                logger.error(f"Failed to classify email: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error classifying email: {str(e)}")
        return False

async def get_email_content(email_id: str) -> Optional[Dict[str, str]]:
    """Get email content from the API"""
    logger.info(f"Fetching email content for: {email_id}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/emails/{email_id}",
                timeout=10.0
            )
            
            if response.status_code == 200:
                email_data = response.json()
                logger.info(f"Successfully retrieved email info")
                return {
                    "subject": email_data.get("subject", ""),
                    "body": email_data.get("body", "")
                }
            else:
                logger.warning(f"Failed to get email content: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error getting email content: {str(e)}")
        return None

async def get_classification(email_id: str) -> Optional[Dict[str, Any]]:
    """Get email classification results"""
    logger.info(f"Fetching classification for: {email_id}")
    
    try:
        async with httpx.AsyncClient() as client:
            # Get the classification from the results endpoint
            response = await client.get(
                f"{BASE_URL}/results/{email_id}?include_content=true",
                timeout=10.0
            )
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Successfully retrieved classification results")
                return results
            else:
                logger.warning(f"Failed to get classification: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error getting classification: {str(e)}")
        return None

async def benchmark_classifier(num_emails=20):
    """Benchmark the classifier against multiple synthetic emails"""
    logger.info(f"Benchmarking classifier with {num_emails} synthetic emails...")
    
    # Generate a variety of emails across all categories
    all_categories = list(EMAIL_CATEGORIES.keys())
    emails = generate_email_batch(num_emails, categories=all_categories)
    
    correct_count = 0
    category_stats = {}
    
    for i, email in enumerate(emails):
        logger.info(f"Testing email {i+1}/{num_emails}: {email['category']}")
        
        # Use our model to predict
        predicted_category, confidence = predict_email_category(
            email['subject'], 
            email['body']
        )
        
        # Map to supported category if needed
        supported_categories = ["meeting", "promotion", "intro", "unknown"]
        category_mapping = {
            "report": "meeting",
            "news": "intro",
            "support": "meeting",
            "billing": "promotion",
            "security": "meeting",
            "invitation": "meeting",
            "feedback": "intro",
            "notification": "promotion",
            "inquiry": "meeting"
        }
        
        expected_category = email['category']
        if expected_category not in supported_categories:
            expected_category = category_mapping.get(expected_category, "unknown")
        
        # For advanced categories, check against mapping for correctness
        is_correct = predicted_category == email['category'] or (
            predicted_category == category_mapping.get(email['category'], "unknown")
        )
        
        if is_correct:
            correct_count += 1
            
        # Track stats by category
        if email['category'] not in category_stats:
            category_stats[email['category']] = {
                'count': 0,
                'correct': 0,
                'confidence': 0
            }
            
        category_stats[email['category']]['count'] += 1
        if is_correct:
            category_stats[email['category']]['correct'] += 1
        category_stats[email['category']]['confidence'] += confidence
        
        logger.info(f"  Original: {email['category']}, Predicted: {predicted_category}, Confidence: {confidence:.2f}, Correct: {is_correct}")
        
    # Calculate accuracy
    accuracy = correct_count / num_emails if num_emails > 0 else 0
    logger.info(f"Overall accuracy: {accuracy:.2f} ({correct_count}/{num_emails})")
    
    # Print category stats
    logger.info("Category statistics:")
    for category, stats in category_stats.items():
        avg_confidence = stats['confidence'] / stats['count'] if stats['count'] > 0 else 0
        cat_accuracy = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
        logger.info(f"  {category}: Accuracy: {cat_accuracy:.2f}, Avg Confidence: {avg_confidence:.2f}, Count: {stats['count']}")
    
    return accuracy, category_stats

async def cross_validation_benchmark(num_folds=5, num_emails_per_fold=200, report_per_category=True):
    """
    Perform cross-validation benchmark to ensure model doesn't overfit
    
    Args:
        num_folds: Number of cross-validation folds
        num_emails_per_fold: Number of emails to generate per fold
        report_per_category: Whether to report metrics per category
        
    Returns:
        Tuple of (average_accuracy, fold_metrics)
    """
    logger.info(f"Running {num_folds}-fold cross-validation benchmark with {num_emails_per_fold} emails per fold...")
    
    # Track metrics
    fold_metrics = []
    all_category_stats = {}
    
    # Generate all unique categories
    all_categories = list(EMAIL_CATEGORIES.keys())
    
    # Generate a large pool of test emails for cross-validation
    total_emails = num_folds * num_emails_per_fold
    logger.info(f"Generating {total_emails} synthetic emails for cross-validation...")
    
    # Generate the emails
    all_emails = generate_email_batch(total_emails, categories=all_categories)
    
    # Shuffle the emails to ensure random distribution
    import random
    random.shuffle(all_emails)
    
    # Split into folds
    email_folds = [all_emails[i:i + num_emails_per_fold] for i in range(0, total_emails, num_emails_per_fold)]
    
    # Run cross-validation
    for fold_idx, fold_emails in enumerate(email_folds):
        logger.info(f"Processing fold {fold_idx + 1}/{num_folds} with {len(fold_emails)} emails...")
        
        # Fold metrics
        correct_count = 0
        fold_category_stats = {}
        
        # Process each email in the fold
        for i, email in enumerate(fold_emails):
            # Use our model to predict
            predicted_category, confidence = predict_email_category(
                email['subject'], 
                email['body']
            )
            
            # Check if prediction is correct (direct match)
            is_correct = predicted_category == email['category']
            
            # Track overall correctness
            if is_correct:
                correct_count += 1
                
            # Track stats by category for detailed reporting
            if email['category'] not in fold_category_stats:
                fold_category_stats[email['category']] = {
                    'count': 0,
                    'correct': 0,
                    'confidence': 0
                }
                
            fold_category_stats[email['category']]['count'] += 1
            if is_correct:
                fold_category_stats[email['category']]['correct'] += 1
            fold_category_stats[email['category']]['confidence'] += confidence
            
            # Periodically log progress for large folds
            if (i + 1) % 50 == 0 or i == len(fold_emails) - 1:
                logger.info(f"  Processed {i + 1}/{len(fold_emails)} emails in fold {fold_idx + 1}")
        
        # Calculate fold accuracy
        fold_accuracy = correct_count / len(fold_emails) if fold_emails else 0
        logger.info(f"Fold {fold_idx + 1} accuracy: {fold_accuracy:.4f} ({correct_count}/{len(fold_emails)})")
        
        # Store fold metrics
        fold_metrics.append({
            'fold': fold_idx + 1,
            'accuracy': fold_accuracy,
            'num_emails': len(fold_emails),
            'category_stats': fold_category_stats
        })
        
        # Aggregate category stats across folds
        for category, stats in fold_category_stats.items():
            if category not in all_category_stats:
                all_category_stats[category] = {
                    'count': 0,
                    'correct': 0,
                    'confidence': 0
                }
            
            all_category_stats[category]['count'] += stats['count']
            all_category_stats[category]['correct'] += stats['correct']
            all_category_stats[category]['confidence'] += stats['confidence']
        
    # Calculate average metrics across folds
    avg_accuracy = sum(fold['accuracy'] for fold in fold_metrics) / len(fold_metrics) if fold_metrics else 0
    logger.info(f"Cross-validation complete. Average accuracy across {num_folds} folds: {avg_accuracy:.4f}")
    
    # Calculate standard deviation of accuracy for stability assessment
    if len(fold_metrics) > 1:
        import numpy as np
        accuracy_std = np.std([fold['accuracy'] for fold in fold_metrics])
        logger.info(f"Standard deviation of accuracy: {accuracy_std:.4f}")
        
        # Assess model stability
        if accuracy_std > 0.05:
            logger.warning("High variance between folds (>0.05). Model may be unstable or data distribution varies significantly.")
        else:
            logger.info("Low variance between folds. Model appears stable across different data samples.")
    
    # Report per-category metrics
    if report_per_category:
        logger.info("\nCategory statistics across all folds:")
        for category, stats in sorted(all_category_stats.items(), key=lambda x: x[0]):
            avg_confidence = stats['confidence'] / stats['count'] if stats['count'] > 0 else 0
            cat_accuracy = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
            logger.info(f"  {category}: Accuracy: {cat_accuracy:.4f}, Avg Confidence: {avg_confidence:.4f}, Count: {stats['count']}")
    
    return avg_accuracy, fold_metrics

async def generate_confusion_matrix(num_samples=100):
    """Generate and display a confusion matrix for model evaluation"""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import numpy as np
    except ImportError:
        logger.error("Required packages not installed. Please run: pip install matplotlib")
        return None
    
    logger.info(f"Generating confusion matrix with {num_samples} samples...")
    
    # Generate test data
    all_categories = list(EMAIL_CATEGORIES.keys())
    emails = generate_email_batch(num_samples, categories=all_categories)
    
    true_labels = []
    predicted_labels = []
    
    # Get predictions for all emails
    for email in emails:
        true_labels.append(email['category'])
        predicted_category, _ = predict_email_category(email['subject'], email['body'])
        predicted_labels.append(predicted_category)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_categories)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_categories)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Email Classification Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{MODEL_DIR}/confusion_matrix.png")
    logger.info(f"Confusion matrix saved to {MODEL_DIR}/confusion_matrix.png")
    
    # Calculate per-category metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # Per-category precision and recall
    precision = np.zeros(len(all_categories))
    recall = np.zeros(len(all_categories))
    
    for i in range(len(all_categories)):
        # Precision = TP / (TP + FP)
        precision[i] = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        # Recall = TP / (TP + FN)
        recall[i] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
    
    # Print metrics
    logger.info(f"Overall accuracy: {accuracy:.4f}")
    logger.info("Per-category metrics:")
    
    for i, category in enumerate(all_categories):
        logger.info(f"  {category}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}")
    
    return cm, accuracy, precision, recall

async def interactive_classification():
    """Allow interactive classification of sample emails or custom input"""
    while True:
        print("\n--- Interactive Email Classification ---")
        print("1. Classify a sample email")
        print("2. Enter custom email")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ")
        
        if choice == "1":
            # Generate a random email
            category = random.choice(list(EMAIL_CATEGORIES.keys()))
            emails = generate_email_batch(1, [category])
            email = emails[0]
            
            print("\nSample Email:")
            print(f"Subject: {email['subject']}")
            print(f"Body: {email['body'][:100]}...")
            print(f"Actual Category: {email['category']}")
            
            # Classify with both models
            traditional_prediction, traditional_confidence = predict_with_traditional_models(
                email['subject'], email['body']
            )
            
            print("\nTraditional Model Classification:")
            print(f"Predicted Category: {traditional_prediction}")
            print(f"Confidence: {traditional_confidence:.2f}")
            
            # Try transformer if available
            if USE_TRANSFORMER:
                try:
                    transformer_prediction, transformer_confidence = predict_with_transformer(
                        email['subject'], email['body']
                    )
                    
                    if transformer_prediction is not None:
                        print("\nTransformer Model Classification:")
                        print(f"Predicted Category: {transformer_prediction}")
                        print(f"Confidence: {transformer_confidence:.2f}")
                except Exception as e:
                    print(f"\nTransformer model not available: {e}")
            
            # Get ensemble prediction
            final_prediction, final_confidence = predict_email_category(
                email['subject'], email['body']
            )
            
            print("\nFinal Ensemble Classification:")
            print(f"Predicted Category: {final_prediction}")
            print(f"Confidence: {final_confidence:.2f}")
            
        elif choice == "2":
            # Get custom input
            subject = input("\nEnter email subject: ")
            print("Enter email body (type 'END' on a new line when finished):")
            body_lines = []
            while True:
                line = input()
                if line == "END":
                    break
                body_lines.append(line)
            body = "\n".join(body_lines)
            
            # Classify with both models
            traditional_prediction, traditional_confidence = predict_with_traditional_models(
                subject, body
            )
            
            print("\nTraditional Model Classification:")
            print(f"Predicted Category: {traditional_prediction}")
            print(f"Confidence: {traditional_confidence:.2f}")
            
            # Try transformer if available
            if USE_TRANSFORMER:
                try:
                    transformer_prediction, transformer_confidence = predict_with_transformer(
                        subject, body
                    )
                    
                    if transformer_prediction is not None:
                        print("\nTransformer Model Classification:")
                        print(f"Predicted Category: {transformer_prediction}")
                        print(f"Confidence: {transformer_confidence:.2f}")
                except Exception as e:
                    print(f"\nTransformer model not available: {e}")
            
            # Get ensemble prediction
            final_prediction, final_confidence = predict_email_category(subject, body)
            
            print("\nFinal Ensemble Classification:")
            print(f"Predicted Category: {final_prediction}")
            print(f"Confidence: {final_confidence:.2f}")
            
            # Show the mapping to core categories if needed
            supported_categories = ["meeting", "promotion", "intro", "unknown"]
            if final_prediction not in supported_categories:
                category_mapping = {
                    "report": "meeting",
                    "news": "intro",
                    "support": "meeting",
                    "billing": "promotion",
                    "security": "meeting",
                    "invitation": "meeting",
                    "feedback": "intro",
                    "notification": "promotion",
                    "inquiry": "meeting"
                }
                mapped_category = category_mapping.get(final_prediction, "unknown")
                print(f"\nMapped to core category: {mapped_category}")
            
        elif choice == "3":
            print("Exiting interactive mode.")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

async def main():
    """Main function to run the email classification flow"""
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--train":
            # Train the models with synthetic data
            num_samples = 1000
            balanced = True
            augment = True
            use_transformer = True  # Default to true for better accuracy
            transformer_model = "distilbert-base-uncased"
            batch_size = 16          # Starting batch size (will be adjusted based on dataset size)
            epochs = 3               # Default epochs (will be adjusted based on dataset size)
            use_hyperparameter_tuning = False
            max_length = 512         # Maximum sequence length for transformer
            
            # Parse additional arguments
            for i in range(2, len(sys.argv)):
                arg = sys.argv[i]
                if arg.startswith("--samples="):
                    try:
                        num_samples = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid samples value: {arg}")
                        
                elif arg.startswith("--balanced="):
                    try:
                        balanced = arg.split("=")[1].lower() in ("true", "yes", "1")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid balanced value: {arg}")
                        
                elif arg.startswith("--augment="):
                    try:
                        augment = arg.split("=")[1].lower() in ("true", "yes", "1")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid augment value: {arg}")
                        
                elif arg.startswith("--transformer="):
                    try:
                        use_transformer = arg.split("=")[1].lower() in ("true", "yes", "1")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid transformer value: {arg}")
                        
                elif arg.startswith("--model="):
                    try:
                        transformer_model = arg.split("=")[1]
                    except IndexError:
                        logger.warning(f"Invalid model value: {arg}")
                        
                elif arg.startswith("--batch-size="):
                    try:
                        batch_size = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid batch size value: {arg}")
                        
                elif arg.startswith("--epochs="):
                    try:
                        epochs = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid epochs value: {arg}")
                        
                elif arg.startswith("--max-length="):
                    try:
                        max_length = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid max length value: {arg}")
                        
                elif arg.startswith("--hyperparameter-tuning="):
                    try:
                        use_hyperparameter_tuning = arg.split("=")[1].lower() in ("true", "yes", "1")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid hyperparameter tuning value: {arg}")
                elif arg.isdigit():
                    # Legacy support for specifying samples as a positional argument
                    num_samples = int(arg)
            
            logger.info(f"Training with {num_samples} samples (balanced={balanced}, augment={augment})")
            
            # Create training data with batch processing for large datasets
            data_batch_size = min(10000, num_samples // 10) if num_samples > 10000 else num_samples
            texts, labels = create_training_data(
                num_samples=num_samples,
                augment=augment,
                save=True,
                balanced=balanced,
                batch_size=data_batch_size
            )
            
            # Train traditional models
            logger.info("Training traditional models...")
            train_start_time = time.time()
            ensemble, vectorizer, label_encoder, trained_models = train_traditional_models(
                texts, labels, save_model=True, 
                use_hyperparameter_tuning=use_hyperparameter_tuning
            )
            logger.info(f"Traditional models training completed in {time.time() - train_start_time:.2f} seconds")
            
            # Try to train transformer model if requested and packages are available
            if use_transformer:
                try:
                    import torch
                    from transformers import AutoTokenizer
                    logger.info(f"Training transformer model (model={transformer_model}, batch_size={batch_size}, epochs={epochs}, max_length={max_length})...")
                    transformer_start_time = time.time()
                    model, tokenizer, t_label_encoder = train_transformer_model(
                        texts, labels, save_model=True,
                        batch_size=batch_size,
                        epochs=epochs,
                        model_name=transformer_model,
                        max_length=max_length
                    )
                    logger.info(f"Transformer model training completed in {time.time() - transformer_start_time:.2f} seconds")
                except ImportError:
                    logger.warning("Transformer training packages not available. Skipping transformer training.")
            
            return 0
        
        elif sys.argv[1] == "--benchmark":
            # Benchmark the classifier
            num_emails = 50
            if len(sys.argv) >= 3:
                try:
                    num_emails = int(sys.argv[2])
                except ValueError:
                    pass
            
            await benchmark_classifier(num_emails)
            return 0
            
        elif sys.argv[1] == "--cross-validate":
            # Perform cross-validation benchmark
            num_folds = 5
            emails_per_fold = 100
            
            # Parse additional arguments
            if len(sys.argv) >= 3:
                try:
                    num_folds = int(sys.argv[2])
                except ValueError:
                    pass
                
            if len(sys.argv) >= 4:
                try:
                    emails_per_fold = int(sys.argv[3])
                except ValueError:
                    pass
            
            await cross_validation_benchmark(
                num_folds=num_folds,
                num_emails_per_fold=emails_per_fold,
                report_per_category=True
            )
            return 0
        
        elif sys.argv[1] == "--interactive":
            # Interactive classification mode
            await interactive_classification()
            return 0
            
        elif sys.argv[1] == "--confusion-matrix":
            # Generate confusion matrix
            num_samples = 100
            if len(sys.argv) >= 3:
                try:
                    num_samples = int(sys.argv[2])
                except ValueError:
                    pass
            
            await generate_confusion_matrix(num_samples)
            return 0
            
        elif sys.argv[1] == "--generate-dataset":
            # Generate a dataset without training models
            num_samples = 100000
            balanced = True
            augment = True
            
            # Parse additional arguments
            for i in range(2, len(sys.argv)):
                arg = sys.argv[i]
                if arg.startswith("--samples="):
                    try:
                        num_samples = int(arg.split("=")[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid samples value: {arg}")
                        
                elif arg.startswith("--balanced="):
                    try:
                        balanced = arg.split("=")[1].lower() in ("true", "yes", "1")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid balanced value: {arg}")
                        
                elif arg.startswith("--augment="):
                    try:
                        augment = arg.split("=")[1].lower() in ("true", "yes", "1")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid augment value: {arg}")
                elif arg.isdigit():
                    # Legacy support for specifying samples as a positional argument
                    num_samples = int(arg)
            
            logger.info(f"Generating dataset with {num_samples} samples (balanced={balanced}, augment={augment})")
            
            # Set an optimal batch size based on dataset size
            batch_size = min(10000, num_samples // 10) if num_samples > 10000 else num_samples
            
            # Create training data with batch processing for large datasets
            create_training_data(
                num_samples=num_samples,
                augment=augment,
                save=True,
                balanced=balanced,
                batch_size=batch_size
            )
            
            logger.info("Dataset generation completed")
            return 0
            
        elif sys.argv[1] == "--help":
            # Display help information
            print("\nEmail Classifier - Usage Information")
            print("===================================\n")
            print("Available commands:\n")
            print("  --train [options]             Train the email classification models")
            print("    --samples=N                 Number of training samples (default: 1000)")
            print("    --balanced=true|false       Whether to balance classes (default: true)")
            print("    --augment=true|false        Whether to augment data (default: true)")
            print("    --transformer=true|false    Whether to train transformer model (default: true)")
            print("    --model=NAME                Transformer model to use (default: distilbert-base-uncased)")
            print("    --batch-size=N              Batch size for training (default: 16)")
            print("    --epochs=N                  Number of epochs (default: 3)")
            print("    --max-length=N              Maximum sequence length (default: 512)")
            print("\n  --benchmark [N]               Benchmark the classifier with N emails (default: 50)")
            print("\n  --cross-validate [folds] [emails_per_fold]")
            print("                               Run cross-validation (default: 5 folds, 100 emails per fold)")
            print("\n  --confusion-matrix [N]        Generate a confusion matrix with N samples (default: 100)")
            print("\n  --interactive                 Start interactive classification mode")
            print("\n  --generate-dataset [options]  Generate a dataset without training")
            print("    --samples=N                 Number of samples (default: 100000)")
            print("    --balanced=true|false       Whether to balance classes (default: true)")
            print("    --augment=true|false        Whether to augment data (default: true)")
            print("\n  --help                        Display this help information")
            
            return 0
    
    # Start the server if needed
    if not start_server():
        return 1
    
    try:
        # Create an email
        email_id = await create_email()
        if not email_id:
            logger.error("Failed to create email")
            return 1
        
        # Get email content
        email_content = await get_email_content(email_id)
        if not email_content:
            logger.error("Failed to get email content")
            return 1
        
        # Classify the email using our ML model
        if not await classify_email_api(
            email_id, 
            email_content['subject'], 
            email_content['body']
        ):
            logger.error("Failed to classify email")
            return 1
        
        # Wait a bit for processing
        logger.info("Waiting for email to be processed...")
        await asyncio.sleep(2)
        
        # Get the classification information
        results = await get_classification(email_id)
        if results:
            # Print the results
            logger.info("\n--- Classification Results ---")
            print(json.dumps(results, indent=2))
            return 0
        else:
            logger.warning("Could not retrieve detailed classification results")
            logger.info("\n--- Basic Classification Info ---")
            
            # Use our model to do a direct prediction
            predicted_category, confidence = predict_email_category(
                email_content['subject'], 
                email_content['body']
            )
            
            print(json.dumps({
                "email_id": email_id,
                "classification": predicted_category,
                "confidence": float(confidence),
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "Direct prediction from advanced ML model (not from API)"
            }, indent=2))
            return 0
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1
    
    finally:
        # Clean up
        stop_server()

if __name__ == "__main__":
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Make sure we have the required packages
    try:
        import joblib
    except ImportError:
        logger.error("Required packages not installed. Please run: pip install joblib scikit-learn")
        sys.exit(1)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        stop_server()
        sys.exit(130)