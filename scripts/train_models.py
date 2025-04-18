#!/usr/bin/env python3
"""
Advanced model training script for email classification.
Tests multiple model architectures and dataset configurations to find optimal performance.
"""
import os
import sys
import json
import time
import logging
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Import various classifier options
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Import experimental HistGradientBoosting
try:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    HIST_GBM_AVAILABLE = True
except ImportError:
    HIST_GBM_AVAILABLE = False

# Try importing specialized boosting libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Add parent directory to path to import from parent modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from email_generator import generate_email_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_models")

# Configuration
MODEL_DIR = os.path.join(parent_dir, "data/models")
DATASET_DIR = os.path.join(parent_dir, "data/datasets")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def generate_dataset(num_samples, balanced=True, augment=False, save=True):
    """Generate a synthetic email dataset for training"""
    logger.info(f"Generating {num_samples} email examples (balanced={balanced}, augment={augment})")
    
    if balanced:
        emails = []
        categories = ["meeting", "promotion", "notification", "security", 
                     "support", "introduction", "billing", "report", "survey"]
        per_category = num_samples // len(categories)
        
        for category in categories:
            logger.info(f"Generating {per_category} samples for category: {category}")
            category_emails = generate_email_batch(per_category, [category])
            emails.extend(category_emails)
    else:
        emails = generate_email_batch(num_samples)
    
    # Extract texts and labels
    texts = []
    labels = []
    
    for email in emails:
        combined_text = f"Subject: {email['subject']}\n\nBody: {email['body']}"
        texts.append(combined_text)
        labels.append(email['category'])
    
    logger.info(f"Generated dataset with {len(texts)} samples")
    
    # Save dataset if requested
    if save:
        dataset_path = os.path.join(DATASET_DIR, f"email_dataset_{len(texts)}.json")
        with open(dataset_path, 'w') as f:
            json.dump({
                "texts": texts,
                "labels": labels,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "num_samples": len(texts)
            }, f)
        logger.info(f"Saved dataset to {dataset_path}")
    
    return texts, labels

def load_dataset(dataset_path):
    """Load a dataset from a JSON file"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        texts = dataset.get("texts", [])
        labels = dataset.get("labels", [])
        
        logger.info(f"Loaded dataset with {len(texts)} samples")
        return texts, labels
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return [], []

def create_classifiers():
    """Create a dictionary of various classifier models to try"""
    classifiers = {
        "LogisticRegression": LogisticRegression(
            C=1.0, 
            solver='saga', 
            max_iter=1000, 
            random_state=42, 
            n_jobs=-1, 
            class_weight='balanced'
        ),
        "LinearSVC": LinearSVC(
            C=1.0, 
            class_weight='balanced', 
            random_state=42, 
            max_iter=1000, 
            dual=False
        ),
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=None, 
            min_samples_split=2, 
            random_state=42, 
            n_jobs=-1, 
            class_weight='balanced'
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        )
    }
    
    # Add HistGradientBoosting if available
    if HIST_GBM_AVAILABLE:
        classifiers["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3, 
            learning_rate=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        classifiers["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    logger.info(f"Created {len(classifiers)} classifier models")
    return classifiers

def create_vectorizers():
    """Create a dictionary of various vectorizer options to try"""
    vectorizers = {
        "TfidfVectorizer-10k": TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True
        ),
        "TfidfVectorizer-20k": TfidfVectorizer(
            max_features=20000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True
        ),
        "CountVectorizer-10k": CountVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )
    }
    
    logger.info(f"Created {len(vectorizers)} vectorizer options")
    return vectorizers

def train_and_evaluate_models(texts, labels, test_size=0.2, save_best=True):
    """Train and evaluate multiple model combinations"""
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
    )
    
    logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create vectorizers and classifiers
    vectorizers = create_vectorizers()
    classifiers = create_classifiers()
    
    results = []
    
    # Try all combinations of vectorizers and classifiers
    for vec_name, vectorizer in vectorizers.items():
        logger.info(f"Vectorizing data with {vec_name}")
        
        # Vectorize text data
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        logger.info(f"Vectorized training data shape: {X_train_vec.shape}")
        
        # Try each classifier
        for clf_name, classifier in classifiers.items():
            run_name = f"{vec_name}_{clf_name}"
            logger.info(f"Training model: {run_name}")
            
            # Time training
            start_time = time.time()
            classifier.fit(X_train_vec, y_train)
            train_time = time.time() - start_time
            
            # Predict and evaluate
            y_pred = classifier.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"{run_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, Training time={train_time:.2f}s")
            
            # Store results
            result = {
                "vectorizer": vec_name,
                "classifier": clf_name,
                "accuracy": accuracy,
                "f1": f1,
                "train_time": train_time,
                "model": classifier,
                "vectorizer_instance": vectorizer,
                "label_encoder": label_encoder
            }
            
            results.append(result)
    
    # Sort results by F1 score
    results.sort(key=lambda x: x["f1"], reverse=True)
    
    # Print top 3 models
    logger.info("\nTop 3 Models:")
    for i, result in enumerate(results[:3]):
        logger.info(f"{i+1}. {result['vectorizer']}_{result['classifier']}: F1={result['f1']:.4f}, Accuracy={result['accuracy']:.4f}")
    
    # Save best model if requested
    if save_best and results:
        best_result = results[0]
        best_name = f"{best_result['vectorizer']}_{best_result['classifier']}"
        
        # Save model components
        joblib.dump(best_result['model'], os.path.join(MODEL_DIR, f"best_classifier_{best_name}.joblib"))
        joblib.dump(best_result['vectorizer_instance'], os.path.join(MODEL_DIR, f"best_vectorizer_{best_name}.joblib"))
        joblib.dump(best_result['label_encoder'], os.path.join(MODEL_DIR, f"best_label_encoder_{best_name}.joblib"))
        
        # Save full results for reference
        with open(os.path.join(MODEL_DIR, "model_comparison_results.json"), 'w') as f:
            # Convert results to serializable format
            serializable_results = []
            for res in results:
                serializable_res = {
                    "vectorizer": res["vectorizer"],
                    "classifier": res["classifier"],
                    "accuracy": float(res["accuracy"]),
                    "f1": float(res["f1"]),
                    "train_time": float(res["train_time"])
                }
                serializable_results.append(serializable_res)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved best model '{best_name}' with F1={best_result['f1']:.4f}")
    
    return results

def find_optimal_dataset_size(start_size=100, max_size=10000, steps=5, best_model_only=True):
    """Find the optimal dataset size by training on progressively larger datasets"""
    # Generate sizes in logarithmic scale
    sizes = np.logspace(np.log10(start_size), np.log10(max_size), steps).astype(int)
    
    logger.info(f"Testing {steps} dataset sizes from {start_size} to {max_size} samples")
    
    size_results = []
    best_overall_f1 = 0
    best_size = 0
    best_config = None
    
    for size in sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing dataset size: {size} samples")
        logger.info(f"{'='*50}")
        
        # Generate dataset
        texts, labels = generate_dataset(size, balanced=True, save=True)
        
        # Train and evaluate models
        if best_model_only:
            # Create only the best models from previous research
            vectorizer = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True)
            
            if HIST_GBM_AVAILABLE:
                classifier = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, random_state=42)
                model_name = "HistGradientBoosting"
            elif LIGHTGBM_AVAILABLE:
                classifier = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)
                model_name = "LightGBM"
            else:
                classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                model_name = "GradientBoosting"
            
            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            # Vectorize
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Train
            start_time = time.time()
            classifier.fit(X_train_vec, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            y_pred = classifier.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results = [{
                "vectorizer": "TfidfVectorizer-10k",
                "classifier": model_name,
                "accuracy": accuracy,
                "f1": f1,
                "train_time": train_time,
                "model": classifier,
                "vectorizer_instance": vectorizer,
                "label_encoder": label_encoder
            }]
            
            logger.info(f"Size {size}: Accuracy={accuracy:.4f}, F1={f1:.4f}, Training time={train_time:.2f}s")
            
        else:
            # Test all model combinations
            results = train_and_evaluate_models(texts, labels, save_best=False)
        
        # Get best result for this size
        best_result = max(results, key=lambda x: x["f1"])
        
        size_result = {
            "size": size,
            "best_f1": best_result["f1"],
            "best_accuracy": best_result["accuracy"],
            "best_model": f"{best_result['vectorizer']}_{best_result['classifier']}",
            "train_time": best_result["train_time"]
        }
        
        size_results.append(size_result)
        
        # Keep track of best overall model
        if best_result["f1"] > best_overall_f1:
            best_overall_f1 = best_result["f1"]
            best_size = size
            best_config = best_result
    
    # Print results by size
    logger.info("\nResults by Dataset Size:")
    for result in size_results:
        logger.info(f"Size {result['size']}: F1={result['best_f1']:.4f}, Model={result['best_model']}")
    
    # Save the best overall model
    if best_config:
        best_name = f"{best_config['vectorizer']}_{best_config['classifier']}_size{best_size}"
        
        # Save model components
        joblib.dump(best_config['model'], os.path.join(MODEL_DIR, f"optimal_classifier_{best_name}.joblib"))
        joblib.dump(best_config['vectorizer_instance'], os.path.join(MODEL_DIR, f"optimal_vectorizer_{best_name}.joblib"))
        joblib.dump(best_config['label_encoder'], os.path.join(MODEL_DIR, f"optimal_label_encoder_{best_name}.joblib"))
        
        logger.info(f"\nSaved optimal model with size={best_size}, F1={best_overall_f1:.4f}")
    
    # Save size results
    with open(os.path.join(MODEL_DIR, "dataset_size_results.json"), 'w') as f:
        json.dump(size_results, f, indent=2)
    
    return size_results, best_size, best_overall_f1

def evaluate_model_on_test_dataset(model_path, vectorizer_path, label_encoder_path, test_dataset_path=None):
    """Evaluate a saved model on a separate test dataset"""
    # Load model components
    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    
    # Load or generate test dataset
    if test_dataset_path:
        texts, labels = load_dataset(test_dataset_path)
    else:
        # Generate a separate test dataset
        texts, labels = generate_dataset(2000, balanced=True, save=True)
    
    # Encode labels
    encoded_labels = label_encoder.transform(labels)
    
    # Vectorize text
    X_vec = vectorizer.transform(texts)
    
    # Predict
    y_pred = classifier.predict(X_vec)
    
    # Evaluate
    accuracy = accuracy_score(encoded_labels, y_pred)
    f1 = f1_score(encoded_labels, y_pred, average='weighted')
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Generate classification report
    report = classification_report(encoded_labels, y_pred, target_names=class_names)
    
    logger.info(f"\nModel Evaluation on Test Dataset:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    
    return accuracy, f1, report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate email classification models")
    parser.add_argument("--mode", type=str, choices=["train", "find-optimal", "evaluate"], default="train",
                      help="Mode of operation")
    parser.add_argument("--size", type=int, default=5000,
                      help="Number of samples to generate")
    parser.add_argument("--balanced", action="store_true", default=True,
                      help="Generate a balanced dataset across categories")
    parser.add_argument("--dataset", type=str, default=None,
                      help="Path to existing dataset (if not generating new one)")
    parser.add_argument("--test-dataset", type=str, default=None,
                      help="Path to test dataset for evaluation")
    parser.add_argument("--model", type=str, default=None,
                      help="Path to model file for evaluation")
    parser.add_argument("--vectorizer", type=str, default=None,
                      help="Path to vectorizer file for evaluation")
    parser.add_argument("--label-encoder", type=str, default=None,
                      help="Path to label encoder file for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Train and evaluate models
        if args.dataset:
            texts, labels = load_dataset(args.dataset)
        else:
            texts, labels = generate_dataset(args.size, balanced=args.balanced)
        
        train_and_evaluate_models(texts, labels)
    
    elif args.mode == "find-optimal":
        # Find optimal dataset size
        find_optimal_dataset_size(start_size=100, max_size=args.size, steps=5)
    
    elif args.mode == "evaluate":
        # Evaluate a specific model
        if not (args.model and args.vectorizer and args.label_encoder):
            logger.error("Model, vectorizer, and label encoder paths must be provided for evaluation")
            sys.exit(1)
        
        evaluate_model_on_test_dataset(
            args.model, 
            args.vectorizer, 
            args.label_encoder,
            args.test_dataset
        )