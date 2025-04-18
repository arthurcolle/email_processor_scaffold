#!/usr/bin/env python3
"""
SpamAssassin Integration with Enhanced Email Classification System

This script integrates the SpamAssassin public corpus with our enhanced email
classification system. It applies the existing classification model to the
SpamAssassin dataset and evaluates performance.

Usage:
    python spamassassin_classifier.py [--dataset PATH] [--model PATH] [--evaluate]

Options:
    --dataset PATH    Path to the SpamAssassin dataset JSON file
    --model PATH      Path to the existing email classifier model
    --evaluate        Run evaluation on the dataset
"""

import os
import sys
import argparse
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add parent directory to path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.enhanced_taxonomy import get_taxonomy, GRANULAR_CATEGORIES
from scripts.enhanced_enron_processor import extract_email_features, classify_email
from app.models.email import Email

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('spamassassin_classifier')

# Constants
# Use relative path from script location
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data"
SPAMASSASSIN_DIR = DATA_DIR / "spamassassin"
SAMPLE_DIR = SPAMASSASSIN_DIR / "samples"
DEFAULT_MODEL_PATH = DATA_DIR / "models" / "email_ensemble_model.joblib"
DEFAULT_DATASET_PATH = SAMPLE_DIR / "spamassassin_dataset_5000.json"
RESULTS_DIR = DATA_DIR / "results"


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load the SpamAssassin dataset from a JSON file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} emails from {dataset_path}")
    return dataset


def load_model(model_path: Path) -> Any:
    """Load the email classification model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    import joblib
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def adapt_spamassassin_to_enhanced_format(email_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SpamAssassin email format to a format compatible with our system."""
    # Map known fields
    adapted_email = {
        "id": email_data.get("file_name", ""),
        "subject": email_data.get("subject", ""),
        "body": email_data.get("body", ""),
        "sender": email_data.get("from", ""),
        "recipients": email_data.get("to", ""),
        "date": email_data.get("date", ""),
        "is_spam": email_data.get("is_spam", False),
        # Add additional metadata
        "metadata": {
            "source": "spamassassin",
            "content_type": email_data.get("content_type", ""),
            "message_id": email_data.get("message_id", ""),
            "file_path": email_data.get("file_path", ""),
            "processed_date": email_data.get("processed_date", "")
        }
    }
    
    # Add any header fields
    for key, value in email_data.items():
        if key.startswith("header_"):
            adapted_email["metadata"][key] = value
    
    return adapted_email


def classify_spamassassin_emails(dataset: List[Dict[str, Any]], model: Any) -> List[Dict[str, Any]]:
    """Classify SpamAssassin emails using our enhanced classification system."""
    results = []
    taxonomy = get_taxonomy()
    
    for email_data in tqdm(dataset, desc="Classifying emails"):
        try:
            # Adapt to our format
            adapted_email = adapt_spamassassin_to_enhanced_format(email_data)
            
            # Extract features
            features = extract_email_features(adapted_email)
            
            # Classify using the enhanced system
            classification_result = classify_email(
                email_data=adapted_email,
                extracted_features=features,
                model=model,
                taxonomy=taxonomy
            )
            
            # Add the classification result to the email data
            result = {
                "email_id": email_data.get("file_name", ""),
                "is_spam": email_data.get("is_spam", False),
                "classification": classification_result,
                "features": features
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error classifying email {email_data.get('file_name', '')}: {str(e)}")
    
    logger.info(f"Classified {len(results)} emails")
    return results


def evaluate_classification(classification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the classification results against the ground truth."""
    # Extract spam/ham ground truth and predictions
    y_true = [int(r["is_spam"]) for r in classification_results]
    
    # For spam detection, we'll consider any email with a high likelihood of being
    # one of our spam-related categories as spam
    spam_related_categories = [
        "urgency_high", "urgency_urgent", 
        "promotion", "advertisement", "marketing_campaign", 
        "newsletter", "mass_mailing"
    ]
    
    y_pred = []
    for r in classification_results:
        # Check if any spam-related category has a high confidence
        is_predicted_spam = False
        for category in r["classification"]["categories"]:
            if category["category_id"] in spam_related_categories and category["confidence"] > 0.7:
                is_predicted_spam = True
                break
        y_pred.append(int(is_predicted_spam))
    
    # Compute evaluation metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Log the results
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(results['confusion_matrix'])}")
    
    return results


def analyze_category_distribution(classification_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze the distribution of categories assigned to emails."""
    category_stats = {}
    taxonomy = get_taxonomy()
    
    # Initialize category stats
    for category_id, category_info in taxonomy.items():
        category_stats[category_id] = {
            "name": category_info["name"],
            "count": 0,
            "avg_confidence": 0.0,
            "spam_count": 0,
            "ham_count": 0
        }
    
    # Count occurrences
    for result in classification_results:
        is_spam = result["is_spam"]
        
        for category in result["classification"]["categories"]:
            category_id = category["category_id"]
            confidence = category["confidence"]
            
            if category_id in category_stats:
                category_stats[category_id]["count"] += 1
                category_stats[category_id]["avg_confidence"] += confidence
                
                if is_spam:
                    category_stats[category_id]["spam_count"] += 1
                else:
                    category_stats[category_id]["ham_count"] += 1
    
    # Calculate averages
    for category_id, stats in category_stats.items():
        if stats["count"] > 0:
            stats["avg_confidence"] /= stats["count"]
    
    # Sort by count
    sorted_stats = {k: v for k, v in sorted(
        category_stats.items(), 
        key=lambda item: item[1]["count"], 
        reverse=True
    )}
    
    return sorted_stats


def generate_report(classification_results: List[Dict[str, Any]], evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive report of the classification performance."""
    # Analyze category distribution
    category_stats = analyze_category_distribution(classification_results)
    
    # Prepare report
    report = {
        "dataset_size": len(classification_results),
        "spam_count": sum(1 for r in classification_results if r["is_spam"]),
        "ham_count": sum(1 for r in classification_results if not r["is_spam"]),
        "accuracy": evaluation_results["accuracy"],
        "precision": evaluation_results["precision"],
        "recall": evaluation_results["recall"],
        "f1": evaluation_results["f1"],
        "confusion_matrix": evaluation_results["confusion_matrix"],
        "top_categories": [
            {
                "category_id": cat_id,
                "name": stats["name"],
                "count": stats["count"],
                "percentage": stats["count"] / len(classification_results) * 100,
                "avg_confidence": stats["avg_confidence"],
                "spam_ratio": stats["spam_count"] / (stats["count"] or 1)
            }
            for cat_id, stats in list(category_stats.items())[:20]  # Top 20 categories
        ],
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save report
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    report_path = RESULTS_DIR / "spamassassin_classification_report.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Report saved to {report_path}")
    return report


def main() -> None:
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Integrate SpamAssassin with enhanced email classification")
    parser.add_argument('--dataset', type=str, help='Path to the SpamAssassin dataset',
                        default=str(DEFAULT_DATASET_PATH))
    parser.add_argument('--model', type=str, help='Path to the email classifier model',
                        default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on the dataset')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    
    try:
        # Load dataset and model
        dataset = load_dataset(dataset_path)
        model = load_model(model_path)
        
        # Classify emails
        classification_results = classify_spamassassin_emails(dataset, model)
        
        # Save classification results
        results_path = RESULTS_DIR / "spamassassin_classification_results.json"
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(classification_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Classification results saved to {results_path}")
        
        # Evaluate if requested
        if args.evaluate:
            evaluation_results = evaluate_classification(classification_results)
            report = generate_report(classification_results, evaluation_results)
            
            # Print summary
            logger.info("\nClassification Summary:")
            logger.info(f"Total emails: {report['dataset_size']}")
            logger.info(f"Spam emails: {report['spam_count']} ({report['spam_count']/report['dataset_size']*100:.1f}%)")
            logger.info(f"Ham emails: {report['ham_count']} ({report['ham_count']/report['dataset_size']*100:.1f}%)")
            logger.info(f"Accuracy: {report['accuracy']:.4f}")
            logger.info(f"Precision: {report['precision']:.4f}")
            logger.info(f"Recall: {report['recall']:.4f}")
            logger.info(f"F1 Score: {report['f1']:.4f}")
            
            logger.info("\nTop 5 Categories:")
            for i, cat in enumerate(report['top_categories'][:5]):
                logger.info(f"{i+1}. {cat['name']}: {cat['count']} emails ({cat['percentage']:.1f}%), " +
                           f"Avg Confidence: {cat['avg_confidence']:.3f}, Spam Ratio: {cat['spam_ratio']:.3f}")
    
    except Exception as e:
        logger.error(f"Error in SpamAssassin classification: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
