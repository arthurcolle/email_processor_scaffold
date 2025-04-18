#\!/usr/bin/env python3
"""
Test the email classifier for overfitting using the training dataset.
"""
import os
import sys
import json
import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from classify_email import predict_email_category

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_overfitting")

def load_dataset(path):
    """Load the dataset from a JSON file"""
    logger.info(f"Loading dataset from {path}")
    try:
        with open(path, 'r') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def extract_subject_body(text):
    """Extract subject and body from combined text"""
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        subject = parts[0].replace("Subject: ", "")
        body = parts[1].replace("Body: ", "")
        return subject, body
    else:
        # If the format is unexpected, return the whole text as body
        return "", text

def test_classifier_on_dataset(dataset_path, sample_size=None, k_fold=5, batch_size=50):
    """Test the classifier on the dataset using k-fold cross-validation
    
    Args:
        dataset_path: Path to the dataset JSON file
        sample_size: Number of samples to use (None for all)
        k_fold: Number of folds for cross-validation
        batch_size: Number of samples to process in a batch for efficiency
    """
    # Cache models to avoid reloading for each prediction
    from classify_email import load_traditional_models, predict_email_category
    
    # Preload models once
    _ = load_traditional_models()
    
    dataset = load_dataset(dataset_path)
    if not dataset:
        return
    
    texts = dataset.get("texts", [])
    labels = dataset.get("labels", [])
    
    if not texts or not labels:
        logger.error("Dataset does not contain texts or labels")
        return
    
    logger.info(f"Dataset loaded with {len(texts)} samples")
    
    # Sample a subset if requested
    if sample_size and sample_size < len(texts):
        logger.info(f"Sampling {sample_size} examples from the dataset")
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_f1_scores = []
    all_predictions = []
    all_true_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(texts)):
        logger.info(f"Processing fold {fold+1}/{k_fold}")
        
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        # Test on this fold
        predictions = []
        confidences = []
        
        # Process in batches for efficiency
        for batch_start in range(0, len(test_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(test_texts))
            if batch_start % 100 == 0:
                logger.info(f"  Processed {batch_start}/{len(test_texts)} samples in fold {fold+1}")
            
            batch_texts = test_texts[batch_start:batch_end]
            batch_predictions = []
            batch_confidences = []
            
            for text in batch_texts:
                subject, body = extract_subject_body(text)
                prediction, confidence = predict_email_category(subject, body)
                batch_predictions.append(prediction)
                batch_confidences.append(confidence)
            
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        
        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1)
        
        logger.info(f"Fold {fold+1} results: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        # Save predictions for overall analysis
        all_predictions.extend(predictions)
        all_true_labels.extend(test_labels)
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    
    # Generate detailed report
    logger.info("\n===== OVERALL RESULTS =====")
    logger.info(f"Average accuracy across {k_fold} folds: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
    logger.info(f"Average F1 score across {k_fold} folds: {np.mean(fold_f1_scores):.4f} (±{np.std(fold_f1_scores):.4f})")
    logger.info(f"Overall accuracy on all folds: {overall_accuracy:.4f}")
    logger.info(f"Overall F1 score on all folds: {overall_f1:.4f}")
    
    # Print detailed classification report
    logger.info("\nClassification Report:")
    report = classification_report(all_true_labels, all_predictions)
    print(report)
    
    # Generate confusion matrix for the most confused pairs
    cm = confusion_matrix(all_true_labels, all_predictions)
    unique_labels = list(set(all_true_labels))
    
    # Find the most confused pairs
    n_display = min(10, len(unique_labels))
    confusion_scores = []
    
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if i != j and cm[i][j] > 0:
                # Confusion rate: misclassifications / total samples of true class
                confusion_rate = cm[i][j] / max(1, np.sum(cm[i]))
                confusion_scores.append((unique_labels[i], unique_labels[j], cm[i][j], confusion_rate))
    
    if confusion_scores:
        # Sort by absolute number of confusions
        logger.info("\nTop confused pairs (by count):")
        for true_label, pred_label, count, rate in sorted(confusion_scores, key=lambda x: x[2], reverse=True)[:n_display]:
            logger.info(f"  {true_label} → {pred_label}: {count} samples ({rate:.1%} of {true_label} class)")
        
        # Sort by confusion rate
        logger.info("\nTop confused pairs (by rate):")
        for true_label, pred_label, count, rate in sorted(confusion_scores, key=lambda x: x[3], reverse=True)[:n_display]:
            logger.info(f"  {true_label} → {pred_label}: {rate:.1%} of {true_label} class ({count} samples)")
    else:
        logger.info("\nNo confusion between classes - perfect classification!")
    
    # Analysis of potential overfitting
    if overall_accuracy > 0.99:
        logger.warning("\nPOTENTIAL OVERFITTING DETECTED: Accuracy is extremely high (>99%)")
        logger.warning("This may indicate that the model is memorizing the training data rather than generalizing.")
        logger.warning("Consider using a completely separate test dataset that wasn't used in training.")
    elif overall_accuracy > 0.95:
        logger.info("\nHigh accuracy (>95%) suggests the model is performing very well,")
        logger.info("but should be validated on new, unseen data to ensure it generalizes properly.")
    
    return overall_accuracy, overall_f1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the email classifier for overfitting")
    parser.add_argument("--dataset", type=str, 
                      default=os.path.join("data", "datasets", "email_dataset_12480.json"), 
                      help="Path to the dataset JSON file")
    parser.add_argument("--samples", type=int, default=None, 
                      help="Number of samples to use (None for all)")
    parser.add_argument("--folds", type=int, default=5,
                      help="Number of folds for cross-validation")
    
    args = parser.parse_args()
    
    # Run the test
    test_classifier_on_dataset(args.dataset, args.samples, args.folds)
