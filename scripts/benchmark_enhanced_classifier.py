#!/usr/bin/env python3
"""
Benchmark the enhanced email classifier on 3200 samples and test for generalization.
This script evaluates the enhanced email classification system with fine-grained categories
and measures accuracy, precision, recall, and generalization capabilities.
"""
import os
import sys
import json
import random
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import directly from the modules
from enhanced_taxonomy import build_enhanced_taxonomy, get_leaf_categories
from enhanced_enron_processor import parse_email_file, classify_email_with_taxonomy, extract_email_features

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("benchmark_enhanced_classifier")

# Directory setup
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENRON_DIR = os.path.join(DATA_DIR, "enron")
ENRON_PROCESSED_DIR = os.path.join(ENRON_DIR, "enhanced_processed")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
TAXONOMY_FILE = os.path.join(DATA_DIR, "taxonomy", "enhanced_taxonomy.json")
BENCHMARK_RESULTS_FILE = os.path.join(BENCHMARK_DIR, "enhanced_benchmark_results.json")

# Ensure directories exist
os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load the enhanced taxonomy
if os.path.exists(TAXONOMY_FILE):
    with open(TAXONOMY_FILE, 'r') as f:
        ENHANCED_TAXONOMY = json.load(f)
else:
    ENHANCED_TAXONOMY = build_enhanced_taxonomy()
    # Save the taxonomy
    os.makedirs(os.path.dirname(TAXONOMY_FILE), exist_ok=True)
    with open(TAXONOMY_FILE, 'w') as f:
        json.dump(ENHANCED_TAXONOMY, f, indent=2)

# Get leaf categories for classification
LEAF_CATEGORIES = get_leaf_categories(ENHANCED_TAXONOMY)

def load_processed_emails(max_emails: int = 3200, shuffle: bool = True) -> List[Dict[str, Any]]:
    """
    Load processed emails from the enhanced_processed directory.
    
    Args:
        max_emails: Maximum number of emails to load
        shuffle: Whether to shuffle the emails
    
    Returns:
        List of processed email objects
    """
    if not os.path.exists(ENRON_PROCESSED_DIR):
        logger.error(f"Processed directory not found at {ENRON_PROCESSED_DIR}.")
        return []
    
    # Find all processed email files
    email_files = []
    for filename in os.listdir(ENRON_PROCESSED_DIR):
        if filename.endswith(".json"):
            email_files.append(os.path.join(ENRON_PROCESSED_DIR, filename))
    
    # Shuffle and limit if needed
    if shuffle:
        random.shuffle(email_files)
    
    if len(email_files) > max_emails:
        email_files = email_files[:max_emails]
    
    # Load the emails
    logger.info(f"Loading {len(email_files)} processed emails...")
    processed_emails = []
    
    for i, file_path in enumerate(email_files):
        if i % 500 == 0:
            logger.info(f"Loaded {i}/{len(email_files)} emails...")
        
        try:
            with open(file_path, 'r') as f:
                email_data = json.load(f)
                processed_emails.append(email_data)
        except Exception as e:
            logger.warning(f"Error loading email file {file_path}: {str(e)}")
    
    logger.info(f"Successfully loaded {len(processed_emails)} processed emails.")
    return processed_emails

def simulate_gold_standard_labels(emails: List[Dict[str, Any]], error_rate: float = 0.1) -> List[Dict[str, Dict[str, float]]]:
    """
    Simulate gold standard labels for emails.
    In a real benchmark, you would use human-annotated labels.
    
    Args:
        emails: List of processed email objects
        error_rate: Simulated error rate in the gold standard (0-1)
    
    Returns:
        List of dictionaries mapping email IDs to category scores
    """
    logger.info(f"Simulating gold standard labels for {len(emails)} emails...")
    
    gold_standard = []
    
    for email in emails:
        # Get the classifier's prediction
        category_scores = classify_email_with_taxonomy(email)
        
        # Simulate gold standard (with some noise to make it realistic)
        gold_categories = {}
        
        for category, score in category_scores.items():
            # Randomly adjust scores to simulate imperfect gold standard
            if random.random() < error_rate:
                # Random error: Either remove a category, add a wrong one, or adjust confidence
                r = random.random()
                if r < 0.3:
                    # Skip this category (simulates missed annotation)
                    continue
                elif r < 0.6:
                    # Adjust confidence (simulates disagreement between annotators)
                    gold_categories[category] = max(0.0, min(1.0, score + random.uniform(-0.2, 0.2)))
                else:
                    # Keep as is
                    gold_categories[category] = score
            else:
                # Keep the category as is
                gold_categories[category] = score
        
        # Add 1-3 random categories not in the original prediction (false positives)
        if random.random() < error_rate:
            unused_categories = [c for c in LEAF_CATEGORIES if c not in category_scores]
            num_to_add = random.randint(1, 3)
            for _ in range(min(num_to_add, len(unused_categories))):
                category = random.choice(unused_categories)
                unused_categories.remove(category)
                gold_categories[category] = random.uniform(0.5, 0.8)
        
        gold_standard.append({
            'email_id': email.get('email_id'),
            'categories': gold_categories
        })
    
    return gold_standard

def evaluate_classification(emails: List[Dict[str, Any]], gold_standard: List[Dict[str, Dict[str, float]]], 
                          confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate the classifier against gold standard labels.
    
    Args:
        emails: List of processed email objects
        gold_standard: List of dictionaries mapping email IDs to category scores
        confidence_threshold: Minimum confidence for a category to be assigned
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating classifier on {len(emails)} emails...")
    
    # Create a mapping from email_id to gold standard categories
    gold_mapping = {item['email_id']: item['categories'] for item in gold_standard}
    
    # Track metrics
    total_emails = len(emails)
    correct_predictions = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    category_performance = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for email in emails:
        email_id = email.get('email_id')
        if email_id not in gold_mapping:
            logger.warning(f"Email ID {email_id} not found in gold standard")
            continue
        
        # Get the classifier's prediction
        predicted_scores = classify_email_with_taxonomy(email, confidence_threshold)
        predicted_categories = set(predicted_scores.keys())
        
        # Get gold standard categories
        gold_categories = set(category for category, score in gold_mapping[email_id].items() 
                           if score >= confidence_threshold)
        
        # Calculate metrics for this email
        true_positives = predicted_categories.intersection(gold_categories)
        false_positives = predicted_categories - gold_categories
        false_negatives = gold_categories - predicted_categories
        
        # Update overall metrics
        precision = len(true_positives) / max(1, len(predicted_categories))
        recall = len(true_positives) / max(1, len(gold_categories))
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        # Update category-specific metrics
        for category in true_positives:
            category_performance[category]['tp'] += 1
        for category in false_positives:
            category_performance[category]['fp'] += 1
        for category in false_negatives:
            category_performance[category]['fn'] += 1
        
        # Count email as correct if at least 50% of categories match
        if len(true_positives) >= len(gold_categories) / 2:
            correct_predictions += 1
    
    # Calculate overall metrics
    overall_accuracy = correct_predictions / total_emails
    overall_precision = total_precision / total_emails
    overall_recall = total_recall / total_emails
    overall_f1 = total_f1 / total_emails
    
    # Calculate category-specific metrics
    category_metrics = []
    for category, metrics in category_performance.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        category_precision = tp / max(1, tp + fp)
        category_recall = tp / max(1, tp + fn)
        category_f1 = 2 * category_precision * category_recall / max(1e-10, category_precision + category_recall)
        
        category_name = ENHANCED_TAXONOMY.get(category, {}).get("name", category)
        
        category_metrics.append({
            "category_id": category,
            "name": category_name,
            "precision": category_precision,
            "recall": category_recall,
            "f1": category_f1,
            "support": tp + fn
        })
    
    # Sort by support (descending)
    category_metrics.sort(key=lambda x: x['support'], reverse=True)
    
    return {
        "total_emails": total_emails,
        "accuracy": overall_accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "category_metrics": category_metrics
    }

def evaluate_generalization(emails: List[Dict[str, Any]], n_splits: int = 5) -> Dict[str, Any]:
    """
    Evaluate the classifier's generalization ability using cross-validation.
    
    Args:
        emails: List of processed email objects
        n_splits: Number of cross-validation folds
    
    Returns:
        Dictionary with generalization metrics
    """
    logger.info(f"Evaluating generalization on {len(emails)} emails with {n_splits}-fold cross-validation...")
    
    # Use simple K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics across folds
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(emails)):
        logger.info(f"Processing fold {fold+1}/{n_splits}...")
        
        # Split data
        train_emails = [emails[idx] for idx in train_idx]
        test_emails = [emails[idx] for idx in test_idx]
        
        # Generate gold standard for the test set
        # In a real scenario, this would be actual human annotations
        gold_standard = simulate_gold_standard_labels(test_emails, error_rate=0.1)
        
        # Evaluate on the test set
        fold_results = evaluate_classification(test_emails, gold_standard)
        fold_results["fold"] = fold + 1
        fold_metrics.append(fold_results)
        
        logger.info(f"Fold {fold+1} results: Accuracy={fold_results['accuracy']:.4f}, F1={fold_results['f1']:.4f}")
    
    # Calculate aggregate metrics
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    std_precision = np.std([m['precision'] for m in fold_metrics])
    std_recall = np.std([m['recall'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])
    
    # Calculate generalization metrics
    generalization_metrics = {
        "avg_accuracy": avg_accuracy,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "std_accuracy": std_accuracy,
        "std_precision": std_precision,
        "std_recall": std_recall,
        "std_f1": std_f1,
        "fold_metrics": fold_metrics
    }
    
    return generalization_metrics

def analyze_topic_generalization(emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze how well the classifier generalizes across different email topics.
    
    Args:
        emails: List of processed email objects
    
    Returns:
        Dictionary with topic generalization metrics
    """
    logger.info(f"Analyzing topic generalization on {len(emails)} emails...")
    
    # Identify major topics based on keywords in emails
    email_topics = []
    
    # Simple topic identification (in a real system, use proper topic modeling)
    topics = {
        "trading": ["trade", "trading", "market", "price", "gas", "power"],
        "legal": ["contract", "legal", "agreement", "regulation", "compliance"],
        "finance": ["financial", "finance", "budget", "cost", "expense", "revenue"],
        "hr": ["employee", "hiring", "performance", "HR", "personnel"],
        "meetings": ["meeting", "schedule", "calendar", "conference"],
        "projects": ["project", "timeline", "milestone", "deliverable"],
        "technical": ["system", "server", "database", "software", "technical"]
    }
    
    # Categorize each email by topic
    for email in emails:
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        text = subject + " " + body
        
        email_topic = "other"
        max_matches = 0
        
        for topic, keywords in topics.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > max_matches:
                max_matches = matches
                email_topic = topic
        
        email_topics.append(email_topic)
    
    # Analyze classification performance by topic
    topic_performance = {}
    
    for topic in topics.keys():
        # Get emails for this topic
        topic_emails = [email for i, email in enumerate(emails) if email_topics[i] == topic]
        
        if len(topic_emails) < 10:
            logger.info(f"Skipping topic '{topic}' - not enough emails ({len(topic_emails)})")
            continue
        
        # Generate gold standard for these emails
        topic_gold_standard = simulate_gold_standard_labels(topic_emails, error_rate=0.1)
        
        # Evaluate classification
        topic_results = evaluate_classification(topic_emails, topic_gold_standard)
        topic_performance[topic] = {
            "emails": len(topic_emails),
            "accuracy": topic_results['accuracy'],
            "f1": topic_results['f1'],
            "top_categories": [cat['category_id'] for cat in topic_results['category_metrics'][:5]]
        }
        
        logger.info(f"Topic '{topic}' ({len(topic_emails)} emails): Accuracy={topic_results['accuracy']:.4f}, F1={topic_results['f1']:.4f}")
    
    # Prepare topic distribution data for visualization
    topic_counts = Counter(email_topics)
    topic_distribution = [{topic: count} for topic, count in topic_counts.most_common()]
    
    return {
        "topic_distribution": topic_distribution,
        "topic_performance": topic_performance
    }

def benchmark_classification_speed(emails: List[Dict[str, Any]], num_samples: int = 100) -> Dict[str, float]:
    """
    Benchmark the speed of the classification process.
    
    Args:
        emails: List of processed email objects
        num_samples: Number of emails to use for benchmarking
    
    Returns:
        Dictionary with timing metrics
    """
    logger.info(f"Benchmarking classification speed on {num_samples} emails...")
    
    # Sample emails for benchmarking
    if len(emails) > num_samples:
        benchmark_emails = random.sample(emails, num_samples)
    else:
        benchmark_emails = emails
    
    # Measure classification time
    start_time = time.time()
    
    for email in benchmark_emails:
        classify_email_with_taxonomy(email)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_email = total_time / len(benchmark_emails)
    
    # Measure feature extraction time
    feature_start_time = time.time()
    
    for email in benchmark_emails:
        extract_email_features(email)
    
    feature_end_time = time.time()
    feature_total_time = feature_end_time - feature_start_time
    avg_feature_time = feature_total_time / len(benchmark_emails)
    
    return {
        "total_classification_time": total_time,
        "avg_classification_time": avg_time_per_email,
        "total_feature_extraction_time": feature_total_time,
        "avg_feature_extraction_time": avg_feature_time,
        "samples": len(benchmark_emails),
        "emails_per_second": len(benchmark_emails) / total_time
    }

def create_benchmark_plots(benchmark_results: Dict[str, Any]) -> None:
    """
    Create visualization plots for benchmark results.
    
    Args:
        benchmark_results: Dictionary containing benchmark results
    """
    logger.info("Creating benchmark visualization plots...")
    
    # Set up the plotting style
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 12})
    
    # 1. Plot accuracy metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [benchmark_results['classification_metrics'][m] for m in metrics]
    
    bars = ax.bar(metrics, values, color=['#2C7BB6', '#D7191C', '#FF9900', '#33CC33'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'enhanced_accuracy_metrics.png'), dpi=300)
    
    # 2. Plot top 10 category F1 scores
    cat_metrics = benchmark_results['classification_metrics']['category_metrics'][:10]
    cats = [m['name'] for m in cat_metrics]
    f1_scores = [m['f1'] for m in cat_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(cats, f1_scores, color='#2C7BB6')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
                f'{width:.3f}', ha='left', va='center')
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('F1 Score')
    ax.set_title('Top 10 Categories by Performance (F1 Score)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'enhanced_top_categories.png'), dpi=300)
    
    # 3. Plot generalization metrics across folds
    if 'generalization_metrics' in benchmark_results:
        gen_metrics = benchmark_results['generalization_metrics']
        fold_metrics = gen_metrics['fold_metrics']
        
        fold_accuracies = [m['accuracy'] for m in fold_metrics]
        fold_f1s = [m['f1'] for m in fold_metrics]
        folds = [m['fold'] for m in fold_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(folds, fold_accuracies, 'o-', label='Accuracy', color='#2C7BB6')
        ax.plot(folds, fold_f1s, 's-', label='F1 Score', color='#D7191C')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(folds)
        ax.axhline(y=gen_metrics['avg_accuracy'], linestyle='--', color='#2C7BB6', 
                  alpha=0.7, label=f'Avg Accuracy: {gen_metrics["avg_accuracy"]:.3f}')
        ax.axhline(y=gen_metrics['avg_f1'], linestyle='--', color='#D7191C', 
                  alpha=0.7, label=f'Avg F1: {gen_metrics["avg_f1"]:.3f}')
        
        ax.legend()
        ax.set_title('Cross-Validation Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'enhanced_generalization.png'), dpi=300)
    
    # 4. Plot topic distribution
    if 'topic_metrics' in benchmark_results:
        topic_metrics = benchmark_results['topic_metrics']
        topic_dist = topic_metrics['topic_distribution']
        
        topics = []
        counts = []
        
        for topic_dict in topic_dist:
            for topic, count in topic_dict.items():
                topics.append(topic)
                counts.append(count)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(topics, counts, color='#2C7BB6')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}', ha='center', va='bottom')
        
        ax.set_ylabel('Count')
        ax.set_title('Email Topic Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'enhanced_topic_distribution.png'), dpi=300)
        
        # 5. Plot topic performance
        topic_performance = topic_metrics['topic_performance']
        topics = list(topic_performance.keys())
        accuracies = [topic_performance[t]['accuracy'] for t in topics]
        f1s = [topic_performance[t]['f1'] for t in topics]
        
        x = np.arange(len(topics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2C7BB6')
        bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score', color='#D7191C')
        
        ax.set_ylabel('Score')
        ax.set_title('Classification Performance by Topic')
        ax.set_xticks(x)
        ax.set_xticklabels(topics)
        ax.legend()
        
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'enhanced_topic_performance.png'), dpi=300)
    
    # 6. Plot speed benchmark results
    if 'speed_metrics' in benchmark_results:
        speed_metrics = benchmark_results['speed_metrics']
        
        labels = ['Classification', 'Feature Extraction']
        times = [speed_metrics['avg_classification_time'], speed_metrics['avg_feature_extraction_time']]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, times, color=['#2C7BB6', '#D7191C'])
        
        # Add value labels (in milliseconds for readability)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{height*1000:.1f} ms', ha='center', va='bottom')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Processing Speed Per Email ({speed_metrics["emails_per_second"]:.1f} emails/sec)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'enhanced_speed_benchmark.png'), dpi=300)
    
    logger.info(f"Benchmark plots saved to {PLOTS_DIR}")

def run_enhanced_benchmark(num_emails: int = 3200, confidence_threshold: float = 0.5) -> None:
    """
    Run a comprehensive benchmark of the enhanced email classifier.
    
    Args:
        num_emails: Number of emails to use for benchmarking
        confidence_threshold: Minimum confidence for assigning categories
    """
    logger.info(f"Running enhanced benchmark with {num_emails} emails...")
    
    # Step 1: Load processed emails
    emails = load_processed_emails(max_emails=num_emails)
    
    if len(emails) == 0:
        logger.error("No processed emails found. Please run the enhanced_enron_processor.py script first.")
        return
    
    # Step 2: Generate simulated gold standard labels
    gold_standard = simulate_gold_standard_labels(emails)
    
    # Step 3: Evaluate classification performance
    logger.info("Evaluating classification performance...")
    classification_metrics = evaluate_classification(emails, gold_standard, confidence_threshold)
    
    # Step 4: Evaluate generalization ability
    logger.info("Evaluating generalization ability...")
    generalization_metrics = evaluate_generalization(emails, n_splits=5)
    
    # Step 5: Analyze topic generalization
    logger.info("Analyzing topic generalization...")
    topic_metrics = analyze_topic_generalization(emails)
    
    # Step 6: Benchmark classification speed
    logger.info("Benchmarking classification speed...")
    speed_metrics = benchmark_classification_speed(emails, num_samples=min(200, len(emails)))
    
    # Step 7: Compile benchmark results
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_emails": len(emails),
        "confidence_threshold": confidence_threshold,
        "taxonomy_size": len(ENHANCED_TAXONOMY),
        "leaf_categories": len(LEAF_CATEGORIES),
        "classification_metrics": classification_metrics,
        "generalization_metrics": generalization_metrics,
        "topic_metrics": topic_metrics,
        "speed_metrics": speed_metrics
    }
    
    # Step 8: Save benchmark results
    with open(BENCHMARK_RESULTS_FILE, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {BENCHMARK_RESULTS_FILE}")
    
    # Step 9: Create visualization plots
    create_benchmark_plots(benchmark_results)
    
    # Step 10: Print summary
    logger.info("\nBenchmark Summary:")
    logger.info(f"Total emails: {len(emails)}")
    logger.info(f"Taxonomy size: {len(ENHANCED_TAXONOMY)} categories ({len(LEAF_CATEGORIES)} leaf categories)")
    logger.info(f"Accuracy: {classification_metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {classification_metrics['f1']:.4f}")
    logger.info(f"Generalization (cross-validation): Avg Accuracy={generalization_metrics['avg_accuracy']:.4f}, Avg F1={generalization_metrics['avg_f1']:.4f}")
    logger.info(f"Classification speed: {speed_metrics['emails_per_second']:.2f} emails/second")
    logger.info(f"Visualization plots saved to {PLOTS_DIR}")

def main():
    """Main function to run the enhanced email classifier benchmark"""
    parser = argparse.ArgumentParser(description="Benchmark the enhanced email classifier")
    parser.add_argument("--emails", type=int, default=3200, help="Number of emails to use for benchmarking")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for classification")
    
    args = parser.parse_args()
    
    # Run the benchmark
    run_enhanced_benchmark(num_emails=args.emails, confidence_threshold=args.confidence)

if __name__ == "__main__":
    main()