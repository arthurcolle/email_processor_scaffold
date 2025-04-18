#!/usr/bin/env python3
"""
Analysis script for email classification results.
This script analyzes and visualizes the results from large-scale email classification tests.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Optional visualization capabilities - will skip if not installed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class ClassificationAnalyzer:
    """
    Analyzes the results of email classification performance tests
    """
    def __init__(self, results_file: str):
        """Initialize with a CSV file containing classification results"""
        self.results_file = results_file
        self.results = self.load_results()
        
    def load_results(self) -> List[Dict[str, Any]]:
        """Load results from CSV file"""
        if not os.path.exists(self.results_file):
            print(f"Error: Results file {self.results_file} not found")
            return []
            
        results = []
        with open(self.results_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert numeric fields
                if "confidence" in row:
                    row["confidence"] = float(row["confidence"])
                if "accuracy" in row:
                    row["accuracy"] = int(row["accuracy"])
                results.append(row)
                
        print(f"Loaded {len(results)} classification results from {self.results_file}")
        return results
    
    def basic_stats(self) -> Dict[str, Any]:
        """Generate basic statistics about the results"""
        if not self.results:
            return {}
            
        # Count by categories
        original_categories = Counter(r["original_type"] for r in self.results)
        classified_categories = Counter(r["classification"] for r in self.results)
        
        # Calculate accuracy
        all_accuracy = [r["accuracy"] for r in self.results]
        overall_accuracy = sum(all_accuracy) / len(all_accuracy) if all_accuracy else 0
        
        # Calculate confidence stats
        confidences = [r["confidence"] for r in self.results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        max_confidence = max(confidences) if confidences else 0
        
        # Confidence by category
        confidence_by_category = defaultdict(list)
        for r in self.results:
            confidence_by_category[r["classification"]].append(r["confidence"])
            
        avg_confidence_by_category = {
            cat: sum(confs) / len(confs) if confs else 0
            for cat, confs in confidence_by_category.items()
        }
        
        # Accuracy by category
        accuracy_by_original_category = defaultdict(list)
        for r in self.results:
            accuracy_by_original_category[r["original_type"]].append(r["accuracy"])
            
        avg_accuracy_by_category = {
            cat: sum(accs) / len(accs) if accs else 0
            for cat, accs in accuracy_by_original_category.items()
        }
        
        # Build confusion matrix data
        confusion_data = defaultdict(Counter)
        for r in self.results:
            confusion_data[r["original_type"]][r["classification"]] += 1
        
        return {
            "total_emails": len(self.results),
            "original_categories": dict(original_categories),
            "classified_categories": dict(classified_categories),
            "overall_accuracy": overall_accuracy,
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "avg_confidence_by_category": avg_confidence_by_category,
            "avg_accuracy_by_category": avg_accuracy_by_category,
            "confusion_data": {k: dict(v) for k, v in confusion_data.items()}
        }
    
    def print_summary(self, stats: Dict[str, Any]) -> None:
        """Print a summary of the analysis results"""
        if not stats:
            print("No statistics available to display")
            return
            
        print("\n===== EMAIL CLASSIFICATION ANALYSIS SUMMARY =====\n")
        
        print(f"Total emails analyzed: {stats['total_emails']}")
        print(f"Overall accuracy: {stats['overall_accuracy']*100:.2f}%")
        print(f"Average confidence: {stats['avg_confidence']:.4f}")
        print(f"Confidence range: {stats['min_confidence']:.4f} - {stats['max_confidence']:.4f}")
        
        print("\nCategory Distribution (Original):")
        for cat, count in sorted(stats["original_categories"].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats["total_emails"]) * 100
            print(f"  {cat}: {count} ({percentage:.1f}%)")
            
        print("\nCategory Distribution (Classified):")
        for cat, count in sorted(stats["classified_categories"].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats["total_emails"]) * 100
            print(f"  {cat}: {count} ({percentage:.1f}%)")
            
        print("\nAccuracy by Category:")
        for cat, acc in sorted(stats["avg_accuracy_by_category"].items(), key=lambda x: x[1], reverse=True):
            count = stats["original_categories"].get(cat, 0)
            print(f"  {cat}: {acc*100:.2f}% (of {count} emails)")
            
        print("\nConfidence by Category:")
        for cat, conf in sorted(stats["avg_confidence_by_category"].items(), key=lambda x: x[1], reverse=True):
            count = stats["classified_categories"].get(cat, 0)
            print(f"  {cat}: {conf:.4f} (for {count} emails)")
    
    def generate_visualizations(self, stats: Dict[str, Any], output_dir: str = ".") -> None:
        """Generate visualizations of the analysis results"""
        if not VISUALIZATION_AVAILABLE:
            print("Visualization packages not available. Install matplotlib, seaborn, and pandas to enable visualizations.")
            return
            
        if not stats:
            print("No statistics available for visualization")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(self.results)
        
        # Set up the style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. Original vs Classified Categories
        self._plot_category_comparison(stats, output_dir, timestamp)
        
        # 2. Confusion Matrix
        self._plot_confusion_matrix(stats, output_dir, timestamp)
        
        # 3. Confidence Distribution
        self._plot_confidence_distribution(df, output_dir, timestamp)
        
        # 4. Accuracy by Category
        self._plot_accuracy_by_category(stats, output_dir, timestamp)
        
        # 5. Confidence vs Accuracy
        self._plot_confidence_vs_accuracy(df, output_dir, timestamp)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _plot_category_comparison(self, stats: Dict[str, Any], output_dir: str, timestamp: str) -> None:
        """Plot comparison of original vs classified categories"""
        orig_cats = stats["original_categories"]
        class_cats = stats["classified_categories"]
        
        # Get all unique categories
        all_cats = set(list(orig_cats.keys()) + list(class_cats.keys()))
        
        # Create data for comparison
        cats = sorted(all_cats)
        orig_counts = [orig_cats.get(cat, 0) for cat in cats]
        class_counts = [class_cats.get(cat, 0) for cat in cats]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(cats))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, orig_counts, width, label='Original')
        rects2 = ax.bar(x + width/2, class_counts, width, label='Classified')
        
        ax.set_title('Original vs Classified Category Distribution')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha='right')
        ax.legend()
        
        # Add counts on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
                
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(f"{output_dir}/category_comparison_{timestamp}.png")
        plt.close()
    
    def _plot_confusion_matrix(self, stats: Dict[str, Any], output_dir: str, timestamp: str) -> None:
        """Plot confusion matrix of classification results"""
        confusion_data = stats["confusion_data"]
        
        # Get all unique categories
        all_orig_cats = set(confusion_data.keys())
        all_class_cats = set()
        for orig_cat, class_counts in confusion_data.items():
            all_class_cats.update(class_counts.keys())
            
        # Sort categories alphabetically
        orig_cats = sorted(all_orig_cats)
        class_cats = sorted(all_class_cats)
        
        # Create confusion matrix
        matrix = np.zeros((len(orig_cats), len(class_cats)))
        for i, orig in enumerate(orig_cats):
            total = sum(confusion_data.get(orig, {}).values())
            for j, cls in enumerate(class_cats):
                count = confusion_data.get(orig, {}).get(cls, 0)
                matrix[i, j] = count / total if total > 0 else 0
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                    xticklabels=class_cats, yticklabels=orig_cats,
                    vmin=0, vmax=1)
        
        ax.set_title('Confusion Matrix (Normalized)')
        ax.set_xlabel('Classified As')
        ax.set_ylabel('Original Category')
        
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{timestamp}.png")
        plt.close()
    
    def _plot_confidence_distribution(self, df: pd.DataFrame, output_dir: str, timestamp: str) -> None:
        """Plot distribution of confidence scores"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Overall confidence distribution
        sns.histplot(df["confidence"], kde=True, bins=20, ax=ax)
        
        ax.set_title('Distribution of Confidence Scores')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        
        fig.tight_layout()
        plt.savefig(f"{output_dir}/confidence_distribution_{timestamp}.png")
        plt.close()
        
        # Confidence by classification
        if len(df["classification"].unique()) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(x="classification", y="confidence", data=df, ax=ax)
            
            ax.set_title('Confidence Scores by Category')
            ax.set_xlabel('Category')
            ax.set_ylabel('Confidence')
            
            plt.xticks(rotation=45, ha='right')
            fig.tight_layout()
            plt.savefig(f"{output_dir}/confidence_by_category_{timestamp}.png")
            plt.close()
    
    def _plot_accuracy_by_category(self, stats: Dict[str, Any], output_dir: str, timestamp: str) -> None:
        """Plot accuracy by category"""
        categories = []
        accuracies = []
        counts = []
        
        for cat, acc in stats["avg_accuracy_by_category"].items():
            count = stats["original_categories"].get(cat, 0)
            if count > 0:  # Only include categories with emails
                categories.append(cat)
                accuracies.append(acc)
                counts.append(count)
        
        # Sort by count for better visualization
        sorted_indices = np.argsort(counts)[::-1]
        categories = [categories[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(categories, [acc * 100 for acc in accuracies])
        
        # Add count annotations
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, 5,
                    f'n={counts[i]}', ha='center', va='bottom',
                    color='black', fontweight='bold')
        
        ax.set_title('Classification Accuracy by Category')
        ax.set_xlabel('Original Category')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        
        # Add overall accuracy line
        ax.axhline(y=stats["overall_accuracy"] * 100, color='r', linestyle='--',
                   label=f'Overall: {stats["overall_accuracy"]*100:.1f}%')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_by_category_{timestamp}.png")
        plt.close()
    
    def _plot_confidence_vs_accuracy(self, df: pd.DataFrame, output_dir: str, timestamp: str) -> None:
        """Plot relationship between confidence and accuracy"""
        # Group by confidence bins
        df['confidence_bin'] = pd.cut(df['confidence'], bins=10)
        grouped = df.groupby('confidence_bin').agg({
            'accuracy': 'mean',
            'email_id': 'count'
        }).reset_index()
        
        # Calculate bin centers for plotting
        grouped['bin_center'] = grouped['confidence_bin'].apply(lambda x: x.mid)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Line for accuracy
        ax.plot(grouped['bin_center'], grouped['accuracy'], marker='o', linestyle='-', color='blue')
        
        # Bars for email count
        ax2 = ax.twinx()
        ax2.bar(grouped['bin_center'], grouped['email_id'], alpha=0.3, color='lightblue')
        
        ax.set_title('Accuracy vs Confidence')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Accuracy (0-1)', color='blue')
        ax2.set_ylabel('Email Count', color='lightblue')
        
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='lightblue')
        
        fig.tight_layout()
        plt.savefig(f"{output_dir}/confidence_vs_accuracy_{timestamp}.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze email classification results")
    parser.add_argument("-f", "--file", type=str, default="email_classification_results.csv", 
                        help="Input CSV file with classification results")
    parser.add_argument("-o", "--output-dir", type=str, default="results",
                        help="Directory to save visualization outputs")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--export-json", type=str,
                        help="Export statistics to JSON file")
    
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = ClassificationAnalyzer(args.file)
    
    # Generate and print statistics
    stats = analyzer.basic_stats()
    analyzer.print_summary(stats)
    
    # Export to JSON if requested
    if args.export_json:
        with open(args.export_json, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Statistics exported to {args.export_json}")
    
    # Generate visualizations
    if not args.no_viz:
        analyzer.generate_visualizations(stats, args.output_dir)

if __name__ == "__main__":
    main()