#!/usr/bin/env python3
"""
Medium-sized training script for the email classifier.
Creates a dataset of 2400 emails and trains models with more features.
"""
import os
import sys
import logging
import time
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_email import create_training_data, train_traditional_models, train_transformer_model

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_medium")

def main():
    """Create a medium-sized dataset and train models"""
    parser = argparse.ArgumentParser(description="Train the email classifier on a medium-sized dataset")
    parser.add_argument("--size", type=int, default=2400, help="Number of emails to generate")
    parser.add_argument("--balanced", action="store_true", help="Create a balanced dataset")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--transformer", action="store_true", help="Train transformer model")
    parser.add_argument("--tuning", action="store_true", help="Apply hyperparameter tuning")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for transformer training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for transformer training")
    
    args = parser.parse_args()
    
    # Use supplied parameters
    dataset_size = args.size
    balanced = args.balanced
    augment = args.augment
    
    # Generate synthetic training data
    logger.info(f"Generating {dataset_size} synthetic training emails...")
    logger.info(f"Settings: balanced={balanced}, augment={augment}")
    
    # Track timing
    start_time = time.time()
    
    texts, labels = create_training_data(
        num_samples=dataset_size,
        augment=augment,
        save=True,
        balanced=balanced,
        batch_size=min(1000, dataset_size // 2)  # Use batch processing for larger datasets
    )
    
    dataset_time = time.time() - start_time
    logger.info(f"Dataset generation completed in {dataset_time:.2f} seconds")
    
    # Train traditional models
    logger.info("Training traditional models...")
    traditional_start = time.time()
    
    ensemble, vectorizer, label_encoder, trained_models = train_traditional_models(
        texts, 
        labels, 
        save_model=True,
        use_hyperparameter_tuning=args.tuning
    )
    
    traditional_time = time.time() - traditional_start
    logger.info(f"Traditional model training completed in {traditional_time:.2f} seconds")
    
    # Print a summary of the traditional models
    logger.info("\nTrained Models Summary:")
    logger.info(f"  Number of categories: {len(label_encoder.classes_)}")
    logger.info(f"  Categories: {', '.join(label_encoder.classes_)}")
    logger.info(f"  Vectorizer features: {vectorizer.get_feature_names_out().shape[0]}")
    logger.info(f"  Models in ensemble: {len(ensemble.estimators)}")
    
    # Train transformer model if requested
    if args.transformer:
        try:
            import torch
            from transformers import AutoTokenizer
            
            logger.info("\nTraining transformer model...")
            transformer_start = time.time()
            
            model, tokenizer, t_label_encoder = train_transformer_model(
                texts=texts,
                labels=labels,
                save_model=True,
                batch_size=args.batch_size,
                epochs=args.epochs,
                model_name="distilbert-base-uncased",
                max_length=512
            )
            
            transformer_time = time.time() - transformer_start
            logger.info(f"Transformer model training completed in {transformer_time:.2f} seconds")
            
        except ImportError:
            logger.warning("Transformer libraries not available. Skipping transformer training.")
    
    # Print overall timing
    total_time = time.time() - start_time
    logger.info(f"\nTotal training process completed in {total_time:.2f} seconds")
    
    return ensemble, vectorizer, label_encoder

if __name__ == "__main__":
    main()