#!/usr/bin/env python3
"""
Small training script for the email classifier.
Creates a minimal dataset and trains models quickly for demonstration.
"""
import os
import sys
import logging
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_email import create_training_data, train_traditional_models

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_small")

def main():
    """Create a small dataset and train models"""
    # Use a small dataset size for quick training
    dataset_size = 500
    
    # Generate synthetic training data
    logger.info(f"Generating {dataset_size} synthetic training emails...")
    texts, labels = create_training_data(
        num_samples=dataset_size,
        augment=False,  # Skip augmentation for speed
        save=True,
        balanced=True   # Balance categories
    )
    
    # Train only traditional models (skip transformer for speed)
    logger.info("Training traditional models...")
    start_time = time.time()
    
    # Train with simpler model set and no hyperparameter tuning
    ensemble, vectorizer, label_encoder, trained_models = train_traditional_models(
        texts, 
        labels, 
        save_model=True,
        use_hyperparameter_tuning=False
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Print a summary of the models
    logger.info("\nTrained Models Summary:")
    logger.info(f"  Number of categories: {len(label_encoder.classes_)}")
    logger.info(f"  Categories: {', '.join(label_encoder.classes_)}")
    logger.info(f"  Vectorizer features: {vectorizer.get_feature_names_out().shape[0]}")
    logger.info(f"  Models in ensemble: {len(ensemble.estimators)}")
    
    return ensemble, vectorizer, label_encoder

if __name__ == "__main__":
    main()