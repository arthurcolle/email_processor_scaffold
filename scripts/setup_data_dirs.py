#!/usr/bin/env python3
"""
Setup script to create the necessary data directory structure.
Run this after cloning the repository to ensure all required directories exist.
"""
import os
import sys
import logging
from pathlib import Path

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("setup_data_dirs")

def setup_data_directories():
    """Create the necessary data directory structure"""
    # Get the project root directory
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define the directories to create
    directories = [
        "data/datasets",
        "data/models",
        "data/enron",
        "data/spamassassin/processed",
        "data/spamassassin/samples",
        "data/results",
        "data/plots",
        "data/benchmark",
    ]
    
    # Create each directory
    for directory in directories:
        dir_path = root_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Create a .gitkeep file in the data directory
    gitkeep_path = root_dir / "data" / ".gitkeep"
    with open(gitkeep_path, 'w') as f:
        f.write("# This file ensures the data directory structure is preserved in git\n")
        f.write("# while the actual data files are ignored\n")
    
    logger.info(f"Created {gitkeep_path}")
    logger.info("\nData directory structure setup complete!")
    
    # Create a README file in the data directory
    readme_path = root_dir / "data" / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# Data Directory\n\n")
        f.write("This directory contains data files that are not tracked by git.\n\n")
        f.write("## Directory Structure\n\n")
        f.write("- `datasets/`: Training and test datasets\n")
        f.write("- `models/`: Trained model files\n")
        f.write("- `enron/`: Enron email dataset samples\n")
        f.write("- `spamassassin/`: SpamAssassin corpus\n")
        f.write("  - `processed/`: Processed email files\n")
        f.write("  - `samples/`: Sample datasets\n")
        f.write("- `results/`: Classification results\n")
        f.write("- `plots/`: Generated visualizations\n")
        f.write("- `benchmark/`: Benchmark results\n\n")
        f.write("## Regenerating Data\n\n")
        f.write("Most data can be regenerated using the scripts in the `scripts/` directory:\n\n")
        f.write("- `scripts/train_small.py`: Generate a small dataset and train models\n")
        f.write("- `scripts/train_medium.py`: Generate a medium-sized dataset and train models\n")
        f.write("- `scripts/spamassassin_processor.py`: Download and process the SpamAssassin corpus\n")
    
    logger.info(f"Created {readme_path}")

if __name__ == "__main__":
    setup_data_directories()
