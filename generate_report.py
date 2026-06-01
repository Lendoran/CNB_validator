#!/usr/bin/env python3
"""Script to run evaluations for all trained models and generate/update the HTML report."""

import logging
from pathlib import Path
import pickle
import sys

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.cli import load_config
from src.evaluation.comparison import MethodComparison

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    config = load_config()
    res_dir = Path(config["paths"]["results_dir"])
    categories = config["categories"]["classify"]

    # Import helper from cli to avoid duplication
    from src.cli import compile_reports_from_runs
    
    print("Compiling reports from saved runs...")
    compile_reports_from_runs(res_dir, categories)
    print("Report compilation completed successfully!")

if __name__ == "__main__":
    main()
