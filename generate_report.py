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
    models_dir = Path(config["paths"]["models_dir"])
    splits_dir = Path(config["paths"].get("splits_dir", "data/splits"))
    categories = config["categories"]["classify"]
    res_dir = Path(config["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    splits_path = splits_dir / "splits.pkl"
    if not splits_path.exists():
        print(f"Error: splits file not found at {splits_path}. Please train your models first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading test split from {splits_path}...")
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]

    # 2. Determine which models are on disk
    methods_to_evaluate = []
    
    # Check rule-based
    if (models_dir / "rule_based" / "rules.json").exists():
        methods_to_evaluate.append("rule_based")
        
    # Check TF-IDF models
    if (models_dir / "tfidf").exists():
        for tfidf_file in (models_dir / "tfidf").glob("tfidf_*.pkl"):
            clf_type = tfidf_file.stem.replace("tfidf_", "")
            methods_to_evaluate.append(f"tfidf_{clf_type}")

    # Check czech_bert
    if (models_dir / "bert" / "mappings.json").exists():
        methods_to_evaluate.append("czech_bert")

    if not methods_to_evaluate:
        print("Error: No trained models found in models/ directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating models: {methods_to_evaluate}")

    comparison = MethodComparison(labels=categories)

    # 3. Evaluate each local model
    for m in methods_to_evaluate:
        print(f"Running evaluation for {m}...")
        try:
            if m == "rule_based":
                from src.classifiers.rule_based import RuleBasedClassifier
                clf = RuleBasedClassifier()
                clf.load(models_dir / "rule_based")
                pred_res = clf.predict(test_texts)
                comparison.add_result("Rule-based", test_labels, pred_res.predicted_labels)
                
            elif m.startswith("tfidf_"):
                from src.classifiers.tfidf_ml import TfidfMLClassifier
                clf_type = m.replace("tfidf_", "")
                clf = TfidfMLClassifier(classifier_type=clf_type)
                clf.load(models_dir / "tfidf")
                pred_res = clf.predict(test_texts)
                comparison.add_result(f"TF-IDF ({clf_type.upper()})", test_labels, pred_res.predicted_labels)
                
            elif m == "czech_bert":
                from src.classifiers.czech_bert import CzechBertClassifier
                clf = CzechBertClassifier()
                clf.load(models_dir / "bert")
                pred_res = clf.predict(test_texts)
                comparison.add_result("Czech BERT", test_labels, pred_res.predicted_labels)
        except Exception as e:
            print(f"Failed to evaluate {m}: {e}", file=sys.stderr)

    # 4. Generate comparison report & HTML report (which merges Ollama automatically)
    if comparison.results:
        print("Generating comparison reports, figures, and HTML report...")
        comparison.generate_full_report(res_dir)
        print("Report generation completed successfully!")
        print(f"Interactive report: report/report.html")
    else:
        print("Error: No models were successfully evaluated.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
