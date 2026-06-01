"""Classifiers package for CNB OAM document classification.

Provides the base class, rule-based keyword matching, TF-IDF + scikit-learn ML models,
fine-tuned Czech BERT models, and dataset loading utilities.
"""

from src.classifiers.base import BaseClassifier, ClassificationResult
from src.classifiers.rule_based import RuleBasedClassifier
from src.classifiers.tfidf_ml import TfidfMLClassifier
from src.classifiers.czech_bert import CzechBertClassifier
from src.classifiers.data_loader import (
    load_dataset_from_db,
    create_splits,
    get_class_distribution,
    get_class_weights,
)

__all__ = [
    "BaseClassifier",
    "ClassificationResult",
    "RuleBasedClassifier",
    "TfidfMLClassifier",
    "CzechBertClassifier",
    "load_dataset_from_db",
    "create_splits",
    "get_class_distribution",
    "get_class_weights",
]
