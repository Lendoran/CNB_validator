"""TF-IDF vectorizer combined with traditional ML classifiers (SVM, RF, LogReg).

Implements the TfidfMLClassifier class adhering to the BaseClassifier interface.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.classifiers.base import BaseClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class TfidfMLClassifier(BaseClassifier):
    """TF-IDF + traditional Machine Learning classifier.

    Supports Linear SVM (with calibration for probabilities), Random Forest, and
    Logistic Regression. Uses balanced class weights to address data imbalance.
    """

    def __init__(
        self,
        classifier_type: str = "svm",
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        """Initialise the classifier pipeline template.

        Args:
            classifier_type: One of 'svm', 'random_forest', 'logistic_regression'.
            max_features: Maximum number of features for TF-IDF.
            ngram_range: Gram bounds for TF-IDF.
        """
        self.classifier_type = classifier_type.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pipeline: Pipeline | None = None
        self._classes: list[str] | None = None

        logger.info(
            "TfidfMLClassifier initialised (type='%s', max_features=%d, ngrams=%s)",
            self.classifier_type,
            self.max_features,
            self.ngram_range,
        )

    @property
    def name(self) -> str:
        """Return the class/type name."""
        return f"TfidfMLClassifier({self.classifier_type})"

    def _build_model(self) -> Pipeline:
        """Construct the scikit-learn pipeline based on selected classifier_type."""
        # 1. TF-IDF Vectorizer with sublinear scaling (good for text docs)
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",  # Czech letters include diacritics
        )

        # 2. Classifier Selection
        if self.classifier_type == "svm":
            # LinearSVC does not support predict_proba natively,
            # so we wrap it in CalibratedClassifierCV to get probabilities.
            base_clf = LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                dual="auto",
                random_state=42,
            )
            classifier = CalibratedClassifierCV(estimator=base_clf, n_jobs=-1)
        elif self.classifier_type == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
        elif self.classifier_type == "logistic_regression":
            classifier = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                n_jobs=-1,
                random_state=42,
            )
        else:
            raise ValueError(
                f"Unknown classifier type '{self.classifier_type}'. "
                f"Use 'svm', 'random_forest', or 'logistic_regression'."
            )

        return Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ])

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Fit the TF-IDF Vectorizer and ML Classifier on the training data.

        Args:
            texts: List of document text strings.
            labels: Target labels corresponding to each text.
        """
        self._validate_inputs(texts, labels)
        
        logger.info("Training %s on %d documents...", self.name, len(texts))
        self.pipeline = self._build_model()
        self.pipeline.fit(texts, labels)
        
        # Store class labels
        clf = self.pipeline.named_steps["classifier"]
        self._classes = list(clf.classes_)
        logger.info("Training complete. Classes learned: %s", self._classes)

    def predict(self, texts: list[str]) -> ClassificationResult:
        """Predict categories for a list of input texts.

        Args:
            texts: List of document texts.

        Returns:
            ClassificationResult containing predicted labels and probability distributions.
        """
        self._validate_inputs(texts)
        if self.pipeline is None or self._classes is None:
            raise RuntimeError("Classifier has not been trained or loaded yet.")

        logger.debug("Running prediction for %d documents...", len(texts))
        
        # Predict labels
        predicted_labels = list(self.pipeline.predict(texts))
        
        # Predict probabilities
        probabilities_array = self.pipeline.predict_proba(texts)
        probabilities = []
        
        for probs in probabilities_array:
            prob_dict = {self._classes[i]: float(probs[i]) for i in range(len(self._classes))}
            probabilities.append(prob_dict)

        return ClassificationResult(
            predicted_labels=predicted_labels,
            probabilities=probabilities,
        )

    def save(self, path: Path) -> None:
        """Pickle the pipeline and metadata to disk.

        Args:
            path: Directory path where model file will be saved.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained classifier.")
            
        path.mkdir(parents=True, exist_ok=True)
        model_file = path / f"tfidf_{self.classifier_type}.pkl"
        
        meta = {
            "classifier_type": self.classifier_type,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "pipeline": self.pipeline,
            "classes": self._classes,
        }
        
        with open(model_file, "wb") as f:
            pickle.dump(meta, f)
            
        logger.info("Saved %s pipeline and metadata to %s", self.name, model_file)

    def load(self, path: Path) -> None:
        """Load a saved pipeline and metadata from disk.

        Args:
            path: Directory containing the model pkl file.
        """
        model_file = path / f"tfidf_{self.classifier_type}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(model_file, "rb") as f:
            meta = pickle.load(f)

        self.classifier_type = meta["classifier_type"]
        self.max_features = meta["max_features"]
        self.ngram_range = meta["ngram_range"]
        self.pipeline = meta["pipeline"]
        self._classes = meta["classes"]
        
        logger.info("Loaded %s from %s", self.name, model_file)
