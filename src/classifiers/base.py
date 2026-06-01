"""Base classifier interface for CNB OAM document classification.

Defines the abstract base class that all classifier implementations must follow,
along with the ClassificationResult dataclass for standardized output.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of a classification prediction.

    Attributes:
        predicted_labels: List of predicted category labels, one per input text.
        probabilities: Optional list of dicts mapping each category label to its
            predicted probability. One dict per input text. May be None if the
            classifier does not support probability estimates.
    """

    predicted_labels: list[str]
    probabilities: list[dict[str, float]] | None = None


class BaseClassifier(ABC):
    """Abstract base class for document classifiers.

    All classifier implementations (rule-based, TF-IDF ML, BERT) must inherit
    from this class and implement the required methods.
    """

    @abstractmethod
    def train(self, texts: list[str], labels: list[str]) -> None:
        """Train the classifier on labelled data.

        Args:
            texts: List of document text contents.
            labels: List of category labels corresponding to each text.

        Raises:
            ValueError: If texts and labels have different lengths or are empty.
        """
        ...

    @abstractmethod
    def predict(self, texts: list[str]) -> ClassificationResult:
        """Predict categories for a list of documents.

        Args:
            texts: List of document text contents to classify.

        Returns:
            ClassificationResult with predicted labels and optional probabilities.

        Raises:
            ValueError: If texts is empty.
            RuntimeError: If the classifier has not been trained/loaded.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained classifier to disk.

        Args:
            path: Directory path where the classifier artifacts will be saved.
                  The directory will be created if it does not exist.

        Raises:
            RuntimeError: If the classifier has not been trained.
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a previously saved classifier from disk.

        Args:
            path: Directory path containing the saved classifier artifacts.

        Raises:
            FileNotFoundError: If the path does not exist or required files
                are missing.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this classifier."""
        ...

    def _validate_inputs(self, texts: list[str], labels: list[str] | None = None) -> None:
        """Validate common input constraints.

        Args:
            texts: List of document texts.
            labels: Optional list of labels (required for training).

        Raises:
            ValueError: If inputs are invalid.
        """
        if not texts:
            raise ValueError("Input texts list must not be empty.")
        if labels is not None:
            if not labels:
                raise ValueError("Labels list must not be empty.")
            if len(texts) != len(labels):
                raise ValueError(
                    f"Number of texts ({len(texts)}) must match "
                    f"number of labels ({len(labels)})."
                )
