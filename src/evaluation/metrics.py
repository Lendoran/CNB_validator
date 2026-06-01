"""Classification metrics and visualization for the CNB OAM document classifier.

Provides accuracy, F1 scores, confusion matrices, and per-class metric plots.
All visualizations are tuned for long Czech category names (rotated labels,
adjusted font sizes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")  # Non-interactive backend for headless environments

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Compute, tabulate, and visualise classification metrics.

    All public methods are static / class-level — no internal state is
    required — so callers can use them à la carte without instantiation
    if they prefer, but an instance works equally well.
    """

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
    ) -> dict:
        """Compute a comprehensive dictionary of classification metrics.

        Args:
            y_true: Ground-truth category labels.
            y_pred: Predicted category labels.
            labels: Ordered list of all possible category labels.

        Returns:
            Dictionary with keys ``accuracy``, ``macro_f1``, ``weighted_f1``,
            and ``per_class`` (a dict mapping each label to its precision,
            recall, f1 and support).

        Raises:
            ValueError: If *y_true* and *y_pred* have different lengths.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length, "
                f"got {len(y_true)} and {len(y_pred)}"
            )

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        per_class: dict[str, dict[str, float | int]] = {}
        for i, label in enumerate(labels):
            per_class[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class,
        }

        logger.info(
            "Metrics computed — accuracy=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
            accuracy,
            macro_f1,
            weighted_f1,
        )
        return metrics

    # ------------------------------------------------------------------
    # Classification report as DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def classification_report_table(
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
    ) -> pd.DataFrame:
        """Return a :class:`~pandas.DataFrame` with the sklearn classification report.

        The table contains one row per class plus ``accuracy``,
        ``macro avg`` and ``weighted avg`` summary rows.

        Args:
            y_true: Ground-truth category labels.
            y_pred: Predicted category labels.
            labels: Ordered list of all possible category labels.

        Returns:
            DataFrame indexed by class / summary, with columns
            ``precision``, ``recall``, ``f1-score``, ``support``.
        """
        report_dict: dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        df = pd.DataFrame(report_dict).T
        # Ensure support is integer
        if "support" in df.columns:
            df["support"] = df["support"].astype(int)
        logger.info("Classification report table created with %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Confusion matrix heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def confusion_matrix_plot(
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
        output_path: Path,
        title: str = "Confusion Matrix",
    ) -> None:
        """Save a confusion-matrix heatmap as a PNG file.

        The plot is sized dynamically based on the number of labels and
        uses rotated tick labels so that long Czech category names remain
        readable.

        Args:
            y_true: Ground-truth category labels.
            y_pred: Predicted category labels.
            labels: Ordered list of all possible category labels.
            output_path: Destination file path (parent dirs created as needed).
            title: Plot title.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        n_labels = len(labels)
        # Dynamic sizing: at least 8×8, scale up for many categories
        fig_size = max(8, n_labels * 0.9)
        font_size = max(6, 12 - n_labels // 4)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": font_size},
            linewidths=0.5,
            linecolor="white",
        )

        ax.set_title(title, fontsize=font_size + 4, pad=16)
        ax.set_xlabel("Predicted", fontsize=font_size + 2)
        ax.set_ylabel("True", fontsize=font_size + 2)

        # Rotate labels for Czech category names
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=font_size)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Confusion matrix saved to %s", output_path)

    # ------------------------------------------------------------------
    # Per-class F1 bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def plot_per_class_f1(
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
        output_path: Path,
    ) -> None:
        """Save a horizontal bar chart of per-class F1 scores.

        Args:
            y_true: Ground-truth category labels.
            y_pred: Predicted category labels.
            labels: Ordered list of all possible category labels.
            output_path: Destination PNG file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _, _, f1_scores, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        # Sort by F1 descending for readability
        sorted_indices = np.argsort(f1_scores)
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_f1 = f1_scores[sorted_indices]

        n_labels = len(labels)
        fig_height = max(5, n_labels * 0.45)
        font_size = max(7, 11 - n_labels // 6)

        fig, ax = plt.subplots(figsize=(10, fig_height))

        colors = sns.color_palette("viridis", n_colors=n_labels)
        bars = ax.barh(range(n_labels), sorted_f1, color=colors)

        ax.set_yticks(range(n_labels))
        ax.set_yticklabels(sorted_labels, fontsize=font_size)
        ax.set_xlabel("F1 Score", fontsize=font_size + 2)
        ax.set_title("Per-Category F1 Score", fontsize=font_size + 4)
        ax.set_xlim(0.0, 1.05)

        # Annotate bars with their value
        for bar, value in zip(bars, sorted_f1):
            ax.text(
                value + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=font_size,
            )

        ax.axvline(x=np.mean(f1_scores), color="red", linestyle="--", linewidth=1, label="Mean F1")
        ax.legend(fontsize=font_size)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Per-class F1 chart saved to %s", output_path)
