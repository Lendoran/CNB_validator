"""Dataset loading and train/val/test splitting utilities for CNB document classification.

Loads text and labels from database and prepares stratified splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.scraper.metadata_db import MetadataDB

logger = logging.getLogger(__name__)


def load_dataset_from_db(
    db_path: str | Path,
    categories: list[str],
    min_char_count: int = 50,
) -> tuple[list[str], list[str]]:
    """Load text contents and labels from the SQLite database.

    For documents with multiple files, their extracted texts are concatenated.
    Filters by the list of active classification categories.

    Args:
        db_path: Path to the SQLite metadata database.
        categories: List of category labels to keep.
        min_char_count: Minimum characters to be considered a valid text document.

    Returns:
        Tuple of (texts_list, labels_list).
    """
    db = MetadataDB(db_path)
    
    # Query to join documents with extracted text, concatenating multiple files per document
    query = """
        SELECT 
            d.id, 
            d.typ_informace, 
            GROUP_CONCAT(et.text_content, '\n\n=== ATTACHMENT SPLIT ===\n\n') as full_text
        FROM documents d
        JOIN extracted_text et ON d.id = et.document_id
        GROUP BY d.id
    """
    
    texts: list[str] = []
    labels: list[str] = []

    with db._get_conn() as conn:
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        
        for row in rows:
            label = row["typ_informace"]
            text = row["full_text"]
            
            # Check filter conditions
            if label not in categories:
                continue
                
            if not text or len(text.strip()) < min_char_count:
                logger.debug("Skipping document %s: text too short (%d chars)", row["id"], len(text) if text else 0)
                continue
                
            texts.append(text.strip())
            labels.append(label)

    logger.info("Loaded %d documents from database belonging to %d categories", len(texts), len(set(labels)))
    return texts, labels


def create_splits(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Create stratified train, validation, and test splits from the dataset.

    Args:
        texts: List of document text strings.
        labels: List of corresponding category labels.
        test_size: Ratio of test dataset.
        val_size: Ratio of validation dataset (relative to original total).
        random_seed: Random seed for reproducibility.

    Returns:
        Dict structured as:
        {
            "train": {"texts": [...], "labels": [...]},
            "val": {"texts": [...], "labels": [...]},
            "test": {"texts": [...], "labels": [...]}
        }
    """
    # Step 1: Split off test set
    try:
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_seed,
        )
    except Exception as split_err:
        logger.warning("Stratified test split failed: %s. Falling back to non-stratified.", split_err)
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_seed,
        )

    # Step 2: Split off validation set if val_size > 0
    if val_size > 0:
        val_samples = int(len(texts) * val_size)
        num_classes = len(set(train_val_labels))
        relative_val_size = val_size / (1.0 - test_size)
        
        if val_samples >= num_classes:
            try:
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    train_val_texts,
                    train_val_labels,
                    test_size=relative_val_size,
                    stratify=train_val_labels,
                    random_state=random_seed,
                )
            except Exception as split_err:
                logger.warning("Stratified validation split failed: %s. Falling back to non-stratified.", split_err)
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    train_val_texts,
                    train_val_labels,
                    test_size=relative_val_size,
                    random_state=random_seed,
                )
        else:
            logger.warning(
                "Validation sample size (%d) is smaller than the number of classes (%d). "
                "Bypassing validation split (all remaining samples go to training).", 
                val_samples, 
                num_classes
            )
            train_texts, train_labels = train_val_texts, train_val_labels
            val_texts, val_labels = [], []
    else:
        train_texts, train_labels = train_val_texts, train_val_labels
        val_texts, val_labels = [], []

    logger.info(
        "Created stratified splits — Train: %d, Val: %d, Test: %d",
        len(train_texts),
        len(val_texts),
        len(test_texts),
    )

    return {
        "train": {"texts": train_texts, "labels": train_labels},
        "val": {"texts": val_texts, "labels": val_labels},
        "test": {"texts": test_texts, "labels": test_labels},
    }


def get_class_distribution(labels: list[str]) -> dict[str, int]:
    """Count occurrences of each label in a list.

    Args:
        labels: List of label strings.

    Returns:
        Dictionary mapping label to frequency count, sorted descending.
    """
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))


def get_class_weights(labels: list[str]) -> dict[str, float]:
    """Compute balanced class weights to address label imbalance during training.

    Args:
        labels: List of training label strings.

    Returns:
        Dictionary mapping class label to its computed weight float value.
    """
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels,
    )
    return {cls: float(w) for cls, w in zip(unique_classes, weights)}
