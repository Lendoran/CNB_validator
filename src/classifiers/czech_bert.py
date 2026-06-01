"""Fine-tuning Czech BERT models (RobeCzech, Czert) for document classification.

Conditional imports are used to avoid runtime failures if torch/transformers
are not installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.classifiers.base import BaseClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class CzechBertClassifier(BaseClassifier):
    """Czech BERT classifier utilizing pre-trained RobeCzech or Czert.

    Handles tokenization, fine-tuning via HuggingFace Trainer, custom class
    weights in loss, and prediction. Supports CPU and CUDA acceleration.
    """

    def __init__(
        self,
        model_name: str = "ufal/robeczech-base",
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ) -> None:
        """Initialise classifier hyperparameters.

        Args:
            model_name: HuggingFace model path (e.g. 'ufal/robeczech-base').
            epochs: Training epochs.
            batch_size: Training batch size.
            learning_rate: Optimizer learning rate.
        """
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = None
        self.tokenizer = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

        logger.info(
            "CzechBertClassifier initialised (model='%s', epochs=%d, batch_size=%d, lr=%s)",
            self.model_name,
            self.epochs,
            self.batch_size,
            self.learning_rate,
        )

    @property
    def name(self) -> str:
        """Return the classifier name."""
        return f"CzechBertClassifier({self.model_name})"

    def _check_dependencies(self) -> None:
        """Ensure torch, transformers, and datasets are installed.

        Raises:
            ImportError: If required deep learning libraries are missing.
        """
        try:
            import torch
            import transformers
            import datasets
        except ImportError as e:
            raise ImportError(
                "Deep learning dependencies not found. Please install the project "
                "with [bert] optional dependencies: pip install -e .[bert]"
            ) from e

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Fine-tune the Czech BERT model on the training data.

        This handles tokenization, dataset splitting, compute metrics callback,
        balanced class weights, and training loop.

        Args:
            texts: List of document text strings.
            labels: Category labels corresponding to each text.
        """
        self._validate_inputs(texts, labels)
        self._check_dependencies()

        import torch
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from sklearn.utils.class_weight import compute_class_weight

        # 1. Prepare label mappings
        unique_labels = sorted(list(set(labels)))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        num_labels = len(unique_labels)

        logger.info("Initializing tokenizer and model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
        )

        # 2. Convert texts and labels to HF dataset
        numeric_labels = [self.label2id[label] for label in labels]
        
        # We split the training data to create a validation set if one is not provided,
        # but to keep the interface simple we'll train on everything or do a quick internal split.
        dataset = Dataset.from_dict({
            "text": texts,
            "label": numeric_labels,
        })

        # Tokenize dataset
        def tokenize_func(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        tokenized_dataset = dataset.map(tokenize_func, batched=True)
        # Split internally for validation (90% train, 10% val)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

        # 3. Compute class weights for weighted cross-entropy loss
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_labels),
            y=numeric_labels,
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        logger.info("Computed class weights: %s", class_weights)

        # 4. Set up custom Trainer with weighted loss
        class WeightedLossTrainer(Trainer):
            def __init__(self, loss_weights=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_weights = loss_weights

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                if self.loss_weights is not None:
                    # Move weights to correct device
                    weights = self.loss_weights.to(logits.device)
                    loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                else:
                    loss_fct = torch.nn.CrossEntropyLoss()
                
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        # 5. Define evaluation metrics callback
        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, f1_score
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="weighted", zero_division=0)
            return {"accuracy": acc, "f1": f1}

        # 6. Define training arguments
        output_dir = Path("results/bert_checkpoints")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            report_to="none",  # Avoid external logger prompts (wandb, etc.)
            fp16=torch.cuda.is_available(),  # mixed precision if GPU is present
        )

        # 7. Instantiate and run trainer
        trainer = WeightedLossTrainer(
            loss_weights=class_weights_tensor,
            model=self.model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            compute_metrics=compute_metrics,
        )

        logger.info("Starting model fine-tuning...")
        trainer.train()
        logger.info("BERT fine-tuning completed successfully!")

    def predict(self, texts: list[str]) -> ClassificationResult:
        """Predict document categories for a list of texts.

        Args:
            texts: List of document text strings.

        Returns:
            ClassificationResult with predicted labels and probability distributions.
        """
        self._validate_inputs(texts)
        self._check_dependencies()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        import torch
        import numpy as np

        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        predicted_labels = []
        probabilities = []

        logger.debug("Running BERT predictions on %s for %d texts", device, len(texts))

        from tqdm import tqdm

        # Run predictions in batches to prevent OOM
        batch_size = 16
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Running BERT predictions"):
                batch_texts = texts[i : i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Compute softmax probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                for pred, prob_dist in zip(preds, probs):
                    pred_label = self.id2label[pred]
                    prob_dict = {self.id2label[j]: float(prob_dist[j]) for j in range(len(self.id2label))}
                    
                    predicted_labels.append(pred_label)
                    probabilities.append(prob_dict)

        return ClassificationResult(
            predicted_labels=predicted_labels,
            probabilities=probabilities,
        )

    def save(self, path: Path) -> None:
        """Save model, tokenizer, and mappings to disk.

        Args:
            path: Directory where model folder will be created.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save an untrained model.")
            
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving BERT model and tokenizer to %s", path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save custom mappings alongside
        mappings = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "model_name": self.model_name,
        }
        with open(path / "mappings.json", "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)

    def load(self, path: Path) -> None:
        """Load fine-tuned model, tokenizer, and mappings from disk.

        Args:
            path: Directory containing saved model files.
        """
        self._check_dependencies()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BERT model path not found: {path}")

        # Load mappings
        with open(path / "mappings.json", "r", encoding="utf-8") as f:
            mappings = json.load(f)
            
        self.label2id = mappings["label2id"]
        # Recover integer keys
        self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
        self.model_name = mappings.get("model_name", self.model_name)

        logger.info("Loading BERT model and tokenizer from %s", path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
