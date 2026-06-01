"""Ollama LLM-based classifier for CNB OAM document classification.

Leverages open-source Czech-capable LLMs hosted on an Ollama server
using zero-shot or few-shot structured prompting.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
import httpx

from src.classifiers.base import BaseClassifier, ClassificationResult

logger = logging.getLogger(__name__)


def _is_placeholder(val: str | None) -> bool:
    """Check if value is empty or a placeholder."""
    if not val:
        return True
    val_clean = val.lower().strip()
    return val_clean in {
        "", 
        "username", 
        "password", 
        "uzivatelske_jmeno", 
        "heslo", 
        "your_username", 
        "your_password", 
        "placeholder",
    } or "example.com" in val_clean


class OllamaLLMClassifier(BaseClassifier):
    """Ollama LLM document classifier.

    Communicates with an Ollama endpoint (e.g. hosted at a university server)
    to classify documents by sending Czech prompts and expecting structured JSON responses.
    """

    def __init__(
        self,
        host: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int = 45,
        few_shot_examples: list[dict[str, str]] | None = None,
        categories: list[str] | None = None,
        auth_type: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialise Ollama classifier.

        Args:
            host: Address of the Ollama server.
            model_name: Name of the model to use on the server (e.g., 'llama3', 'mistral').
            timeout_seconds: Timeout for requests.
            few_shot_examples: Optional list of prompt examples.
            categories: List of target categories.
            auth_type: Type of auth ('digest', 'basic', or None).
            username: Username for authentication.
            password: Password for authentication.
        """
        import os
        
        env_host = os.environ.get("OLLAMA_HOST")
        self.host = (host if not _is_placeholder(host) else env_host or "http://localhost:11434").rstrip("/")
        
        self.model_name = model_name or os.environ.get("OLLAMA_MODEL_NAME") or "llama3"
        self.timeout = timeout_seconds
        self.few_shot_examples = few_shot_examples or []
        self.categories = categories or []
        
        env_auth = os.environ.get("OLLAMA_AUTH_TYPE")
        self.auth_type = auth_type if not _is_placeholder(auth_type) else env_auth
        
        env_user = os.environ.get("OLLAMA_USERNAME")
        self.username = username if not _is_placeholder(username) else env_user
        
        env_pass = os.environ.get("OLLAMA_PASSWORD")
        self.password = password if not _is_placeholder(password) else env_pass

        logger.info(
            "OllamaLLMClassifier initialised (host='%s', model='%s', timeout=%ds, few_shots=%d, auth='%s')",
            self.host,
            self.model_name,
            self.timeout,
            len(self.few_shot_examples),
            self.auth_type,
        )

    @property
    def name(self) -> str:
        """Return human-readable classifier name."""
        return f"OllamaLLMClassifier({self.model_name})"

    def train(self, texts: list[str], labels: list[str]) -> None:
        """No-op — LLM classifier does not require standard parameter training.

        However, we store the list of distinct categories learned from the training labels
        to inject into the prompt.

        Args:
            texts: Training text list (unused).
            labels: Training label list (used to extract unique categories).
        """
        self._validate_inputs(texts, labels)
        self.categories = sorted(list(set(labels)))
        logger.info("OllamaLLMClassifier configured with %d categories from labels.", len(self.categories))

    def _build_prompt(self, text: str) -> str:
        """Construct the Czech classification instruction prompt."""
        categories_str = "\n".join(f"- {cat}" for cat in self.categories)
        
        examples_str = ""
        if self.few_shot_examples:
            examples_str = "\nZde jsou příklady správného zařazení:\n"
            for ex in self.few_shot_examples:
                examples_str += (
                    f"--- Пример ---\n"
                    f"Text: {ex.get('text', '')[:300]}...\n"
                    f"Výstup JSON: {{\"category\": \"{ex.get('label', '')}\", \"reason\": \"Příklad\"}}\n"
                )
        
        prompt = (
            "Jsi bankovní expert a klasifikátor dokumentů České národní banky (ČNB).\n"
            "Tvým úkolem je zařadit zadaný text do jedné z následujících kategorií regulovaných informací ČNB:\n"
            f"{categories_str}\n\n"
            "Pravidla:\n"
            "- Musíš zvolit přesně jeden název kategorie ze seznamu výše.\n"
            "- Odpověz VÝHRADNĚ ve formátu JSON s následujícími klíči:\n"
            "  - 'category': název zvolené kategorie\n"
            "  - 'reason': stručné odůvodnění v češtině\n"
            f"{examples_str}\n"
            "Text k zařazení:\n"
            f"{text[:6000]}\n\n"
            "Odpověď JSON:"
        )
        return prompt

    def predict(self, texts: list[str]) -> ClassificationResult:
        """Predict categories by sending REST requests to the Ollama server.

        Args:
            texts: List of document texts.

        Returns:
            ClassificationResult containing predicted labels.
        """
        self._validate_inputs(texts)
        predicted_labels = []
        probabilities = []

        endpoint = f"{self.host}/api/generate"
        logger.info("Running Ollama LLM predictions on %s...", endpoint)

        auth = None
        if self.auth_type == "digest" and self.username and self.password:
            auth = httpx.DigestAuth(self.username, self.password)
        elif self.auth_type == "basic" and self.username and self.password:
            auth = httpx.BasicAuth(self.username, self.password)

        with httpx.Client(auth=auth, timeout=self.timeout) as client:
            for idx, text in enumerate(texts):
                prompt = self._build_prompt(text)
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                    }
                }
                
                try:
                    logger.debug("[%d/%d] Querying Ollama for text of length %d", idx+1, len(texts), len(text))
                    response = client.post(endpoint, json=payload)
                    response.raise_for_status()
                    
                    res_json = response.json()
                    response_text = res_json.get("response", "").strip()
                    
                    # Parse output JSON
                    data = json.loads(response_text)
                    pred_cat = data.get("category", "").strip()
                    reason = data.get("reason", "")
                    
                    # Fuzzy match predicted category to valid list
                    matched_cat = self._fuzzy_match_category(pred_cat)
                    logger.debug("Ollama prediction: '%s' -> matched: '%s' (Reason: %s)", pred_cat, matched_cat, reason)
                    
                    predicted_labels.append(matched_cat)
                    # We can assign dummy probability (1.0 for predicted, 0.0 others)
                    prob_dict = {cat: (1.0 if cat == matched_cat else 0.0) for cat in self.categories}
                    probabilities.append(prob_dict)

                except Exception as e:
                    logger.warning("Ollama prediction failed for text index %d: %s. Falling back to default.", idx, e)
                    # Use fallback category
                    fallback = self.categories[0] if self.categories else "Vnitřní informace"
                    predicted_labels.append(fallback)
                    probabilities.append({fallback: 1.0})

        return ClassificationResult(
            predicted_labels=predicted_labels,
            probabilities=probabilities,
        )

    def _fuzzy_match_category(self, pred_cat: str) -> str:
        """Find the closest matching category from the valid categories list."""
        if not self.categories:
            return pred_cat
            
        pred_clean = pred_cat.lower().strip().replace(":", "")
        for cat in self.categories:
            if cat.lower() == pred_clean:
                return cat
                
        # Look for partial matches
        for cat in self.categories:
            if cat.lower() in pred_clean or pred_clean in cat.lower():
                return cat
                
        # Default fallback to first category
        return self.categories[0]

    def save(self, path: Path) -> None:
        """Save Ollama settings and category configs to a JSON file.

        Args:
            path: Directory where configs will be saved.
        """
        path.mkdir(parents=True, exist_ok=True)
        config_file = path / "ollama_config.json"
        
        data = {
            "host": self.host,
            "model_name": self.model_name,
            "timeout": self.timeout,
            "few_shot_examples": self.few_shot_examples,
            "categories": self.categories,
            "auth_type": self.auth_type,
            "username": "",
            "password": "",
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Ollama configs saved to %s (credentials omitted)", config_file)

    def load(self, path: Path) -> None:
        """Load Ollama settings from JSON config file.

        Args:
            path: Directory containing the JSON file.
        """
        import os

        config_file = path / "ollama_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Ollama config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_host = data.get("host")
        env_host = os.environ.get("OLLAMA_HOST")
        if env_host:
            self.host = env_host.rstrip("/")
        elif loaded_host and not _is_placeholder(loaded_host):
            self.host = loaded_host.rstrip("/")
        else:
            self.host = "http://localhost:11434"

        loaded_model = data.get("model_name")
        env_model = os.environ.get("OLLAMA_MODEL_NAME")
        if env_model:
            self.model_name = env_model
        elif loaded_model and not _is_placeholder(loaded_model):
            self.model_name = loaded_model
        else:
            self.model_name = "llama3"

        self.timeout = data.get("timeout", self.timeout)
        self.few_shot_examples = data.get("few_shot_examples", self.few_shot_examples)
        self.categories = data.get("categories", self.categories)

        loaded_auth = data.get("auth_type")
        env_auth = os.environ.get("OLLAMA_AUTH_TYPE")
        if env_auth:
            self.auth_type = env_auth
        elif loaded_auth and not _is_placeholder(loaded_auth):
            self.auth_type = loaded_auth

        loaded_user = data.get("username")
        env_user = os.environ.get("OLLAMA_USERNAME")
        if env_user:
            self.username = env_user
        elif loaded_user and not _is_placeholder(loaded_user):
            self.username = loaded_user

        loaded_pass = data.get("password")
        env_pass = os.environ.get("OLLAMA_PASSWORD")
        if env_pass:
            self.password = env_pass
        elif loaded_pass and not _is_placeholder(loaded_pass):
            self.password = loaded_pass

        logger.info("Ollama configs loaded from %s", config_file)
