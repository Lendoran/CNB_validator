"""Rule-based classifier for CNB OAM document classification.

Uses weighted keyword dictionaries for each document category. Classification
is performed by scoring each category based on case-insensitive keyword matches
in the input text, then selecting the category with the highest score.
"""

import json
import logging
from pathlib import Path

from src.classifiers.base import BaseClassifier, ClassificationResult

logger = logging.getLogger(__name__)

# Type alias for keyword rules: mapping from category to list of (keyword, weight) tuples.
KeywordRules = dict[str, list[tuple[str, float]]]

# Default keyword rules for each OAM document category.
# Higher weight = more distinctive / specific keyword.
_DEFAULT_RULES: KeywordRules = {
    "Oznámení o konání valné hromady": [
        ("pozvánka na valnou hromadu", 3.0),
        ("oznámení o konání valné hromady", 3.0),
        ("valná hromada", 2.0),
        ("řádná valná hromada", 2.5),
        ("mimořádná valná hromada", 2.5),
        ("program valné hromady", 2.0),
        ("pořad jednání", 1.5),
        ("návrh usnesení", 1.0),
        ("registrace akcionářů", 1.5),
        ("hlasování", 0.5),
    ],
    "Informace související s valnou hromadou": [
        ("rozhodnutí valné hromady", 3.0),
        ("usnesení valné hromady", 3.0),
        ("výsledky hlasování valné hromady", 3.0),
        ("zápis z valné hromady", 2.5),
        ("valná hromada", 1.0),
        ("dividenda", 1.5),
        ("výplata dividendy", 2.0),
        ("rozhodný den", 1.0),
        ("stanovy", 1.0),
    ],
    "Informace související s emisí dluhopisů": [
        ("emise dluhopisů", 3.0),
        ("emisní podmínky", 3.0),
        ("dluhopis", 2.0),
        ("dluhopisy", 2.0),
        ("bond", 1.5),
        ("bonds", 1.5),
        ("kupón", 1.5),
        ("kupónová sazba", 2.0),
        ("jmenovitá hodnota dluhopisu", 2.5),
        ("výnos dluhopisu", 2.0),
        ("splatnost dluhopisu", 2.0),
        ("prospekt", 1.0),
        ("ISIN", 1.0),
    ],
    "Výroční finanční zpráva": [
        ("výroční zpráva", 3.0),
        ("výroční finanční zpráva", 3.5),
        ("roční finanční", 2.5),
        ("annual report", 2.5),
        ("annual financial report", 3.0),
        ("konsolidovaná výroční zpráva", 3.0),
        ("roční účetní závěrka", 2.5),
        ("účetní závěrka za rok", 2.5),
        ("zpráva auditora", 1.5),
        ("zpráva nezávislého auditora", 2.0),
        ("hospodaření za rok", 1.5),
        ("audit", 0.5),
    ],
    "Pololetní finanční zpráva": [
        ("pololetní zpráva", 3.0),
        ("pololetní finanční zpráva", 3.5),
        ("pololetní finanční", 3.0),
        ("half-year", 2.5),
        ("half-year financial report", 3.0),
        ("interim report", 2.0),
        ("pololetní účetní závěrka", 3.0),
        ("za první pololetí", 2.5),
        ("za 1. pololetí", 2.5),
        ("hospodaření za pololetí", 2.0),
    ],
    "Vnitřní informace": [
        ("vnitřní informace", 3.0),
        ("inside information", 3.0),
        ("insider information", 2.5),
        ("insider", 1.5),
        ("MAR", 2.0),
        ("nařízení MAR", 2.5),
        ("článek 17", 2.0),
        ("article 17", 2.0),
        ("ad hoc", 1.5),
        ("regulované informace", 1.0),
        ("cenotvorná informace", 2.0),
        ("kurzově relevantní", 2.0),
    ],
    "Oznámení podílu na hlasovacích právech": [
        ("podíl na hlasovacích právech", 3.0),
        ("oznámení podílu na hlasovacích", 3.5),
        ("hlasovací práva", 2.0),
        ("voting rights", 2.0),
        ("překročení prahové hodnoty", 2.5),
        ("major holdings", 2.0),
        ("podíl na hlasovacích", 2.5),
        ("nabytí podílu", 1.5),
        ("pozbytí podílu", 1.5),
        ("prahová hodnota", 1.5),
    ],
    "Informace o celkovém počtu hlasovacích práv a výši základního kapitálu": [
        ("celkový počet hlasovacích práv", 3.5),
        ("výše základního kapitálu", 3.0),
        ("základní kapitál", 2.0),
        ("total number of voting rights", 3.0),
        ("počet hlasovacích práv", 2.5),
        ("celkový počet hlasů", 2.5),
        ("akcie s hlasovacími právy", 2.0),
        ("share capital", 1.5),
    ],
    "Oznámení o konání schůze vlastníků": [
        ("schůze vlastníků", 3.0),
        ("schůze vlastníků dluhopisů", 3.5),
        ("svolání schůze vlastníků", 3.5),
        ("bondholders meeting", 3.0),
        ("bondholders' meeting", 3.0),
        ("meeting of bondholders", 3.0),
        ("program schůze", 2.0),
        ("vlastníci dluhopisů", 2.0),
    ],
    "Informace o nabytí nebo pozbytí vlastních akcií emitenta": [
        ("vlastní akcie", 3.0),
        ("nabytí vlastních akcií", 3.5),
        ("pozbytí vlastních akcií", 3.5),
        ("treasury shares", 3.0),
        ("buy-back", 2.5),
        ("buyback", 2.5),
        ("zpětný odkup akcií", 3.0),
        ("nabytí", 1.0),
        ("pozbytí", 1.0),
        ("share buyback", 2.5),
    ],
    "Zpráva o úhradách placených státu": [
        ("úhrady placené státu", 3.5),
        ("úhrady placených státu", 3.5),
        ("payments to governments", 3.0),
        ("report on payments", 2.5),
        ("zpráva o úhradách", 3.0),
        ("platby vládám", 2.5),
        ("těžební průmysl", 1.5),
        ("přírodní zdroje", 1.0),
    ],
    "Samostatná zpráva o nefinančních informacích": [
        ("nefinanční informace", 3.0),
        ("nefinančních informacích", 3.0),
        ("nefinanční", 2.0),
        ("non-financial", 2.5),
        ("non-financial information", 3.0),
        ("ESG", 2.5),
        ("udržitelnost", 2.0),
        ("sustainability", 2.0),
        ("sustainable", 1.5),
        ("společenská odpovědnost", 2.0),
        ("CSR", 2.0),
        ("životní prostředí", 1.5),
        ("environment", 1.0),
    ],
}


class RuleBasedClassifier(BaseClassifier):
    """Rule-based classifier using weighted keyword matching.

    Each category is associated with a set of keywords and their weights.
    For a given document, each category's score is computed as the sum of
    weights for all keywords found in the (lowercased) text. The category
    with the highest score is selected as the prediction.

    If no keywords match for any category, the classifier falls back to a
    configurable default category.
    """

    def __init__(
        self,
        rules: KeywordRules | None = None,
        default_category: str = "Vnitřní informace",
    ) -> None:
        """Initialise the rule-based classifier.

        Args:
            rules: Optional custom keyword rules. If None, built-in defaults
                are used.
            default_category: Category to assign when no keywords match.
        """
        self._rules: KeywordRules = rules if rules is not None else _DEFAULT_RULES.copy()
        self._default_category = default_category
        logger.info(
            "RuleBasedClassifier initialised with %d categories.", len(self._rules)
        )

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name for this classifier."""
        return "RuleBasedClassifier"

    def train(self, texts: list[str], labels: list[str]) -> None:
        """No-op — rule-based classifier does not require training.

        Args:
            texts: List of document texts (unused).
            labels: List of labels (unused).
        """
        self._validate_inputs(texts, labels)
        logger.info(
            "train() called on RuleBasedClassifier — no-op (rules are predefined)."
        )

    def predict(self, texts: list[str]) -> ClassificationResult:
        """Predict document categories using keyword scoring.

        Args:
            texts: List of document text contents.

        Returns:
            ClassificationResult with predicted labels and normalised score
            distributions as probabilities.

        Raises:
            ValueError: If texts is empty.
        """
        self._validate_inputs(texts)
        predicted_labels: list[str] = []
        probabilities: list[dict[str, float]] = []

        for text in texts:
            scores = self._score_text(text)
            total_score = sum(scores.values())

            if total_score > 0:
                probs = {cat: score / total_score for cat, score in scores.items()}
                best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
            else:
                # No keywords matched — uniform distribution, use default
                n_cats = len(self._rules) if self._rules else 1
                probs = {cat: 1.0 / n_cats for cat in self._rules}
                best_category = self._default_category

            predicted_labels.append(best_category)
            probabilities.append(probs)

        logger.debug("Predicted %d documents.", len(predicted_labels))
        return ClassificationResult(
            predicted_labels=predicted_labels,
            probabilities=probabilities,
        )

    def save(self, path: Path) -> None:
        """Save the keyword rules to a JSON file.

        Args:
            path: Directory where ``rules.json`` will be written.
        """
        path.mkdir(parents=True, exist_ok=True)
        rules_file = path / "rules.json"
        # Convert list-of-tuples to list-of-lists for JSON serialisation.
        serialisable = {
            cat: [[kw, w] for kw, w in kwlist]
            for cat, kwlist in self._rules.items()
        }
        meta = {
            "default_category": self._default_category,
            "rules": serialisable,
        }
        rules_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Rules saved to %s", rules_file)

    def load(self, path: Path) -> None:
        """Load keyword rules from a JSON file.

        Args:
            path: Directory containing ``rules.json``.

        Raises:
            FileNotFoundError: If the rules file does not exist.
        """
        rules_file = path / "rules.json"
        if not rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_file}")

        meta = json.loads(rules_file.read_text(encoding="utf-8"))
        self._default_category = meta.get("default_category", self._default_category)
        raw_rules = meta.get("rules", {})
        self._rules = {
            cat: [(kw, float(w)) for kw, w in kwlist]
            for cat, kwlist in raw_rules.items()
        }
        logger.info(
            "Loaded rules for %d categories from %s", len(self._rules), rules_file
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_text(self, text: str) -> dict[str, float]:
        """Score all categories for a single document.

        Args:
            text: Document text content.

        Returns:
            Dict mapping each category to its weighted keyword score.
        """
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for category, keywords in self._rules.items():
            score = 0.0
            for keyword, weight in keywords:
                # Count occurrences of the keyword in the text.
                count = text_lower.count(keyword.lower())
                if count > 0:
                    # Use log-dampened counts to avoid over-counting repeated
                    # keywords, but always count at least 1.
                    import math
                    dampened = 1.0 + math.log(count)
                    score += weight * dampened
            scores[category] = score

        return scores
