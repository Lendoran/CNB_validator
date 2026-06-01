"""Unit tests for CNB document classifier components.

Tests database operations, parser functionality, preprocessing extraction,
and classifiers.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

# Scraper tests
from src.scraper.metadata_db import MetadataDB
from src.scraper.parser import parse_results_html, parse_detail_html

# Preprocessing tests
from src.preprocessing.text_cleaner import clean_text, truncate_for_bert, get_text_stats
from src.preprocessing.pdf_extractor import extract_text_from_pdf
from src.preprocessing.xhtml_extractor import extract_text_from_xhtml

# Classifiers tests
from src.classifiers.rule_based import RuleBasedClassifier
from src.classifiers.tfidf_ml import TfidfMLClassifier
from src.classifiers.data_loader import create_splits, get_class_distribution


class TestMetadataDB(unittest.TestCase):
    """Test SQLite database operations."""

    def setUp(self) -> None:
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.temp_db.name)
        self.db = MetadataDB(self.db_path)

    def tearDown(self) -> None:
        try:
            self.temp_db.close()
            self.db_path.unlink()
        except OSError:
            pass

    def test_document_and_file_operations(self) -> None:
        doc = {
            "id": "S12345",
            "emitter_name": "Test Banka",
            "emitter_ico": "12345678",
            "emitter_lei": "12345678901234567890",
            "typ_informace": "Vnitřní informace",
            "typ_zpravy": "Vnitřní informace",
            "strucny_popis": "Nějaké vnitřní sdělení",
            "datum_prijeti": "30.05.2026",
            "posledni_den_obdobi": None,
            "section": "Informace uveřejňované emitentem",
        }
        self.db.insert_document(doc)

        f_rec = {
            "file_id": 999,
            "document_id": "S12345",
            "filename": "report.pdf",
            "language": "CZ",
            "file_extension": "pdf",
            "download_url": "http://example.com/file?id=999",
            "local_path": "",
            "downloaded": 0,
        }
        self.db.insert_file(f_rec)

        # Check records
        docs = self.db.get_documents_by_category("Vnitřní informace")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["emitter_name"], "Test Banka")

        to_download = self.db.get_files_to_download()
        self.assertEqual(len(to_download), 1)
        self.assertEqual(to_download[0]["filename"], "report.pdf")

        # Update download status
        self.db.update_file_downloaded(999, 1, "/path/to/local/report.pdf")
        to_download_after = self.db.get_files_to_download()
        self.assertEqual(len(to_download_after), 0)

        # Check stats
        stats = self.db.get_stats()
        self.assertEqual(stats["total_documents"], 1)
        self.assertEqual(stats["downloaded_files"], 1)


class TestParsers(unittest.TestCase):
    """Test HTML parsing functionality."""

    def test_results_parsing(self) -> None:
        mock_html = """
        <html>
            <body>
                <h2>Informace uveřejňované emitentem</h2>
                <table>
                    <tr>
                        <td>Test Emitter a.s. (IČ: 87654321, LEI: 99990000000000000099)</td>
                        <td>30.05.2026</td>
                        <td>Vnitřní informace</td>
                        <td>Brief Description</td>
                        <td><a href="R2_FXX.xdo?par_obf_id=S98765">Otevřít</a></td>
                        <td><a href="DWNL_FILE?file_id=1111">(CZ) Attachment.pdf</a></td>
                    </tr>
                </table>
            </body>
        </html>
        """
        docs = parse_results_html(mock_html, "Vnitřní informace")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["id"], "S98765")
        self.assertEqual(docs[0]["emitter_name"], "Test Emitter a.s.")
        self.assertEqual(docs[0]["emitter_ico"], "87654321")
        self.assertEqual(len(docs[0]["files"]), 1)
        self.assertEqual(docs[0]["files"][0]["file_id"], 1111)
        self.assertEqual(docs[0]["files"][0]["language"], "CZ")

    def test_detail_parsing(self) -> None:
        mock_html = """
        <table>
            <tr><td>Emitent:</td><td>Target Emitter</td></tr>
            <tr><td>IČ:</td><td>11223344</td></tr>
            <tr><td>ID informace:</td><td>S55555</td></tr>
            <tr><td>Typ zprávy:</td><td>Výroční finanční zpráva</td></tr>
        </table>
        <a href="DWNL_FILE?file_id=2222">(EN) Annual_Report.pdf</a>
        """
        meta = parse_detail_html(mock_html)
        self.assertEqual(meta["emitter_name"], "Target Emitter")
        self.assertEqual(meta["emitter_ico"], "11223344")
        self.assertEqual(meta["id"], "S55555")
        self.assertEqual(len(meta["files"]), 1)
        self.assertEqual(meta["files"][0]["file_id"], 2222)
        self.assertEqual(meta["files"][0]["language"], "EN")


class TestPreprocessing(unittest.TestCase):
    """Test text cleaning and normalization."""

    def test_clean_text(self) -> None:
        raw = "Nějaký   text s  diakritikou: ř, š, č, ž. \r\nStrana 1 z 5\n\n\nNový odstavec."
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Nějaký text s diakritikou: ř, š, č, ž.\n\nNový odstavec.")

    def test_truncate_for_bert(self) -> None:
        words = ["slovo"] * 500
        text = " ".join(words)
        truncated = truncate_for_bert(text, max_tokens=100)
        self.assertEqual(len(truncated.split()), 75)  # 100 * 0.75 = 75 words

    def test_get_text_stats(self) -> None:
        text = "Příliš žluťoučký kůň úpěl ďábelské ódy."
        stats = get_text_stats(text)
        self.assertEqual(stats["language_hint"], "cs")
        self.assertGreater(stats["word_count"], 0)


class TestClassifiers(unittest.TestCase):
    """Test machine learning and rule-based classifiers."""

    def test_rule_based_classifier(self) -> None:
        clf = RuleBasedClassifier()
        text1 = "Tato pozvánka na valnou hromadu obsahuje pořad jednání a usnesení valné hromady."
        text2 = "Zveřejňujeme výroční finanční zprávu a auditovanou roční účetní závěrku."
        
        results = clf.predict([text1, text2])
        self.assertEqual(results.predicted_labels[0], "Oznámení o konání valné hromady")
        self.assertEqual(results.predicted_labels[1], "Výroční finanční zpráva")

    def test_tfidf_ml_classifier(self) -> None:
        texts = [
            "pozvánka na valnou hromadu a program jednání",
            "valná hromada akcionářů",
            "výroční zpráva společnosti za loňský rok",
            "konsolidovaná výroční finanční zpráva",
        ]
        labels = [
            "Oznámení o konání valné hromady",
            "Oznámení o konání valné hromady",
            "Výroční finanční zpráva",
            "Výroční finanční zpráva",
        ]
        
        # Train Logistic Regression model (fast training)
        clf = TfidfMLClassifier(classifier_type="logistic_regression")
        clf.train(texts, labels)
        
        # Predict
        res = clf.predict(["řádná valná hromada společnosti"])
        self.assertEqual(res.predicted_labels[0], "Oznámení o konání valné hromady")
        self.assertIsNotNone(res.probabilities)
        self.assertIn("Oznámení o konání valné hromady", res.probabilities[0])

    def test_data_loader_split(self) -> None:
        texts = [f"doc_{i}" for i in range(20)]
        # Balanced 2 classes
        labels = ["ClassA"] * 10 + ["ClassB"] * 10
        
        splits = create_splits(texts, labels, test_size=0.2, val_size=0.1)
        self.assertEqual(len(splits["test"]["texts"]), 4)
        self.assertEqual(len(splits["val"]["texts"]), 2)
        self.assertEqual(len(splits["train"]["texts"]), 14)


if __name__ == "__main__":
    unittest.main()
