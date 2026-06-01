# CNB OAM Document Category Classifier

Predict the category of documents submitted to the Czech National Bank's OAM
(Centrální úložiště regulovaných informací) using text content extracted from
file attachments.

## Project Structure

```
src/
├── scraper/          # Data acquisition from OAM website
├── preprocessing/    # Text extraction from PDF, XBRL, XHTML, DOCX, ZIP
├── classifiers/      # Classification approaches
├── evaluation/       # Metrics and cross-method comparison
└── cli.py            # Command-line interface
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Install Playwright browsers
playwright install chromium

# Optional: install BERT dependencies
pip install -e ".[bert]"
```

## Usage

```bash
# 1. Scrape metadata from OAM website
python -m src.cli scrape --output data/

# 2. Download file attachments
python -m src.cli download --resume

# 3. Extract text from downloaded files
python -m src.cli extract

# 4. Train classifiers
python -m src.cli train --method rule_based
python -m src.cli train --method tfidf --classifier svm
python -m src.cli train --method czech_bert

# 5. Evaluate and compare all methods
python -m src.cli evaluate --all --output results/
```

## Classification Approaches

| Method | Description | Expected Accuracy |
|--------|-------------|-------------------|
| Rule-based | Keyword/regex pattern matching | ~60-75% |
| TF-IDF + ML | TF-IDF vectorization + SVM/RF/LogReg | ~80-90% |
| Czech BERT | Fine-tuned RobeCzech/Czert | ~88-95% |

## Document Categories

Categories based on "Typ informace" from the OAM website, with "Výroční/pololetní
finanční zpráva" split into separate annual and semi-annual categories. Very small
categories are excluded from classification.
