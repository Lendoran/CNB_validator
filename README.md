# CNB OAM Document Category Classifier

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Lendoran/CNB_validator)

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
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

## Configuration

1. Copy the default configuration template:
   ```bash
   cp config-default.yaml config.yaml
   ```
2. Open `config.yaml` and fill in your settings.
3. **Important:** If using the **Ollama LLM classifier**, you must specify your own Ollama `host` and authentication `credentials` (username/password) under the `ollama` section in `config.yaml`.

## Usage (CLI Utility)

The project includes a unified CLI for the entire pipeline.

### Core Pipeline
```bash
# 1. Scrape metadata from OAM website
python -m src.cli scrape --output data/

# 2. Download file attachments
python -m src.cli download --resume

# 3. Extract text from downloaded files (PDF, DOCX, XHTML, XBRL, ZIP)
python -m src.cli extract

# 4. Train classifiers
python -m src.cli train --method rule_based
python -m src.cli train --method tfidf --classifier svm
python -m src.cli train --method czech_bert

# 5. Evaluate and compare all methods
python -m src.cli evaluate --all --output results/

# 6. Compile reports from saved runs (generates report/report.html)
python -m src.cli report
```

### Utility Commands
```bash
# Predict category for a single file on disk
python -m src.cli predict --file path/to/doc.pdf --method tfidf

# Validate if file content matches its declared category
python -m src.cli validate --file doc.pdf --declared-category "Vnitřní informace" --method bert

# Show database and collection statistics
python -m src.cli stats

# Run K-Fold cross-validation on the dataset
python -m src.cli cross-validate --method tfidf --folds 5
```

## Results & Performance

Based on our evaluation on the test split (20% of the dataset), the models achieved the following performance:

| Method | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| TF-IDF (SVM) | 95.0 % | 0.788 | 0.949 |
| Czech BERT (RobeCzech) | 94.7 % | 0.681 | 0.944 |
| TF-IDF (Random Forest) | 93.3 % | 0.734 | 0.930 |
| TF-IDF (Logistic Regression) | 89.8 % | 0.760 | 0.908 |
| Rule-based | 72.2 % | 0.488 | 0.724 |
| Ollama LLM (Gemma 3) | 66.2 % | 0.520 | 0.678 |

*Note: TF-IDF with a Linear Support Vector Machine (SVM) proved to be the most accurate model overall. Czech BERT achieved very comparable performance with higher potential for robust contextual generalization on new formats.*

## Document Categories

Categories based on "Typ informace" from the OAM website, with "Výroční/pololetní
finanční zpráva" split into separate annual and semi-annual categories. Some
categories are excluded from classification.