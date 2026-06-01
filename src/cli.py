"""Click command-line interface for the CNB OAM document classifier.

Aggregates commands for scraping, downloading, extracting, training, and evaluating.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
import click
import yaml

logger = logging.getLogger("cnb_classifier")


def load_config() -> dict:
    """Load configuration from config.yaml at project root, with defaults."""
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            click.echo(f"Warning: Failed to load config.yaml: {e}. Using default settings.")
    
    # Standard fallback defaults
    return {
        "paths": {
            "data_dir": "data",
            "raw_dir": "data/raw",
            "metadata_db": "data/metadata.db",
            "results_dir": "results",
            "models_dir": "models",
        },
        "scraper": {
            "rate_limit_seconds": 1.5,
            "timeout_seconds": 60,
        },
        "categories": {
            "classify": [
                "Oznámení o konání valné hromady",
                "Informace související s valnou hromadou",
                "Informace související s emisí dluhopisů",
                "Výroční finanční zpráva",
                "Pololetní finanční zpráva",
                "Vnitřní informace",
                "Oznámení podílu na hlasovacích právech",
                "Informace o celkovém počtu hlasovacích práv a výši základního kapitálu",
                "Oznámení o konání schůze vlastníků",
                "Informace o nabytí nebo pozbytí vlastních akcií emitenta",
                "Zpráva o úhradách placených státu",
                "Samostatná zpráva o nefinančních informacích",
            ]
        },
        "preprocessing": {
            "min_text_length": 50,
        }
    }


# Click group entrypoint
@click.group()
@click.option("--debug", is_flag=True, help="Show verbose debug logs.")
def cli(debug: bool) -> None:
    """CNB OAM Document Category Classifier CLI.

    Allows automated document collection, parsing, and multi-approach category prediction.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

@cli.command()
@click.option(
    "--categories",
    default="all",
    help="Comma-separated categories to scrape. Defaults to all active configured ones.",
)
@click.option("--limit", type=int, default=None, help="Max records to fetch per category.")
@click.option(
    "--years",
    default=None,
    help="Comma-separated list of years to scrape (e.g. 2025,2026). Defaults to all history (2010 to current year).",
)
def scrape(categories: str, limit: int | None, years: str | None) -> None:
    """Scrape metadata from the CNB OAM web forms."""
    config = load_config()
    db_path = Path(config["paths"]["metadata_db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine categories list
    if categories.lower() == "all":
        target_cats = config["categories"]["classify"]
    else:
        target_cats = [c.strip() for c in categories.split(",") if c.strip()]

    # Parse years list if provided
    years_list = None
    if years:
        try:
            years_list = [int(y.strip()) for y in years.split(",") if y.strip()]
        except ValueError:
            click.echo("Error: --years must be a comma-separated list of integers.", err=True)
            sys.exit(1)

    # To scrape Výroční vs Pololetní, the OAM form mapping will select "Výroční/pololetní..."
    # Deduplicate form targets
    from src.scraper.browser import CATEGORY_MAPPING
    form_targets = set()
    for cat in target_cats:
        form_targets.add(CATEGORY_MAPPING.get(cat, cat))

    click.echo(f"Scraping {len(form_targets)} OAM search categories...")

    async def run_scrape():
        from src.scraper.browser import OAMBrowser
        from src.scraper.metadata_db import MetadataDB
        
        db = MetadataDB(db_path)
        
        async with OAMBrowser(
            rate_limit_seconds=config["scraper"]["rate_limit_seconds"],
            timeout_seconds=config["scraper"]["timeout_seconds"],
        ) as browser:
            existing_ids = db.get_all_document_ids()
            click.echo(f"Loaded {len(existing_ids)} existing document IDs from database.")
            
            for form_cat in form_targets:
                click.echo(f"Processing category: {form_cat}")
                try:
                    docs = await browser.scrape_category(
                        form_cat, limit=limit, existing_ids=existing_ids, years=years_list
                    )
                    click.echo(f"Found {len(docs)} new document records for {form_cat}.")
                    
                    # Store to database in a single batch transaction
                    docs_to_insert = []
                    files_to_insert = []
                    for doc in docs:
                        files = doc.pop("files", [])
                        docs_to_insert.append(doc)
                        for f in files:
                            f["document_id"] = doc["id"]
                            files_to_insert.append(f)
                            
                    db.insert_documents_batch(docs_to_insert, files_to_insert)
                    click.echo(f"Database updated. Inserted/Updated {len(docs_to_insert)} docs and {len(files_to_insert)} files in batch.")
                except Exception as e:
                    logger.error("Scraping failed for category %s: %s", form_cat, e, exc_info=True)
                    click.echo(f"Error scraping category {form_cat}: {e}", err=True)

    asyncio.run(run_scrape())
    click.echo("Scraping completed.")


@cli.command()
@click.option("--resume/--no-resume", default=True, help="Skip already downloaded files.")
@click.option("--rate-limit", type=float, default=None, help="Rate limit delay in seconds.")
def download(resume: bool, rate_limit: float | None) -> None:
    """Download pending document attachments from CNB."""
    config = load_config()
    db_path = config["paths"]["metadata_db"]
    raw_dir = config["paths"]["raw_dir"]
    rl = rate_limit if rate_limit is not None else config["scraper"]["rate_limit_seconds"]

    from src.scraper.downloader import FileDownloader
    downloader = FileDownloader(
        db_path=db_path,
        raw_dir=raw_dir,
        rate_limit_seconds=rl,
    )

    click.echo("Starting downloads...")
    dl_count = asyncio.run(downloader.download_all(resume=resume))
    click.echo(f"Download process complete. Succeeded downloads: {dl_count}")


@cli.command()
@click.option("--force", is_flag=True, help="Re-extract even if text already exists.")
def extract(force: bool) -> None:
    """Extract text from downloaded documents (PDF, ZIP, DOCX, XHTML)."""
    config = load_config()
    db_path = config["paths"]["metadata_db"]

    from src.preprocessing.pipeline import TextExtractionPipeline
    pipeline = TextExtractionPipeline(db_path=db_path)

    click.echo("Running text extraction pipeline...")
    extracted_count = pipeline.process_all_files(force=force)
    click.echo(f"Text extraction complete. Processed {extracted_count} files successfully.")


@cli.command()
@click.option(
    "--method",
    required=True,
    type=click.Choice(["rule_based", "tfidf", "czech_bert", "ollama"]),
    help="Classification method to train.",
)
@click.option(
    "--classifier",
    default="svm",
    type=click.Choice(["svm", "random_forest", "logistic_regression"]),
    help="Classifier type for TF-IDF method.",
)
@click.option(
    "--model",
    default="ufal/robeczech-base",
    help="Pretrained model name for czech_bert fine-tuning.",
)
def train(method: str, classifier: str, model: str) -> None:
    """Train/Fine-tune a classification model."""
    config = load_config()
    db_path = config["paths"]["metadata_db"]
    models_dir = Path(config["paths"]["models_dir"])
    categories = config["categories"]["classify"]
    
    # 1. Load data
    from src.classifiers.data_loader import load_dataset_from_db, create_splits
    click.echo("Loading dataset from database...")
    texts, labels = load_dataset_from_db(db_path, categories)
    
    if not texts:
        click.echo("No documents with extracted text found in database. Run 'extract' first.", err=True)
        sys.exit(1)
        
    splits = create_splits(
        texts,
        labels,
        test_size=config.get("training", {}).get("test_size", 0.2),
        val_size=config.get("training", {}).get("val_size", 0.1),
        random_seed=config.get("training", {}).get("random_seed", 42),
    )
    
    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]

    # Save splits for evaluation consistency
    splits_dir = Path(config["paths"].get("splits_dir", "data/splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(splits_dir / "splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    click.echo(f"Train/val/test splits cached at {splits_dir / 'splits.pkl'}")

    # 2. Train model
    if method == "rule_based":
        from src.classifiers.rule_based import RuleBasedClassifier
        clf = RuleBasedClassifier()
        clf.train(train_texts, train_labels)
        clf.save(models_dir / "rule_based")
        click.echo(f"RuleBasedClassifier rules saved to {models_dir / 'rule_based'}")
        
    elif method == "tfidf":
        from src.classifiers.tfidf_ml import TfidfMLClassifier
        clf = TfidfMLClassifier(
            classifier_type=classifier,
            max_features=config.get("training", {}).get("tfidf", {}).get("max_features", 10000),
            ngram_range=tuple(config.get("training", {}).get("tfidf", {}).get("ngram_range", [1, 2])),
        )
        clf.train(train_texts, train_labels)
        clf.save(models_dir / "tfidf")
        click.echo(f"TF-IDF model saved to {models_dir / 'tfidf'}")
        
    elif method == "czech_bert":
        from src.classifiers.czech_bert import CzechBertClassifier
        clf = CzechBertClassifier(
            model_name=model,
            epochs=config.get("training", {}).get("bert", {}).get("epochs", 5),
            batch_size=config.get("training", {}).get("bert", {}).get("batch_size", 8),
            learning_rate=config.get("training", {}).get("bert", {}).get("learning_rate", 2e-5),
        )
        clf.train(train_texts, train_labels)
        clf.save(models_dir / "bert")
        click.echo(f"BERT fine-tuned model saved to {models_dir / 'bert'}")
        
    elif method == "ollama":
        from src.classifiers.ollama_llm import OllamaLLMClassifier
        ollama_config = config.get("ollama", {})
        clf = OllamaLLMClassifier(
            host=ollama_config.get("host", "http://localhost:11434"),
            model_name=ollama_config.get("model_name", "llama3"),
            timeout_seconds=ollama_config.get("timeout_seconds", 45),
            few_shot_examples=ollama_config.get("few_shot_examples", []),
            auth_type=ollama_config.get("auth_type"),
            username=ollama_config.get("username"),
            password=ollama_config.get("password"),
        )
        clf.train(train_texts, train_labels)
        clf.save(models_dir / "ollama")
        click.echo(f"OllamaLLMClassifier configs saved to {models_dir / 'ollama'}")

def save_evaluation_run(method_key: str, method_name: str, y_true: list[str], y_pred: list[str]) -> None:
    """Save raw true/predicted labels to results/eval_runs/{method_key}.json."""
    import json
    run_dir = Path("results/eval_runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_data = {
        "method_name": method_name,
        "y_true": y_true,
        "y_pred": y_pred
    }
    with open(run_dir / f"{method_key}.json", "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2)


def compile_reports_from_runs(res_dir: Path, categories: list[str]) -> None:
    """Aggregate all runs in results/eval_runs/*.json and compile reports/charts/HTML."""
    import json
    from src.evaluation.comparison import MethodComparison
    
    run_dir = Path("results/eval_runs")
    if not run_dir.exists() or not list(run_dir.glob("*.json")):
        click.echo("Warning: No saved evaluation runs found in results/eval_runs/. Run evaluate first.")
        return
        
    comparison = MethodComparison(labels=categories)
    
    for run_file in sorted(run_dir.glob("*.json")):
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                run_data = json.load(f)
            method_name = run_data["method_name"]
            y_true = run_data["y_true"]
            y_pred = run_data["y_pred"]
            comparison.add_result(method_name, y_true, y_pred)
        except Exception as e:
            click.echo(f"Failed to load run from {run_file.name}: {e}", err=True)
            
    if len(comparison.results) > 0:
        comparison.generate_full_report(res_dir)
        click.echo(f"Evaluation report and charts generated inside {res_dir}")
        click.echo("\n--- Performance Summary (Aggregated from all saved runs) ---")
        click.echo(comparison.comparison_table().to_string())
    else:
        click.echo("No successful evaluation runs were loaded.")


@cli.command()
@click.option(
    "--method",
    type=click.Choice(["rule_based", "tfidf", "czech_bert", "ollama"]),
    help="Evaluate a specific method.",
)
@click.option(
    "--all",
    "evaluate_all",
    is_flag=True,
    help="Evaluate all trained models and generate comparison reports.",
)
@click.option(
    "--output",
    default=None,
    help="Results output directory.",
)
def evaluate(method: str | None, evaluate_all: bool, output: str | None) -> None:
    """Evaluate classifier performance on the test split."""
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    splits_dir = Path(config["paths"].get("splits_dir", "data/splits"))
    categories = config["categories"]["classify"]
    
    res_dir = Path(output) if output else Path(config["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    splits_path = splits_dir / "splits.pkl"
    if not splits_path.exists():
        click.echo("Splits file splits.pkl not found. Run 'train' first to generate splits.", err=True)
        sys.exit(1)

    import pickle
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]

    methods_to_evaluate = []
    if evaluate_all:
        # Check which models exist on disk
        if (models_dir / "rule_based" / "rules.json").exists():
            methods_to_evaluate.append("rule_based")
        
        # Check tfidf models
        for tfidf_file in (models_dir / "tfidf").glob("tfidf_*.pkl") if (models_dir / "tfidf").exists() else []:
            clf_type = tfidf_file.stem.replace("tfidf_", "")
            methods_to_evaluate.append(f"tfidf_{clf_type}")

        if (models_dir / "bert" / "mappings.json").exists():
            methods_to_evaluate.append("czech_bert")
    elif method:
        if method == "tfidf":
            # If TF-IDF general requested, check and add all found tfidf submodels
            for tfidf_file in (models_dir / "tfidf").glob("tfidf_*.pkl") if (models_dir / "tfidf").exists() else []:
                clf_type = tfidf_file.stem.replace("tfidf_", "")
                methods_to_evaluate.append(f"tfidf_{clf_type}")
        else:
            methods_to_evaluate.append(method)
    else:
        click.echo("Error: Please specify either --method or --all.", err=True)
        sys.exit(1)

    if not methods_to_evaluate:
        click.echo("No trained models found to evaluate. Run 'train' first.", err=True)
        sys.exit(1)

    click.echo(f"Evaluating methods: {methods_to_evaluate}")

    for m in methods_to_evaluate:
        click.echo(f"Running evaluation for: {m}")
        try:
            if m == "rule_based":
                from src.classifiers.rule_based import RuleBasedClassifier
                clf = RuleBasedClassifier()
                clf.load(models_dir / "rule_based")
                pred_res = clf.predict(test_texts)
                save_evaluation_run("rule_based", "Rule-based", test_labels, pred_res.predicted_labels)
                
            elif m.startswith("tfidf_") or m == "tfidf":
                from src.classifiers.tfidf_ml import TfidfMLClassifier
                clf_type = m.replace("tfidf_", "") if "_" in m else "svm"
                clf = TfidfMLClassifier(classifier_type=clf_type)
                clf.load(models_dir / "tfidf")
                pred_res = clf.predict(test_texts)
                save_evaluation_run(f"tfidf_{clf_type}", f"TF-IDF ({clf_type.upper()})", test_labels, pred_res.predicted_labels)
                
            elif m == "czech_bert":
                from src.classifiers.czech_bert import CzechBertClassifier
                clf = CzechBertClassifier()
                clf.load(models_dir / "bert")
                pred_res = clf.predict(test_texts)
                save_evaluation_run("czech_bert", "Czech BERT", test_labels, pred_res.predicted_labels)
                
            elif m == "ollama":
                from src.classifiers.ollama_llm import OllamaLLMClassifier
                ollama_config = config.get("ollama", {})
                clf = OllamaLLMClassifier(
                    host=ollama_config.get("host"),
                    model_name=ollama_config.get("model_name"),
                    timeout_seconds=ollama_config.get("timeout_seconds", 45),
                    few_shot_examples=ollama_config.get("few_shot_examples", []),
                    auth_type=ollama_config.get("auth_type"),
                    username=ollama_config.get("username"),
                    password=ollama_config.get("password"),
                )
                clf.load(models_dir / "ollama")
                pred_res = clf.predict(test_texts)
                save_evaluation_run("ollama", "Ollama LLM", test_labels, pred_res.predicted_labels)
                
        except Exception as e:
            logger.error("Failed evaluating %s: %s", m, e, exc_info=True)
            click.echo(f"Failed evaluating {m}: {e}", err=True)

    # Recompile reports dynamically using all available run files
    click.echo("\nCompiling aggregated reports...")
    compile_reports_from_runs(res_dir, categories)


@cli.command()
@click.option(
    "--output",
    default=None,
    help="Results output directory.",
)
def report(output: str | None) -> None:
    """Generate combined comparison report from saved evaluation runs."""
    config = load_config()
    res_dir = Path(output) if output else Path(config["paths"]["results_dir"])
    categories = config["categories"]["classify"]
    
    click.echo("Compiling reports from saved runs...")
    compile_reports_from_runs(res_dir, categories)


@cli.command()
@click.option("--file", required=True, type=click.Path(exists=True), help="Document file path to predict.")
@click.option(
    "--method",
    required=True,
    type=click.Choice(["rule_based", "tfidf", "czech_bert", "ollama"]),
    help="Classification model to use.",
)
@click.option(
    "--classifier",
    default="svm",
    help="TF-IDF classifier subtype (svm, random_forest, logistic_regression).",
)
def predict(file: str, method: str, classifier: str) -> None:
    """Predict the document category for a single file on disk."""
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    
    file_path = Path(file)

    # 1. Extract text from file
    click.echo(f"Extracting text from {file_path.name}...")
    from src.preprocessing.pipeline import TextExtractionPipeline
    # Temp extraction (no DB needed for single file prediction)
    temp_pipeline = TextExtractionPipeline(db_path=":memory:")
    raw_text, _ = temp_pipeline.extract_text(file_path)
    
    from src.preprocessing.text_cleaner import clean_text
    cleaned = clean_text(raw_text)

    if not cleaned:
        click.echo("Error: Text extraction returned empty text. File is empty or format unsupported.", err=True)
        sys.exit(1)

    click.echo(f"Extracted {len(cleaned)} characters. Loading {method} classifier...")

    # 2. Load model
    try:
        if method == "rule_based":
            from src.classifiers.rule_based import RuleBasedClassifier
            clf = RuleBasedClassifier()
            clf.load(models_dir / "rule_based")
        elif method == "tfidf":
            from src.classifiers.tfidf_ml import TfidfMLClassifier
            clf = TfidfMLClassifier(classifier_type=classifier)
            clf.load(models_dir / "tfidf")
        elif method == "czech_bert":
            from src.classifiers.czech_bert import CzechBertClassifier
            clf = CzechBertClassifier()
            clf.load(models_dir / "bert")
        elif method == "ollama":
            from src.classifiers.ollama_llm import OllamaLLMClassifier
            clf = OllamaLLMClassifier()
            clf.load(models_dir / "ollama")
            
        # 3. Predict
        res = clf.predict([cleaned])
        predicted_class = res.predicted_labels[0]
        
        click.echo("\n================ Prediction Result ================")
        click.echo(f"File: {file_path.name}")
        click.echo(f"Predicted Category: {predicted_class}")
        
        if res.probabilities:
            click.echo("\n--- Top Probabilities ---")
            sorted_probs = sorted(res.probabilities[0].items(), key=lambda x: x[1], reverse=True)
            for cat, prob in sorted_probs[:5]:
                click.echo(f"  {cat}: {prob * 100:.2f}%")
        click.echo("===================================================\n")
    except Exception as e:
        click.echo(f"Error loading/predicting with {method}: {e}", err=True)
        logger.error("Predict command failed", exc_info=True)


@cli.command()
@click.option("--file", required=True, type=click.Path(exists=True), help="Document file path to validate.")
@click.option("--declared-category", required=True, help="Declared category from metadata.")
@click.option(
    "--method",
    required=True,
    type=click.Choice(["rule_based", "tfidf", "czech_bert", "ollama"]),
    help="Classification model to use.",
)
@click.option(
    "--classifier",
    default="svm",
    help="TF-IDF classifier subtype (svm, random_forest, logistic_regression).",
)
def validate(file: str, declared_category: str, method: str, classifier: str) -> None:
    """Validate if the file contents match the declared category."""
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    file_path = Path(file)

    # 1. Extract text from file
    click.echo(f"Extracting text from {file_path.name}...")
    from src.preprocessing.pipeline import TextExtractionPipeline
    temp_pipeline = TextExtractionPipeline(db_path=":memory:")
    raw_text, _ = temp_pipeline.extract_text(file_path)
    
    from src.preprocessing.text_cleaner import clean_text
    cleaned = clean_text(raw_text)

    if not cleaned:
        click.echo("Error: Text extraction returned empty text. File is empty or format unsupported.", err=True)
        sys.exit(1)

    click.echo(f"Extracted {len(cleaned)} characters. Loading {method} classifier...")

    # 2. Load model
    try:
        if method == "rule_based":
            from src.classifiers.rule_based import RuleBasedClassifier
            clf = RuleBasedClassifier()
            clf.load(models_dir / "rule_based")
        elif method == "tfidf":
            from src.classifiers.tfidf_ml import TfidfMLClassifier
            clf = TfidfMLClassifier(classifier_type=classifier)
            clf.load(models_dir / "tfidf")
        elif method == "czech_bert":
            from src.classifiers.czech_bert import CzechBertClassifier
            clf = CzechBertClassifier()
            clf.load(models_dir / "bert")
        elif method == "ollama":
            from src.classifiers.ollama_llm import OllamaLLMClassifier
            clf = OllamaLLMClassifier()
            clf.load(models_dir / "ollama")
            
        # 3. Predict
        res = clf.predict([cleaned])
        predicted_category = res.predicted_labels[0]
        
        click.echo("\n================ Validation Result ================")
        click.echo(f"File:              {file_path.name}")
        click.echo(f"Declared Category: {declared_category}")
        click.echo(f"Predicted Category: {predicted_category}")
        
        # Match check
        if predicted_category.lower().strip() == declared_category.lower().strip():
            click.echo("\nStatus: [VALID]")
            click.echo("Success: The document content matches the declared submission category.")
        else:
            click.echo("\nStatus: [MISMATCH / WARNING]")
            click.echo(f"Warning: The document content looks like '{predicted_category}' but was submitted under '{declared_category}'!")
            
        if res.probabilities:
            click.echo("\n--- Model Confidence Scores ---")
            sorted_probs = sorted(res.probabilities[0].items(), key=lambda x: x[1], reverse=True)
            for cat, prob in sorted_probs[:3]:
                click.echo(f"  {cat}: {prob * 100:.2f}%")
        click.echo("===================================================\n")
    except Exception as e:
        click.echo(f"Error loading/predicting with {method}: {e}", err=True)
        logger.error("Validate command failed", exc_info=True)


@cli.command()
@click.option("--db", default=None, help="Database path. Defaults to config settings.")
def stats(db: str | None) -> None:
    """Print database and collection statistics."""
    config = load_config()
    db_p = Path(db) if db else Path(config["paths"]["metadata_db"])

    if not db_p.exists():
        click.echo(f"Metadata database not found at {db_p}. Run 'scrape' first.", err=True)
        sys.exit(1)

    from src.scraper.metadata_db import MetadataDB
    metadata_db = MetadataDB(db_p)
    stats_dict = metadata_db.get_stats()

    click.echo("\n================ Collection Statistics ================")
    click.echo(f"SQLite DB Path:      {db_p.resolve()}")
    click.echo(f"Total Documents:     {stats_dict['total_documents']}")
    click.echo(f"Total File Entries:  {stats_dict['total_files']}")
    click.echo(f"Downloaded Files:    {stats_dict['downloaded_files']} / {stats_dict['total_files']} "
               f"({stats_dict['downloaded_files'] / stats_dict['total_files'] * 100:.1f}%)")
    click.echo(f"Extracted Texts:     {stats_dict['extracted_texts']}")
    
    click.echo("\n--- Document Counts Per Category ---")
    for cat, count in stats_dict["category_counts"].items():
        click.echo(f"  {cat:<70}: {count}")
    click.echo("=======================================================\n")


@cli.command()
@click.option(
    "--method",
    required=True,
    type=click.Choice(["rule_based", "tfidf"]),
    help="Classification method to cross-validate.",
)
@click.option(
    "--classifier",
    default="svm",
    type=click.Choice(["svm", "random_forest", "logistic_regression"]),
    help="Classifier type for TF-IDF method.",
)
@click.option(
    "--folds",
    default=5,
    type=int,
    help="Number of folds for cross-validation.",
)
def cross_validate(method: str, classifier: str, folds: int) -> None:
    """Run K-Fold Cross-Validation on the dataset (split 4:1, K iterations)."""
    config = load_config()
    db_path = config["paths"]["metadata_db"]
    categories = config["categories"]["classify"]
    
    from src.classifiers.data_loader import load_dataset_from_db
    click.echo("Loading dataset from database...")
    texts, labels = load_dataset_from_db(db_path, categories)
    
    if not texts:
        click.echo("No documents with extracted text found in database. Run 'extract' first.", err=True)
        sys.exit(1)
        
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    click.echo(f"Running stratified {folds}-fold cross-validation for {method} ({classifier if method=='tfidf' else ''})...")
    
    # Convert to numpy arrays for split indexing
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    all_preds = []
    all_true = []
    
    # Try stratified split first; fallback to standard K-Fold if some rare classes have only 1 member
    try:
        splits = list(skf.split(texts_arr, labels_arr))
    except Exception as e:
        click.echo(f"Stratified split failed: {e}. Falling back to standard K-Fold.", err=True)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        splits = list(kf.split(texts_arr))
        
    fold = 0
    for train_idx, test_idx in splits:
        fold += 1
        click.echo(f"Processing Fold {fold}/{folds}...")
        
        train_texts = texts_arr[train_idx].tolist()
        train_labels = labels_arr[train_idx].tolist()
        test_texts = texts_arr[test_idx].tolist()
        test_labels = labels_arr[test_idx].tolist()
        
        if method == "rule_based":
            from src.classifiers.rule_based import RuleBasedClassifier
            clf = RuleBasedClassifier()
            clf.train(train_texts, train_labels)
            pred_res = clf.predict(test_texts)
            preds = pred_res.predicted_labels
        elif method == "tfidf":
            from src.classifiers.tfidf_ml import TfidfMLClassifier
            clf = TfidfMLClassifier(
                classifier_type=classifier,
                max_features=config.get("training", {}).get("tfidf", {}).get("max_features", 10000),
                ngram_range=tuple(config.get("training", {}).get("tfidf", {}).get("ngram_range", [1, 2])),
            )
            clf.train(train_texts, train_labels)
            pred_res = clf.predict(test_texts)
            preds = pred_res.predicted_labels
            
        all_preds.extend(preds)
        all_true.extend(test_labels)
        
    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro")
    weighted_f1 = f1_score(all_true, all_preds, average="weighted")
    
    click.echo("\n================ Cross-Validation Results ================")
    click.echo(f"Accuracy:    {acc * 100:.2f}%")
    click.echo(f"Macro F1:    {macro_f1 * 100:.2f}%")
    click.echo(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    click.echo(f"Total Evaluated Documents: {len(all_true)}")
    click.echo("\nClassification Report:")
    click.echo(classification_report(all_true, all_preds))
    click.echo("=========================================================\n")


if __name__ == "__main__":
    cli()
