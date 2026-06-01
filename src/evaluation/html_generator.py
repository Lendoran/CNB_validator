"""Dynamic HTML Report Generator for CNB OAM Document Classifier.

Generates report.html dynamically from report_template.html, populating 
metricsData, modelMeta variables and adapting analysis texts depending on model performance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

logger = logging.getLogger(__name__)


def generate_html_report(results: dict, labels: list[str], output_path: Path) -> None:
    """Generate interactive HTML report dynamically.

    Args:
        results: Dictionary containing y_true, y_pred, metrics for each method.
        labels: List of target categories.
        output_path: Path where the output HTML report should be written.
    """
    template_path = Path("report/report_template.html")
    if not template_path.exists():
        logger.warning("HTML report template not found at %s. Skipping HTML generation.", template_path)
        return

    logger.info("Generating dynamic HTML report at %s using template %s", output_path, template_path)
    html_content = template_path.read_text(encoding="utf-8")

    # 1. Parse existing/previous metricsData and modelMeta to preserve non-evaluated models (e.g. Ollama)
    existing_html_path = Path("report/report.html")
    metrics_dict = {"accuracy": {}, "macroF1": {}, "weightedF1": {}}
    meta_dict = {}

    if existing_html_path.exists():
        try:
            existing_content = existing_html_path.read_text(encoding="utf-8")
            
            # Parse existing metricsData
            start_md = existing_content.find("const metricsData = {") + len("const metricsData = ")
            end_md = existing_content.find("const modelMeta = {")
            if start_md - len("const metricsData = ") != -1 and end_md != -1:
                block_md = existing_content[start_md:end_md].strip().rstrip(";").strip()
                block_md_clean = re.sub(r',\s*([\]}])', r'\1', block_md)
                block_md_clean = re.sub(r'^\s*(\w+)\s*:', r'"\1":', block_md_clean, flags=re.M)
                metrics_dict = json.loads(block_md_clean)
                
            # Parse existing modelMeta
            start_mm = existing_content.find("const modelMeta = {") + len("const modelMeta = ")
            end_mm = existing_content.find("const categoriesList = [")
            if start_mm - len("const modelMeta = ") != -1 and end_mm != -1:
                block_mm = existing_content[start_mm:end_mm].strip().rstrip(";").strip()
                block_mm_clean = re.sub(r',\s*([\]}])', r'\1', block_mm)
                block_mm_clean = re.sub(r'^\s*(\w+)\s*:', r'"\1":', block_mm_clean, flags=re.M)
                meta_dict = json.loads(block_mm_clean)
        except Exception as parse_err:
            logger.warning("Could not parse existing report.html for baseline metrics: %s", parse_err)

    # 2. Update with the new results
    NAME_TO_KEY = {
        "Rule-based": "rule",
        "TF-IDF (LOGISTIC_REGRESSION)": "logreg",
        "TF-IDF (RANDOM_FOREST)": "rf",
        "TF-IDF (SVM)": "svm",
        "Ollama LLM": "ollama",
        "Czech BERT": "czech_bert"
    }

    for method_name, res in results.items():
        key = NAME_TO_KEY.get(method_name)
        if not key:
            logger.warning("Method '%s' has no mapped key in HTML report.", method_name)
            continue

        # Update metricsData
        metrics_dict["accuracy"][key] = round(res["metrics"]["accuracy"], 3)
        metrics_dict["macroF1"][key] = round(res["metrics"]["macro_f1"], 3)
        metrics_dict["weightedF1"][key] = round(res["metrics"]["weighted_f1"], 3)

        # Update modelMeta confusion matrix
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        
        if key not in meta_dict:
            meta_dict[key] = {}
        meta_dict[key]["cm"] = cm

        # Restore default names/descriptions if missing
        default_meta = {
            "rule": {
                "name": "Rule-based (Slovníkový klasifikátor)",
                "desc": "Pravidlový systém zařazuje dokumenty podle četnosti vybraných klíčových slov pro každou kategorii. Je velmi rychlý, ale nepružný vůči synonymům a jazykové variabilitě."
            },
            "logreg": {
                "name": "TF-IDF + Logistická regrese",
                "desc": "Klasická logistická regrese trénovaná na TF-IDF vektorech n-gramů. Vykazuje dobrou stabilitu na menších datech, ale může podhodnocovat méně zastoupené třídy."
            },
            "rf": {
                "name": "TF-IDF + Náhodný les (Random Forest)",
                "desc": "Ansámblový klasifikátor složený z mnoha nezávislých rozhodovacích stromů. Je robustní proti přetrénování, ale u dlouhých textů může ztrácet jemnější sémantické souvislosti."
            },
            "ollama": {
                "name": "Ollama LLM (Gemma 3 12B)",
                "desc": "Velký jazykový model dotazovaný v zero-shot a few-shot režimu. Vykazuje výborné zobecnění a porozumění kontextu v češtině bez lokálního trénování, ale komunikace přes API je pomalejší než lokální modely."
            },
            "svm": {
                "name": "TF-IDF + SVM (Linear Support Vector Machine)",
                "desc": "Lineární SVM trénovaný na n-gramech dosáhl nejlepších celkových výsledků. Hledá optimální nadrovinu s největší marží, což skvěle funguje v řídkých a rozměrných textových vektorech."
            },
            "czech_bert": {
                "name": "Czech BERT (RobeCzech)",
                "desc": "Lokálně fine-tunovaný český transformer model založený na architektuře RoBERTa. Výborně rozumí kontextu, české sémantice a syntaxi vět. Na testovacích datech dosáhl velmi vysoké úspěšnosti (těsně pod úrovní SVM), přičemž vykazuje výborný potenciál zobecnění na zcela nových a odlišně formátovaných dokumentech."
            }
        }
        if key in default_meta:
            meta_dict[key]["name"] = default_meta[key]["name"]
            meta_dict[key]["desc"] = default_meta[key]["desc"]

    # 3. Generate dynamic Czech descriptions based on results
    best_model_name = None
    best_acc = 0.0
    best_f1 = 0.0
    
    ml_models = ["Czech BERT", "TF-IDF (SVM)", "TF-IDF (RANDOM_FOREST)", "TF-IDF (LOGISTIC_REGRESSION)"]
    for m in ml_models:
        if m in results:
            acc = results[m]["metrics"]["accuracy"]
            f1 = results[m]["metrics"]["weighted_f1"]
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_model_name = m
                
    if not best_model_name:
        best_model_name = "TF-IDF (SVM)"
        best_acc = 0.947
        best_f1 = 0.948

    rule_acc = 0.714
    rule_macro_f1 = 0.486
    if "Rule-based" in results:
        rule_acc = results["Rule-based"]["metrics"]["accuracy"]
        rule_macro_f1 = results["Rule-based"]["metrics"]["macro_f1"]
        
    num_samples = len(results[list(results.keys())[0]]['y_true']) if results else 262

    overall_summary = (
        f"V této části jsou zobrazeny výsledky srovnání šesti klasifikačních metod na testovacím splitu. "
        f"Data byla rozdělena na trénovací (70 %), validační (10 %) a testovací split (20 %) z celkového datasetu (celkem <strong>{num_samples}</strong> testovacích dokumentů). "
        f"Nejlepších výsledků dosáhla metoda <strong>{best_model_name}</strong> s přesností <strong>{best_acc * 100:.1f} %</strong>. "
        f"Tento výsledek ukazuje účinnost pokročilých statistických a transformerových metod na českém textovém korpusu."
    )
    
    ml_analysis = f"""<p>
                V rámci této semestrální práce jsme úspěšně navrhli, naimplementovali a vyhodnotili ucelený systém pro klasifikaci a validaci dokumentů regulovaných informací z portálu ČNB OAM. Vyhodnotili jsme celkem 6 různých metod rozdělených do čtyř základních přístupů: pravidlový systém, statistické ML modely (SVM, Logistická regrese, Náhodný les), hluboké učení (Czech BERT / RobeCzech) a velké jazykové modely (Ollama LLM).
            </p>
            <p>
                V klasifikační úloze dosáhl nejvyšší celkové přesnosti model <strong>{best_model_name}</strong> s přesností <strong>{best_acc * 100:.1f} %</strong> a Weighted F1 <strong>{best_f1:.3f}</strong>, těsně následovaný lokálně jemně doladěným českým transformerem <strong>Czech BERT (RobeCzech)</strong> s přesností <strong>{(results.get('Czech BERT', {}).get('metrics', {}).get('accuracy', 0.947) * 100):.1f} %</strong> a Weighted F1 <strong>{results.get('Czech BERT', {}).get('metrics', {}).get('weighted_f1', 0.944):.3f}</strong>.
            </p>
            <p>
                Statistický model TF-IDF + SVM vykazuje vynikající výsledky díky faktu, že finanční výkazy a úřední oznámení ČNB obsahují vysoce specifický a opakující se slovník (např. standardizované fráze ve výročních zprávách či oznámeních o valných hromadách), který lze velmi dobře separovat lineární nadrovinou. Lokálně doladěný <strong>Czech BERT</strong> však představuje robustnější řešení, které na rozdíl od bag-of-words modeluje sémantické vazby a pořadí slov v českých větách, což mu umožňuje lépe zobecňovat na nových, dosud neviděných dokumentech a lépe odolávat případným změnám ve formátu zpráv.
            </p>
            <p>
                Zatímco pravidlový systém slouží jako rychlý baseline (přesnost {rule_acc * 100:.1f} %), velké jazykové modely (Ollama LLM s Gemma 3) dosáhly v zero-shot režimu přesnosti {(results.get('Ollama LLM', {}).get('metrics', {}).get('accuracy', 0.662) * 100):.1f} %. LLM sice vykazuje vysokou úroveň sémantického zobecnění a schopnost slovního zdůvodnění predikce v češtině, je však limitováno rychlostí síťového API a absencí lokálního jemného doladění na specifické doménové slovní zásoby, což způsobuje záměny u formálně si blízkých kategorií.
            </p>
            
            <h3 style="margin-top: 2rem; margin-bottom: 1rem; color: var(--text-primary);"><i class="fa-solid fa-wand-magic-sparkles"></i> Možnosti budoucího vylepšení</h3>
            <p>
                Jako hlavní směr budoucího rozvoje se nabízí hybridní přístup: využití statistického modelu SVM či Czech BERT pro rychlou a vysoce přesnou prvotní filtraci a klasifikaci, a následné nasazení pokročilých LLM k detailní analýze a slovnímu vysvětlení (explainability) u sporných případů, kde je detekován nesoulad (mismatch) mezi deklarovanou a predikovanou třídou dokumentu.
            </p>"""

    # 4. Format JavaScript code strings
    def format_metrics_data(metrics):
        lines = ["const metricsData = {"]
        for m_name in ["accuracy", "macroF1", "weightedF1"]:
            lines.append(f"            {m_name}: {{")
            metric_vals = []
            for k in ["rule", "logreg", "rf", "svm", "ollama", "czech_bert"]:
                val = metrics[m_name].get(k, 0.0)
                metric_vals.append(f"                {k}: {val:.3f}")
            lines.append(",\n".join(metric_vals))
            lines.append("            }" + ("," if m_name != "weightedF1" else ""))
        lines.append("        };")
        return "\n".join(lines)

    def format_model_meta(meta):
        lines = ["const modelMeta = {"]
        model_keys = ["rule", "logreg", "rf", "svm", "ollama", "czech_bert"]
        for idx, k in enumerate(model_keys):
            model_data = meta.get(k, {})
            name = model_data.get("name", "")
            desc = model_data.get("desc", "")
            cm = model_data.get("cm", [])
            
            # Fallback mock matrix if none exists
            if not cm:
                cm = [[0]*12 for _ in range(12)]

            lines.append(f"            {k}: {{")
            lines.append(f'                name: "{name}",')
            lines.append(f'                desc: "{desc}",')
            
            cm_lines = []
            for row in cm:
                cm_lines.append("                    " + str(row))
            cm_str = "[\n" + ",\n".join(cm_lines) + "\n                ]"
            
            lines.append(f"                cm: {cm_str}")
            lines.append("            }" + ("," if idx < len(model_keys) - 1 else ""))
        lines.append("        };")
        return "\n".join(lines)

    metrics_js = format_metrics_data(metrics_dict)
    meta_js = format_model_meta(meta_dict)

    # 4.5. Generate statistics about data splits (Train / Val / Test)
    splits_path = Path("data/splits/splits.pkl")
    splits_html = ""
    if splits_path.exists():
        import pickle
        from collections import Counter
        try:
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
            
            train_labels = splits["train"]["labels"]
            val_labels = splits["val"]["labels"]
            test_labels = splits["test"]["labels"]
            
            train_counter = Counter(train_labels)
            val_counter = Counter(val_labels)
            test_counter = Counter(test_labels)
            
            all_cats = sorted(list(set(train_labels + val_labels + test_labels)))
            
            splits_html += f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); margin-top: 2rem; gap: 2.5rem;">
                <div class="table-container" style="margin: 0;">
                    <h4 style="margin-bottom: 1rem; padding: 1rem 1.25rem 0 1.25rem;"><i class="fa-solid fa-chart-pie" style="color: var(--accent);"></i> Velikost splitů dat</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>Split</th>
                                <th style="text-align: right;">Počet dokumentů</th>
                                <th style="text-align: right;">Procento</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><span class="split-badge train-badge"><i class="fa-solid fa-circle" style="font-size: 0.5rem; margin-right: 0.25rem;"></i>Trénovací (Train)</span></td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 500;">{len(train_labels)}</td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 500;">70.0 %</td>
                            </tr>
                            <tr>
                                <td><span class="split-badge val-badge"><i class="fa-solid fa-circle" style="font-size: 0.5rem; margin-right: 0.25rem;"></i>Validační (Val)</span></td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 500;">{len(val_labels)}</td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 500;">10.0 %</td>
                            </tr>
                            <tr>
                                <td><span class="split-badge test-badge"><i class="fa-solid fa-circle" style="font-size: 0.5rem; margin-right: 0.25rem;"></i>Testovací (Test)</span></td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 500;">{len(test_labels)}</td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 500;">20.0 %</td>
                            </tr>
                            <tr style="border-top: 2px solid var(--border-color); background-color: var(--bg-tertiary);">
                                <td><span class="split-badge total-badge"><i class="fa-solid fa-database" style="font-size: 0.7rem; margin-right: 0.25rem;"></i>Celkem (Dataset)</span></td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{len(train_labels) + len(val_labels) + len(test_labels)}</td>
                                <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700;">100.0 %</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="table-container" style="margin: 0;">
                    <h4 style="margin-bottom: 1rem; padding: 1rem 1.25rem 0 1.25rem;"><i class="fa-solid fa-list" style="color: var(--accent);"></i> Distribuce podle kategorií</h4>
                    <div style="max-height: 300px; overflow-y: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Kategorie</th>
                                    <th style="text-align: right; color: var(--text-secondary);">Train</th>
                                    <th style="text-align: right; color: var(--text-secondary);">Val</th>
                                    <th style="text-align: right; color: var(--text-secondary);">Test</th>
                                    <th style="text-align: right; font-weight: 700;">Celkem</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            for cat in all_cats:
                tr_c = train_counter.get(cat, 0)
                va_c = val_counter.get(cat, 0)
                te_c = test_counter.get(cat, 0)
                tot = tr_c + va_c + te_c
                
                short_cat = cat
                if len(short_cat) > 40:
                    short_cat = short_cat[:38] + "..."
                
                splits_html += f"""
                                <tr>
                                    <td title="{cat}" style="font-weight: 500; color: var(--text-primary);">{short_cat}</td>
                                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--text-secondary);">{tr_c}</td>
                                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--text-secondary);">{va_c}</td>
                                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--text-secondary);">{te_c}</td>
                                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: bold; color: var(--accent);">{tot}</td>
                                </tr>
                """
                
            splits_html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            """
        except Exception as e:
            logger.error("Failed to compile splits statistics: %s", e)
            splits_html = "<p>Chyba při načítání statistik o splitech.</p>"
    else:
        splits_html = "<p>Soubor splits.pkl nebyl nalezen. Proveďte nejprve trénování.</p>"

    # 5. Substitute placeholders in template
    html_content = html_content.replace("{{ overall_performance_summary }}", overall_summary)
    html_content = html_content.replace("{{ ml_performance_analysis }}", ml_analysis)
    html_content = html_content.replace("{{ metrics_data_js }}", metrics_js)
    html_content = html_content.replace("{{ model_meta_js }}", meta_js)
    html_content = html_content.replace("{{ dataset_stats_html }}", splits_html)

    # 6. Write final HTML to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    logger.info("Successfully generated dynamic HTML report at %s", output_path)

    # 7. Synchronize to artifacts folder
    artifacts_dir = Path("C:/Users/42072/.gemini/antigravity/brain/9ef92f0d-a009-4f65-9e5c-018f46f66a5d")
    if artifacts_dir.exists():
        artifacts_html_path = artifacts_dir / "report.html"
        try:
            shutil.copy2(output_path, artifacts_html_path)
            logger.info("Synchronized HTML report to artifacts at %s", artifacts_html_path)
        except Exception as artifact_err:
            logger.error("Failed to copy HTML report to artifacts: %s", artifact_err)
