"""Comparison utilities to evaluate different classification methods side-by-side.

Provides performance tables, grouped metric charts, LaTeX exports for university
papers, and comprehensive report generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class MethodComparison:
    """Stores classification results across different models and generates comparisons."""

    def __init__(self, labels: list[str]) -> None:
        """Initialise comparison workspace.

        Args:
            labels: List of target category names.
        """
        self.labels = labels
        self.results: dict[str, dict] = {}
        logger.info("MethodComparison initialized for %d labels.", len(self.labels))

    def add_result(self, method_name: str, y_true: list[str], y_pred: list[str]) -> None:
        """Calculate and store metrics for a specific classification method.

        Args:
            method_name: Unique identifier for the method (e.g. 'Rule-based', 'TF-IDF SVM').
            y_true: True document category labels.
            y_pred: Predicted document category labels.
        """
        metrics = ClassificationMetrics.compute_metrics(y_true, y_pred, self.labels)
        report_df = ClassificationMetrics.classification_report_table(y_true, y_pred, self.labels)
        
        self.results[method_name] = {
            "metrics": metrics,
            "report_table": report_df,
            "y_true": y_true,
            "y_pred": y_pred,
        }
        logger.info("Added classification results for method: %s", method_name)

    def comparison_table(self) -> pd.DataFrame:
        """Compile a summary table of all added methods.

        Returns:
            DataFrame with rows as methods and columns as Accuracy, Macro F1, Weighted F1.
        """
        data = []
        for name, res in self.results.items():
            metrics = res["metrics"]
            data.append({
                "Method": name,
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
            })
        df = pd.DataFrame(data).set_index("Method")
        return df

    def plot_comparison_chart(self, output_path: Path) -> None:
        """Save a grouped bar chart comparing methods on main metrics.

        Args:
            output_path: Destination PNG path.
        """
        df = self.comparison_table()
        if df.empty:
            logger.warning("No results to plot.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Melt DataFrame for seaborn grouped bar chart
        df_melt = df.reset_index().melt(id_vars="Method", var_name="Metric", value_name="Score")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Metric", y="Score", hue="Method", palette="Set2", ax=ax)
        
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Method Performance Comparison")
        ax.legend(loc="lower right")

        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="center",
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=8,
                )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Comparison chart saved to %s", output_path)

    def plot_per_class_comparison(self, output_path: Path) -> None:
        """Save a horizontal grouped bar chart showing per-class F1 for each method.

        Args:
            output_path: Destination PNG path.
        """
        if not self.results:
            logger.warning("No results to plot.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for method_name, res in self.results.items():
            per_class = res["metrics"]["per_class"]
            for label, scores in per_class.items():
                data.append({
                    "Method": method_name,
                    "Category": label,
                    "F1 Score": scores["f1"],
                })

        df = pd.DataFrame(data)

        n_labels = len(self.labels)
        fig_height = max(6, n_labels * 0.75)
        
        fig, ax = plt.subplots(figsize=(12, fig_height))
        sns.barplot(
            data=df,
            y="Category",
            x="F1 Score",
            hue="Method",
            palette="Set2",
            orient="h",
            ax=ax,
        )

        ax.set_xlim(0.0, 1.05)
        ax.set_title("Per-Category F1 Score Comparison")
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Per-class comparison chart saved to %s", output_path)

    def export_results(self, output_dir: Path, formats: list[str]) -> None:
        """Export comparison table to LaTeX (for university report) and CSV.

        Args:
            output_dir: Folder where table outputs will be saved.
            formats: List of formats, e.g. ['csv', 'latex'].
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.comparison_table()

        if "csv" in formats:
            csv_path = output_dir / "method_comparison.csv"
            df.to_csv(csv_path)
            logger.info("Exported comparison table to CSV: %s", csv_path)

        if "latex" in formats:
            latex_path = output_dir / "method_comparison.tex"
            try:
                # Format numbers to 3 decimal places in LaTeX
                df.to_latex(
                    latex_path,
                    float_format="%.3f",
                    column_format="lrrr",
                    caption="Srovnání klasifikačních metod regulovaných informací ČNB OAM",
                    label="tab:method_comparison",
                )
            except Exception as latex_err:
                logger.warning("pandas to_latex failed: %s. Using custom fallback LaTeX exporter.", latex_err)
                # Simple fallback LaTeX generator
                lines = [
                    "\\begin{table}[h]",
                    "\\centering",
                    "\\caption{Srovnání klasifikačních metod regulovaných informací ČNB OAM}",
                    "\\label{tab:method_comparison}",
                    "\\begin{tabular}{lrrr}",
                    "\\hline",
                    "Method & Accuracy & Macro F1 & Weighted F1 \\\\",
                    "\\hline"
                ]
                for method_name, row in df.iterrows():
                    lines.append(f"{method_name} & {row['Accuracy']:.3f} & {row['Macro F1']:.3f} & {row['Weighted F1']:.3f} \\\\")
                lines.extend([
                    "\\hline",
                    "\\end{tabular}",
                    "\\end{table}"
                ])
                latex_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Exported comparison table to LaTeX: %s", latex_path)

    def generate_full_report(self, output_dir: Path) -> None:
        """Run all plots, confusion matrices for each method, and generate a markdown summary report.

        Args:
            output_dir: Root results folder.
        """
        output_dir = Path(output_dir)
        fig_dir = output_dir / "figures"
        report_dir = output_dir / "reports"
        
        fig_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating full comparison report inside %s", output_dir)

        # 1. Generate summary plots
        self.plot_comparison_chart(fig_dir / "method_comparison.png")
        self.plot_per_class_comparison(fig_dir / "per_class_f1_comparison.png")

        # 2. Export table formats
        self.export_results(report_dir, ["csv", "latex"])

        # 3. Generate confusion matrix for each individual method
        for name, res in self.results.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            ClassificationMetrics.confusion_matrix_plot(
                res["y_true"],
                res["y_pred"],
                self.labels,
                fig_dir / f"confusion_matrix_{safe_name}.png",
                title=f"Confusion Matrix - {name}",
            )
            ClassificationMetrics.plot_per_class_f1(
                res["y_true"],
                res["y_pred"],
                self.labels,
                fig_dir / f"per_class_f1_{safe_name}.png",
            )

        # Helper to output markdown tables without tabulate package dependency
        def _to_markdown(df_to_convert: pd.DataFrame) -> str:
            try:
                return df_to_convert.to_markdown()
            except Exception:
                cols = [df_to_convert.index.name or ""] + list(df_to_convert.columns)
                lines = [
                    "| " + " | ".join(str(c) for c in cols) + " |",
                    "| " + " | ".join("---" for _ in cols) + " |"
                ]
                for idx, row in df_to_convert.iterrows():
                    row_vals = []
                    for col_name in df_to_convert.columns:
                        val = row[col_name]
                        if isinstance(val, (float, np.floating)):
                            row_vals.append(f"{val:.4f}")
                        else:
                            row_vals.append(str(val))
                    lines.append(f"| {idx} | " + " | ".join(row_vals) + " |")
                return "\n".join(lines)

        # 4. Generate a comprehensive Markdown comparison report
        report_path = report_dir / "summary_report.md"
        df = self.comparison_table()
        
        md_lines = [
            "# CNB OAM Category Classifier — Method Evaluation Report\n\n",
            "This report evaluates multiple classification models for Czech National Bank OAM documents:\n\n",
            "1. **Rule-based**: Custom keyword/regex matching dictionaries in Czech.\n",
            "2. **TF-IDF + ML**: Traditional text classification using TF-IDF and SVM/RandomForest/Logistic Regression.\n\n",
            "## Summary Metrics Comparison\n\n",
            _to_markdown(df),
            "\n\n## Per-Method Detailed Classification Reports\n\n",
        ]

        for name, res in self.results.items():
            md_lines.append(f"### Method: {name}\n\n")
            md_lines.append(_to_markdown(res["report_table"]))
            md_lines.append("\n\n")

        report_path.write_text("".join(md_lines), encoding="utf-8")
        logger.info("Markdown summary report saved to %s", report_path)

        # 5. Automatically generate/update interactive HTML report
        try:
            from src.evaluation.html_generator import generate_html_report
            generate_html_report(self.results, self.labels, Path("report/report.html"))
        except Exception as e:
            logger.error("Failed to generate HTML report: %s", e, exc_info=True)

