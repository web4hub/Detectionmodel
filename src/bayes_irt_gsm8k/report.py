"""Post-processing reports for Bayesian IRT sensitivity suites."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_suite_outputs(output_dir: str | Path, *, top_k: int = 100) -> dict[str, Path]:
    """Create comparison and stability summaries for a completed suite."""

    output_path = Path(output_dir)
    summary_path = output_path / "summary_ranked.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")

    summary = pd.read_csv(summary_path)
    written: dict[str, Path] = {}
    best_by_model = _best_by_model(summary)
    best_path = output_path / "best_by_model.csv"
    best_by_model.to_csv(best_path, index=False)
    written["best_by_model"] = best_path

    mixture_top = _load_mixture_top_tables(output_path, top_k=top_k)
    if not mixture_top.empty:
        item_stability = _stability_table(mixture_top, "item")
        model_stability = _stability_table(mixture_top, "model")
        family_stability = _stability_table(mixture_top, "family")
        run_probability_summary = _run_probability_summary(mixture_top)

        item_path = output_path / f"mixture_item_stability_top{top_k}.csv"
        model_path = output_path / f"mixture_model_stability_top{top_k}.csv"
        family_path = output_path / f"mixture_family_stability_top{top_k}.csv"
        prob_path = output_path / f"mixture_probability_summary_top{top_k}.csv"
        item_stability.to_csv(item_path, index=False)
        model_stability.to_csv(model_path, index=False)
        family_stability.to_csv(family_path, index=False)
        run_probability_summary.to_csv(prob_path, index=False)
        written.update(
            {
                "mixture_item_stability": item_path,
                "mixture_model_stability": model_path,
                "mixture_family_stability": family_path,
                "mixture_probability_summary": prob_path,
            }
        )

    report_path = output_path / "model_comparison_report.md"
    report_path.write_text(_markdown_report(summary, best_by_model, mixture_top, top_k=top_k))
    written["markdown_report"] = report_path
    return written


def _best_by_model(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["status"] == "ok"].copy()
    if ok.empty:
        return ok
    idx = ok.groupby("model_kind")["holdout_mean_log_likelihood"].idxmax()
    return ok.loc[idx].sort_values("holdout_mean_log_likelihood", ascending=False)


def _load_mixture_top_tables(output_dir: Path, *, top_k: int) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.glob("dmixture*/top_preknowledge_probabilities_full.csv")):
        frame = pd.read_csv(path).head(top_k).copy()
        if frame.empty:
            continue
        frame["run_name"] = path.parent.name
        frame["rank"] = range(1, len(frame) + 1)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _stability_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(column)
        .agg(
            n_runs=("run_name", "nunique"),
            n_rows=("run_name", "size"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
            mean_preknowledge_probability=("preknowledge_probability", "mean"),
            min_preknowledge_probability=("preknowledge_probability", "min"),
        )
        .reset_index()
        .sort_values(["n_runs", "n_rows", "mean_rank"], ascending=[False, False, True])
    )
    return grouped


def _run_probability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("run_name")
        .agg(
            n_rows=("run_name", "size"),
            min_preknowledge_probability=("preknowledge_probability", "min"),
            mean_preknowledge_probability=("preknowledge_probability", "mean"),
            median_preknowledge_probability=("preknowledge_probability", "median"),
            n_unique_items=("item", "nunique"),
            n_unique_models=("model", "nunique"),
            n_unique_families=("family", "nunique"),
        )
        .reset_index()
        .sort_values("run_name")
    )


def _markdown_report(
    summary: pd.DataFrame,
    best_by_model: pd.DataFrame,
    mixture_top: pd.DataFrame,
    *,
    top_k: int,
) -> str:
    ok = summary[summary["status"] == "ok"].copy()
    status_counts = summary["status"].value_counts().to_dict()
    lines = [
        "# Bayesian IRT Model Comparison",
        "",
        f"Runs completed: {status_counts}",
        "",
        "## Top models by held-out mean log likelihood",
        "",
    ]
    cols = [
        "run_name",
        "model_kind",
        "prior_name",
        "holdout_mean_log_likelihood",
        "holdout_brier",
        "holdout_accuracy_0_5",
    ]
    lines.append(_markdown_table(ok.sort_values("holdout_mean_log_likelihood", ascending=False)[cols].head(10)))
    lines.extend(["", "## Best prior variant within each model family", ""])
    lines.append(_markdown_table(best_by_model[cols]) if not best_by_model.empty else "No successful runs.")

    if not mixture_top.empty:
        lines.extend(["", f"## Stable top preknowledge items across mixture runs, top {top_k}", ""])
        lines.append(_markdown_table(_stability_table(mixture_top, "item").head(10)))
        lines.extend(["", f"## Stable top preknowledge families across mixture runs, top {top_k}", ""])
        lines.append(_markdown_table(_stability_table(mixture_top, "family").head(10)))

    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""

    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a completed Bayesian IRT suite.")
    parser.add_argument("output_dir", type=Path, help="Suite output directory.")
    parser.add_argument("--top-k", type=int, default=100, help="Top rows per mixture run for stability summaries.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    written = summarize_suite_outputs(args.output_dir, top_k=args.top_k)
    for name, path in written.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
