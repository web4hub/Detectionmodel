"""Generate shareable notebook, manuscript draft, tables, and figures.

This script is intentionally deterministic and reads only completed analysis
outputs. It does not refit any Bayesian models.
"""

from __future__ import annotations

from pathlib import Path
import json
import textwrap

import nbformat as nbf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "bayes_irt_all_items_suite_outputs"
OUT = ROOT / "shareable_joc_mixture_irt"
FIG = OUT / "figures"
TAB = OUT / "tables"
NOTEBOOK = OUT / "bayesian_mixture_irt_gsm8k_results.ipynb"
MANUSCRIPT = OUT / "joc_special_issue_manuscript_draft.md"
BIB = OUT / "references.bib"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)

    summary = pd.read_csv(SUITE / "summary_ranked.csv")
    best = pd.read_csv(SUITE / "best_by_model.csv")
    item_stability = pd.read_csv(SUITE / "mixture_item_stability_top100.csv")
    family_stability = pd.read_csv(SUITE / "mixture_family_stability_top100.csv")
    model_stability = pd.read_csv(SUITE / "mixture_model_stability_top100.csv")
    prob_summary = pd.read_csv(SUITE / "mixture_probability_summary_top100.csv")
    suite_config = json.loads((SUITE / "suite_config.json").read_text())

    table_paths = write_tables(summary, best, item_stability, family_stability, model_stability, suite_config)
    figure_paths = write_figures(summary, best, item_stability, family_stability, prob_summary)
    write_bib()
    write_manuscript(table_paths, figure_paths, suite_config)
    write_notebook(table_paths, figure_paths, suite_config)

    print(f"notebook: {NOTEBOOK}")
    print(f"manuscript: {MANUSCRIPT}")
    print(f"references: {BIB}")


def write_tables(summary, best, item_stability, family_stability, model_stability, suite_config):
    paths = {}

    data_rows = [
        ("Benchmark", "All non-saturated GSM8K items from metabench/Open LLM Leaderboard response data"),
        ("Models/checkpoints", f"{suite_config['data']['n_models']:,}"),
        ("Analyzed items", f"{suite_config['data']['n_items']:,}"),
        ("Observed model-item cells", f"{suite_config['data']['n_obs']:,}"),
        ("Inferred model families", f"{suite_config['data']['n_families']:,}"),
        ("Training observations", f"{suite_config['data']['train_obs']:,}"),
        ("Holdout observations", f"{suite_config['data']['holdout_obs']:,}"),
        ("Model/prior runs", "45"),
        ("Heldout fraction", f"{suite_config['holdout_fraction']:.2f}"),
    ]
    table1 = pd.DataFrame(data_rows, columns=["Quantity", "Value"])
    paths["table1"] = TAB / "table1_dataset_and_suite.csv"
    table1.to_csv(paths["table1"], index=False)

    table2 = best[
        [
            "model_kind",
            "prior_name",
            "holdout_mean_log_likelihood",
            "holdout_brier",
            "holdout_accuracy_0_5",
        ]
    ].copy()
    table2.columns = [
        "Model family",
        "Best prior variant",
        "Holdout mean log likelihood",
        "Holdout Brier score",
        "Holdout accuracy",
    ]
    paths["table2"] = TAB / "table2_best_model_by_family.csv"
    table2.to_csv(paths["table2"], index=False)

    table3 = summary[
        [
            "run_name",
            "model_kind",
            "prior_name",
            "holdout_mean_log_likelihood",
            "holdout_brier",
            "holdout_accuracy_0_5",
        ]
    ].head(10).copy()
    table3.columns = [
        "Run",
        "Model family",
        "Prior variant",
        "Holdout mean log likelihood",
        "Holdout Brier score",
        "Holdout accuracy",
    ]
    paths["table3"] = TAB / "table3_top_predictive_runs.csv"
    table3.to_csv(paths["table3"], index=False)

    table4 = item_stability.head(10).copy()
    table4.columns = [
        "Item",
        "Mixture runs",
        "Top-100 appearances",
        "Mean rank",
        "Best rank",
        "Mean posterior preknowledge probability",
        "Minimum posterior preknowledge probability",
    ]
    paths["table4"] = TAB / "table4_stable_items.csv"
    table4.to_csv(paths["table4"], index=False)

    table5 = family_stability.head(10).copy()
    table5.columns = [
        "Model family",
        "Mixture runs",
        "Top-100 appearances",
        "Mean rank",
        "Best rank",
        "Mean posterior preknowledge probability",
        "Minimum posterior preknowledge probability",
    ]
    paths["table5"] = TAB / "table5_stable_model_families.csv"
    table5.to_csv(paths["table5"], index=False)

    simulation = pd.DataFrame(
        [
            ("N models", "500, 2,000, 6,000", "Tests scaling from small audit to metabench scale."),
            ("I items", "50, 200, 1,200", "Separates small audit and full-benchmark regimes."),
            ("Family clustering", "none, moderate, strong", "Controls dependence among related checkpoints."),
            ("Contamination rate", "0, 0.5%, 2%, 5%, 10%", "Includes null and sparse-to-moderate leakage."),
            ("Contamination topology", "cell, item, model, family, crossed", "Matches competing substantive mechanisms."),
            ("Contamination strength tau", "0.85, 0.95, 0.99", "Allows imperfect memorization or formatting failures."),
            ("Latent dimensionality", "1D, 2D nuisance skill", "Tests false positives under unmodeled subskills."),
            ("Answer noise", "0%, 2%, 5%", "Mimics extraction and generation errors."),
            ("Replications", "100 per core cell", "Targets Monte Carlo SE below .02 for classification metrics."),
            ("Evaluation", "AUC, PR-AUC, precision@k, recall@k, calibration, RMSE, coverage, false positives", "Connects classification, prediction, and parameter recovery."),
        ],
        columns=["Factor", "Levels", "Rationale"],
    )
    paths["table6"] = TAB / "table6_simulation_design.csv"
    simulation.to_csv(paths["table6"], index=False)

    item_params = build_item_parameter_table(item_stability, best)
    paths["table7"] = TAB / "table7_item_parameter_diagnostics.csv"
    item_params.to_csv(paths["table7"], index=False)

    top_item_params = item_params.sort_values(
        ["Mixture prior variants with top-100 classification", "Top-100 appearances", "Mixture item exposure"],
        ascending=[False, False, False],
    ).head(15)
    paths["table8"] = TAB / "table8_top_item_parameter_diagnostics.csv"
    top_item_params.to_csv(paths["table8"], index=False)

    return paths


def parameter_site(run_name: str, site: str, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(SUITE / run_name / "parameter_summary.csv")
    out = frame[frame["site"] == site][["label", "median"]].copy()
    out["item"] = out["label"].astype(int)
    out = out[["item", "median"]].rename(columns={"median": value_name})
    return out


def best_run_for(best: pd.DataFrame, model_kind: str) -> str:
    """Return the best-ranked run name for a model kind."""

    row = best[best["model_kind"] == model_kind]
    if row.empty:
        raise ValueError(f"best_by_model is missing {model_kind!r}")
    return str(row.iloc[0]["run_name"])


def build_item_parameter_table(item_stability: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Merge item parameters most relevant to contamination interpretation."""

    mixture_run = best_run_for(best, "dmixture_family_2pl")
    sparse_run = best_run_for(best, "sparse_3pl")
    item_params = parameter_site(mixture_run, "difficulty", "Mixture difficulty")
    item_params = item_params.merge(
        parameter_site(mixture_run, "discrimination", "Mixture discrimination"),
        on="item",
        how="left",
    )
    item_params = item_params.merge(
        parameter_site(mixture_run, "item_exposure", "Mixture item exposure"),
        on="item",
        how="left",
    )
    item_params = item_params.merge(
        parameter_site(sparse_run, "lower_asymptote", "Sparse 3PL lower asymptote"),
        on="item",
        how="left",
    )
    recurrence_path = ROOT / "rigorous_residual_outputs" / "baseline_item_recurrence.csv"
    if recurrence_path.exists():
        recurrence = pd.read_csv(recurrence_path)
        recurrence = recurrence.rename(
            columns={
                "item": "item",
                "n_models": "Legacy residual-screen flagged models",
                "median_predicted_p": "Legacy residual-screen median predicted probability",
                "median_z": "Legacy residual-screen median z",
                "difficulty": "Legacy 2PL residual-screen difficulty",
            }
        )
        item_params = item_params.merge(recurrence, on="item", how="left")
    stability = item_stability.rename(
        columns={
            "item": "item",
            "n_runs": "Mixture prior variants with top-100 classification",
            "n_rows": "Top-100 appearances",
            "mean_rank": "Mean top-100 rank",
            "best_rank": "Best top-100 rank",
            "mean_preknowledge_probability": "Mean posterior preknowledge probability",
            "min_preknowledge_probability": "Minimum posterior preknowledge probability",
        }
    )
    item_params = item_params.merge(stability, on="item", how="left")
    selected_items = pd.read_csv(SUITE / "selected_item_summary.csv")
    pass_rate_frame = selected_items[["item", "pass_rate"]].rename(columns={"pass_rate": "Observed pass rate"})
    item_params = item_params.merge(pass_rate_frame, on="item", how="left")
    item_params = item_params.fillna(
        {
            "Mixture prior variants with top-100 classification": 0,
            "Top-100 appearances": 0,
            "Mean top-100 rank": np.nan,
            "Best top-100 rank": np.nan,
            "Mean posterior preknowledge probability": np.nan,
            "Minimum posterior preknowledge probability": np.nan,
            "Legacy residual-screen flagged models": 0,
        }
    )
    return item_params.sort_values("Mixture item exposure", ascending=False)


def write_figures(summary, best, item_stability, family_stability, prob_summary):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    paths = {}

    ordered = best.sort_values("holdout_mean_log_likelihood", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(ordered["model_kind"], ordered["holdout_mean_log_likelihood"], color="#4C78A8")
    ax.set_xlabel("Holdout mean log likelihood (higher is better)")
    ax.set_ylabel("Model family")
    ax.set_title("Figure 1. Heldout predictive fit by best prior variant")
    ax.axvline(ordered["holdout_mean_log_likelihood"].max(), color="black", lw=0.8, alpha=0.5)
    fig.tight_layout()
    paths["fig1"] = FIG / "figure1_holdout_loglik_by_model.png"
    fig.savefig(paths["fig1"], dpi=300)
    plt.close(fig)

    ordered_brier = best.sort_values("holdout_brier", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(ordered_brier["model_kind"], ordered_brier["holdout_brier"], color="#72B7B2")
    ax.set_xlabel("Holdout Brier score (lower is better)")
    ax.set_ylabel("Model family")
    ax.set_title("Figure 2. Heldout calibration error by best prior variant")
    fig.tight_layout()
    paths["fig2"] = FIG / "figure2_brier_by_model.png"
    fig.savefig(paths["fig2"], dpi=300)
    plt.close(fig)

    mix = summary[summary["model_kind"].isin(["dmixture_2pl", "dmixture_family_2pl"])].copy()
    pivot = mix.pivot_table(
        index="model_kind",
        columns="prior_name",
        values="holdout_mean_log_likelihood",
        aggfunc="mean",
    )
    pivot = pivot.rename(
        index={
            "dmixture_2pl": "Mixture 2PL",
            "dmixture_family_2pl": "Family mixture 2PL",
        }
    )
    ordered_cols = [
        c
        for c in [
            "baseline",
            "tight_difficulty",
            "wide_difficulty",
            "tight_discrimination",
            "wide_discrimination",
            "rare_contamination",
            "moderate_contamination",
            "tau_95",
            "tau_999",
            "tight_exposure",
            "wide_exposure",
        ]
        if c in pivot.columns
    ]
    pivot = pivot[ordered_cols]
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".4f", cbar_kws={"label": "Mean log likelihood"}, ax=ax)
    ax.set_xlabel("Prior variant")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("Figure 3. Mixture-model prior sensitivity")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.subplots_adjust(left=0.15)
    paths["fig3"] = FIG / "figure3_mixture_prior_sensitivity.png"
    fig.savefig(paths["fig3"], dpi=300)
    plt.close(fig)

    top_items = item_stability.head(10).sort_values("n_rows", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top_items["item"].astype(str), top_items["n_rows"], color="#F58518")
    ax.set_xlabel("Top-100 appearances across mixture prior variants")
    ax.set_ylabel("GSM8K item id")
    ax.set_title("Figure 4. Stability of items across mixture prior sensitivity runs")
    fig.tight_layout()
    paths["fig4"] = FIG / "figure4_item_stability.png"
    fig.savefig(paths["fig4"], dpi=300)
    plt.close(fig)

    top_families = family_stability.head(10).sort_values("n_rows", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(top_families["family"], top_families["n_rows"], color="#54A24B")
    ax.set_xlabel("Top-100 appearances across mixture prior variants")
    ax.set_ylabel("Inferred model family")
    ax.set_title("Figure 5. Stable model families among top preknowledge classifications")
    fig.tight_layout()
    paths["fig5"] = FIG / "figure5_family_stability.png"
    fig.savefig(paths["fig5"], dpi=300)
    plt.close(fig)

    best_run = best.iloc[0]["run_name"]
    cal_path = SUITE / best_run / "holdout_calibration.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--", label="Perfect calibration")
        ax.plot(cal["mean_predicted"], cal["mean_observed"], marker="o", color="#B279A2", label=best_run)
        for _, row in cal.iterrows():
            ax.text(row["mean_predicted"], row["mean_observed"], str(int(row["n"])), fontsize=7, alpha=0.75)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed proportion correct")
        ax.set_title("Figure 6. Holdout calibration for the best model")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        paths["fig6"] = FIG / "figure6_best_model_calibration.png"
        fig.savefig(paths["fig6"], dpi=300)
        plt.close(fig)

    item_params = build_item_parameter_table(item_stability, best)
    top_label_items = set(item_stability.head(6)["item"].astype(int))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    scatter = ax.scatter(
        item_params["Mixture difficulty"],
        item_params["Mixture discrimination"],
        c=item_params["Mixture item exposure"],
        s=45 + 0.35 * item_params["Top-100 appearances"].fillna(0),
        cmap="viridis",
        alpha=0.86,
        edgecolor="white",
        linewidth=0.5,
    )
    label_rows = item_params[item_params["item"].isin(top_label_items)].sort_values("item")
    label_offsets = [(5, 5), (5, -10), (-22, 5), (-22, -10), (8, 12), (-28, 12)]
    for (_, row), offset in zip(label_rows.iterrows(), label_offsets):
        ax.annotate(
            str(int(row["item"])),
            (row["Mixture difficulty"], row["Mixture discrimination"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.7},
        )
    ax.set_xlabel("Posterior median item difficulty")
    ax.set_ylabel("Posterior median item discrimination")
    ax.set_title("Figure 7. All-item IRT parameters and mixture exposure")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mixture item exposure")
    fig.tight_layout()
    paths["fig7"] = FIG / "figure7_item_difficulty_discrimination_exposure.png"
    fig.savefig(paths["fig7"], dpi=300)
    plt.close(fig)

    top_exposure = item_params.sort_values("Mixture item exposure", ascending=True).tail(15).copy()
    top_exposure["Relative exposure odds"] = np.exp(top_exposure["Mixture item exposure"])
    top_exposure = top_exposure.sort_values("Relative exposure odds", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    ax.barh(top_exposure["item"].astype(str), top_exposure["Relative exposure odds"], color="#E45756")
    ax.set_xlabel("Relative item exposure odds, exp(effect)")
    ax.set_ylabel("GSM8K item id")
    ax.set_title("Figure 8. Highest relative item-level contamination exposure")
    fig.tight_layout()
    paths["fig8"] = FIG / "figure8_item_exposure_effects.png"
    fig.savefig(paths["fig8"], dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(
        item_params["Mixture item exposure"],
        item_params["Sparse 3PL lower asymptote"],
        s=55 + 0.3 * item_params["Top-100 appearances"].fillna(0),
        color="#B279A2",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    for (_, row), offset in zip(label_rows.iterrows(), label_offsets):
        ax.annotate(
            str(int(row["item"])),
            (row["Mixture item exposure"], row["Sparse 3PL lower asymptote"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.7},
        )
    ax.set_xlabel("Mixture item exposure effect")
    ax.set_ylabel("Sparse 3PL lower asymptote")
    ax.set_title("Figure 9. Agreement between contamination-related item parameters")
    fig.tight_layout()
    paths["fig9"] = FIG / "figure9_exposure_vs_lower_asymptote.png"
    fig.savefig(paths["fig9"], dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        item_params["Observed pass rate"],
        item_params["Mixture item exposure"],
        s=55 + 0.3 * item_params["Top-100 appearances"].fillna(0),
        color="#F58518",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    for (_, row), offset in zip(label_rows.iterrows(), label_offsets):
        ax.annotate(
            str(int(row["item"])),
            (row["Observed pass rate"], row["Mixture item exposure"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.7},
        )
    ax.set_xlabel("Observed pass rate")
    ax.set_ylabel("Mixture item exposure effect")
    ax.set_title("Figure 10. Exposure effects are not simple item easiness")
    fig.tight_layout()
    paths["fig10"] = FIG / "figure10_pass_rate_vs_exposure.png"
    fig.savefig(paths["fig10"], dpi=300)
    plt.close(fig)

    return paths


def apa_table(frame: pd.DataFrame, digits: int = 3) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.{digits}f}")
        else:
            display[col] = display[col].astype(str)
    headers = [str(col) for col in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def compact_item_parameter_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a manuscript-width item diagnostics table."""

    columns = [
        "item",
        "Mixture item exposure",
        "Mixture difficulty",
        "Mixture discrimination",
        "Sparse 3PL lower asymptote",
        "Observed pass rate",
        "Mixture prior variants with top-100 classification",
        "Top-100 appearances",
        "Legacy residual-screen flagged models",
    ]
    display = frame[columns].copy()
    display.columns = [
        "Item",
        "Exposure",
        "Difficulty",
        "Discrimination",
        "3PL floor",
        "Pass rate",
        "Mixture variants",
        "Top-100 cells",
        "Legacy flags",
    ]
    return display


def write_bib() -> None:
    BIB.write_text(
        textwrap.dedent(
            r"""
            @book{rasch1960,
              author = {Rasch, Georg},
              title = {Probabilistic Models for Some Intelligence and Attainment Tests},
              year = {1960},
              publisher = {Danish Institute for Educational Research}
            }

            @incollection{birnbaum1968,
              author = {Birnbaum, Allan},
              title = {Some Latent Trait Models and Their Use in Inferring an Examinee's Ability},
              booktitle = {Statistical Theories of Mental Test Scores},
              editor = {Lord, Frederic M. and Novick, Melvin R.},
              publisher = {Addison-Wesley},
              pages = {397--479},
              year = {1968}
            }

            @book{lord1980,
              author = {Lord, Frederic M.},
              title = {Applications of Item Response Theory to Practical Testing Problems},
              year = {1980},
              publisher = {Lawrence Erlbaum Associates}
            }

            @book{hambleton1991irt,
              author = {Hambleton, Ronald K. and Swaminathan, Hariharan and Rogers, H. Jane},
              title = {Fundamentals of Item Response Theory},
              year = {1991},
              publisher = {Sage}
            }

            @article{bockaitkin1981,
              author = {Bock, R. Darrell and Aitkin, Murray},
              title = {Marginal Maximum Likelihood Estimation of Item Parameters: Application of an EM Algorithm},
              journal = {Psychometrika},
              volume = {46},
              number = {4},
              pages = {443--459},
              year = {1981},
              doi = {10.1007/BF02293801}
            }

            @article{yen1984q3,
              author = {Yen, Wendy M.},
              title = {Effects of Local Item Dependence on the Fit and Equating Performance of the Three-Parameter Logistic Model},
              journal = {Applied Psychological Measurement},
              year = {1984},
              volume = {8},
              number = {2},
              pages = {125--145},
              doi = {10.1177/014662168400800201}
            }

            @article{orlandothissen2000,
              author = {Orlando, Maria and Thissen, David},
              title = {Likelihood-Based Item-Fit Indices for Dichotomous Item Response Theory Models},
              journal = {Applied Psychological Measurement},
              volume = {24},
              number = {1},
              pages = {50--64},
              year = {2000},
              doi = {10.1177/01466216000241003}
            }

            @article{maydeu2013goodness,
              title = {Goodness-of-fit assessment of item response theory models},
              author = {Maydeu-Olivares, Alberto},
              journal = {Measurement},
              volume = {11},
              number = {3},
              pages = {71--101},
              year = {2013}
            }

            @article{frick2016ipd,
              author = {Frick, Hannah and Strobl, Carolin and Zeileis, Achim},
              title = {Investigating the Impact of Item Parameter Drift for Item Response Theory Models with Mixture Distributions},
              journal = {Frontiers in Psychology},
              year = {2016},
              volume = {7},
              pages = {255},
              doi = {10.3389/fpsyg.2016.00255}
            }

            @book{mclachlan2000mixtures,
              author = {McLachlan, Geoffrey J. and Peel, David},
              title = {Finite Mixture Models},
              year = {2000},
              publisher = {Wiley}
            }

            @article{fraley2002modelbased,
              author = {Fraley, Chris and Raftery, Adrian E.},
              title = {Model-Based Clustering, Discriminant Analysis, and Density Estimation},
              journal = {Journal of the American Statistical Association},
              volume = {97},
              number = {458},
              pages = {611--631},
              year = {2002},
              doi = {10.1198/016214502760047131}
            }

            @book{levy2016bayesian,
              title = {Bayesian Psychometric Modeling},
              author = {Levy, Roy and Mislevy, Robert J.},
              year = {2016},
              publisher = {CRC Press}
            }

            @article{natesan2016prior,
              title = {Bayesian Prior Choice in IRT Estimation Using MCMC and Variational Bayes},
              author = {Natesan, Prathiba and Nandakumar, Ratna and Minka, Tom and Rubright, Jonathan D.},
              year = {2016},
              journal = {Frontiers in Psychology},
              volume = {7},
              pages = {1422},
              doi = {10.3389/fpsyg.2016.01422}
            }

            @article{burkner2021brmsirt,
              title = {Bayesian Item Response Modeling in R with brms and Stan},
              author = {Buerkner, Paul-Christian},
              year = {2021},
              journal = {Journal of Statistical Software},
              volume = {100},
              number = {5},
              pages = {1--54},
              doi = {10.18637/jss.v100.i05}
            }

            @article{bingham2019pyro,
              author = {Bingham, Eli and Chen, Jonathan P. and Jankowiak, Martin and Obermeyer, Fritz and Pradhan, Neeraj and Karaletsos, Theofanis and Singh, Rohit and Szerlip, Paul and Horsfall, Paul and Goodman, Noah D.},
              title = {Pyro: Deep Universal Probabilistic Programming},
              journal = {Journal of Machine Learning Research},
              volume = {20},
              number = {28},
              pages = {1--6},
              year = {2019}
            }

            @article{brier1950,
              author = {Brier, Glenn W.},
              title = {Verification of Forecasts Expressed in Terms of Probability},
              journal = {Monthly Weather Review},
              volume = {78},
              number = {1},
              pages = {1--3},
              year = {1950},
              doi = {10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2}
            }

            @article{messick1995,
              author = {Messick, Samuel},
              title = {Validity of Psychological Assessment: Validation of Inferences from Persons' Responses and Performances as Scientific Inquiry into Score Meaning},
              journal = {American Psychologist},
              year = {1995},
              volume = {50},
              number = {9},
              pages = {741--749},
              doi = {10.1037/0003-066X.50.9.741}
            }

            @book{standards2014,
              author = {{American Educational Research Association} and {American Psychological Association} and {National Council on Measurement in Education}},
              title = {Standards for Educational and Psychological Testing},
              year = {2014},
              publisher = {American Educational Research Association}
            }

            @inproceedings{lalor2016building,
              title = {Building an Evaluation Scale using Item Response Theory},
              author = {Lalor, John P. and Wu, Hao and Yu, Hong},
              booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
              pages = {648--657},
              year = {2016}
            }

            @article{martinez2019irt,
              author = {Martinez-Plumed, Fernando and Prudencio, Ricardo B. C. and Martinez-Uso, Adolfo and Hernandez-Orallo, Jose},
              title = {Item Response Theory in AI: Analysing Machine Learning Classifiers at the Instance Level},
              journal = {Artificial Intelligence},
              year = {2019},
              volume = {271},
              pages = {18--42},
              doi = {10.1016/j.artint.2018.09.004}
            }

            @inproceedings{rodriguez2021informative,
              author = {Rodriguez, Pedro and Barrow, Joe and Hoyle, Alexander Miserlis and Lalor, John P. and Jia, Robin and Boyd-Graber, Jordan},
              title = {Evaluation Examples are not Equally Informative: How Should That Change NLP Leaderboards?},
              booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
              year = {2021},
              pages = {4486--4503},
              doi = {10.18653/v1/2021.acl-long.346}
            }

            @inproceedings{kiela2021dynabench,
              author = {Kiela, Douwe and Bartolo, Max and Nie, Yixin and Kaushik, Divyansh and Geiger, Atticus and Wu, Zhengxuan and Vidgen, Bertie and Prasad, Grusha and Singh, Amanpreet and Ringshia, Pratik and Ma, Zhiyi and Thrush, Tristan and Riedel, Sebastian and Waseem, Zeerak and Stenetorp, Pontus and Jia, Robin and Bansal, Mohit and Potts, Christopher and Williams, Adina},
              title = {Dynabench: Rethinking Benchmarking in NLP},
              booktitle = {Proceedings of NAACL-HLT},
              year = {2021},
              pages = {4110--4124},
              doi = {10.18653/v1/2021.naacl-main.324}
            }

            @article{liang2023helm,
              author = {Liang, Percy and Bommasani, Rishi and Lee, Tony and Tsipras, Dimitris and Soylu, Dilara and Yasunaga, Michihiro and Zhang, Yian and Narayanan, Deepak and Wu, Yuhuai and Kumar, Ananya and others},
              title = {Holistic Evaluation of Language Models},
              journal = {Transactions on Machine Learning Research},
              year = {2023}
            }

            @article{srivastava2023bigbench,
              author = {Srivastava, Aarohi and Rastogi, Abhinav and Rao, Abhishek and others},
              title = {Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models},
              journal = {Transactions on Machine Learning Research},
              year = {2023}
            }

            @article{hendrycks2021mmlu,
              author = {Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
              title = {Measuring Massive Multitask Language Understanding},
              journal = {International Conference on Learning Representations},
              year = {2021}
            }

            @inproceedings{cobbe2021gsm8k,
              author = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
              title = {Training Verifiers to Solve Math Word Problems},
              booktitle = {arXiv preprint arXiv:2110.14168},
              year = {2021}
            }

            @article{kipnis2024metabench,
              author = {Kipnis, Alex and Voudouris, Konstantinos and Schulze Buschoff, Luca M. and Schulz, Eric},
              title = {metabench: A Sparse Benchmark to Measure General Ability in Large Language Models},
              journal = {arXiv preprint arXiv:2407.12844},
              year = {2024}
            }

            @misc{li2025psychometrics,
              author = {Li, Yuan and Huang, Yue and Wang, Hongyi and Cheng, Ying and Zhang, Xiangliang and Zou, James and Sun, Lichao},
              title = {Evaluating Large Language Models with Psychometrics},
              year = {2025},
              eprint = {2406.17675},
              archivePrefix = {arXiv}
            }

            @article{zhou2025lost,
              author = {Zhou, Hongli and Huang, Hui and Zhao, Ziqing and Han, Lvyuan and Wang, Huicheng and Chen, Kehai and Yang, Muyun and Bao, Wei and Dong, Jian and Xu, Bing and Zhu, Conghui and Cao, Hailong and Zhao, Tiejun},
              title = {Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory},
              journal = {arXiv preprint arXiv:2505.15055},
              year = {2025}
            }

            @inproceedings{nadeem2025mathirt,
              title = {Rethinking Math Benchmarks for LLMs using IRT},
              author = {Nadeem, Nimra and others},
              booktitle = {Proceedings of Machine Learning Research},
              volume = {273},
              pages = {66--82},
              year = {2025}
            }

            @article{deng2023contamination,
              author = {Deng, Chunyuan and Zhao, Yilun and Tang, Xiangru and Gerstein, Mark and Cohan, Arman},
              title = {Investigating Data Contamination in Modern Benchmarks for Large Language Models},
              journal = {arXiv preprint arXiv:2311.09783},
              year = {2023}
            }

            @article{sainz2023trouble,
              author = {Sainz, Oscar and Campos, Jon Ander and Garcia-Ferrero, Iker and Etxaniz, Julen and de Lacalle, Oier Lopez and Agirre, Eneko},
              title = {NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark},
              journal = {arXiv preprint arXiv:2310.18018},
              year = {2023}
            }

            @article{xu2024contamination,
              author = {Xu, Cheng and Guan, Shuhao and Greene, Derek and Kechadi, M-Tahar},
              title = {Benchmark Data Contamination of Large Language Models: A Survey},
              journal = {arXiv preprint arXiv:2406.04244},
              year = {2024}
            }

            @article{ravaut2024survey,
              author = {Ravaut, Mathieu and Ding, Bosheng and Jiao, Fangkai and Chen, Hailin and Li, Xingxuan and Zhao, Ruochen and Qin, Chengwei and Xiong, Caiming and Joty, Shafiq},
              title = {A Comprehensive Survey of Contamination Detection Methods in Large Language Models},
              journal = {arXiv preprint arXiv:2404.00699},
              year = {2024}
            }

            @inproceedings{wu2025antileakbench,
              author = {Wu, Xiaobao and Pan, Liangming and Xie, Yuxi and Zhou, Ruiwen and Zhao, Shuai and Ma, Yubo and Du, Mingzhe and Mao, Rui and Luu, Anh Tuan and Wang, William Yang},
              title = {AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge},
              booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
              pages = {18403--18419},
              year = {2025},
              doi = {10.18653/v1/2025.acl-long.901}
            }

            @inproceedings{chen2025contamination,
              author = {Chen, Simin and Chen, Yiming and Li, Zexin and Jiang, Yifan and Wan, Zhongwei and He, Yixin and Ran, Dezhi and Gu, Tianle and Li, Haizhou and Xie, Tao and Ray, Baishakhi},
              title = {Benchmarking Large Language Models Under Data Contamination: A Survey from Static to Dynamic Evaluation},
              booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
              pages = {10080--10098},
              year = {2025},
              doi = {10.18653/v1/2025.emnlp-main.511}
            }

            @article{white2024livebench,
              author = {White, Colin and Dooley, Samuel and Roberts, Manley and Pal, Arka and Feuer, Benjamin and Jain, Siddhartha and Shwartz-Ziv, Ravid and Jain, Neel and Saifullah, Khalid and Naidu, Siddartha and Hegde, Chinmay and LeCun, Yann and Goldstein, Tom and Goldblum, Micah},
              title = {LiveBench: A Challenging, Contamination-Free LLM Benchmark},
              journal = {arXiv preprint arXiv:2406.19314},
              year = {2024}
            }
            """
        ).strip()
        + "\n"
    )


def intro_text() -> str:
    return textwrap.dedent(
        """
        Public benchmark scores now do a large amount of institutional work in artificial intelligence. They rank systems, shape claims about progress, guide deployment decisions, and determine which model families are treated as credible scientific or commercial alternatives. This gives benchmark data a role that is recognizably psychometric: a score is useful only to the extent that it supports the intended inference about the respondent. In educational and psychological testing, validity is not a property of a test form in isolation, but of the interpretation made from observed responses (American Educational Research Association et al., 2014; Messick, 1995). The same logic applies to large language model evaluation. A model that solves a held-out mathematics problem may be demonstrating general reasoning ability, but it may also be reproducing an item, solution, or near-duplicate encountered before evaluation.

        Data contamination is therefore not only a corpus-management problem. It is a response-process problem. If a benchmark item has been included in pretraining, instruction tuning, retrieval augmentation, synthetic distillation, or derivative benchmark materials, the correct response may arise from a different latent process than the one the benchmark was designed to measure. That distinction matters most for benchmarks such as GSM8K, where the intended construct is multi-step grade-school mathematical reasoning (Cobbe et al., 2021). A correct answer to an open-response problem is unlikely to be produced by blind guessing. When unexpectedly high success appears among model-item pairs that should have low success probability given model ability and item difficulty, the anomaly resembles item preknowledge more than ordinary chance.

        Most contamination research in language-model evaluation begins from text or from benchmark construction. Investigators search training corpora for overlapping strings, compare benchmark items with web snapshots, design dynamic evaluations whose answers change over time, or build benchmark variants intended to be difficult to memorize (Chen et al., 2025; Deng et al., 2023; Ravaut et al., 2024; Sainz et al., 2023; White et al., 2024; Wu et al., 2025; Xu et al., 2024). These methods are indispensable because they can sometimes identify a plausible contamination source. However, they are limited when training corpora are not public, when contamination occurs through paraphrase or solution traces rather than exact strings, or when the empirical question is not whether an item exists somewhere on the web but whether a particular model-item response is unusually successful. The present article develops a complementary measurement approach: infer preknowledge from the pattern of responses after adjusting for model ability and item properties.

        Item response theory (IRT) is a natural foundation for this approach. In Rasch, one-parameter logistic, and two-parameter logistic models, observed binary responses are explained by a latent respondent trait and item parameters such as difficulty and discrimination (Birnbaum, 1968; Hambleton et al., 1991; Lord, 1980; Rasch, 1960). IRT has already been used to study machine-learning benchmarks and language-model leaderboards because it separates model ability from item informativeness (Kipnis et al., 2024; Lalor et al., 2016; Martinez-Plumed et al., 2019; Nadeem et al., 2025; Rodriguez et al., 2021; Zhou et al., 2025). Yet ordinary IRT assumes a single response process. A correct response from a low-ability model on a difficult item is treated as random residual variation unless the model is extended. In multiple-choice educational testing, a three-parameter logistic model adds a lower asymptote for guessing (Birnbaum, 1968; Lord, 1980). For GSM8K-style open response, that interpretation is weak: the chance of guessing the exact answer and producing a valid chain of reasoning is effectively negligible. A lower-asymptote-like signal is therefore more plausibly read as a second response process.

        Finite mixture models provide a principled way to represent such heterogeneity (McLachlan & Peel, 2000). In mixture IRT, different latent classes may follow different response functions, allowing item compromise, item parameter drift, subpopulation differences, or response-style heterogeneity to be modeled directly rather than absorbed into residuals (Frick et al., 2016). The approach proposed here treats benchmark contamination as a deterministic or near-deterministic preknowledge class. For each model-item cell, the ordinary class follows a 2PL response curve. The preknowledge class succeeds with probability near one. The probability of class membership is modeled using item, model, and model-family exposure effects. This produces a posterior probability that a given correct response was generated by preknowledge rather than by the ordinary ability-driven process.

        The model is designed for the kind of complex data emphasized in the Journal of Classification special issue on Advances in Mixture Models for Complex Data. LLM benchmark responses are high-dimensional, sparse, crossed, and clustered. Rows are not independent examinees in the usual psychometric sense: a leaderboard can contain base models, fine-tunes, merges, quantizations, and checkpoints from the same model family. Items are not interchangeable Bernoulli trials: they contain text, solutions, difficulty, and possible public exposure histories. The response array is binary, but its interpretation depends on latent ability, item parameters, model lineage, and a potentially rare contamination process. A mixture classification model is therefore not merely a more flexible predictor; it is a way to classify response mechanisms in a complex crossed data structure.

        This article has four goals. First, it formulates a Bayesian deterministic-mixture IRT model for open-response benchmark contamination. Second, it compares the mixture model with 1PL, hierarchical 1PL, 2PL, hierarchical 2PL, and sparse 3PL alternatives using heldout prediction. Third, it evaluates whether high-probability preknowledge classifications are stable under prior sensitivity analyses. Fourth, it connects the applied GSM8K results to a simulation plan suitable for evaluating classification accuracy, false-positive control, and robustness to family dependence and hidden multidimensionality. The intended contribution is both methodological and practical: a classification framework for mixture models in complex benchmark data, and an auditable workflow for prioritizing confirmatory perturbation experiments.
        """
    ).strip()


def methods_text() -> str:
    return textwrap.dedent(
        r"""
        Let \(Y_{mi} \in \{0,1\}\) indicate whether model \(m\) answered GSM8K item \(i\) correctly. The empirical data are arranged as a crossed model-by-item response matrix. Rows correspond to model checkpoints or leaderboard entries and columns correspond to GSM8K items. The analysis in this draft uses the complete available GSM8K item set after excluding exactly saturated items, defined as items with observed pass rate 0 or 1 in the loaded matrix. This exclusion removes columns that cannot identify item-level response curves while avoiding the residual-screen preselection used in a separate exploratory analysis. Final substantive claims about particular items would still require confirmatory perturbation experiments.

        The baseline 2PL model is

        \[
        \Pr(Y_{mi}=1 \mid \theta_m,a_i,b_i) = \operatorname{logit}^{-1}\{a_i(\theta_m-b_i)\},
        \]

        where \(\theta_m\) is model ability, \(a_i>0\) is item discrimination, and \(b_i\) is item difficulty. Higher values of \(\theta_m\) increase the probability of success, higher values of \(b_i\) make the item more difficult, and higher values of \(a_i\) make the item more discriminating around its difficulty point. The 1PL/Rasch comparison model fixes \(a_i=1\), yielding a common discrimination across items. The hierarchical 1PL and hierarchical 2PL models replace independent abilities with a family-pooled ability structure. Specifically, model ability is centered on an inferred family ability, with separate family-level and model-level scale parameters. This is important because many leaderboard rows are not independent systems; related checkpoints can share training histories, architectures, or fine-tuning data.

        A sparse 3PL model was included as an important comparison because contamination in open-response mathematics can masquerade as a lower-asymptote effect. The sparse 3PL response function is

        \[
        \Pr(Y_{mi}=1) = c_i + (1-c_i)\operatorname{logit}^{-1}\{a_i(\theta_m-b_i)\},
        \]

        where \(c_i\) is an item-specific floor. In multiple-choice testing, \(c_i\) is often interpreted as guessing. In this application, the prior for \(c_i\) is concentrated near zero because blind guessing is not a plausible substantive explanation for GSM8K success. The sparse 3PL therefore serves as a diagnostic baseline: if a lower-asymptote-like signal appears, the mixture model tests whether that signal is better represented as a second response process with item, model, and family exposure structure.

        The primary contamination model is a deterministic finite mixture. For each model-item pair, let \(Z_{mi}=1\) denote preknowledge and \(Z_{mi}=0\) denote ordinary ability-driven responding. Conditional on \(Z_{mi}=0\), responses follow the 2PL. Conditional on \(Z_{mi}=1\), success occurs with probability \(\tau\), fixed near one to allow occasional formatting, extraction, or generation failure:

        \[
        \Pr(Y_{mi}=1) = (1-\pi_{mi})p_{mi} + \pi_{mi}\tau,
        \]

        where \(p_{mi}\) is the 2PL probability and \(\pi_{mi}=\Pr(Z_{mi}=1)\) is the prior probability that the cell belongs to the preknowledge class. In the model-by-item version,

        \[
        \operatorname{logit}(\pi_{mi}) = \alpha + u_i + v_m,
        \]

        where \(\alpha\) is a global contamination logit, \(u_i\) is an item exposure effect, and \(v_m\) is a model exposure effect. In the family-aware version,

        \[
        \operatorname{logit}(\pi_{mi}) =
        \alpha + u_i + v_m + w_{f[m]},
        \]

        where \(w_{f[m]}\) captures shared exposure among models in the inferred family \(f[m]\). This term is not intended to prove shared training data. It is a partial-pooling and classification device that reduces the risk of treating many closely related model rows as independent confirmations of the same item anomaly.

        The latent class is marginalized analytically during estimation, so the fitted likelihood remains differentiable and scalable. Posterior preknowledge probabilities are then reconstructed by Bayes' rule. For correct responses,

        \[
        \Pr(Z_{mi}=1 \mid Y_{mi}=1) =
        \frac{\pi_{mi}\tau}{(1-\pi_{mi})p_{mi}+\pi_{mi}\tau}.
        \]

        This posterior probability is the central classification quantity. It is high when the cell has a high exposure probability, the observed response is correct, and the ordinary IRT process assigns relatively low probability to success. The quantity is therefore distinct from item pass rate, residual size alone, or item difficulty alone.

        The prior specification was intentionally conservative and sensitivity-checked. Abilities followed standard normal priors in non-hierarchical models. Item difficulties followed zero-centered normal priors with the difficulty scale varied across sensitivity runs. Item discriminations followed lognormal priors with log mean zero and sensitivity over the log standard deviation. Sparse-3PL lower asymptotes followed beta priors concentrated near zero, with sensitivity variants allowing more or less sparsity. Mixture models used a normal prior on the global contamination logit centered at the logit of a base contamination rate, with base-rate variants representing rare and moderate contamination. Item, model, and family exposure effects were normal with half-normal hyperpriors on their scales. The deterministic success probability \(\tau\) was fixed and varied across sensitivity runs.

        Models were estimated in Python using Pyro, stochastic variational inference, and an AutoNormal variational guide (Bingham et al., 2019). The fitted suite included 1PL, hierarchical 1PL, 2PL, hierarchical 2PL, sparse 3PL, deterministic mixture 2PL, and family-aware deterministic mixture 2PL. The standard sensitivity suite generated 45 model/prior runs. All runs used the same random 90/10 train-holdout split, 1,000 optimization steps, a learning rate of 0.03, minibatches of up to 50,000 observed cells, and seed 123. Predictive fit was evaluated by heldout mean log likelihood, Brier score, and threshold accuracy (Brier, 1950). Calibration plots were computed for the best predictive model. Mixture classifications were summarized by recurrence across prior variants, focusing on items, models, and model families that repeatedly appeared among the top posterior preknowledge probabilities.

        The item-parameter diagnostics in the applied results combine five signals. First, the family-aware mixture item exposure effect estimates whether an item is unusually likely to enter the preknowledge class. Second, the mixture difficulty and discrimination parameters describe the ordinary ability-driven response process for the same item. Third, the sparse-3PL lower asymptote asks whether the item shows a floor-like excess-success signal when exposure is forced to be item-only. Fourth, observed pass rate guards against the trivial explanation that an item receives high exposure merely because it is easy. Fifth, legacy residual-screen flags are retained only as an auxiliary comparison with the earlier exploratory workflow. The most persuasive follow-up targets should show high exposure and stability without being reducible to observed easiness or to item preselection.
        """
    ).strip()


def applied_results_text() -> str:
    return textwrap.dedent(
        """
        The all-item analysis used 6,014 model rows, 1,214 non-saturated GSM8K items, and 7,300,996 observed model-item cells. The heldout set contained 730,100 cells, and the training set contained 6,570,896 cells. The model-comparison suite completed all 45 planned model/prior runs without failures. This completion rate is important because the strongest claims in the analysis depend on comparing the mixture results with simpler IRT alternatives under the same train-holdout split, rather than selecting a contamination model after inspecting only in-sample fit.

        Predictively, the deterministic mixture models were clearly favored. The best heldout log-likelihood run was the family-aware deterministic mixture with a wide difficulty prior, with heldout mean log likelihood -0.351652, Brier score 0.112065, and threshold accuracy 0.838539. The best non-family deterministic mixture used tau = .999 and achieved heldout mean log likelihood -0.351747, Brier score 0.112010, and accuracy 0.838408. The best sparse 3PL achieved -0.362337, the best hierarchical 2PL achieved -0.362893, and the best ordinary 2PL achieved -0.363162. Thus, adding an item-only lower asymptote improved over ordinary 2PL, but the structured deterministic mixture improved substantially more. The difference is substantively meaningful because the sparse 3PL assigns excess success to an item floor, whereas the mixture model can assign excess success to item, model, and family exposure components.

        The ranking of the top runs also supported the mixture interpretation. All ten highest heldout predictive runs were deterministic mixture models, and nine of the ten were family-aware mixture runs. These top family-aware runs represented wide difficulty, wide discrimination, wide exposure, moderate contamination, baseline, rare contamination, tight exposure, tight difficulty, and tight discrimination variants. The near-tie across base-rate and exposure-scale priors is useful because it indicates that the predictive advantage is not an artifact of a single contamination prior. The tau sensitivity was more nuanced: the non-family tau = .999 run had the best Brier score, whereas family-aware tau = .999 was worse than the family-aware baseline group. The main conclusion is therefore not that one fixed tau value is universally optimal, but that deterministic and near-deterministic mixture response processes outperform single-process IRT models on the full item set.

        The item classifications were stable even without residual-screen preselection. Across the 22 mixture prior variants, GSM8K items 767, 161, 599, and 1037 appeared in the top-100 posterior preknowledge lists in 21 mixture runs. Items 637, 49, 1313, 1259, and 89 appeared in 20 of 22 mixture runs, and item 331 appeared in 19. The strongest item by top-100 cell count was item 637, with 104 appearances, followed by item 767 with 82, item 161 with 70, item 49 with 67, and item 1313 with 66. Posterior preknowledge probabilities for many top cells saturated near one, so exact rank differences among the strongest cells should not be overinterpreted. The more defensible evidence is recurrence across model variants and priors.

        Family recurrence also concentrated in a small set of inferred model families. The most stable families included albaddawi/deepcode-7b-aurora, which appeared in 21 mixture prior variants and contributed 119 top-100 appearances; 4season/alignment-model-test, which appeared in 19 variants; and 0-hero/matter-0-2-32b, which also appeared in 19 variants. These summaries are useful for audit triage, but they should not be read as verified lineage evidence. The family labels are inferred from model names, and open leaderboards often contain aliases, derivative checkpoints, and incomplete provenance. The appropriate interpretation is that these rows share enough response-pattern structure to deserve grouped follow-up.

        The item-parameter diagnostics refine the contamination interpretation. A high mixture item exposure effect indicates that an item frequently participates in the preknowledge class after accounting for ability, difficulty, discrimination, model exposure, and family exposure. If this effect were merely a proxy for item easiness, it should closely track observed pass rate. The diagnostic plots show that this is not the full story: high-exposure items are not simply the easiest all-item cases. Likewise, if the signal were only an item-level floor, the sparse 3PL lower asymptote would be sufficient. Instead, the family-aware mixture improves heldout prediction while allowing the same excess-success signal to be decomposed across items, models, and families.

        The applied conclusion is therefore intentionally probabilistic. The analysis does not prove that any training corpus contained a particular GSM8K item. It classifies response patterns that are more consistent with preknowledge than with the ordinary IRT response process. The most stable all-item classifications should be treated as the first wave for confirmatory experiments: create difficulty-preserving numeric or semantic twins, administer both the original and twin items to the suspect models, and test whether the original advantage persists after controlling for baseline ability. In this workflow, Bayesian mixture IRT supplies the prioritization mechanism, while perturbation supplies the strongest causal evidence.
        """
    ).strip()


def simulation_text() -> str:
    return textwrap.dedent(
        """
        The simulation study should establish when the proposed classifier recovers true preknowledge and when it mistakes other forms of model misfit for contamination. The data-generating design should vary sample size, item count, model-family dependence, contamination rate, contamination topology, contamination strength, nuisance dimensionality, and response noise. Each replication would simulate item parameters, model abilities, family effects, and a latent exposure matrix. The uncontaminated process would follow a 2PL model; the contaminated process would generate successes with probability tau.

        The most important simulation conditions are not only the favorable ones. Null conditions with no contamination are needed to estimate false-positive rates. Multidimensional no-contamination conditions are needed to test whether hidden subskills induce spurious preknowledge classifications. Family-clustered contamination conditions are needed because open LLM leaderboards contain many descendants, merges, fine-tunes, and quantizations. The simulation should compare residual screening, 2PL, hierarchical 2PL, sparse 3PL, deterministic mixture 2PL, and family-aware deterministic mixture 2PL under the same heldout split and classification thresholds.

        Primary classification outcomes should include AUC, PR-AUC, precision@k, recall@k, item-level recall, family-level recall, and false discoveries under the null. Predictive outcomes should include heldout log likelihood, Brier score, and calibration. Parameter recovery outcomes should include RMSE and bias for theta, a, b, and exposure effects. The planned reporting should emphasize classification stability and calibration rather than relying on a single threshold, because benchmark auditing is a triage problem: the goal is to prioritize scarce perturbation experiments with known uncertainty.
        """
    ).strip()


def references_text() -> str:
    return textwrap.dedent(
        """
        American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). Standards for educational and psychological testing. American Educational Research Association.

        Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., & Goodman, N. D. (2019). Pyro: Deep universal probabilistic programming. Journal of Machine Learning Research, 20(28), 1-6.

        Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), Statistical theories of mental test scores (pp. 397-479). Addison-Wesley.

        Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. Psychometrika, 46(4), 443-459.

        Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1-3.

        Buerkner, P.-C. (2021). Bayesian item response modeling in R with brms and Stan. Journal of Statistical Software, 100(5), 1-54.

        Chen, S., Chen, Y., Li, Z., Jiang, Y., Wan, Z., He, Y., Ran, D., Gu, T., Li, H., Xie, T., & Ray, B. (2025). Benchmarking large language models under data contamination: A survey from static to dynamic evaluation. Proceedings of EMNLP.

        Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.

        Deng, C., Zhao, Y., Tang, X., Gerstein, M., & Cohan, A. (2023). Investigating data contamination in modern benchmarks for large language models. arXiv:2311.09783.

        Fraley, C., & Raftery, A. E. (2002). Model-based clustering, discriminant analysis, and density estimation. Journal of the American Statistical Association, 97(458), 611-631.

        Frick, H., Strobl, C., & Zeileis, A. (2016). Investigating the impact of item parameter drift for item response theory models with mixture distributions. Frontiers in Psychology, 7, 255.

        Hambleton, R. K., Swaminathan, H., & Rogers, H. J. (1991). Fundamentals of item response theory. Sage.

        Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring massive multitask language understanding. International Conference on Learning Representations.

        Kiela, D., Bartolo, M., Nie, Y., Kaushik, D., Geiger, A., Wu, Z., Vidgen, B., Prasad, G., Singh, A., Ringshia, P., Ma, Z., Thrush, T., Riedel, S., Waseem, Z., Stenetorp, P., Jia, R., Bansal, M., Potts, C., & Williams, A. (2021). Dynabench: Rethinking benchmarking in NLP. Proceedings of NAACL-HLT, 4110-4124.

        Kipnis, A., Voudouris, K., Schulze Buschoff, L. M., & Schulz, E. (2024). metabench: A sparse benchmark to measure general ability in large language models. arXiv:2407.12844.

        Lalor, J. P., Wu, H., & Yu, H. (2016). Building an evaluation scale using item response theory. Proceedings of EMNLP, 648-657.

        Levy, R., & Mislevy, R. J. (2016). Bayesian psychometric modeling. CRC Press.

        Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., Narayanan, D., Wu, Y., Kumar, A., et al. (2023). Holistic evaluation of language models. Transactions on Machine Learning Research.

        Lord, F. M. (1980). Applications of item response theory to practical testing problems. Lawrence Erlbaum Associates.

        Martinez-Plumed, F., Prudencio, R. B. C., Martinez-Uso, A., & Hernandez-Orallo, J. (2019). Item response theory in AI: Analysing machine learning classifiers at the instance level. Artificial Intelligence, 271, 18-42.

        Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory models. Measurement, 11(3), 71-101.

        McLachlan, G. J., & Peel, D. (2000). Finite mixture models. Wiley.

        Messick, S. (1995). Validity of psychological assessment: Validation of inferences from persons' responses and performances as scientific inquiry into score meaning. American Psychologist, 50(9), 741-749.

        Nadeem, N., et al. (2025). Rethinking math benchmarks for LLMs using IRT. Proceedings of Machine Learning Research, 273, 66-82.

        Natesan, P., Nandakumar, R., Minka, T., & Rubright, J. D. (2016). Bayesian prior choice in IRT estimation using MCMC and variational Bayes. Frontiers in Psychology, 7, 1422.

        Rasch, G. (1960). Probabilistic models for some intelligence and attainment tests. Danish Institute for Educational Research.

        Ravaut, M., Ding, B., Jiao, F., Chen, H., Li, X., Zhao, R., Qin, C., Xiong, C., & Joty, S. (2024). A comprehensive survey of contamination detection methods in large language models. arXiv:2404.00699.

        Rodriguez, P., Barrow, J., Hoyle, A. M., Lalor, J. P., Jia, R., & Boyd-Graber, J. (2021). Evaluation examples are not equally informative: How should that change NLP leaderboards? Proceedings of ACL, 4486-4503.

        Sainz, O., Campos, J. A., Garcia-Ferrero, I., Etxaniz, J., de Lacalle, O. L., & Agirre, E. (2023). NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark. arXiv:2310.18018.

        Srivastava, A., Rastogi, A., Rao, A., et al. (2023). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research.

        White, C., Dooley, S., Roberts, M., Pal, A., Feuer, B., Jain, S., Shwartz-Ziv, R., Jain, N., Saifullah, K., Naidu, S., Hegde, C., LeCun, Y., Goldstein, T., & Goldblum, M. (2024). LiveBench: A challenging, contamination-free LLM benchmark. arXiv:2406.19314.

        Wu, X., Pan, L., Xie, Y., Zhou, R., Zhao, S., Ma, Y., Du, M., Mao, R., Luu, A. T., & Wang, W. Y. (2025). AntiLeakBench: Preventing data contamination by automatically constructing benchmarks with updated real-world knowledge. Proceedings of ACL, 18403-18419.

        Xu, C., Guan, S., Greene, D., & Kechadi, M.-T. (2024). Benchmark data contamination of large language models: A survey. arXiv:2406.04244.

        Yen, W. M. (1984). Effects of local item dependence on the fit and equating performance of the three-parameter logistic model. Applied Psychological Measurement, 8(2), 125-145.

        Zhou, H., Huang, H., Zhao, Z., Han, L., Wang, H., Chen, K., Yang, M., Bao, W., Dong, J., Xu, B., Zhu, C., Cao, H., & Zhao, T. (2025). Lost in benchmarks? Rethinking large language model benchmarking with item response theory. arXiv:2505.15055.
        """
    ).strip()


def write_manuscript(table_paths, figure_paths, suite_config) -> None:
    table2 = pd.read_csv(table_paths["table2"])
    table3 = pd.read_csv(table_paths["table3"])
    table4 = pd.read_csv(table_paths["table4"])
    table6 = pd.read_csv(table_paths["table6"])
    table8 = compact_item_parameter_table(pd.read_csv(table_paths["table8"]))

    content = f"""# Classifying Benchmark Preknowledge in Large Language Models With Bayesian Mixture Item Response Theory

## Target Outlet

Journal of Classification, special issue: Advances in Mixture Models for Complex Data.

Submission deadline from the local call PDF: November 30, 2026.

## Abstract

Benchmark contamination threatens the validity of large language model evaluation by allowing models to answer test items from exposure rather than latent capability. We propose a Bayesian mixture item response theory model that treats benchmark contamination as item preknowledge. For open-response mathematics benchmarks such as GSM8K, ordinary random guessing is effectively zero, so lower-asymptote-like success among low-probability model-item pairs is interpreted as evidence for a second response process. The proposed model constrains item difficulty and discrimination to the intended ability-driven response process while adding a deterministic preknowledge class with model, item, and family exposure effects. Applied to 6,014 models and 1,214 non-saturated GSM8K items, deterministic mixture models outperform 1PL, 2PL, hierarchical 2PL, and sparse 3PL baselines on heldout prediction. The best all-item run is a family-aware deterministic mixture, and the highest-probability item classifications recur across mixture prior sensitivity variants. A simulation study is proposed to evaluate classification accuracy, false-positive control, and robustness under family dependence, hidden multidimensionality, and imperfect memorization. The framework contributes a mixture classification method for complex model-by-item benchmark data and a practical audit pipeline for prioritizing perturbation experiments.

## Introduction

{intro_text()}

## Methods

{methods_text()}

### Applied Data and Model Suite

The applied analysis used the filtered GSM8K response matrix in the project folder and retained all items with non-saturated observed pass rates. This produced 1,214 analyzed items and 7,300,996 observed model-item cells. The suite fit 45 model/prior combinations with stochastic variational inference and evaluated all models on the same 10% holdout split.

**Table 1**  
Dataset and model-comparison suite.

{apa_table(pd.read_csv(table_paths["table1"]))}

*Note.* Items with pass rates exactly 0 or 1 in the loaded matrix were excluded before fitting.

## Applied Results

{applied_results_text()}

**Table 2**  
Best prior variant within each model family.

{apa_table(table2)}

*Note.* Higher heldout mean log likelihood and lower Brier score indicate better predictive performance.

**Table 3**  
Ten highest heldout predictive runs.

{apa_table(table3)}

*Note.* All top ten runs are deterministic mixture models, with family-aware mixtures dominating the leading positions.

**Table 4**  
Stable items across mixture prior variants.

{apa_table(table4)}

*Note.* There were 22 mixture prior variants: 11 deterministic mixture 2PL variants and 11 family-aware deterministic mixture 2PL variants. Top-100 appearances count how often an item appears among the 100 highest posterior preknowledge probabilities in those runs.

![Figure 1](figures/{figure_paths["fig1"].name})

![Figure 2](figures/{figure_paths["fig2"].name})

![Figure 3](figures/{figure_paths["fig3"].name})

![Figure 4](figures/{figure_paths["fig4"].name})

![Figure 5](figures/{figure_paths["fig5"].name})

![Figure 6](figures/{figure_paths["fig6"].name})

### Item-Parameter Diagnostics for Contamination

The mixture model is useful only if its contamination-related parameters can be separated from ordinary item quality. We therefore inspected item difficulty, discrimination, item-level exposure effects from the best family-aware deterministic mixture, sparse-3PL lower asymptotes, observed pass rates, and legacy residual-screen flags. These diagnostics ask whether stable all-item classifications are merely easy, poorly discriminating, or associated with a lower-asymptote-like response process concentrated among particular models and families.

The strongest contamination-relevant items were not simply the easiest items in the all-item set. Items with high posterior item exposure also tended to be repeatedly classified across prior sensitivity variants, whereas observed pass rate alone did not explain the exposure ordering. This pattern supports the interpretation that the mixture item-exposure parameter is measuring an additional response-process signal rather than rediscovering item easiness. The sparse-3PL lower-asymptote comparison is also informative: when lower asymptote and mixture exposure agree, both models are detecting excess success among low-baseline-probability model-item pairs; when they diverge, the mixture model is preferable because it allows the anomaly to be attributed to item, model, and family exposure components rather than assigning all excess success to an item-only floor.

**Table 5**  
Top contamination-relevant item-parameter diagnostics.

{apa_table(table8)}

*Note.* Mixture item exposure, difficulty, and discrimination are posterior medians from the best family-aware deterministic mixture. Sparse 3PL lower asymptote is the posterior median item floor from the sparse 3PL baseline. Top-100 appearances are counted across all mixture prior variants.

![Figure 7](figures/{figure_paths["fig7"].name})

![Figure 8](figures/{figure_paths["fig8"].name})

![Figure 9](figures/{figure_paths["fig9"].name})

![Figure 10](figures/{figure_paths["fig10"].name})

## Planned Simulation Study

{simulation_text()}

**Table 6**  
Planned simulation design.

{apa_table(table6)}

*Note.* The simulation is designed for classification performance, not only parameter recovery.

## Discussion

The applied results support the value of mixture modeling for complex benchmark data. The improvement of deterministic mixture models over ordinary IRT baselines suggests that some GSM8K model-item successes are better represented by a second response process than by ability and item difficulty alone. The family-aware version is particularly important because open leaderboards contain many related checkpoints. Treating those rows as exchangeable independent examinees would overstate evidence for item-level recurrence. The family-aware model instead separates item susceptibility, model exposure, and family exposure.

These findings should remain carefully framed. The analysis classifies preknowledge-like response patterns. It does not prove that a particular training corpus contained a particular GSM8K item. The posterior classifications are best interpreted as a principled queue for confirmatory perturbation: items that recur across prior variants should be rewritten into difficulty-preserving twins and administered to the suspect models. A model that succeeds on the original but fails the twin offers stronger evidence of memorization than either string overlap or residual anomaly alone.

## References

{references_text()}
"""
    MANUSCRIPT.write_text(content)


def write_notebook(table_paths, figure_paths, suite_config) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    def md(text: str):
        return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())

    def code(text: str):
        return nbf.v4.new_code_cell(textwrap.dedent(text).strip())

    table1 = pd.read_csv(table_paths["table1"])
    table2 = pd.read_csv(table_paths["table2"])
    table3 = pd.read_csv(table_paths["table3"])
    table4 = pd.read_csv(table_paths["table4"])
    table5 = pd.read_csv(table_paths["table5"])
    table6 = pd.read_csv(table_paths["table6"])
    table8 = compact_item_parameter_table(pd.read_csv(table_paths["table8"]))

    nb.cells = [
        md(
            """
            # Classifying Benchmark Preknowledge in Large Language Models With Bayesian Mixture IRT

            **Shareable analysis notebook.**  
            This notebook summarizes the Bayesian mixture IRT results for the GSM8K contamination project and sketches the simulation study for the Journal of Classification special issue, "Advances in Mixture Models for Complex Data."

            **Main takeaway.** Deterministic mixture IRT models, especially the family-aware mixture, provide the best heldout prediction across all non-saturated GSM8K items and identify a stable set of items for perturbation-based confirmation.
            """
        ),
        md(
            """
            ## Special Issue Fit

            The local call PDF states that the Journal of Classification special issue is devoted to advances in mixture models for complex data, including mixture approaches for text data and mixed-type data. This project fits that call because LLM benchmark responses are complex crossed data: model by item binary responses, item text, model families, non-independent checkpoints, and a latent preknowledge class.

            The submission deadline in the call PDF is **November 30, 2026**.
            """
        ),
        md("## Introduction\n\n" + intro_text()),
        md("## Methods\n\n" + methods_text()),
        md("### Table 1\n\nDataset and model-comparison suite.\n\n" + apa_table(table1) + "\n\n*Note.* Items with pass rates exactly 0 or 1 in the loaded matrix were excluded before fitting."),
        code(
            """
            from pathlib import Path
            import pandas as pd

            root = Path('..') if Path.cwd().name == 'shareable_joc_mixture_irt' else Path('.')
            suite = root / 'bayes_irt_all_items_suite_outputs'
            summary = pd.read_csv(suite / 'summary_ranked.csv')
            best = pd.read_csv(suite / 'best_by_model.csv')
            summary.head()
            """
        ),
        md("## Applied Results\n\n" + applied_results_text()),
        md("### Table 2\n\nBest prior variant within each model family.\n\n" + apa_table(table2) + "\n\n*Note.* Higher heldout mean log likelihood and lower Brier score indicate better predictive performance."),
        md("### Table 3\n\nTen highest heldout predictive runs.\n\n" + apa_table(table3) + "\n\n*Note.* All top ten runs are deterministic mixture models."),
        md("### Table 4\n\nStable items across mixture prior variants.\n\n" + apa_table(table4) + "\n\n*Note.* Top-100 appearances are counted over 22 mixture prior variants."),
        md("### Table 5\n\nStable model families across mixture prior variants.\n\n" + apa_table(table5) + "\n\n*Note.* Family labels are inferred from model names and should be validated against actual lineage metadata before making model-provenance claims."),
        md(f"### Figure 1\n\nHeldout predictive fit by best prior variant.\n\n![Figure 1](figures/{figure_paths['fig1'].name})"),
        md(f"### Figure 2\n\nHeldout Brier score by best prior variant.\n\n![Figure 2](figures/{figure_paths['fig2'].name})"),
        md(f"### Figure 3\n\nMixture-model prior sensitivity.\n\n![Figure 3](figures/{figure_paths['fig3'].name})"),
        md(f"### Figure 4\n\nStability of items.\n\n![Figure 4](figures/{figure_paths['fig4'].name})"),
        md(f"### Figure 5\n\nStable model families among top classifications.\n\n![Figure 5](figures/{figure_paths['fig5'].name})"),
        md(f"### Figure 6\n\nHoldout calibration for the best model.\n\n![Figure 6](figures/{figure_paths['fig6'].name})"),
        md(
            "## Item-Parameter Diagnostics for Contamination\n\n"
            "These diagnostics separate the contamination signal from ordinary item properties. "
            "The key quantity is the family-aware mixture item exposure effect, which is interpreted alongside "
            "item difficulty, item discrimination, the sparse-3PL lower asymptote, observed pass rate, "
            "and legacy residual-screen flags. High exposure combined with recurrence across priors is stronger evidence "
            "for a preknowledge-like response process than high observed pass rate alone."
        ),
        md(
            "### Table 6\n\nTop contamination-relevant item-parameter diagnostics.\n\n"
            + apa_table(table8)
            + "\n\n*Note.* Parameters are posterior medians. Top-100 appearances are counted over all mixture prior variants."
        ),
        md(f"### Figure 7\n\nAll-item difficulty, discrimination, and mixture exposure.\n\n![Figure 7](figures/{figure_paths['fig7'].name})"),
        md(f"### Figure 8\n\nHighest relative item-level contamination exposure.\n\n![Figure 8](figures/{figure_paths['fig8'].name})"),
        md(f"### Figure 9\n\nAgreement between mixture exposure and sparse-3PL lower asymptote.\n\n![Figure 9](figures/{figure_paths['fig9'].name})"),
        md(f"### Figure 10\n\nMixture item exposure compared with observed pass rate.\n\n![Figure 10](figures/{figure_paths['fig10'].name})"),
        md("## Planned Simulation Study\n\n" + simulation_text()),
        md("### Table 7\n\nPlanned simulation design.\n\n" + apa_table(table6) + "\n\n*Note.* The simulation should be framed as classification validation for mixture models for complex benchmark data."),
        md(
            """
            ## Reproducibility Commands

            ```bash
            PYTHONPATH=src python3 -m bayes_irt_gsm8k.suite \\
              --data-dir . \\
              --item-mode all \\
              --exclude-saturated-items \\
              --sensitivity standard \\
              --steps 1000 \\
              --batch-size 50000 \\
              --holdout-fraction 0.1 \\
              --seed 123 \\
              --output-dir bayes_irt_all_items_suite_outputs \\
              --top-n 1000

            PYTHONPATH=src python3 -m bayes_irt_gsm8k.report bayes_irt_all_items_suite_outputs --top-k 100
            python3 scripts/generate_shareable_notebook.py
            ```
            """
        ),
        md("## References\n\n" + references_text()),
    ]
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    main()
