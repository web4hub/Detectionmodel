from __future__ import annotations

from pathlib import Path

import pandas as pd

from bayes_irt_gsm8k.report import summarize_suite_outputs


def test_summarize_suite_outputs(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "run_name": "1pl__baseline",
                "model_kind": "1pl",
                "prior_name": "baseline",
                "status": "ok",
                "holdout_mean_log_likelihood": -0.5,
                "holdout_brier": 0.2,
                "holdout_accuracy_0_5": 0.7,
            },
            {
                "run_name": "dmixture_2pl__baseline",
                "model_kind": "dmixture_2pl",
                "prior_name": "baseline",
                "status": "ok",
                "holdout_mean_log_likelihood": -0.3,
                "holdout_brier": 0.1,
                "holdout_accuracy_0_5": 0.8,
            },
        ]
    )
    summary.to_csv(tmp_path / "summary_ranked.csv", index=False)
    run_dir = tmp_path / "dmixture_2pl__baseline"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model": "org/model-a",
                "item": 101,
                "family": "org/model",
                "response": 1,
                "base_probability": 0.01,
                "mixture_probability": 0.9,
                "preknowledge_probability": 0.99,
            }
        ]
    ).to_csv(run_dir / "top_preknowledge_probabilities_full.csv", index=False)

    written = summarize_suite_outputs(tmp_path, top_k=1)

    assert written["best_by_model"].exists()
    assert written["mixture_item_stability"].exists()
    assert written["markdown_report"].exists()
