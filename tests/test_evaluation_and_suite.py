from __future__ import annotations

from pathlib import Path

import pandas as pd

from bayes_irt_gsm8k.data import (
    filter_items_by_pass_rate,
    item_pass_rate_summary,
    response_data_from_matrix,
    split_response_data,
)
from bayes_irt_gsm8k.evaluation import predictive_metrics
from bayes_irt_gsm8k.fit import FitConfig, fit_svi
from bayes_irt_gsm8k.models import IRTModelConfig
from bayes_irt_gsm8k.suite import SuiteConfig, build_model_specs, run_suite


def make_suite_data():
    matrix = pd.DataFrame(
        [
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        index=[
            "alpha/model-a-v1",
            "alpha/model-a-v2",
            "beta/model-b-v1",
            "beta/model-b-v2",
            "gamma/model-c-v1",
        ],
        columns=[11, 22, 33, 44],
    )
    return response_data_from_matrix(matrix)


def test_split_response_data_keeps_dimensions() -> None:
    data = make_suite_data()
    train, holdout = split_response_data(data, holdout_fraction=0.25, seed=4)

    assert train.n_obs + holdout.n_obs == data.n_obs
    assert train.n_models == holdout.n_models == data.n_models
    assert train.n_items == holdout.n_items == data.n_items


def test_item_pass_rate_filter_drops_saturated_items() -> None:
    matrix = pd.DataFrame(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        columns=[10, 20, 30, 40],
    )

    summary = item_pass_rate_summary(matrix)
    filtered = filter_items_by_pass_rate(matrix, exclude_saturated_items=True)

    assert summary["pass_rate"].tolist() == [0.0, 1 / 3, 1.0, 1.0]
    assert list(filtered.columns) == [20]


def test_item_pass_rate_filter_supports_near_saturation_bounds() -> None:
    matrix = pd.DataFrame(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        columns=[10, 20, 30, 40],
    )

    filtered = filter_items_by_pass_rate(
        matrix,
        min_item_pass_rate=0.3,
        max_item_pass_rate=0.8,
    )

    assert list(filtered.columns) == [20, 30]


def test_predictive_metrics_smoke() -> None:
    data = make_suite_data()
    fit = fit_svi(
        data,
        IRTModelConfig(kind="2pl"),
        FitConfig(steps=3, batch_size=None, progress=False, seed=9),
    )
    metrics = predictive_metrics(fit, data)

    assert metrics.n_obs == data.n_obs
    assert metrics.mean_log_likelihood < 0
    assert 0 <= metrics.brier <= 1
    assert 0 <= metrics.accuracy_0_5 <= 1


def test_build_model_specs_baseline_has_one_prior_per_model() -> None:
    specs = build_model_specs(("1pl", "dmixture_2pl"), sensitivity="baseline")

    assert [spec.run_name for spec in specs] == ["1pl__baseline", "dmixture_2pl__baseline"]
    assert all(spec.prior_name == "baseline" for spec in specs)


def test_run_suite_smoke(tmp_path: Path) -> None:
    data = make_suite_data()
    summary = run_suite(
        data,
        SuiteConfig(
            output_dir=tmp_path,
            model_kinds=("1pl", "dmixture_2pl"),
            sensitivity="baseline",
            steps=2,
            batch_size=None,
            progress=False,
            holdout_fraction=0.2,
            top_n=3,
        ),
    )

    assert len(summary) == 2
    assert set(summary["status"]) == {"ok"}
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary_ranked.csv").exists()
    assert (tmp_path / "selected_item_summary.csv").exists()
    assert (tmp_path / "dmixture_2pl__baseline" / "top_preknowledge_probabilities_full.csv").exists()
