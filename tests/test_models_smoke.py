from __future__ import annotations

import pandas as pd

from bayes_irt_gsm8k.data import response_data_from_matrix
from bayes_irt_gsm8k.fit import FitConfig, fit_svi, top_preknowledge_probabilities
from bayes_irt_gsm8k.models import IRTModelConfig


def make_tiny_data():
    matrix = pd.DataFrame(
        [
            [0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        index=[
            "alpha/model-a-v1",
            "alpha/model-a-v2",
            "beta/model-b-v1",
            "beta/model-b-v2",
            "gamma/model-c-v1",
            "delta/model-d-v1",
        ],
        columns=[101, 102, 103, 104, 105],
    )
    return response_data_from_matrix(matrix)


def smoke_fit(kind: str):
    data = make_tiny_data()
    fit = fit_svi(
        data,
        IRTModelConfig(kind=kind),
        FitConfig(steps=3, learning_rate=0.02, batch_size=None, progress=False, seed=7),
    )
    assert len(fit.losses) == 3
    assert all(loss == loss for loss in fit.losses)
    assert "theta" in fit.median()
    return data, fit


def test_1pl_smoke_fit() -> None:
    smoke_fit("1pl")


def test_2pl_smoke_fit() -> None:
    data, fit = smoke_fit("2pl")
    assert fit.median()["difficulty"].shape[0] == data.n_items
    assert fit.median()["discrimination"].shape[0] == data.n_items


def test_hier_2pl_smoke_fit() -> None:
    data, fit = smoke_fit("hier_2pl")
    assert fit.median()["family_ability"].shape[0] == data.n_families


def test_sparse_3pl_smoke_fit() -> None:
    data, fit = smoke_fit("sparse_3pl")
    lower = fit.median()["lower_asymptote"]
    assert lower.shape[0] == data.n_items
    assert bool(((lower > 0) & (lower < 1)).all())


def test_dmixture_family_smoke_fit_and_scores() -> None:
    data, fit = smoke_fit("dmixture_family_2pl")
    ranked = top_preknowledge_probabilities(fit, data, top_n=5)
    assert list(ranked.columns) == [
        "model",
        "item",
        "family",
        "response",
        "base_probability",
        "mixture_probability",
        "preknowledge_probability",
    ]
    assert len(ranked) == 5
    assert ranked["preknowledge_probability"].between(0, 1).all()
