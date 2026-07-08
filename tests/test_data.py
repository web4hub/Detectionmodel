from __future__ import annotations

import pandas as pd

from bayes_irt_gsm8k.data import infer_model_family, response_data_from_matrix


def test_response_data_from_matrix_builds_indices() -> None:
    matrix = pd.DataFrame(
        [[1, 0, 1], [0, 1, 0]],
        index=["Org/Model-v1-Q4", "Org/Model-v2-Q4"],
        columns=[10, 20, 30],
    )

    data = response_data_from_matrix(matrix)

    assert data.n_models == 2
    assert data.n_items == 3
    assert data.n_obs == 6
    assert data.model_idx.tolist() == [0, 0, 0, 1, 1, 1]
    assert data.item_idx.tolist() == [0, 1, 2, 0, 1, 2]
    assert data.y.tolist() == [1, 0, 1, 0, 1, 0]


def test_infer_model_family_is_stable_for_common_suffixes() -> None:
    assert infer_model_family("Qwen/Qwen1.5-72B-GPTQ") == infer_model_family("qwen/qwen1-5-72b")
