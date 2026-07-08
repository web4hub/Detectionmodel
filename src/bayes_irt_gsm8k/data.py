"""Data loading and indexing helpers for GSM8K IRT models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ResponseData:
    """Long-form binary response data used by the Pyro models.

    The original GSM8K matrix is wide: models by item ids. Pyro fitting is
    simpler and cheaper when the matrix is converted once to three tensors:
    model index, item index, and observed correctness.
    """

    y: torch.Tensor
    model_idx: torch.Tensor
    item_idx: torch.Tensor
    n_models: int
    n_items: int
    model_names: tuple[str, ...]
    item_ids: tuple[int | str, ...]
    family_idx: torch.Tensor
    family_names: tuple[str, ...]

    @property
    def n_obs(self) -> int:
        return int(self.y.numel())

    @property
    def n_families(self) -> int:
        return len(self.family_names)

    def to(self, device: str | torch.device) -> "ResponseData":
        """Return a copy with tensors moved to ``device``."""

        return ResponseData(
            y=self.y.to(device),
            model_idx=self.model_idx.to(device),
            item_idx=self.item_idx.to(device),
            n_models=self.n_models,
            n_items=self.n_items,
            model_names=self.model_names,
            item_ids=self.item_ids,
            family_idx=self.family_idx.to(device),
            family_names=self.family_names,
        )

    def with_observations(self, obs_idx: torch.Tensor | np.ndarray | Sequence[int]) -> "ResponseData":
        """Return a copy containing only selected long-form observations."""

        idx = torch.as_tensor(obs_idx, dtype=torch.long, device=self.y.device)
        return ResponseData(
            y=self.y[idx],
            model_idx=self.model_idx[idx],
            item_idx=self.item_idx[idx],
            n_models=self.n_models,
            n_items=self.n_items,
            model_names=self.model_names,
            item_ids=self.item_ids,
            family_idx=self.family_idx,
            family_names=self.family_names,
        )


def infer_model_family(model_name: str) -> str:
    """Infer a coarse model family label from a Hugging Face style model name.

    This is intentionally conservative. It captures shared repository prefixes
    and strips common version/quantization suffixes, but it does not claim to be
    a provenance graph. Better family labels can be supplied by replacing this
    function's output upstream.
    """

    normalized = model_name.lower().strip()
    normalized = normalized.replace("_", "-")
    owner, _, repo = normalized.partition("/")
    if not repo:
        owner, repo = "unknown", owner

    repo = re.sub(r"[-_.]+", "-", repo).strip("-")
    removable_tokens = {"gguf", "awq", "gptq", "exl2", "fp16", "bf16", "preview"}
    tokens = []
    for token in repo.split("-"):
        if not token:
            continue
        if token in removable_tokens or re.fullmatch(r"(q|int)\d+", token):
            continue
        if re.fullmatch(r"v\d+(\.\d+)*|r\d+(\.\d+)*", token):
            continue
        tokens.append(token)
    family_repo = "-".join(tokens[:4]) if tokens else "unknown"
    return f"{owner}/{family_repo}"


def load_item_list(path: str | Path) -> list[int | str]:
    """Load an item list from a CSV with an ``item`` column or a plain text file."""

    item_path = Path(path)
    if item_path.suffix.lower() == ".csv":
        df = pd.read_csv(item_path)
        if "item" not in df.columns:
            raise ValueError(f"{item_path} must contain an 'item' column")
        values = df["item"].tolist()
    else:
        values = [line.strip() for line in item_path.read_text().splitlines() if line.strip()]
    return [_coerce_item_id(value) for value in values]


def load_gsm8k_response_data(
    data_dir: str | Path,
    matrix_name: str = "gsm8k_matrix_filtered.parquet",
    *,
    item_ids: Sequence[int | str] | None = None,
    model_names: Sequence[str] | None = None,
    exclude_saturated_items: bool = False,
    min_item_pass_rate: float | None = None,
    max_item_pass_rate: float | None = None,
    max_models: int | None = None,
    max_items: int | None = None,
    max_obs: int | None = None,
    seed: int = 123,
    dtype: torch.dtype = torch.float32,
) -> ResponseData:
    """Load the GSM8K response matrix and convert it to indexed long form.

    Parameters are applied in this order: explicit model/item filters, item
    pass-rate filters, random row/column subsampling, then optional observation
    subsampling. Omitting ``item_ids`` intentionally uses every matrix item.
    """

    data_dir = Path(data_dir)
    matrix = pd.read_parquet(data_dir / matrix_name)
    matrix = _filter_matrix(matrix, item_ids=item_ids, model_names=model_names)
    matrix = filter_items_by_pass_rate(
        matrix,
        exclude_saturated_items=exclude_saturated_items,
        min_item_pass_rate=min_item_pass_rate,
        max_item_pass_rate=max_item_pass_rate,
    )
    matrix = _sample_matrix(matrix, max_models=max_models, max_items=max_items, seed=seed)
    return response_data_from_matrix(matrix, max_obs=max_obs, seed=seed, dtype=dtype)


def response_data_from_matrix(
    matrix: pd.DataFrame,
    *,
    max_obs: int | None = None,
    seed: int = 123,
    dtype: torch.dtype = torch.float32,
) -> ResponseData:
    """Convert a dense model-by-item matrix into ``ResponseData``."""

    if matrix.empty:
        raise ValueError("response matrix is empty after filtering")
    if matrix.isna().any().any():
        raise ValueError("response matrix contains missing values; impute or filter before fitting")

    values = matrix.to_numpy(dtype=np.float32, copy=True)
    unique_values = np.unique(values)
    if not np.isin(unique_values, [0.0, 1.0]).all():
        raise ValueError(f"response matrix must be binary 0/1; found {unique_values[:10]}")

    n_models, n_items = values.shape
    model_idx_np, item_idx_np = np.indices(values.shape)
    y_np = values.reshape(-1)
    model_idx_np = model_idx_np.reshape(-1)
    item_idx_np = item_idx_np.reshape(-1)

    if max_obs is not None and max_obs < y_np.size:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(y_np.size, size=max_obs, replace=False))
        y_np = y_np[chosen]
        model_idx_np = model_idx_np[chosen]
        item_idx_np = item_idx_np[chosen]

    family_labels = [infer_model_family(name) for name in matrix.index.astype(str)]
    family_names = tuple(dict.fromkeys(family_labels))
    family_lookup = {name: idx for idx, name in enumerate(family_names)}
    family_idx_np = np.array([family_lookup[name] for name in family_labels], dtype=np.int64)

    return ResponseData(
        y=torch.as_tensor(y_np, dtype=dtype),
        model_idx=torch.as_tensor(model_idx_np, dtype=torch.long),
        item_idx=torch.as_tensor(item_idx_np, dtype=torch.long),
        n_models=n_models,
        n_items=n_items,
        model_names=tuple(matrix.index.astype(str)),
        item_ids=tuple(_coerce_item_id(col) for col in matrix.columns),
        family_idx=torch.as_tensor(family_idx_np, dtype=torch.long),
        family_names=family_names,
    )


def item_pass_rate_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarize item-level pass rates for a wide response matrix."""

    if matrix.empty:
        raise ValueError("response matrix is empty")
    n_obs = int(matrix.shape[0])
    n_correct = matrix.sum(axis=0).astype(float)
    return pd.DataFrame(
        {
            "item": [_coerce_item_id(col) for col in matrix.columns],
            "matrix_column": [str(col) for col in matrix.columns],
            "n_obs": n_obs,
            "n_correct": n_correct.to_numpy(),
            "pass_rate": (n_correct / n_obs).to_numpy(),
        }
    )


def response_item_summary(data: ResponseData) -> pd.DataFrame:
    """Summarize item-level pass rates for indexed ``ResponseData``."""

    item_idx = data.item_idx.detach().cpu()
    y = data.y.detach().cpu()
    counts = torch.bincount(item_idx, minlength=data.n_items).to(torch.float32)
    correct = torch.zeros(data.n_items, dtype=torch.float32)
    correct.scatter_add_(0, item_idx, y.to(torch.float32))
    pass_rate = correct / counts.clamp_min(1.0)
    return pd.DataFrame(
        {
            "item": list(data.item_ids),
            "n_obs": counts.numpy().astype(int),
            "n_correct": correct.numpy(),
            "pass_rate": pass_rate.numpy(),
        }
    )


def filter_items_by_pass_rate(
    matrix: pd.DataFrame,
    *,
    exclude_saturated_items: bool = False,
    min_item_pass_rate: float | None = None,
    max_item_pass_rate: float | None = None,
) -> pd.DataFrame:
    """Return a matrix with optional saturated or near-saturated items removed.

    Saturated items have pass rates exactly 0 or 1 within the loaded model set.
    The optional pass-rate bounds can be used to remove near-saturated items,
    for example ``min_item_pass_rate=0.01`` and ``max_item_pass_rate=0.99``.
    """

    _validate_item_pass_rate_bounds(min_item_pass_rate, max_item_pass_rate)
    if not exclude_saturated_items and min_item_pass_rate is None and max_item_pass_rate is None:
        return matrix

    summary = item_pass_rate_summary(matrix)
    keep = np.ones(len(summary), dtype=bool)
    pass_rate = summary["pass_rate"].to_numpy()
    if exclude_saturated_items:
        keep &= (pass_rate > 0.0) & (pass_rate < 1.0)
    if min_item_pass_rate is not None:
        keep &= pass_rate >= min_item_pass_rate
    if max_item_pass_rate is not None:
        keep &= pass_rate <= max_item_pass_rate
    if not bool(keep.any()):
        raise ValueError("item pass-rate filters removed every item")
    return matrix.iloc[:, keep]


def split_response_data(
    data: ResponseData,
    *,
    holdout_fraction: float = 0.1,
    seed: int = 123,
) -> tuple[ResponseData, ResponseData]:
    """Split observations into train and holdout ``ResponseData`` objects."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    n_holdout = max(1, int(round(data.n_obs * holdout_fraction)))
    if n_holdout >= data.n_obs:
        raise ValueError("holdout split would leave no training observations")
    generator = torch.Generator(device=data.y.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(data.n_obs, generator=generator, device=data.y.device)
    holdout_idx = permutation[:n_holdout]
    train_idx = permutation[n_holdout:]
    return data.with_observations(train_idx), data.with_observations(holdout_idx)


def _filter_matrix(
    matrix: pd.DataFrame,
    *,
    item_ids: Sequence[int | str] | None,
    model_names: Sequence[str] | None,
) -> pd.DataFrame:
    filtered = matrix
    if model_names is not None:
        model_set = set(model_names)
        missing = sorted(model_set.difference(filtered.index.astype(str)))
        if missing:
            raise ValueError(f"{len(missing)} requested models are absent; first missing: {missing[:3]}")
        filtered = filtered.loc[list(model_names)]

    if item_ids is not None:
        wanted = [_coerce_item_id(item) for item in item_ids]
        column_lookup = {_coerce_item_id(col): col for col in filtered.columns}
        missing = [item for item in wanted if item not in column_lookup]
        if missing:
            raise ValueError(f"{len(missing)} requested items are absent; first missing: {missing[:5]}")
        filtered = filtered[[column_lookup[item] for item in wanted]]
    return filtered


def _sample_matrix(
    matrix: pd.DataFrame,
    *,
    max_models: int | None,
    max_items: int | None,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = matrix
    if max_models is not None and max_models < sampled.shape[0]:
        positions = np.sort(rng.choice(sampled.shape[0], size=max_models, replace=False))
        sampled = sampled.iloc[positions, :]
    if max_items is not None and max_items < sampled.shape[1]:
        positions = np.sort(rng.choice(sampled.shape[1], size=max_items, replace=False))
        sampled = sampled.iloc[:, positions]
    return sampled


def _validate_item_pass_rate_bounds(
    min_item_pass_rate: float | None,
    max_item_pass_rate: float | None,
) -> None:
    for name, value in (
        ("min_item_pass_rate", min_item_pass_rate),
        ("max_item_pass_rate", max_item_pass_rate),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if min_item_pass_rate is not None and max_item_pass_rate is not None:
        if min_item_pass_rate > max_item_pass_rate:
            raise ValueError("min_item_pass_rate cannot exceed max_item_pass_rate")


def _coerce_item_id(value: object) -> int | str:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def ensure_iterable(value: Iterable[str] | None) -> Iterable[str]:
    """Small helper retained for CLI call sites that may pass ``None``."""

    return () if value is None else value
