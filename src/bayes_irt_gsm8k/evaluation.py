"""Predictive evaluation helpers for Bayesian IRT model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from bayes_irt_gsm8k.data import ResponseData
from bayes_irt_gsm8k.fit import FitResult, posterior_median_probabilities


@dataclass(frozen=True)
class PredictiveMetrics:
    """Held-out metrics computed from posterior-median probabilities."""

    n_obs: int
    log_likelihood: float
    mean_log_likelihood: float
    brier: float
    accuracy_0_5: float
    observed_mean: float
    predicted_mean: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_obs": self.n_obs,
            "log_likelihood": self.log_likelihood,
            "mean_log_likelihood": self.mean_log_likelihood,
            "brier": self.brier,
            "accuracy_0_5": self.accuracy_0_5,
            "observed_mean": self.observed_mean,
            "predicted_mean": self.predicted_mean,
        }


def predictive_metrics(fit: FitResult, data: ResponseData) -> PredictiveMetrics:
    """Evaluate posterior-median predictions on a response dataset."""

    data_cpu = data.to("cpu")
    y = data_cpu.y
    probs = posterior_median_probabilities(fit, data_cpu)
    log_likelihood = y * torch.log(probs) + (1.0 - y) * torch.log1p(-probs)
    predictions = (probs >= 0.5).to(y.dtype)
    return PredictiveMetrics(
        n_obs=data_cpu.n_obs,
        log_likelihood=float(log_likelihood.sum().item()),
        mean_log_likelihood=float(log_likelihood.mean().item()),
        brier=float(torch.mean((y - probs) ** 2).item()),
        accuracy_0_5=float(torch.mean((predictions == y).to(torch.float32)).item()),
        observed_mean=float(y.mean().item()),
        predicted_mean=float(probs.mean().item()),
    )


def calibration_table(
    fit: FitResult,
    data: ResponseData,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create a simple probability-bin calibration table."""

    data_cpu = data.to("cpu")
    probs = posterior_median_probabilities(fit, data_cpu)
    y = data_cpu.y
    bins = torch.clamp((probs * n_bins).floor().long(), max=n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if not bool(mask.any()):
            continue
        rows.append(
            {
                "bin": bin_id,
                "n": int(mask.sum().item()),
                "mean_predicted": float(probs[mask].mean().item()),
                "mean_observed": float(y[mask].mean().item()),
            }
        )
    return pd.DataFrame(rows)
