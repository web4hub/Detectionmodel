"""Fitting and posterior utilities for Bayesian GSM8K IRT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Iterable

import numpy as np
import pandas as pd
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import torch
from tqdm.auto import trange

from bayes_irt_gsm8k.data import ResponseData
from bayes_irt_gsm8k.models import (
    IRTModelConfig,
    apply_response_process,
    base_irt_probability,
    irt_model,
)


@dataclass(frozen=True)
class FitConfig:
    """Configuration for stochastic variational inference."""

    steps: int = 1_000
    learning_rate: float = 0.03
    batch_size: int | None = 50_000
    seed: int = 123
    device: str = "cpu"
    progress: bool = True

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")


@dataclass
class FitResult:
    """Result object returned by ``fit_svi``."""

    guide: AutoNormal
    model_config: IRTModelConfig
    fit_config: FitConfig
    losses: list[float]

    def median(self) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.guide.median()


def posterior_median_probabilities(fit: FitResult, data: ResponseData) -> torch.Tensor:
    """Return posterior-median response probabilities for ``data`` observations."""

    data_cpu = data.to("cpu")
    med = {key: value.detach().cpu() for key, value in fit.median().items()}
    base_p = base_irt_probability(
        theta=med["theta"],
        difficulty=med["difficulty"],
        discrimination=med.get("discrimination", 1.0),
        model_idx=data_cpu.model_idx,
        item_idx=data_cpu.item_idx,
    )

    lower_asymptote = med.get("lower_asymptote")
    exposure_terms = None
    if fit.model_config.kind in {"dmixture_2pl", "dmixture_family_2pl"}:
        exposure_terms = {
            "global_logit_pi": med["global_logit_pi"],
            "item_exposure": med["item_exposure"],
            "model_exposure": med["model_exposure"],
        }
        if fit.model_config.kind == "dmixture_family_2pl":
            exposure_terms["family_exposure"] = med["family_exposure"]

    probs = apply_response_process(
        base_p=base_p,
        lower_asymptote=lower_asymptote,
        exposure_terms=exposure_terms,
        data=data_cpu,
        config=fit.model_config,
        model_idx=data_cpu.model_idx,
        item_idx=data_cpu.item_idx,
    )
    return probs.clamp(fit.model_config.probability_epsilon, 1.0 - fit.model_config.probability_epsilon)


def fit_svi(
    data: ResponseData,
    model_config: IRTModelConfig,
    fit_config: FitConfig,
) -> FitResult:
    """Fit a configured IRT model with Pyro SVI and an AutoNormal guide."""

    pyro.clear_param_store()
    pyro.set_rng_seed(fit_config.seed)
    torch.manual_seed(fit_config.seed)
    data = data.to(fit_config.device)

    batch_size = fit_config.batch_size
    if batch_size is not None:
        batch_size = min(batch_size, data.n_obs)

    def model() -> None:
        irt_model(data, model_config, batch_size=batch_size)

    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam({"lr": fit_config.learning_rate}), loss=Trace_ELBO())
    losses: list[float] = []

    iterator: Iterable[int]
    if fit_config.progress:
        iterator = trange(fit_config.steps, desc=f"fit {model_config.kind}", leave=False)
    else:
        iterator = range(fit_config.steps)

    for step in iterator:
        loss = float(svi.step())
        losses.append(loss)
        if fit_config.progress and hasattr(iterator, "set_postfix") and (step == 0 or (step + 1) % 25 == 0):
            iterator.set_postfix(loss=f"{loss:,.1f}")

    return FitResult(guide=guide, model_config=model_config, fit_config=fit_config, losses=losses)


def parameter_summary(
    fit: FitResult,
    data: ResponseData,
    *,
    max_rows_per_site: int = 2_000,
) -> pd.DataFrame:
    """Return a compact posterior-median summary for core parameter sites."""

    med = fit.median()
    rows: list[dict[str, object]] = []
    for site, values in med.items():
        values = values.detach().cpu().reshape(-1)
        if values.numel() > max_rows_per_site:
            idx = torch.linspace(0, values.numel() - 1, steps=max_rows_per_site).long()
            values = values[idx]
            indices = idx.tolist()
            truncated = True
        else:
            indices = list(range(values.numel()))
            truncated = False
        for index, value in zip(indices, values.tolist()):
            rows.append(
                {
                    "site": site,
                    "index": index,
                    "label": _site_label(site, index, data),
                    "median": float(value),
                    "truncated": truncated,
                }
            )
    return pd.DataFrame(rows)


def top_preknowledge_probabilities(
    fit: FitResult,
    data: ResponseData,
    *,
    only_correct: bool = True,
    top_n: int = 1_000,
) -> pd.DataFrame:
    """Rank observations by posterior-median preknowledge probability.

    This is defined for deterministic mixture models. The latent preknowledge
    indicator is marginalized during fitting, then reconstructed via Bayes' rule
    using posterior median parameters.
    """

    if fit.model_config.kind not in {"dmixture_2pl", "dmixture_family_2pl"}:
        raise ValueError("preknowledge probabilities are only available for dmixture models")

    data_cpu = data.to("cpu")
    med = {key: value.detach().cpu() for key, value in fit.median().items()}
    base_p = base_irt_probability(
        theta=med["theta"],
        difficulty=med["difficulty"],
        discrimination=med.get("discrimination", 1.0),
        model_idx=data_cpu.model_idx,
        item_idx=data_cpu.item_idx,
    )
    exposure_terms = {
        "global_logit_pi": med["global_logit_pi"],
        "item_exposure": med["item_exposure"],
        "model_exposure": med["model_exposure"],
    }
    if fit.model_config.kind == "dmixture_family_2pl":
        exposure_terms["family_exposure"] = med["family_exposure"]

    mixture_p = posterior_median_probabilities(fit, data_cpu)
    logit_pi = exposure_terms["global_logit_pi"]
    logit_pi = logit_pi + exposure_terms["item_exposure"][data_cpu.item_idx]
    logit_pi = logit_pi + exposure_terms["model_exposure"][data_cpu.model_idx]
    if fit.model_config.kind == "dmixture_family_2pl":
        logit_pi = logit_pi + exposure_terms["family_exposure"][data_cpu.family_idx[data_cpu.model_idx]]
    pi = torch.sigmoid(logit_pi)

    y = data_cpu.y
    tau = fit.model_config.tau
    numerator = torch.where(y > 0.5, pi * tau, pi * (1.0 - tau))
    denominator = torch.where(y > 0.5, mixture_p, 1.0 - mixture_p)
    posterior_z = (numerator / denominator.clamp_min(fit.model_config.probability_epsilon)).clamp(0.0, 1.0)

    mask = y > 0.5 if only_correct else torch.ones_like(y, dtype=torch.bool)
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    if selected.numel() == 0:
        return pd.DataFrame()
    scores = posterior_z[selected]
    n = min(top_n, selected.numel())
    top_scores, top_positions = torch.topk(scores, k=n)
    obs_positions = selected[top_positions]

    rows = []
    for score, obs_pos in zip(top_scores.tolist(), obs_positions.tolist()):
        model_position = int(data_cpu.model_idx[obs_pos])
        item_position = int(data_cpu.item_idx[obs_pos])
        rows.append(
            {
                "model": data_cpu.model_names[model_position],
                "item": data_cpu.item_ids[item_position],
                "family": data_cpu.family_names[int(data_cpu.family_idx[model_position])],
                "response": int(data_cpu.y[obs_pos].item()),
                "base_probability": float(base_p[obs_pos].item()),
                "mixture_probability": float(mixture_p[obs_pos].item()),
                "preknowledge_probability": float(score),
            }
        )
    return pd.DataFrame(rows)


def save_fit_artifacts(
    fit: FitResult,
    data: ResponseData,
    output_dir: str | Path,
    *,
    include_preknowledge: bool = True,
) -> None:
    """Save losses, config, parameter summary, and optional preknowledge ranks."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": np.arange(1, len(fit.losses) + 1), "loss": fit.losses}).to_csv(
        output_path / "losses.csv",
        index=False,
    )
    parameter_summary(fit, data).to_csv(output_path / "parameter_summary.csv", index=False)
    config_payload = {
        "model": asdict(fit.model_config),
        "fit": asdict(fit.fit_config),
        "data": {
            "n_models": data.n_models,
            "n_items": data.n_items,
            "n_obs": data.n_obs,
            "n_families": data.n_families,
        },
    }
    (output_path / "config.json").write_text(json.dumps(config_payload, indent=2))

    if include_preknowledge and fit.model_config.kind in {"dmixture_2pl", "dmixture_family_2pl"}:
        top_preknowledge_probabilities(fit, data).to_csv(
            output_path / "top_preknowledge_probabilities.csv",
            index=False,
        )


def _site_label(site: str, index: int, data: ResponseData) -> str:
    if site == "theta" and index < len(data.model_names):
        return data.model_names[index]
    if site in {"difficulty", "discrimination", "lower_asymptote", "item_exposure"} and index < len(data.item_ids):
        return str(data.item_ids[index])
    if site in {"family_ability", "family_exposure"} and index < len(data.family_names):
        return data.family_names[index]
    if site == "model_exposure" and index < len(data.model_names):
        return data.model_names[index]
    return str(index)
