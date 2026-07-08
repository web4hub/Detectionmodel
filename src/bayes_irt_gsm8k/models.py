"""Pyro model definitions for Bayesian GSM8K IRT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import pyro
import pyro.distributions as dist
import torch

from bayes_irt_gsm8k.data import ResponseData


SUPPORTED_MODEL_KINDS = (
    "1pl",
    "hier_1pl",
    "2pl",
    "hier_2pl",
    "sparse_3pl",
    "dmixture_2pl",
    "dmixture_family_2pl",
)

ModelKind = Literal[
    "1pl",
    "hier_1pl",
    "2pl",
    "hier_2pl",
    "sparse_3pl",
    "dmixture_2pl",
    "dmixture_family_2pl",
]


@dataclass(frozen=True)
class IRTModelConfig:
    """Configuration for Bayesian IRT model structure and priors."""

    kind: ModelKind = "2pl"
    difficulty_sd: float = 2.0
    discrimination_log_mean: float = 0.0
    discrimination_log_sd: float = 0.5
    lower_asymptote_alpha: float = 0.5
    lower_asymptote_beta: float = 30.0
    base_contamination_rate: float = 0.01
    contamination_logit_sd: float = 1.0
    exposure_effect_sd: float = 0.75
    tau: float = 0.99
    probability_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_MODEL_KINDS:
            raise ValueError(f"unknown model kind {self.kind!r}; expected one of {SUPPORTED_MODEL_KINDS}")
        if not 0.0 < self.base_contamination_rate < 1.0:
            raise ValueError("base_contamination_rate must be between 0 and 1")
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must be between 0 and 1")
        if self.probability_epsilon <= 0.0:
            raise ValueError("probability_epsilon must be positive")


def irt_model(data: ResponseData, config: IRTModelConfig, batch_size: int | None = None) -> None:
    """Pyro model used by SVI.

    The contamination mixture models marginalize the latent preknowledge
    indicator analytically, so the model remains differentiable and scalable.
    """

    theta = _sample_ability(data, config)
    difficulty = pyro.sample(
        "difficulty",
        dist.Normal(0.0, config.difficulty_sd).expand([data.n_items]).to_event(1),
    )
    discrimination = _sample_discrimination(data, config)
    lower_asymptote = _sample_lower_asymptote(data, config)
    exposure_terms = _sample_exposure_terms(data, config)

    with pyro.plate("obs", data.n_obs, subsample_size=batch_size) as obs_idx:
        model_idx = data.model_idx[obs_idx]
        item_idx = data.item_idx[obs_idx]
        base_p = base_irt_probability(theta, difficulty, discrimination, model_idx, item_idx)
        probs = apply_response_process(
            base_p=base_p,
            lower_asymptote=lower_asymptote,
            exposure_terms=exposure_terms,
            data=data,
            config=config,
            model_idx=model_idx,
            item_idx=item_idx,
        )
        probs = probs.clamp(config.probability_epsilon, 1.0 - config.probability_epsilon)
        pyro.sample("response", dist.Bernoulli(probs=probs), obs=data.y[obs_idx])


def base_irt_probability(
    theta: torch.Tensor,
    difficulty: torch.Tensor,
    discrimination: torch.Tensor | float,
    model_idx: torch.Tensor,
    item_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute the 1PL/2PL probability before contamination terms."""

    if isinstance(discrimination, torch.Tensor):
        slopes = discrimination[item_idx]
    else:
        slopes = torch.as_tensor(discrimination, dtype=theta.dtype, device=theta.device)
    logits = slopes * (theta[model_idx] - difficulty[item_idx])
    return torch.sigmoid(logits)


def apply_response_process(
    *,
    base_p: torch.Tensor,
    lower_asymptote: torch.Tensor | None,
    exposure_terms: dict[str, torch.Tensor] | None,
    data: ResponseData,
    config: IRTModelConfig,
    model_idx: torch.Tensor,
    item_idx: torch.Tensor,
) -> torch.Tensor:
    """Apply 3PL or deterministic-mixture response process to base IRT rates."""

    if config.kind == "sparse_3pl":
        if lower_asymptote is None:
            raise ValueError("sparse_3pl requires lower_asymptote")
        c_i = lower_asymptote[item_idx]
        return c_i + (1.0 - c_i) * base_p

    if config.kind in {"dmixture_2pl", "dmixture_family_2pl"}:
        if exposure_terms is None:
            raise ValueError(f"{config.kind} requires exposure terms")
        logit_pi = exposure_terms["global_logit_pi"]
        logit_pi = logit_pi + exposure_terms["item_exposure"][item_idx]
        logit_pi = logit_pi + exposure_terms["model_exposure"][model_idx]
        if config.kind == "dmixture_family_2pl":
            family_idx = data.family_idx[model_idx]
            logit_pi = logit_pi + exposure_terms["family_exposure"][family_idx]
        pi = torch.sigmoid(logit_pi)
        return (1.0 - pi) * base_p + pi * config.tau

    return base_p


def _sample_ability(data: ResponseData, config: IRTModelConfig) -> torch.Tensor:
    if config.kind in {"hier_1pl", "hier_2pl"}:
        family_sd = pyro.sample("family_ability_sd", dist.HalfNormal(1.0))
        model_sd = pyro.sample("model_ability_sd", dist.HalfNormal(1.0))
        family_ability = pyro.sample(
            "family_ability",
            dist.Normal(0.0, family_sd).expand([data.n_families]).to_event(1),
        )
        return pyro.sample(
            "theta",
            dist.Normal(family_ability[data.family_idx], model_sd).to_event(1),
        )

    return pyro.sample("theta", dist.Normal(0.0, 1.0).expand([data.n_models]).to_event(1))


def _sample_discrimination(data: ResponseData, config: IRTModelConfig) -> torch.Tensor | float:
    if config.kind in {"1pl", "hier_1pl"}:
        return 1.0
    return pyro.sample(
        "discrimination",
        dist.LogNormal(config.discrimination_log_mean, config.discrimination_log_sd)
        .expand([data.n_items])
        .to_event(1),
    )


def _sample_lower_asymptote(data: ResponseData, config: IRTModelConfig) -> torch.Tensor | None:
    if config.kind != "sparse_3pl":
        return None
    return pyro.sample(
        "lower_asymptote",
        dist.Beta(config.lower_asymptote_alpha, config.lower_asymptote_beta)
        .expand([data.n_items])
        .to_event(1),
    )


def _sample_exposure_terms(
    data: ResponseData,
    config: IRTModelConfig,
) -> dict[str, torch.Tensor] | None:
    if config.kind not in {"dmixture_2pl", "dmixture_family_2pl"}:
        return None

    base_logit = math.log(config.base_contamination_rate / (1.0 - config.base_contamination_rate))
    global_logit_pi = pyro.sample(
        "global_logit_pi",
        dist.Normal(base_logit, config.contamination_logit_sd),
    )
    item_exposure_sd = pyro.sample("item_exposure_sd", dist.HalfNormal(config.exposure_effect_sd))
    model_exposure_sd = pyro.sample("model_exposure_sd", dist.HalfNormal(config.exposure_effect_sd))
    terms = {
        "global_logit_pi": global_logit_pi,
        "item_exposure": pyro.sample(
            "item_exposure",
            dist.Normal(0.0, item_exposure_sd).expand([data.n_items]).to_event(1),
        ),
        "model_exposure": pyro.sample(
            "model_exposure",
            dist.Normal(0.0, model_exposure_sd).expand([data.n_models]).to_event(1),
        ),
    }
    if config.kind == "dmixture_family_2pl":
        family_exposure_sd = pyro.sample("family_exposure_sd", dist.HalfNormal(config.exposure_effect_sd))
        terms["family_exposure"] = pyro.sample(
            "family_exposure",
            dist.Normal(0.0, family_exposure_sd).expand([data.n_families]).to_event(1),
        )
    return terms
