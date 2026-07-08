"""End-to-end model-comparison and prior-sensitivity suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import re
import time
import traceback

import pandas as pd

from bayes_irt_gsm8k.data import (
    load_gsm8k_response_data,
    load_item_list,
    response_item_summary,
    split_response_data,
)
from bayes_irt_gsm8k.evaluation import calibration_table, predictive_metrics
from bayes_irt_gsm8k.fit import FitConfig, fit_svi, parameter_summary, top_preknowledge_probabilities
from bayes_irt_gsm8k.models import IRTModelConfig, SUPPORTED_MODEL_KINDS


DEFAULT_SUITE_MODELS = SUPPORTED_MODEL_KINDS


@dataclass(frozen=True)
class ModelSpec:
    """A single model/prior run in the suite."""

    run_name: str
    prior_name: str
    model_config: IRTModelConfig


@dataclass(frozen=True)
class SuiteConfig:
    """Configuration for model-comparison and sensitivity runs."""

    output_dir: Path
    model_kinds: tuple[str, ...] = DEFAULT_SUITE_MODELS
    sensitivity: str = "standard"
    steps: int = 1_000
    learning_rate: float = 0.03
    batch_size: int | None = 50_000
    seed: int = 123
    device: str = "cpu"
    holdout_fraction: float = 0.1
    progress: bool = False
    continue_on_error: bool = True
    top_n: int = 1_000
    item_mode: str = "all"
    items_file: str | None = None
    exclude_saturated_items: bool = False
    min_item_pass_rate: float | None = None
    max_item_pass_rate: float | None = None

    def __post_init__(self) -> None:
        if self.sensitivity not in {"baseline", "standard", "expanded"}:
            raise ValueError("sensitivity must be baseline, standard, or expanded")
        if self.item_mode not in {"all", "file"}:
            raise ValueError("item_mode must be all or file")


def build_model_specs(
    model_kinds: tuple[str, ...] = DEFAULT_SUITE_MODELS,
    *,
    sensitivity: str = "standard",
) -> list[ModelSpec]:
    """Build model/prior variants for the requested sensitivity level."""

    specs: list[ModelSpec] = []
    for kind in model_kinds:
        base = IRTModelConfig(kind=kind)
        for prior_name, overrides in _prior_variants(kind, sensitivity):
            config = replace(base, **overrides)
            run_name = _safe_name(f"{kind}__{prior_name}")
            specs.append(ModelSpec(run_name=run_name, prior_name=prior_name, model_config=config))
    return specs


def run_suite(data, suite_config: SuiteConfig) -> pd.DataFrame:
    """Run all configured models and save per-run plus aggregate outputs."""

    suite_config.output_dir.mkdir(parents=True, exist_ok=True)
    response_item_summary(data).to_csv(suite_config.output_dir / "selected_item_summary.csv", index=False)
    train_data, holdout_data = split_response_data(
        data,
        holdout_fraction=suite_config.holdout_fraction,
        seed=suite_config.seed,
    )
    specs = build_model_specs(suite_config.model_kinds, sensitivity=suite_config.sensitivity)
    (suite_config.output_dir / "suite_config.json").write_text(
        json.dumps(
            {
                **asdict(suite_config),
                "output_dir": str(suite_config.output_dir),
                "data": {
                    "n_models": data.n_models,
                    "n_items": data.n_items,
                    "n_obs": data.n_obs,
                    "n_families": data.n_families,
                    "train_obs": train_data.n_obs,
                    "holdout_obs": holdout_data.n_obs,
                },
                "runs": [spec.run_name for spec in specs],
            },
            indent=2,
        )
    )

    rows = []
    for index, spec in enumerate(specs, start=1):
        run_dir = suite_config.output_dir / spec.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(specs)}] {spec.run_name}")
        started = time.time()
        row = _base_summary_row(spec, suite_config, data, train_data, holdout_data, run_dir)
        try:
            fit_config = FitConfig(
                steps=suite_config.steps,
                learning_rate=suite_config.learning_rate,
                batch_size=suite_config.batch_size,
                seed=suite_config.seed,
                device=suite_config.device,
                progress=suite_config.progress,
            )
            fit = fit_svi(train_data, spec.model_config, fit_config)
            losses = pd.DataFrame({"step": range(1, len(fit.losses) + 1), "loss": fit.losses})
            losses.to_csv(run_dir / "losses.csv", index=False)
            parameter_summary(fit, train_data).to_csv(run_dir / "parameter_summary.csv", index=False)

            holdout_metrics = predictive_metrics(fit, holdout_data)
            train_metrics = predictive_metrics(fit, train_data)
            calibration_table(fit, holdout_data).to_csv(run_dir / "holdout_calibration.csv", index=False)

            if spec.model_config.kind in {"dmixture_2pl", "dmixture_family_2pl"}:
                top_preknowledge_probabilities(fit, data, top_n=suite_config.top_n).to_csv(
                    run_dir / "top_preknowledge_probabilities_full.csv",
                    index=False,
                )

            run_payload = {
                "run_name": spec.run_name,
                "prior_name": spec.prior_name,
                "model": asdict(spec.model_config),
                "fit": asdict(fit_config),
                "train_metrics": train_metrics.as_dict(),
                "holdout_metrics": holdout_metrics.as_dict(),
            }
            (run_dir / "run_config.json").write_text(json.dumps(run_payload, indent=2))

            row.update(
                {
                    "status": "ok",
                    "final_loss": fit.losses[-1],
                    "min_loss": min(fit.losses),
                    "tail_100_loss_mean": sum(fit.losses[-100:]) / min(100, len(fit.losses)),
                    **{f"train_{k}": v for k, v in train_metrics.as_dict().items()},
                    **{f"holdout_{k}": v for k, v in holdout_metrics.as_dict().items()},
                }
            )
        except Exception as exc:  # pragma: no cover - exercised in real failure cases.
            row.update({"status": "error", "error": repr(exc)})
            (run_dir / "error.txt").write_text(traceback.format_exc())
            if not suite_config.continue_on_error:
                rows.append(row)
                _write_summary(rows, suite_config.output_dir)
                raise
        finally:
            row["elapsed_seconds"] = time.time() - started
            rows.append(row)
            _write_summary(rows, suite_config.output_dir)

    summary = _write_summary(rows, suite_config.output_dir)
    _write_ranked_summary(summary, suite_config.output_dir)
    return summary


def _prior_variants(kind: str, sensitivity: str) -> list[tuple[str, dict[str, float]]]:
    variants: list[tuple[str, dict[str, float]]] = [("baseline", {})]
    if sensitivity == "baseline":
        return variants

    variants.extend(
        [
            ("tight_difficulty", {"difficulty_sd": 1.0}),
            ("wide_difficulty", {"difficulty_sd": 3.0}),
        ]
    )

    if kind not in {"1pl", "hier_1pl"}:
        variants.extend(
            [
                ("tight_discrimination", {"discrimination_log_sd": 0.3}),
                ("wide_discrimination", {"discrimination_log_sd": 0.8}),
            ]
        )

    if kind == "sparse_3pl":
        variants.extend(
            [
                ("very_sparse_c", {"lower_asymptote_alpha": 0.25, "lower_asymptote_beta": 50.0}),
                ("less_sparse_c", {"lower_asymptote_alpha": 1.0, "lower_asymptote_beta": 20.0}),
            ]
        )

    if kind in {"dmixture_2pl", "dmixture_family_2pl"}:
        variants.extend(
            [
                ("rare_contamination", {"base_contamination_rate": 0.001}),
                ("moderate_contamination", {"base_contamination_rate": 0.05}),
                ("tau_95", {"tau": 0.95}),
                ("tau_999", {"tau": 0.999}),
                ("tight_exposure", {"exposure_effect_sd": 0.4}),
                ("wide_exposure", {"exposure_effect_sd": 1.25}),
            ]
        )
        if sensitivity == "expanded":
            variants.extend(
                [
                    ("rare_tau_95", {"base_contamination_rate": 0.001, "tau": 0.95}),
                    ("moderate_tau_999", {"base_contamination_rate": 0.05, "tau": 0.999}),
                    (
                        "wide_exposure_moderate",
                        {"exposure_effect_sd": 1.25, "base_contamination_rate": 0.05},
                    ),
                ]
            )

    if sensitivity == "expanded" and kind not in {"1pl", "hier_1pl"}:
        variants.extend(
            [
                (
                    "wide_diff_wide_disc",
                    {"difficulty_sd": 3.0, "discrimination_log_sd": 0.8},
                ),
                (
                    "tight_diff_tight_disc",
                    {"difficulty_sd": 1.0, "discrimination_log_sd": 0.3},
                ),
            ]
        )

    return variants


def _base_summary_row(
    spec: ModelSpec,
    suite_config: SuiteConfig,
    data,
    train_data,
    holdout_data,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "run_name": spec.run_name,
        "model_kind": spec.model_config.kind,
        "prior_name": spec.prior_name,
        "status": "started",
        "n_models": data.n_models,
        "n_items": data.n_items,
        "n_families": data.n_families,
        "n_obs": data.n_obs,
        "train_obs": train_data.n_obs,
        "holdout_obs": holdout_data.n_obs,
        "steps": suite_config.steps,
        "batch_size": suite_config.batch_size,
        "seed": suite_config.seed,
        "run_dir": str(run_dir),
        "error": "",
    }


def _write_summary(rows: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def _write_ranked_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if "holdout_mean_log_likelihood" not in summary.columns:
        return
    ranked = summary[summary["status"] == "ok"].copy()
    if ranked.empty:
        return
    ranked = ranked.sort_values(
        ["holdout_mean_log_likelihood", "holdout_brier"],
        ascending=[False, True],
    )
    ranked.to_csv(output_dir / "summary_ranked.csv", index=False)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Bayesian IRT model-comparison and prior sensitivity suite.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing GSM8K artifacts.")
    parser.add_argument("--matrix-name", default="gsm8k_matrix_filtered.parquet", help="Response matrix parquet file.")
    parser.add_argument(
        "--item-mode",
        choices=["all", "file"],
        default=None,
        help="Use all matrix items, or item ids from --items-file. Defaults to file when --items-file is set.",
    )
    parser.add_argument("--items-file", type=Path, default=None, help="CSV/text file specifying item ids for file mode.")
    parser.add_argument(
        "--exclude-saturated-items",
        action="store_true",
        help="Drop items with pass rate exactly 0 or 1 in the loaded matrix.",
    )
    parser.add_argument(
        "--min-item-pass-rate",
        type=float,
        default=None,
        help="Drop items with pass rate below this threshold after explicit model/item filters.",
    )
    parser.add_argument(
        "--max-item-pass-rate",
        type=float,
        default=None,
        help="Drop items with pass rate above this threshold after explicit model/item filters.",
    )
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODEL_KINDS, default=list(DEFAULT_SUITE_MODELS))
    parser.add_argument("--sensitivity", choices=["baseline", "standard", "expanded"], default="standard")
    parser.add_argument("--max-models", type=int, default=None, help="Randomly keep at most this many models.")
    parser.add_argument("--max-items", type=int, default=None, help="Randomly keep at most this many items.")
    parser.add_argument("--max-obs", type=int, default=None, help="Randomly keep at most this many cells before splitting.")
    parser.add_argument("--steps", type=int, default=1_000, help="SVI optimization steps per run.")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Observation minibatch size.")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--holdout-fraction", type=float, default=0.1, help="Observation holdout fraction.")
    parser.add_argument("--output-dir", type=Path, default=Path("bayes_irt_suite_outputs"))
    parser.add_argument("--top-n", type=int, default=1_000, help="Top preknowledge rows to save for mixture models.")
    parser.add_argument("--progress", action="store_true", help="Show per-model SVI progress bars.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the suite at the first failed run.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    item_ids, item_mode = _resolve_item_selection(args, parser)
    data = load_gsm8k_response_data(
        args.data_dir,
        matrix_name=args.matrix_name,
        item_ids=item_ids,
        exclude_saturated_items=args.exclude_saturated_items,
        min_item_pass_rate=args.min_item_pass_rate,
        max_item_pass_rate=args.max_item_pass_rate,
        max_models=args.max_models,
        max_items=args.max_items,
        max_obs=args.max_obs,
        seed=args.seed,
    )
    config = SuiteConfig(
        output_dir=args.output_dir,
        model_kinds=tuple(args.models),
        sensitivity=args.sensitivity,
        steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        holdout_fraction=args.holdout_fraction,
        progress=args.progress,
        continue_on_error=not args.stop_on_error,
        top_n=args.top_n,
        item_mode=item_mode,
        items_file=str(args.items_file) if args.items_file else None,
        exclude_saturated_items=args.exclude_saturated_items,
        min_item_pass_rate=args.min_item_pass_rate,
        max_item_pass_rate=args.max_item_pass_rate,
    )
    summary = run_suite(data, config)
    ok = int((summary["status"] == "ok").sum())
    print(f"completed {ok}/{len(summary)} runs; outputs saved to {config.output_dir.resolve()}")


def _resolve_item_selection(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[list[int | str] | None, str]:
    item_mode = args.item_mode or ("file" if args.items_file else "all")
    if item_mode == "file":
        if args.items_file is None:
            parser.error("--item-mode file requires --items-file")
        return load_item_list(args.items_file), "file"
    if args.items_file is not None:
        parser.error("--items-file can only be used with --item-mode file")
    return None, "all"


if __name__ == "__main__":
    main()
