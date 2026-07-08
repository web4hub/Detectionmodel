"""Command line interface for Bayesian GSM8K IRT fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayes_irt_gsm8k.data import load_gsm8k_response_data, load_item_list, response_item_summary
from bayes_irt_gsm8k.fit import FitConfig, fit_svi, save_fit_artifacts
from bayes_irt_gsm8k.models import IRTModelConfig, SUPPORTED_MODEL_KINDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit Bayesian IRT models to GSM8K response data.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing GSM8K artifacts.")
    parser.add_argument("--matrix-name", default="gsm8k_matrix_filtered.parquet", help="Response matrix parquet file.")
    parser.add_argument("--model", choices=SUPPORTED_MODEL_KINDS, default="2pl", help="Model family to fit.")
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
    parser.add_argument("--max-models", type=int, default=None, help="Randomly keep at most this many models.")
    parser.add_argument("--max-items", type=int, default=None, help="Randomly keep at most this many items.")
    parser.add_argument("--max-obs", type=int, default=None, help="Randomly keep at most this many cells.")
    parser.add_argument("--steps", type=int, default=1_000, help="SVI optimization steps.")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Observation minibatch size.")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--tau", type=float, default=0.99, help="Success probability in deterministic mixture class.")
    parser.add_argument(
        "--base-contamination-rate",
        type=float,
        default=0.01,
        help="Prior mean contamination rate for deterministic mixture models.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bayes_irt_outputs"), help="Where artifacts are saved.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
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
    model_config = IRTModelConfig(
        kind=args.model,
        tau=args.tau,
        base_contamination_rate=args.base_contamination_rate,
    )
    fit_config = FitConfig(
        steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        progress=not args.no_progress,
    )
    fit = fit_svi(data, model_config, fit_config)

    output_dir = args.output_dir / args.model
    save_fit_artifacts(fit, data, output_dir)
    response_item_summary(data).to_csv(output_dir / "selected_item_summary.csv", index=False)
    (output_dir / "item_selection_config.txt").write_text(
        "\n".join(
            [
                f"item_mode={item_mode}",
                f"items_file={args.items_file or ''}",
                f"exclude_saturated_items={args.exclude_saturated_items}",
                f"min_item_pass_rate={args.min_item_pass_rate}",
                f"max_item_pass_rate={args.max_item_pass_rate}",
            ]
        )
        + "\n"
    )
    print(f"saved artifacts to {output_dir.resolve()}")


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
