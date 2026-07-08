# IRT-based contamination detection

This project aims to detect benchmark contamination in large language models by treating it as item preknowledge, a concept borrowed from psychometric test security. Under the proposed hypothesis, a model is a suspect on an item when it succeeds far beyond what its latent ability predicts. This repository holds a first stage that screens for such cases on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) using item response theory (IRT), together with the groundwork for a perturbation stage that would confirm them.

## Idea

Contamination is a property of a model-item pair, not of an item alone, in the same way that a student having seen an exam question is a property of that student and that question. The method runs in two stages.

- Screen: First, we fit a 2PL IRT model to a large model-by-item response matrix, then compute the standardized residual for each cell. A large positive residual is a success the item difficulty cannot explain. This is a cheap correlational filter that surfaces suspects.
- Identify: Second, we perturb flagged items while holding its difficulty fixed, then re-run the suspect models. The idea is that a model that passes the original and fails the surface-changed twin may have "memorized" the item. A model that passes both has the underlying skill. This stage is future work.

The screen on its own cannot separate genuine narrow skill from exposure, so no flagged item is a finding until perturbation confirms it.

## Data

- Response data comes from metabench (Kipnis et al.), which assembled item-wise correctness for the Open LLM Leaderboard v1 benchmarks across more than five thousand models. The GSM8K slice is used here. To reproduce the analysis, download data.tar.gz from the metabench Zenodo record 12819251 and extract it to benchmark-data/.
- Question text and gold worked answers come from the GSM8K dataset on Hugging Face.

## Contents

- gsm8k_2pl_irt.ipynb. The analysis notebook, end to end, from loading the matrix through the residual screen and the handoff to perturbation.
- gsm8k_matrix_filtered.parquet. The filtered binary response matrix, 6,014 models by 1,214 items.
- gsm8k_2pl_item_params.csv. Item difficulty and discrimination from the 2PL fit.
- gsm8k_2pl_abilities.csv. Estimated latent ability for each model.
- gsm8k_preknowledge_screen.csv. The one-sided preknowledge score per model.
- gsm8k_flagged_cells.csv. The flagged model-item cells from the screen.
- gsm8k_flagged_with_gold.csv. The recurring flagged items with question text and gold worked answers, which is the substrate for the perturbation stage.
- src/bayes_irt_gsm8k/. Bayesian IRT package using Pyro, including 1PL, 2PL, hierarchical models, sparse 3PL, and deterministic preknowledge mixtures.
- tests/. Lightweight smoke tests for data loading, model construction, SVI fitting, and preknowledge scoring.

## Results so far

- GSM8K survives as a near-unidimensional construct on this pool. The variance filter retains 1,214 of 1,319 items, and the item correlation scree gives a first-to-second eigenvalue ratio of about 11.9.
- The 2PL fit is well behaved, with a median discrimination of 1.83 and item parameters that reproduce observed item difficulty at a correlation of 0.999.
- The screen flags 848 model-item cells across 424 models and 63 distinct items, 57 of which are flagged by three or more models. The recurring items are the strongest contamination candidates.

## Running it

1. Install the dependencies, numpy, pandas, matplotlib, pyarrow, scipy, datasets, and py-irt. Note that `py-irt` needs Python 3.9 to 3.11, since newer Python silently installs an old release without the trainer.
2. Download the metabench data and extract it to benchmark-data/.
3. Open the notebook and run it from top to bottom.

## Bayesian IRT models

The `bayes_irt_gsm8k` package fits Bayesian models with Pyro. Supported model names are:

- `1pl`: Rasch baseline with item difficulty only.
- `hier_1pl`: 1PL with partial pooling by inferred model family.
- `2pl`: item difficulty plus item discrimination.
- `hier_2pl`: 2PL with partial pooling by inferred model family.
- `sparse_3pl`: 2PL plus a lower-asymptote parameter with a strong near-zero prior.
- `dmixture_2pl`: deterministic preknowledge mixture with model and item exposure effects.
- `dmixture_family_2pl`: deterministic preknowledge mixture with model, item, and family exposure effects.

Quick smoke run on a real-data subset:

```bash
PYTHONPATH=src python3 -m bayes_irt_gsm8k.cli \
  --data-dir . \
  --model dmixture_family_2pl \
  --item-mode all \
  --exclude-saturated-items \
  --max-models 100 \
  --max-items 25 \
  --steps 100 \
  --batch-size 5000 \
  --output-dir bayes_irt_outputs
```

Full all-item run, excluding only exactly saturated items:

```bash
PYTHONPATH=src python3 -m bayes_irt_gsm8k.cli \
  --data-dir . \
  --model dmixture_family_2pl \
  --item-mode all \
  --exclude-saturated-items \
  --steps 3000 \
  --batch-size 50000 \
  --output-dir bayes_irt_all_items_outputs
```

For a stricter near-saturation filter, add bounds such as
`--min-item-pass-rate 0.01 --max-item-pass-rate 0.99`.

Full candidate-item run:

```bash
PYTHONPATH=src python3 -m bayes_irt_gsm8k.cli \
  --data-dir . \
  --model dmixture_family_2pl \
  --item-mode file \
  --items-file rigorous_residual_outputs/perturbation_candidate_items.csv \
  --steps 3000 \
  --batch-size 50000 \
  --output-dir bayes_irt_outputs
```

Outputs include `losses.csv`, `parameter_summary.csv`, `config.json`, and, for deterministic mixture models, `top_preknowledge_probabilities.csv`.

Run the full model-comparison and prior-sensitivity suite on all non-saturated items:

```bash
PYTHONPATH=src python3 -m bayes_irt_gsm8k.suite \
  --data-dir . \
  --item-mode all \
  --exclude-saturated-items \
  --sensitivity standard \
  --steps 1000 \
  --batch-size 50000 \
  --holdout-fraction 0.1 \
  --seed 123 \
  --output-dir bayes_irt_all_items_suite_outputs \
  --top-n 1000
```

Run the full model-comparison and prior-sensitivity suite on the candidate items:

```bash
PYTHONPATH=src python3 -m bayes_irt_gsm8k.suite \
  --data-dir . \
  --item-mode file \
  --items-file rigorous_residual_outputs/perturbation_candidate_items.csv \
  --sensitivity standard \
  --steps 1000 \
  --batch-size 50000 \
  --holdout-fraction 0.1 \
  --seed 123 \
  --output-dir bayes_irt_suite_outputs \
  --top-n 1000
```

The suite fits `1pl`, `hier_1pl`, `2pl`, `hier_2pl`, `sparse_3pl`, `dmixture_2pl`, and `dmixture_family_2pl` under multiple prior variants, using the same train/holdout split for all runs. It writes `summary.csv` and `summary_ranked.csv` for held-out predictive comparison.

Generate stability reports from a completed suite:

```bash
PYTHONPATH=src python3 -m bayes_irt_gsm8k.report bayes_irt_suite_outputs --top-k 100
```

Report outputs include `best_by_model.csv`, `mixture_item_stability_top100.csv`, `mixture_model_stability_top100.csv`, `mixture_family_stability_top100.csv`, `mixture_probability_summary_top100.csv`, and `model_comparison_report.md`.

Run smoke tests:

```bash
python3 -m pytest -q
```

## Status

The screen is complete. The perturbation stage, which builds numeric twins from each item's worked solution and re-runs the suspect models, is laid out in the final notebook cell and is the next step. Before twins, the immediate task is a cheap reproduction check that re-runs a few flagged models on the original items and confirms the scores match the stored matrix.
