# Technical Assessment of the GSM8K IRT Contamination Proposal

## Executive verdict

The proposal is mathematically sound as a screening method, provided it is interpreted narrowly. Modeling benchmark performance with item response theory (IRT), then looking for model-item successes that are much larger than the fitted model predicts, is a principled way to search for possible item preknowledge. This is closely aligned with psychometric test-security logic: preknowledge is not an item-only property or a test-taker-only property, but an anomalous interaction between a test-taker and a specific item.

The current implementation should not be treated as a confirmatory statistical test or as proof of contamination. The residual scores are useful triage statistics, not calibrated p-values. The method depends on IRT assumptions, many model-item tests are performed, fitted parameters are reused to compute residuals, and recurring item flags can arise from item misfit, answer-parsing artifacts, shared model families, or narrow legitimate skill. The proposed perturbation stage is therefore essential: the IRT screen can nominate suspects, but perturbation is what could turn a suspicion into evidence.

In short: the mathematical idea is good; the present notebook is a reasonable first-stage filter; the evidential claim should remain modest until the method is calibrated, stress-tested, and followed by controlled perturbation experiments.

## The proposal

The notebook fits a two-parameter logistic IRT model to a GSM8K response matrix. Rows are models, columns are GSM8K items, and each entry is a binary correctness indicator.

Let

```text
U_mi in {0, 1}
```

denote whether model `m` answered item `i` correctly. The 2PL model assumes

```text
Pr(U_mi = 1 | theta_m, a_i, b_i) = sigmoid(a_i * (theta_m - b_i))
```

where:

- `theta_m` is the latent ability of model `m`,
- `b_i` is the difficulty of item `i`,
- `a_i` is the discrimination of item `i`,
- `sigmoid(x) = 1 / (1 + exp(-x))`.

The contamination hypothesis is then framed as a local deviation from this model. If a model has seen a particular item, it may answer that item correctly even when its fitted ability and the item's fitted difficulty imply a low success probability.

The notebook operationalizes this by computing a standardized residual:

```text
Z_mi = (U_mi - P_mi) / sqrt(P_mi * (1 - P_mi))
```

where `P_mi` is the fitted 2PL probability. A large positive residual means the model got an item right despite the model assigning low probability to that success.

The screen then focuses on:

- hard items, defined as the top quartile of effective difficulty,
- positive residuals only,
- models at or above median estimated ability,
- cells with residuals at least `4.0`,
- recurring item-level hits across multiple models.

The produced artifacts show:

- filtered response matrix: 6,014 models by 1,214 retained GSM8K items,
- 848 flagged model-item cells,
- 424 models with at least one flagged cell,
- 63 distinct flagged items,
- 57 items flagged by at least three models.

## Why the proposal is mathematically coherent

### 1. The unit of suspicion is correct

Benchmark contamination is naturally a model-item phenomenon. A model may have seen item `i` but not item `j`; another model may have seen neither. Looking only at item-level pass rates or model-level aggregate accuracy loses this interaction structure.

The residual

```text
U_mi - P_mi
```

is therefore a good primitive object. It asks: after accounting for model ability and item difficulty, is this specific success surprising?

This is a more appropriate quantity than raw correctness, raw item pass rate, or raw model accuracy.

### 2. The 2PL model is a defensible baseline

The 2PL model is better suited than a one-parameter Rasch model because GSM8K items likely differ not only in difficulty but also in how strongly they separate high-performing and low-performing models. The discrimination parameter `a_i` allows items to vary in slope. This matters because an easy but low-discrimination item and an easy high-discrimination item should not induce the same residual behavior.

The saved item parameters look numerically reasonable:

```text
median discrimination: about 1.83
difficulty range: about -2.20 to 1.97
```

The notebook also reports a strong first-to-second eigenvalue ratio of about 11.9 in the item-correlation scree check. That does not prove unidimensionality, but it supports the use of a single dominant ability dimension as a first approximation.

### 3. The residual direction matches the contamination hypothesis

The method looks for large positive residuals, not just large absolute residuals.

That is mathematically appropriate. Contamination should mainly show up as unexpectedly correct answers, not unexpectedly wrong answers. Negative residuals may indicate other problems, such as brittle reasoning, prompt sensitivity, or item misfit, but they are not direct evidence of preknowledge.

### 4. Hard-item conditioning is directionally sensible

The method emphasizes hard items because preknowledge is most visible when the baseline probability of success is low. If `P_mi` is already 0.85, a correct answer carries little evidence. If `P_mi` is 0.03 and the model succeeds, the observation is much more suspicious.

The flagged cells have median predicted probabilities around 0.05, which is exactly the intended regime for detecting surprising successes.

### 5. The proposed perturbation stage is the right confirmatory idea

The notebook does not claim that a residual flag proves contamination. It proposes perturbing flagged items and rerunning suspect models.

This is the key mathematically sound move. If a model solves the original item but fails a matched numerical twin, that is much stronger evidence of memorization than the residual alone. If it solves both, the residual was likely genuine skill or some broader competence not captured by the fitted IRT model.

## What the current screen actually establishes

The current analysis establishes that some model-item successes are extreme relative to the fitted 2PL baseline. It does not establish why they are extreme.

The screen can support statements like:

```text
Under this fitted 2PL model, these model-item successes have unusually large positive residuals, especially on hard items.
```

It should not yet support statements like:

```text
These items are contaminated.
These models memorized these items.
These items leaked into training data.
```

Those stronger claims require either perturbation evidence, known contamination labels, external data-provenance evidence, or a calibrated statistical model whose false-positive behavior is understood.

## Main mathematical and statistical issues

### 1. IRT assumptions may be violated

The 2PL model assumes a single latent ability dimension, monotonic item response curves, and local independence conditional on ability. GSM8K performance by LLMs may violate these assumptions.

Possible violations include:

- items requiring different skills, such as arithmetic, reading comprehension, equation setup, or multi-step planning,
- strong family-level dependence between related model checkpoints,
- local dependence among similar GSM8K items,
- prompt-format artifacts,
- answer-extraction artifacts,
- benchmark-specific fine-tuning behavior.

If these effects are present, a residual may reflect model misfit rather than preknowledge.

### 2. Residuals are computed using parameters fitted on the same data

The same responses are used to estimate model ability, item difficulty, item discrimination, and residual surprise. This creates dependence between the fitted probability `P_mi` and the observation `U_mi`.

For a single cell, a correct answer slightly increases the fitted ability of the model and may affect the fitted item parameters. This usually makes the cell less surprising than it would be under a leave-one-out fit, but the exact effect depends on the item and model.

The result is that `Z_mi` should not be interpreted as exactly standard normal. It is a diagnostic residual, not a clean test statistic.

### 3. Multiple testing is substantial

The filtered matrix has roughly

```text
6014 * 1214 = 7,300,996
```

model-item cells. Even after restricting to plausible models and hard items, the effective number of tested cells is roughly:

```text
3007 models * about 304 hard items = about 914,000 tests
```

For a one-sided standard normal threshold of `Z >= 4`, the nominal tail probability is about `3.17e-5`. Under independence and perfect calibration, this restricted search would still produce about:

```text
914,000 * 3.17e-5 = about 29
```

false positives by chance. The observed 848 flagged cells is far above that simple null expectation, which is interesting, but the independence and calibration assumptions are not guaranteed. Dependence between related models and item misfit can create clusters of flags without contamination.

### 4. The preknowledge score is a heuristic, not a likelihood ratio

The notebook defines a model-level score by summing capped positive residuals on hard items. This is a sensible ranking statistic, but it is not derived as an optimal test under an explicit alternative model.

Consequences:

- scores are not p-values,
- the cap at 4.0 is arbitrary,
- the hard-item threshold is arbitrary,
- the median-ability plausibility filter is arbitrary,
- scores may depend on missingness or uneven item exposure,
- models with many mildly surprising successes can outrank models with fewer but more extreme successes.

That does not make the score useless. It only means the score should be used for prioritization, not inference.

### 5. Recurring item flags can mean item misfit

The strongest item-level candidates are items flagged across many models. That is useful for triage, but it has an ambiguity:

```text
many models unexpectedly solve item i
```

could mean:

- item `i` is contaminated,
- item `i` is easier for certain model families than the one-dimensional IRT model predicts,
- item `i` has an answer-pattern or parsing artifact,
- item `i` is overrepresented in similar training data without being exactly memorized,
- item `i` belongs to a subtype not captured by the global ability dimension,
- item `i` has a flawed difficulty estimate.

This is why recurring flags are candidates, not findings.

### 6. Model dependence is not handled

The response matrix contains many related models: base models, instruction-tuned variants, DPO variants, quantizations, merges, and checkpoints. Treating all model rows as independent test-takers can overstate the evidence for recurring item-level anomalies.

If 20 related models all inherit the same capability or data exposure from a shared ancestor, this is not the same evidential weight as 20 independent models from unrelated training pipelines.

### 7. Perturbation is necessary but delicate

The proposed numeric-twin perturbation is the right next step, but it must preserve item difficulty. If the perturbed version is accidentally harder, easier, more awkwardly worded, or less natural than the original, failures on the twin may reflect distribution shift rather than memorization.

The perturbation stage therefore needs validation, ideally by testing twins on a broad set of non-suspect models and checking whether original and twin difficulty are matched.

### 8. The notebook has reproducibility issues

The saved notebook references helper names that are not defined in the notebook file:

```text
pick
ordered_ids
norm
allp
extract_target
```

The generated CSV outputs exist, so the analysis was evidently run in some working state, but the notebook as saved may not execute from top to bottom without reconstructing these helpers. This is not a mathematical flaw in the proposal, but it is a practical issue for auditability.

## Ways to make the proposal stronger

### 1. Use cross-fitted or leave-one-item-out residuals

Estimate a model's ability without using the target item, then compute the residual for that target item. For each cell, replace:

```text
P_mi = P(U_mi = 1 | theta_m fitted using all items)
```

with an approximation to:

```text
P_mi^LOO = P(U_mi = 1 | theta_m fitted without item i)
```

This makes residuals cleaner and reduces self-influence. A practical compromise is K-fold cross-fitting over items.

### 2. Calibrate the null distribution by parametric bootstrap

Simulate response matrices from the fitted 2PL model, rerun the same residual screen, and measure how many cells, models, and recurring items are flagged under the null.

This would answer questions like:

- How many `Z >= 4` cells should this pipeline produce by chance?
- How many recurring items should appear under model misfit alone?
- Is 848 flagged cells extreme relative to the fitted null?
- Is 57 recurring items extreme?

This is probably the single most useful mathematical improvement.

### 3. Add posterior predictive checks

Check whether the 2PL model reproduces important observed summaries:

- item pass-rate distribution,
- model accuracy distribution,
- residual distribution by predicted probability,
- residual distribution by model family,
- residual distribution by item difficulty,
- number of extreme positive residuals,
- number of recurring item-level flags.

If the fitted model cannot reproduce these summaries under non-contamination simulation, the residual screen needs adjustment before being interpreted as contamination evidence.

### 4. Normalize model-level scores by exposure

If missingness differs by model or item, normalize the preknowledge score by the number of eligible hard items observed. For example:

```text
S_m = sum_i max(0, capped Z_mi) / sqrt(number of eligible hard items for model m)
```

or report both the sum and an exposure-normalized score.

### 5. Control for model families

Group related models by base model, organization, architecture, or lineage where possible. Then report item recurrence both:

- across all model rows,
- across independent model families.

An item flagged by 30 variants of one lineage is less compelling than an item flagged by 30 unrelated lineages.

### 6. Use empirical false-discovery control

After bootstrap calibration, set thresholds based on desired false discovery rates at the cell level and item level. The current `Z >= 4` threshold is intuitive but arbitrary.

A stronger pipeline would predefine something like:

```text
Flag recurring items at estimated item-level FDR <= 10%.
```

### 7. Compare 1PL, 2PL, 3PL, and multidimensional variants

The 2PL model is a good first choice, but sensitivity checks would be valuable:

- 1PL/Rasch: tests whether item discrimination is driving flags,
- 3PL: allows item-specific lower asymptotes, similar to guessing or shortcut behavior,
- multidimensional IRT: tests whether anomalies disappear when multiple GSM8K skill dimensions are modeled,
- hierarchical IRT: allows family-level effects.

If the same items remain suspicious across model classes, confidence increases.

### 8. Add negative and positive controls

Useful controls include:

- synthetic injected contamination: force selected cells to correct and verify the screen recovers them,
- random label perturbation: verify the screen does not hallucinate structure,
- easy-item controls: verify the method does not overclaim on items where success is expected,
- known duplicated or widely circulated GSM8K-like examples, if any can be documented externally.

These controls would make the detection claims much easier to trust.

### 9. Validate perturbation difficulty

For each numeric twin, estimate whether its difficulty matches the original. A practical design:

1. Generate several twins per candidate item.
2. Run a diverse reference set of non-suspect models.
3. Keep twins whose pass rates are close to the original among reference models.
4. Then evaluate suspect models on original-versus-twin flips.

The target estimand should be something like:

```text
flip_rate = Pr(correct on original, incorrect on matched twin)
```

compared against matched non-suspect controls.

### 10. Turn the heuristic into an explicit mixture model

A more ambitious extension is a latent contamination model:

```text
K_mi ~ Bernoulli(pi_mi)

if K_mi = 0:
    Pr(U_mi = 1) = sigmoid(a_i * (theta_m - b_i))

if K_mi = 1:
    Pr(U_mi = 1) = q_i
```

where `K_mi` is latent preknowledge and `q_i` is a high success probability under preknowledge. This would directly estimate posterior probabilities of preknowledge rather than using residual thresholds.

This is harder to fit and easier to overfit, so it should come after the simpler residual screen is calibrated.

## Bottom line

The proposal is mathematically sound as a first-stage anomaly detector. It uses the right object of analysis, model-item residuals, and the right broad logic, item preknowledge as unexpected success conditional on estimated ability and difficulty.

The current implementation is not yet mathematically sufficient to prove contamination. It should be described as a screening pipeline whose output is a prioritized list of model-item and item-level candidates. To make it stronger, the next work should focus on cross-fitted residuals, bootstrap calibration, model-family dependence, reproducible notebook execution, and carefully validated perturbation twins.

