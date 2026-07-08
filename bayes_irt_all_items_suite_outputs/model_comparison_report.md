# Bayesian IRT Model Comparison

Runs completed: {'ok': 45}

## Top models by held-out mean log likelihood

| run_name | model_kind | prior_name | holdout_mean_log_likelihood | holdout_brier | holdout_accuracy_0_5 |
| --- | --- | --- | --- | --- | --- |
| dmixture_family_2pl__wide_difficulty | dmixture_family_2pl | wide_difficulty | -0.351652 | 0.112065 | 0.838539 |
| dmixture_family_2pl__wide_discrimination | dmixture_family_2pl | wide_discrimination | -0.351656 | 0.112065 | 0.838534 |
| dmixture_family_2pl__wide_exposure | dmixture_family_2pl | wide_exposure | -0.351656 | 0.112067 | 0.838533 |
| dmixture_family_2pl__moderate_contamination | dmixture_family_2pl | moderate_contamination | -0.351656 | 0.112067 | 0.838533 |
| dmixture_family_2pl__baseline | dmixture_family_2pl | baseline | -0.351656 | 0.112067 | 0.838533 |
| dmixture_family_2pl__rare_contamination | dmixture_family_2pl | rare_contamination | -0.351656 | 0.112067 | 0.838533 |
| dmixture_family_2pl__tight_exposure | dmixture_family_2pl | tight_exposure | -0.351656 | 0.112067 | 0.838533 |
| dmixture_family_2pl__tight_difficulty | dmixture_family_2pl | tight_difficulty | -0.351679 | 0.112074 | 0.838519 |
| dmixture_family_2pl__tight_discrimination | dmixture_family_2pl | tight_discrimination | -0.35168 | 0.112094 | 0.838478 |
| dmixture_2pl__tau_999 | dmixture_2pl | tau_999 | -0.351747 | 0.11201 | 0.838408 |

## Best prior variant within each model family

| run_name | model_kind | prior_name | holdout_mean_log_likelihood | holdout_brier | holdout_accuracy_0_5 |
| --- | --- | --- | --- | --- | --- |
| dmixture_family_2pl__wide_difficulty | dmixture_family_2pl | wide_difficulty | -0.351652 | 0.112065 | 0.838539 |
| dmixture_2pl__tau_999 | dmixture_2pl | tau_999 | -0.351747 | 0.11201 | 0.838408 |
| sparse_3pl__tight_discrimination | sparse_3pl | tight_discrimination | -0.362337 | 0.115711 | 0.832396 |
| hier_2pl__tight_discrimination | hier_2pl | tight_discrimination | -0.362893 | 0.115964 | 0.832401 |
| 2pl__tight_discrimination | 2pl | tight_discrimination | -0.363162 | 0.116056 | 0.831989 |
| hier_1pl__tight_difficulty | hier_1pl | tight_difficulty | -0.373823 | 0.118675 | 0.830427 |
| 1pl__wide_difficulty | 1pl | wide_difficulty | -0.373953 | 0.118707 | 0.830841 |

## Stable top preknowledge items across mixture runs, top 100

| item | n_runs | n_rows | mean_rank | best_rank | mean_preknowledge_probability | min_preknowledge_probability |
| --- | --- | --- | --- | --- | --- | --- |
| 767 | 21 | 82 | 49.1098 | 2 | 1 | 1 |
| 161 | 21 | 70 | 50.1857 | 2 | 1 | 1 |
| 599 | 21 | 60 | 58.3833 | 4 | 1 | 1 |
| 1037 | 21 | 58 | 49.3103 | 4 | 1 | 1 |
| 637 | 20 | 104 | 49.3558 | 1 | 1 | 1 |
| 49 | 20 | 67 | 52.8209 | 1 | 1 | 1 |
| 1313 | 20 | 66 | 48.7727 | 1 | 1 | 1 |
| 1259 | 20 | 60 | 50.5 | 3 | 1 | 1 |
| 89 | 20 | 59 | 44.8305 | 1 | 1 | 1 |
| 331 | 19 | 42 | 54.4048 | 6 | 1 | 1 |

## Stable top preknowledge families across mixture runs, top 100

| family | n_runs | n_rows | mean_rank | best_rank | mean_preknowledge_probability | min_preknowledge_probability |
| --- | --- | --- | --- | --- | --- | --- |
| albaddawi/deepcode-7b-aurora | 21 | 119 | 45.8067 | 2 | 1 | 1 |
| 4season/alignment-model-test | 19 | 40 | 45.275 | 1 | 1 | 1 |
| 0-hero/matter-0-2-32b | 19 | 33 | 51.9091 | 9 | 1 | 1 |
| aa051611/a0118 | 17 | 42 | 55 | 1 | 1 | 1 |
| causallm/35b-beta2ep | 16 | 35 | 45.9714 | 4 | 1 | 1 |
| azure99/blossom-1-yi-34b | 16 | 35 | 60.1143 | 4 | 1 | 1 |
| apmic/caigun-lora-model-34b | 15 | 25 | 42.52 | 1 | 1 | 1 |
| changgil/k2s3-solar-11b-0 | 15 | 23 | 44.3478 | 1 | 1 | 1 |
| chih-hung/llama-2-13b-finetune4 | 14 | 30 | 55.9333 | 3 | 1 | 1 |
| causallm/7b-dpo-alpha | 14 | 20 | 37.7 | 3 | 1 | 1 |
