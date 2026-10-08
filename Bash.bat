python3 -m pytest -q
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
