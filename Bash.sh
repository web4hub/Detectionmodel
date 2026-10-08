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
