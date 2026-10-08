#!/bin/bash
docker run --rm --gpus all --ipc=host \
  -p 30000:30000 \
  lmsysorg/sglang:latest \
  sglang serve MODEL_PATH --host 0.0.0.0 --port 30000
curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=train&offset=0&length=100"
     curl -X GET \
     "https://datasets-server.huggingface.co/splits?dataset=openai%2Fgsm8k"
     curl -X GET \
     "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/train"
