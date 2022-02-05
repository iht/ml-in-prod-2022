gcloud ai hp-tuning-jobs create \
  --region=europe-west4 \
  --display-name=ht-my-first-model \
  --max-trial-count=56 \
  --parallel-trial-count=2 \
  --config=./bin/training_config_ht.yaml
