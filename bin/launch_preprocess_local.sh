#!/bin/bash

# exit when any command fails
set -e

INPUT_DATA=data/
OUTPUT_DATA=output/

python pipeline/preprocess_pipeline.py \
  --runner=DirectRunner \
  --data-location=$INPUT_DATA \
  --output-location=$OUTPUT_DATA
