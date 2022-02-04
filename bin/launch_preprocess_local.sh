#!/bin/bash

# exit when any command fails
set -e

INPUT_DATA=data/aclImdb/
OUTPUT_DATA=output/

python pipeline/preprocess_pipeline.py \
  --runner=DirectRunner \
  --temp_location=/tmp/ \
  --data-location=$INPUT_DATA \
  --output-location=$OUTPUT_DATA
