export AIP_MODEL_DIR=training-output/

mkdir -p $AIP_MODEL_DIR

python trainer/task.py \
    --data-location=gs://ferrous-linker-339609/output/ \
    --tft-location=gs://ferrous-linker-339609/output/transform_fn/ \
    --batch-size=1024 \
    --epochs=10
