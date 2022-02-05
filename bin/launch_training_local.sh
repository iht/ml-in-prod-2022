export AIP_MODEL_DIR=training-output/

mkdir -p $AIP_MODEL_DIR

python trainer/task.py \
    --data-location=gs://class-ml-prod-2022/data/preprocessed/tf_record_with_text/ \
    --tft-location=gs://class-ml-prod-2022/data/preprocessed/tf_record_with_text/transform_fn/ \
    --batch-size=4096 \
    --epochs=10
