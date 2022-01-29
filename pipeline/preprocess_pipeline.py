import os

from typing import List, Dict

import apache_beam as beam
import tensorflow_transform.beam as tft_beam
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
from apache_beam import PCollection, Pipeline
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions


def get_train_and_test(p: Pipeline, data_location: str) -> (PCollection[Dict], PCollection[Dict]):
    train_pos_location = os.path.join(data_location, "train/pos/")
    train_neg_location = os.path.join(data_location, "train/neg")
    test_pos_location = os.path.join(data_location, "test/pos/")
    test_neg_location = os.path.join(data_location, "test/neg/")

    train_pos: PCollection[str] = p | "Train pos" >> beam.io.ReadFromText(train_pos_location)
    train_neg: PCollection[str] = p | "Train neg" >> beam.io.ReadFromText(train_neg_location)

    test_pos: PCollection[str] = p | "Test pos" >> beam.io.ReadFromText(test_pos_location)
    test_neg: PCollection[str] = p | "Test neg" >> beam.io.ReadFromText(test_neg_location)

    train_pos_dicts: PCollection[Dict] = train_pos | "Add label train pos" >> beam.Map(
        lambda t: {'text': t, 'target': 1})
    train_neg_dicts: PCollection[Dict] = train_neg | "Add label train neg" >> beam.Map(
        lambda t: {'text': t, 'target': 0})

    train_dicts: PCollection[Dict] = (train_pos_dicts, train_neg_dicts) | "Train set" >> beam.Flatten()

    test_pos_dicts = test_pos | "Add label test pos" >> beam.Map(
        lambda t: {'text': t, 'target': 1})
    test_neg_dicts = test_neg | "Add label test neg" >> beam.Map(
        lambda t: {'text': t, 'target': 0})

    test_dicts = (test_pos_dicts, test_neg_dicts) | "Test set" >> beam.Flatten()

    return train_dicts, test_dicts


def preprocessing_fn(inputs):
    # texts = inputs['text']
    # targets = inputs['target']

    outputs = inputs.copy()

    return outputs


def run_pipeline(argv: List[str], data_location: str, output_location: str):
    feature_spec = {
        'text': tf.io.FixedLenFeature([], tf.strings),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }

    metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_as_feature_spec(feature_spec)
    )

    options = PipelineOptions(argv)
    gcp_options = options.view_as(GoogleCloudOptions)
    temp_dir = gcp_options.temp_location

    with beam.Pipeline(options=options) as p, tft_beam.Context(temp_dir=temp_dir):
        train_set, test_set = get_train_and_test(p, data_location)

        transf_train_ds, transform_fn = (train_set, metadata) | "TFT train" >> \
                                        tft_beam.AnalyzeAndTransformDataset(
                                            preprocessing_fn=preprocessing_fn,
                                            output_record_batches=True)

        transf_train_pcoll, _ = transf_train_ds

        test_set_ds = (test_set, metadata)

        transf_test_ds = \
            (test_set_ds, transform_fn) | "TFT test" >> tft_beam.TransformDataset(output_record_batches=True)

        transf_test_pcoll, _ = transf_test_ds