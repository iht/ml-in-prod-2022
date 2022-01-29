import os

from typing import List, Dict

import apache_beam as beam
from apache_beam import PCollection, Pipeline
from apache_beam.options.pipeline_options import PipelineOptions


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


def run_pipeline(argv: List[str], data_location: str, output_location: str):
    options = PipelineOptions(argv)

    with beam.Pipeline(options=options) as p:
        train_set, test_set = get_train_and_test(p, data_location)
