import os

from typing import List, Dict

import apache_beam as beam
from apache_beam import PCollection
from apache_beam.options.pipeline_options import PipelineOptions


def run_pipeline(argv: List[str], data_location: str, output_location: str):
    options = PipelineOptions(argv)

    with beam.Pipeline(options=options) as p:
        # train -
        #       -> pos
        #       -> neg
        # test ....

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

        train_neg_dicts: PCollection[Dict] = train_pos | "Add label train pos" >> beam.Map(
            lambda t: {'text': t, 'target': 0})

        # "Esta peli es genial", "Esta peli vale un montón", ....
        # {'text': "Esta peli es genial", 'target': 1}, ......

        # Lo mismo para train_neg, ....
