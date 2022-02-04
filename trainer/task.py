"""A simple main file to showcase the template."""

import logging
import os
import sys
from typing import List, Dict, Union
from urllib.parse import urlparse

import tensorflow as tf
import tensorflow_transform as tft
from google.cloud import storage
from google.cloud.storage import Blob


def read_dataset(filenames: List[str], feature_spec) -> tf.data.TFRecordDataset:
    dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda r: tf.io.parse_single_example(r, feature_spec))
    dataset = dataset.map(lambda d: (d['text'], d['target']))

    return dataset


def get_filename_list(file_pattern: str) -> List[str]:
    sc = storage.Client()
    url_parts = urlparse(file_pattern)
    bucket_name = url_parts.hostname
    location = url_parts.path[1:]
    output: List[Blob] = sc.list_blobs(bucket_name, prefix=location)
    paths: List[str] = [f"gs://{b.bucket.name}/{b.name}" for b in output]
    return paths


def train_and_evaluate(data_location: str,
                       tft_location: str):

    train_location = os.path.join(data_location, "train/")
    test_location = os.path.join(data_location, "test/")

    tft_output = tft.TFTransformOutput(tft_location)
    feature_spec = tft_output.transformed_feature_spec()

    filenames_train = [""]  # ???
    filenames_test = [""]  # ???

    train_ds: tf.data.TFRecordDataset = read_dataset(filenames_train, feature_spec)
    test_ds: tf.data.TFRecordDataset = read_dataset(filenames_test, feature_spec)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-location', default=None, required=True)
    parser.add_argument('--tft-location', required=True)

    args = parser.parse_args()

    loglevel = 'INFO'
    logging.basicConfig(stream=sys.stdout, level=loglevel)

    train_and_evaluate(args.data_location, args.tft_location)
