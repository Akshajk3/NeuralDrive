import os
import torch
import tensorflow as tf
from waymo_open_dataset import e2ed_pb2

def convert_tfrecord(record_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)