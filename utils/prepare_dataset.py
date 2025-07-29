import os
from pathlib import Path
import numpy as np


def get_split(path):
    dataset_path = Path(path)

    train_file_names = []
    val_file_names = []
    test_file_names = []
    sets = os.listdir(dataset_path)
    for set in sets:
        if set == "train":
            train_file_names = list((dataset_path / (set) / 'images').glob('*'))
        elif set == "test":
            test_file_names = list((dataset_path / (set) / 'images').glob('*'))
        else:
            val_file_names = list((dataset_path / (set) / 'images').glob('*'))
            
    return train_file_names, test_file_names, val_file_names
