import h5py
import numpy as np


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 3, 2, 1))
        train_label = np.transpose(label, (0, 3, 2, 1))
        return train_data, train_label
