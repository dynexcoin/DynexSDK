__copyright__ = "© Accenture Latvijas filiāle, 2022"

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from operator import mul

class CXRDataset_pca:

    def __init__(self, root_dir, dataset_type='train', dataset_file=None, num_features = -1, include_classes = (0, None)):

        data_dir = os.path.join(root_dir, dataset_file)
        self.pca_data = np.load(data_dir)# , mmap_mode='r'
        self.index_dir = os.path.join(root_dir, dataset_type + '_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, :].to_numpy()[1:9]
        self.classes = self.classes[include_classes[0]:include_classes[1]]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        self.num_features = num_features
        self.include_classes = include_classes

    def __len__(self):
        return len(self.label_index)

    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        label = self.label_index.iloc[idx, 1:9].to_numpy().astype('int')
        data = self.pca_data[idx]
        data = data[:self.num_features]
        label = label[self.include_classes[0]:self.include_classes[1]]

        return data, label, name

def load_data(data_root_dir, pca_features, encoding=None):
    data_root_dir = Path(data_root_dir)

    datasets = {
        'train': CXRDataset_pca(data_root_dir, dataset_type='train', dataset_file='train.npy', num_features=pca_features),
        'val': CXRDataset_pca(data_root_dir, dataset_type='val', dataset_file='val.npy', num_features=pca_features)
    }

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = datasets['train'].classes

    train_dataset = datasets['train']
    val_dataset = datasets['val']

    X = train_dataset.pca_data[:,:pca_features]
    y = train_dataset.label_index[train_dataset.classes]
    X_val = val_dataset.pca_data[:,:pca_features]
    y_val = val_dataset.label_index[train_dataset.classes]
    
    if encoding is None:
        return X, y, X_val, y_val

    if (data_root_dir / f"encoded_{encoding.__name__}.npy").is_file():
        with open(data_root_dir / f"encoded_{encoding.__name__}.npy", "rb") as f:
            X_encoded = pickle.load(f)
        with open(data_root_dir / f"encoded_val_{encoding.__name__}.npy", "rb") as f:
            X_val_encoded = pickle.load(f)
    else:
        X_encoded = encode(X, encoding)
        X_val_encoded = encode(X_val, encoding)
        with open(data_root_dir / f"encoded_{encoding.__name__}.npy", "wb") as f:
            pickle.dump(X_encoded, f)
        with open(data_root_dir / f"encoded_val_{encoding.__name__}.npy", "wb") as f:
            pickle.dump(X_val_encoded, f)
    
    return X_encoded, y, X_val_encoded, y_val

def binary(x):
    return [x // 2, x % 2]

def binary_three_bits(x):
    return [x//4, (x%4)//2, x%2]

def encode(x, func):
    num_rows = x.shape[0]
    return np.array(list(map(func, x.flatten().astype(np.int8)))).reshape(num_rows, -1)

def l2_regularizer(weight_cost=0.0002):
    return partial(mul, weight_cost)
