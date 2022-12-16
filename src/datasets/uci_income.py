from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from easyfl.datasets import FederatedTensorDataset

from utils import one_hot_encoding, normalize


def get_dataset(is_norm = True, train_size = 0.8, labeled_size = 0.1, num_of_client = 10) -> Tuple[FederatedTensorDataset, FederatedTensorDataset, FederatedTensorDataset]:
    if train_size <= 0.0 or train_size >= 1.0:
        raise ValueError("train_size must be between (0,1)")
    if labeled_size <= 0.0 or labeled_size >= 1.0:
        raise ValueError("labeled_size must be between (0,1)")

    target_column_name = "salary"

    # read csv
    df: pd.DataFrame = pd.read_csv("./data/uci_income/adult.csv", sep=",")

    # one-hot encoding
    df = one_hot_encoding(df, target_column_name)

    # normalization
    if is_norm:
        df = normalize(df)

    # data split
    train_data          : np.ndarray
    train_labeled_data  : np.ndarray
    train_unlabeled_data: np.ndarray
    test_data           : np.ndarray
    # train & test split
    train_data, test_data = train_test_split(df.values, train_size=train_size)
    # labeled & unlabeled split
    train_labeled_data, train_unlabeled_data = train_test_split(train_data, train_size=labeled_size)

    # easyfl dataset
    train_labeled_data   = {"x": train_labeled_data[:,:-1],   "y": train_labeled_data[:,-1]}
    train_unlabeled_data = {"x": train_unlabeled_data[:,:-1], "y": train_unlabeled_data[:,-1]}
    test_data            = {"x": test_data[:,:-1],            "y": test_data[:,-1]}

    fl_labeled_data   = FederatedTensorDataset(data=train_labeled_data, num_of_clients=num_of_client)
    fl_unlabeled_data = FederatedTensorDataset(data=train_unlabeled_data, num_of_clients=num_of_client)
    fl_test_data      = FederatedTensorDataset(data=test_data, num_of_clients=num_of_client)

    return fl_labeled_data, fl_unlabeled_data, fl_test_data
