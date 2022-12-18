from typing import Tuple, Dict
import math

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from easyfl.datasets import FederatedTensorDataset
from easyfl.datasets.dataset import default_process_x, default_process_y
from easyfl.datasets.dataset_util import TransformDataset
from easyfl.datasets.simulation import SIMULATE_IID

from utils import one_hot_encoding, normalize


class FederatedUCIIncomeDataset(FederatedTensorDataset):
    def __init__(
        self,
        data,
        labeled_size=1.0,
        transform=None,
        target_transform=None,
        process_x=default_process_x,
        process_y=default_process_y,
        simulated=False,
        do_simulate=True,
        num_of_clients=10,
        simulation_method=SIMULATE_IID,
        weights=None,
        alpha=0.5,
        min_size=10,
        class_per_client=1,
    ):
        super().__init__(
            data,
            transform,
            target_transform,
            process_x,
            process_y,
            simulated,
            do_simulate,
            num_of_clients,
            simulation_method,
            weights,
            alpha,
            min_size,
            class_per_client,
        )

        # validation
        if labeled_size > 1.0 or labeled_size < 0.0:
            raise ValueError("labeled_size must be in (0,1]")
        self.labeled_size = labeled_size

    def loader(
        self,
        batch_size,
        client_id=None,
        shuffle=True,
        seed=0,
        transform=None,
        drop_last=False,
    ) -> DataLoader | Dict[str, DataLoader]:
        """Get dataset loader.
        Args:
            batch_size (int): The batch size.
            client_id (str, optional): The id of client.
            shuffle (bool, optional): Whether to shuffle before batching.
            seed (int, optional): The shuffle seed.
            transform (torchvision.transforms.transforms.Compose, optional): Data transformation.
            drop_last (bool, optional): Whether to drop the last batch if its size is smaller than batch size.
        Returns:
            torch.utils.data.DataLoader: The data loader to load data.
        """
        # Simulation need to be done before creating a data loader
        if client_id is None:
            data = self.data
        else:
            data = self.data[client_id]

        data_x = data["x"]
        data_y = data["y"]

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        data_x = self._input_process(data_x)
        data_y = self._label_process(data_y)
        if shuffle:
            np.random.seed(seed)
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        transform = self.transform if transform is None else transform
        if transform is not None:
            dataset = TransformDataset(
                data_x, data_y, transform_x=transform, transform_y=self.target_transform
            )
        else:
            dataset = TensorDataset(data_x, data_y)

        if self.labeled_size == 1.0:
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            return loader
        else:
            # label & unlabel split
            labeled_num = math.floor(self.labeled_size * len(dataset))
            unlabeled_num = len(dataset) - labeled_num
            labeled_dataset, unlabeled_dataset = random_split(
                dataset, lengths=[labeled_num, unlabeled_num]
            )

            # data loader
            labeled_dataloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            unlabeled_dataloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )

            return {"labeled": labeled_dataloader, "unlabeled": unlabeled_dataloader}


def get_dataset(
    labeled_size: float, is_norm=True, train_size=0.8, num_of_clients=10
) -> Tuple[FederatedUCIIncomeDataset, FederatedUCIIncomeDataset]:
    if train_size <= 0.0 or train_size >= 1.0:
        raise ValueError("train_size must be between (0,1)")

    target_column_name = "salary"

    # read csv
    df: pd.DataFrame = pd.read_csv("./data/uci_income/adult.csv", sep=",")

    # one-hot encoding
    df = one_hot_encoding(df, target_column_name)

    # normalization
    if is_norm:
        df = normalize(df)

    # train & test split
    train_data, test_data = train_test_split(df.values, train_size=train_size)

    train_dataset = {
        "x": train_data[:, :-1].astype(np.float32),
        "y": train_data[:, -1].astype(np.int64),
    }
    test_dataset = {
        "x": test_data[:, :-1].astype(np.float32),
        "y": test_data[:, -1].astype(np.int64),
    }

    # federated dataset
    fed_train_dataset = FederatedUCIIncomeDataset(
        data=train_dataset, labeled_size=labeled_size, num_of_clients=num_of_clients
    )
    fed_test_dataset = FederatedUCIIncomeDataset(
        data=test_dataset, num_of_clients=num_of_clients
    )

    return fed_train_dataset, fed_test_dataset
