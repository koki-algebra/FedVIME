import warnings

from torch import nn, Tensor
import easyfl
from easyfl.datasets import FederatedTensorDataset
from easyfl.models import BaseModel

from datasets import UCIIncome


class MLP(BaseModel):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, X: Tensor) -> Tensor:
        logits = self.mlp(X)
        return logits


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    num_of_client = 10
    X_l_train, y_train, X_u_train, X_test, y_test = UCIIncome.get_dataset(labeled_size=0.9)

    train_labeled_data = {"x": X_l_train, "y": y_train}
    test_data          = {"x": X_test,    "y": y_test}

    # training
    train_labeled_dataset = FederatedTensorDataset(
        data=train_labeled_data,
        num_of_clients=num_of_client
    )

    # test
    test_dataset = FederatedTensorDataset(
        data=test_data,
        num_of_clients=num_of_client
    )

    # register dataset
    easyfl.register_dataset(train_data=train_labeled_dataset, test_data=test_dataset)

    # register model
    easyfl.register_model(MLP(input_dim=X_l_train.size(1), output_dim=2))

    # load configuration
    config = easyfl.load_config("./config/config.yaml")

    # initialization
    easyfl.init(config)

    # run simulation
    easyfl.run()
