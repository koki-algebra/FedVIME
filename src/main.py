import warnings

from torch import nn, Tensor
import easyfl

from datasets import uci_income


class MLP(nn.Module):
    def __init__(self, feature_num: int, class_num: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_num, feature_num),
            nn.ReLU(),
            nn.Linear(feature_num, class_num),
        )

    def forward(self, X: Tensor) -> Tensor:
        logits = self.mlp(X)
        return logits


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    train_data, test_data = uci_income.get_dataset(
        num_of_clients=100, labeled_size=1.0, train_size=0.8
    )

    print(train_data.size("f0000000"))

    config = easyfl.load_config("./config/main.yaml")

    easyfl.register_dataset(train_data, test_data)
    easyfl.register_model(model=MLP(feature_num=105, class_num=2))

    easyfl.init(config)

    easyfl.run()
