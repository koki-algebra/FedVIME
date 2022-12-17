from torch import nn, Tensor
from easyfl.models import BaseModel


class SemiSLNetworks(BaseModel):
    def __init__(self, encoder: nn.Module, dim_z: int, dim_y: int) -> None:
        super().__init__()
        # pre-trained encoder
        self.encoder = encoder
        # predictor
        self.predictor = nn.Sequential(
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_z * 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_z * 2),
            nn.Linear(dim_z * 2, dim_z),
            nn.ReLU(),
            nn.Linear(dim_z, dim_y),
        )

    def forward(self, X: Tensor) -> Tensor:
        Z = self.encoder(X)
        logits = self.predictor(Z)
        return logits
