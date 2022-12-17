from typing import Tuple

from torch import nn
from torch import Tensor, sigmoid
from easyfl.models import BaseModel


class SelfSLNetworks(BaseModel):
    def __init__(self, dim_x: int, dim_z: int, is_norm=True) -> None:
        super().__init__()
        self.is_norm = is_norm

        # encoder e: X -> Z
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(dim_x),
            nn.Linear(dim_x, dim_x * 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_x * 2),
            nn.Linear(dim_x * 2, dim_x * 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_x * 2),
            nn.Linear(dim_x * 2, dim_z),
        )

        # mask vector estimator s_m: Z -> {0,1}^d
        self.mask_estimator = nn.Sequential(
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_z),
            nn.ReLU(),
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_x),
        )

        # feature vector estimator s_r: Z -> X
        self.feature_estimator = nn.Sequential(
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_z),
            nn.ReLU(),
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_x),
        )

    def forward(self, X_tilde: Tensor) -> Tuple[Tensor, Tensor]:
        # encode collapsed feature matrix
        Z: Tensor = self.encoder(X_tilde)
        # mask prediction
        M_pred: Tensor = sigmoid(self.mask_estimator(Z))
        # feature prediction
        X_pred: Tensor = self.feature_estimator(Z)

        # is normalization
        if self.is_norm:
            X_pred = sigmoid(X_pred)

        return M_pred, X_pred

    def get_encoder(self) -> nn.Module:
        return self.encoder
