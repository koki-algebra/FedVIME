import time
import logging

from torch import Tensor, nn
from easyfl.client.base import BaseClient

from utils import pretext_generator


class SelfSLClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, sleep_time=0, is_remote=False, local_port=23000, server_addr="localhost:22999", tracker_addr="localhost:12666"):
        super().__init__(cid, conf, train_data, test_data, device, sleep_time, is_remote, local_port, server_addr, tracker_addr)

    def load_loss_fn(self, conf):
        return SelfSLLoss(alpha=conf.alpha)

    def train(self, conf, device):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []

        # training
        for i in range(conf.local_epoch):
            batch_loss = []
            for X, _ in self.train_loader:
                X: Tensor = X.to(device)

                # pretext generate
                M, X_tilde = pretext_generator(X, conf.p_m)

                # compute prediction and loss
                M_pred, X_pred = self.model(X_tilde)
                loss: Tensor = loss_fn(M_pred, M, X_pred, X)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # batch loss
                batch_loss.append(loss.item())

            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logging.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        
        self.train_time = time.time() - start_time
        logging.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))


# SelfSLLoss: Self-SL loss function
class SelfSLLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, M_pred: Tensor, M_target: Tensor, X_pred: Tensor, X_target: Tensor) -> float:
        mask_loss = nn.BCELoss()
        feature_loss = nn.MSELoss()
        return mask_loss(M_pred, M_target) + self.alpha * feature_loss(X_pred, X_target)