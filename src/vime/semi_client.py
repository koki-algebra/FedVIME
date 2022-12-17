import time
import logging
from typing import List

import torch
from torch import Tensor, nn
from easyfl.client.base import BaseClient

from utils import pretext_generator

logger = logging.getLogger(__name__)


class SemiSLClient(BaseClient):
    def __init__(
        self,
        cid,
        conf,
        train_data,
        test_data,
        device,
        sleep_time=0,
        is_remote=False,
        local_port=23000,
        server_addr="localhost:22999",
        tracker_addr="localhost:12666",
    ):
        super().__init__(
            cid,
            conf,
            train_data,
            test_data,
            device,
            sleep_time,
            is_remote,
            local_port,
            server_addr,
            tracker_addr,
        )

    def load_loss_fn(self, conf):
        # hyperparams
        beta = conf.hyperparams.beta

        return SemiSLLoss(beta)

    def train(self, conf, device):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []

        # training
        for i in range(conf.local_epoch):
            batch_loss = []
            labeled_loader = self.train_loader["labeled"]
            unlabeled_loader = self.train_loader["unlabeled"]
            for (X_l, y), (X_u, _) in zip(labeled_loader, unlabeled_loader):
                X_l: Tensor = X_l.to(device)
                y: Tensor = y.to(device)
                X_u: Tensor = X_u.to(device)

                # hyperparams
                p_m = conf.hyperparams.p_m
                K = conf.hyperparams.K

                # labeled data prediction
                y_l_pred = self.model(X_l)

                # unlabel
                # K times Data Augmentation
                y_u_tilde_pred_list: List[Tensor] = []
                y_u_pred_list: List[Tensor] = []
                for _ in range(K):
                    # pretext generate
                    _, X_u_tilde = pretext_generator(X_u, p_m, device)

                    # predict
                    y_u_tilde_pred = self.model(X_u_tilde)
                    y_u_pred = self.model(X_u)

                    # append
                    y_u_tilde_pred_list.append(y_u_tilde_pred)
                    y_u_pred_list.append(y_u_pred)

                y_u_tilde_pred = torch.cat(y_u_tilde_pred_list)
                y_u_pred = torch.cat(y_u_pred_list)

                # loss
                loss: Tensor = loss_fn(y_l_pred, y, y_u_tilde_pred, y_u_pred)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # batch loss
                batch_loss.append(loss.item())

            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.info(
                "Client {}, local epoch: {}, loss: {}".format(
                    self.cid, i, current_epoch_loss
                )
            )

        self.train_time = time.time() - start_time
        logger.info("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def test(self, conf, device):
        """Execute client testing.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        begin_test_time = time.time()
        self.model.eval()
        self.model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        if self.test_loader is None:
            self.test_loader = self.test_data.loader(
                conf.test_batch_size, self.cid, shuffle=False, seed=conf.seed
            )
        # TODO: make evaluation metrics a separate package and apply it here.
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for batched_x, batched_y in self.test_loader:
                x = batched_x.to(device)
                y = batched_y.to(device)
                log_probs = self.model(x)
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                self.test_loss += loss.item()
            test_size = self.test_data.size(self.cid)
            self.test_loss /= test_size
            self.test_accuracy = 100.0 * float(correct) / test_size

        logger.debug(
            "Client {}, testing -- Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                self.cid, self.test_loss, correct, test_size, self.test_accuracy
            )
        )

        self.test_time = time.time() - begin_test_time
        self.model = self.model.cpu()


# SemiSLLoss: Semi-SL loss function
class SemiSLLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        y_pred: Tensor,
        y_target: Tensor,
        unlabeled_y_tilde_pred: Tensor,
        unlabeled_y_pred: Tensor,
    ) -> float:
        supervised_loss = nn.CrossEntropyLoss()
        unsupervised_loss = nn.MSELoss()

        return supervised_loss(y_pred, y_target) + self.beta * unsupervised_loss(
            unlabeled_y_tilde_pred, unlabeled_y_pred
        )
