import time
import logging

from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from easyfl.client.base import BaseClient
from pytorch_tabnet.utils import check_input

logger = logging.getLogger(__name__)


class PretrainerClient(BaseClient):
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

    def train(self, conf, device):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []

        # model parameters
        clip_value: float = conf.model_params.clip_value

        for i in range(conf.local_epoch):
            train_dataloader = self.train_loader["unlabeled"]

            batch_loss = []
            for X, _ in train_dataloader:
                X = X.to(device).float()

                check_input(X)

                for param in self.model.parameters():
                    param.grad = None

                output, embedded_X, obf_vars = self.model(X)
                loss: Tensor = loss_fn(output, embedded_X, obf_vars)

                # Perform backward pass and optimization
                loss.backward()
                if clip_value:
                    clip_grad_norm_(self.model.parameters(), clip_value)
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

    def test(self, conf, device=...):
        pass
