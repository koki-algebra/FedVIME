import warnings

import easyfl
from pytorch_tabnet.tab_network import TabNetPretraining

from datasets import uci_income
from tabnet import PretrainerClient


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    train_data, test_data = uci_income.get_dataset(labeled_size=0.1)

    input_dim = train_data.data["f0000000"]["x"].shape[1]

    config = easyfl.load_config("./config/tabnet_pretrain.yaml")

    # model parameters
    params = config.model_parameters

    easyfl.register_dataset(train_data=train_data, test_data=test_data)
    easyfl.register_model(
        model=TabNetPretraining(
            input_dim=input_dim,
            pretraining_ratio=params.pretraining_ratio,
            n_d=params.n_d,
            n_a=params.n_a,
            n_steps=params.n_steps,
            gamma=params.gamma,
            n_independent=params.n_independent,
            n_shared=params.n_shared,
            epsilon=params.epsilon,
            virtual_batch_size=params.virtual_batch_size,
            momentum=params.momentum,
            mask_type=params.mask_type,
            n_shared_decoder=params.n_shared_decoder,
            n_indep_decoder=params.n_indep_decoder,
        )
    )
    easyfl.register_client(client=PretrainerClient)

    easyfl.init(config)

    easyfl.run()
