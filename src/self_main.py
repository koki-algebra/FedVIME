import warnings

import easyfl

from datasets import uci_income
from vime import SelfSLClient, SelfSLNetworks


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    # get dataset
    train_data, test_data = uci_income.get_dataset(labeled_size=0.1)
    feature_num = 105

    easyfl.register_dataset(train_data=train_data, test_data=test_data)
    easyfl.register_model(
        model=SelfSLNetworks(dim_x=feature_num, dim_z=feature_num * 2)
    )
    easyfl.register_client(client=SelfSLClient)

    # configuration
    config = easyfl.load_config("./config/config.yaml")

    # initialization
    easyfl.init(config)

    easyfl.run()
