from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message


class CustomFedAvg(FedAvg):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.99 every 1 round
        if server_round % 1 == 0 and server_round > 1:
            config["lr"] *= 0.99
            print("LR decreased to:", config["lr"])
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)