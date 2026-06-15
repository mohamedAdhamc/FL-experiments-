# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower message-based FedAvg strategy."""


from collections.abc import Callable, Iterable
from logging import INFO, WARNING
import time
import io

import numpy as np
import torch

from flwr.app import MessageType
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
    NDArray
)
from flwr.serverapp.strategy.result import Result
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords, sample_nodes, validate_message_reply_consistency, log_strategy_start_info
from typing import cast


#plain old fedavg stuff, extracts weighting factors and then aggregates
#we don't really need the weight factor stuff in scaffold
#
def aggregate_arrayrecords(
    records: list[RecordDict], weighting_metric_name: str
):
    """Perform weighted aggregation all ArrayRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    # weights: list[float] = []
    # for record in records:
    #     # Get the first (and only) MetricRecord in the record
    #     metricrecord = next(iter(record.metric_records.values()))
    #     # Because replies have been checked for consistency,
    #     # we can safely cast the weighting factor to float
    #     w = cast(float, metricrecord[weighting_metric_name])
    #     weights.append(w)

    # Average
    # total_weight = sum(weights)
    # weight_factors = [w / total_weight for w in weights]

    # Perform weighted aggregation
    # aggregated_np_arrays: dict[str, NDArray] = {}

    # for record, weight in zip(records, weight_factors, strict=True):
    #     for record_item in record.array_records.values():
    #         # aggregate in-place
    #         for key, value in record_item.items():
    #             if key not in aggregated_np_arrays:
    #                 aggregated_np_arrays[key] = value.numpy() * weight
    #             else:
    #                 aggregated_np_arrays[key] += value.numpy() * weight
    #aggregate deltax 
    delta_x = [torch.zeros_like(torch.tensor(records[0].array_records["deltay_i"][str(i)].numpy())) for i in range(len(records[0].array_records["deltay_i"]))]
    for record in records:
        #list of pytorch tensors of each param
        deltayi = [ torch.tensor(record.array_records["deltay_i"][str(i)].numpy()) for i in range(len(record.array_records["deltay_i"]))] 
        delta_x += deltayi
    for i in range(len(delta_x)):
        delta_x[i] = delta_x[i]/len(records) #len(records) should be the number of selected clients
    
    #aggregate deltac
    delta_c = [torch.zeros_like(torch.tensor(records[0].array_records["deltaci_plus"][str(i)].numpy())) for i in range(len(records[0].array_records["deltaci_plus"]))]
    for record in records:
        #list of pytorch tensors of each param
        delta_ci = [ torch.tensor(record.array_records["deltaci_plus"][str(i)].numpy()) for i in range(len(record.array_records["deltaci_plus"]))] 
        delta_c += delta_ci
    for i in range(len(delta_c)):
        delta_c[i] = delta_c[i]/len(records) #len(records) should be the number of selected clients


    return delta_x, delta_c



class Scaffold(Strategy):
    """Scaffold Strategy.

    Implementation based on https://arxiv.org/pdf/1910.06378

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation. In case `min_evaluate_nodes`
        is larger than `fraction_evaluate * total_connected_nodes`,
        `min_evaluate_nodes` will still be sampled.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for both ArrayRecords and MetricRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    train_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        model, #all we need from the model is basically the dimensionlity so maybe better passs the dim but for now this is fine
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        server_learning_rate=1.0,
        num_clients=10,#adjust if running with a diff number
    ) -> None:
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighted_by_key = weighted_by_key
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key
        self.train_metrics_aggr_fn = train_metrics_aggr_fn or aggregate_metricrecords
        self.evaluate_metrics_aggr_fn = (
            evaluate_metrics_aggr_fn or aggregate_metricrecords
        )
        self.server_learning_rate = server_learning_rate
        self.num_clients = num_clients
        # a list of zeroed np array equivalents of each parameter tensor
        self.server_control_variate = [np.zeros_like(p.detach().cpu().numpy()) for p in model.parameters()]
        self.x = [p.detach().cpu().numpy() for p in model.parameters()]
            
        if self.fraction_evaluate == 0.0:
            self.min_evaluate_nodes = 0
            log(
                WARNING,
                "fraction_evaluate is set to 0.0. "
                "Federated evaluation will be skipped.",
            )
        if self.fraction_train == 0.0:
            self.min_train_nodes = 0
            log(
                WARNING,
                "fraction_train is set to 0.0. Federated training will be skipped.",
            )



    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> Sampling:")
        log(
            INFO,
            "\t│\t├──Fraction: train (%.2f) | evaluate ( %.2f)",
            self.fraction_train,
            self.fraction_evaluate,
        )  # pylint: disable=line-too-long
        log(
            INFO,
            "\t│\t├──Minimum nodes: train (%d) | evaluate (%d)",
            self.min_train_nodes,
            self.min_evaluate_nodes,
        )  # pylint: disable=line-too-long
        log(INFO, "\t│\t└──Minimum available nodes: %d", self.min_available_nodes)
        log(INFO, "\t└──> Keys in records:")
        log(INFO, "\t\t├── Weighted by: '%s'", self.weighted_by_key)
        log(INFO, "\t\t├── ArrayRecord key: '%s'", self.arrayrecord_key)
        log(INFO, "\t\t└── ConfigRecord key: '%s'", self.configrecord_key)


    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages



    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round


        print("\n=== Server control variate shapes ===")
        for i, c in enumerate(self.server_control_variate):
            print(f"layer {i}: shape = {c.shape}")

        # Test inject server control variate
        # get server control variate
        # log(INFO, f"before at server round {server_round}: {self.server_control_variate}")
        #self.server_control_variate += 0.1
        #log(INFO, f"after at server round {server_round}: {self.server_control_variate}")
        # send new server control variate


        # Construct messages
        record = RecordDict(
            {"x": ArrayRecord(self.x), self.configrecord_key: config, "c": ArrayRecord(self.server_control_variate)}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)


    def _check_and_log_replies(
        self, replies: Iterable[Message], is_train: bool, validate: bool = True
    ) -> tuple[list[Message], list[Message]]:
        """Check replies for errors and log them.

        Parameters
        ----------
        replies : Iterable[Message]
            Iterable of reply Messages.
        is_train : bool
            Set to True if the replies are from a training round; False otherwise.
            This impacts logging and validation behavior.
        validate : bool (default: True)
            Whether to validate the reply contents for consistency.

        Returns
        -------
        tuple[list[Message], list[Message]]
            A tuple containing two lists:
            - Messages with valid contents.
            - Messages with errors.
        """
        if not replies:
            return [], []

        # Filter messages that carry content
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(
            INFO,
            "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies),
            len(error_replies),
        )

        # Log errors
        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        # Ensure expected ArrayRecords and MetricRecords are received
        # if validate and valid_replies:
        #     validate_message_reply_consistency(
        #         replies=[msg.content for msg in valid_replies],
        #         weighted_by_key=self.weighted_by_key,
        #         check_arrayrecord=is_train,
        #     )

        return valid_replies, error_replies



    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords 
            delta_x, delta_c = aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )

            #update x and c

            #self.x is a list of numpy arrays
            delta_x_np = [delta_x[i].numpy() for i in range(len(delta_x))]
            for i in range(len(delta_x_np)):
                self.x[i] += (delta_x_np[i] * self.server_learning_rate)
            #server control variate is list of np_arrays
            delta_c_np = [delta_c[i].numpy() for i in range(len(delta_c))]
            num_selected_clients = len(reply_contents)
            total_num_clients = self.num_clients
            frac = float(num_selected_clients/total_num_clients)
            for i in range(delta_c_np):
                self.server_control_variate[i] += (frac * delta_c_np[i])


            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return metrics




    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)




    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return metrics


    #override the default start method to allow for our custom aggregation of delta x and delta c
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )
            # for reply in train_replies:
            #     print(reply.content)#reply.content should have the recorddict that the client sent

            # Aggregate train
            agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            # if agg_arrays is not None:
            #     result.arrays = agg_arrays
            #     arrays = agg_arrays

            #

            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result

