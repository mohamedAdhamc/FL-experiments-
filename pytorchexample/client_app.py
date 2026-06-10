"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log

from pytorchexample.task import Net, load_data, load_nonIID_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.task import train_scaffold_client as scaffold_train_fn
import random
import numpy as np

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #ToDo: check type, perhaps it is safer and better to also cast to a float
    # Load the server control variate
    vals = msg.content['config']["server-control-variate"]["values"]
    shapes = msg.content['config']["server-control-variate"]["shapes"]

    server_control_variate = []
    idx = 0
    for s in shapes:
        server_control_variate.append(torch.tensor(vals[idx:idx+s]))
        idx += s

    #log(20, f"obtained {msg.content['config']['server-control-variate']}")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_nonIID_data(partition_id, num_partitions, batch_size)
    

    # Although the prints were kind of sus not gonna lie
    if "client-state" not in context.state:
        context.state["client-state"] = MetricRecord({"client-control-variate": torch.zeros_like(p).tolist() for p in model.parameters()})
    #ToDo: check type, perhaps it is safer and better to also cast to a float
    client_control_variate = context.state["client-state"]["client-control-variate"]

    # Call the training function
    train_loss = scaffold_train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        server_control_variate=server_control_variate,
        client_control_variate=client_control_variate
    )

    # test update client control variate
    # context.state["client-state"]["client-control-variate"] += random.randint(0, 10)
    # get ci and log it
    c_i  = [torch.tensor(layer) for layer in context.state['client-state']["client-control-variate"]]
    log(20, f"for client {context.node_config['partition-id']} the client control variate is: {c_i}")


    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "delta_client_control_variate": 3.2 #change to actual variable later
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_nonIID_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
