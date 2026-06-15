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
from collections import OrderedDict

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    x_arr_rec = msg.content["x"]
    #x as a list of np arrays
    x = [ x_arr_rec[str(i)].numpy() for i in range(len(x_arr_rec))] 
    layer_names = list(model.state_dict().keys())

    state_dict = OrderedDict(
        (name, torch.from_numpy(val))
        for name, val in zip(layer_names, x)
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #x as a torch tensors
    x = [torch.tensor(x_layer, device=device) for x_layer in x]

    model.load_state_dict(state_dict)
    model.to(device)
    
    #ToDo: check type, perhaps it is safer and better to also cast to a float
    # Load the server control variate
    c_arrRec = msg.content['c']

    # print("cuda available:", torch.cuda.is_available())
    # print("cuda device count:", torch.cuda.device_count())
    # print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
    
    #list of pytorch tensors of each param
    c = [ torch.tensor(c_arrRec[str(i)].numpy(), device=device) for i in range(len(c_arrRec))] 
    
    print("\n=== Server control variate c at client after processing ===")
    for i, ci in enumerate(c):
        print(f"layer {i}: shape = {tuple(ci.shape)} dtype = {ci.dtype} device = {ci.device}")
    # log(20, f"obtained c from server {c}")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_nonIID_data(partition_id, num_partitions, batch_size)
    

    # Although the prints were kind of sus not gonna lie
    if "client-state" not in context.state:
        context.state["client-state"] = ArrayRecord([np.zeros_like(p.detach().cpu().numpy()) for p in model.parameters()])
    #ToDo: check type, perhaps it is safer and better to also cast to a float
    client_control_variate = context.state["client-state"]
    #list of tensors form
    client_control_variate = [torch.tensor(client_control_variate[str(i)].numpy(), device=device) for i in range(len(client_control_variate)) ]

    print("\n=== Client control variate c at client check ===")
    for i, ci in enumerate(client_control_variate):
        print(f"layer {i}: shape = {tuple(ci.shape)} dtype = {ci.dtype} device = {ci.device}")


    # Call the training function
    train_loss = scaffold_train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        c=c,#passes as list of tensors form
        ci=client_control_variate#passed as list of tensors form
    )

    y = [ torch.tensor(p.detach().clone().numpy(), device=device) for p in model.parameters() ]
    lr = msg.content["config"]["lr"]
    K = context.run_config["local-epochs"]
    # calc ciplus client control variate
    ci_plus = []

    for x_layer, y_layer, c_layer, ci_layer in zip(
        x,#list of torch tensors
        y,#list of torch tensors
        c,#list of torch tensors
        client_control_variate,#list of torch tensors
    ):
        ci_new = ci_layer - c_layer + (x_layer - y_layer) / (lr * K)
        ci_plus.append(ci_new)


    #comm to server the delta c and delta yi
    delta_ci_plus = []
    for ci_plus_layer, ci_layer in zip(ci_plus, client_control_variate):
        delta_ci_plus.append(ci_plus_layer - ci_layer)
    #restore list of numpy form for comm
    delta_ci_plus_np = [delta_ci_plus[i].numpy() for i in range(len(delta_ci_plus))]
    delta_ci_plus_np_arrrec = ArrayRecord(delta_ci_plus_np)

    delta_y_i = []
    for x_layer, y_layer in zip(x,y):
        delta_y_i.append(y_layer-x_layer)
    delta_y_i_np = [delta_y_i[i].numpy() for i in range(len(delta_y_i))]
    delta_yi_np_arrrec = ArrayRecord(delta_y_i_np)

    #update client control variate 
    client_control_variate = ci_plus
    #restore list of numpy form to store in state
    client_control_variate_np = [client_control_variate[i].numpy() for i in range(len(client_control_variate))]
    context.state["client-state"] = ArrayRecord(client_control_variate_np)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "delta_client_control_variate": 3.2 #change to actual variable later
    }
    metric_record = MetricRecord(metrics)
    #arrays key is no longer used
    content = RecordDict({"arrays": model_record, "metrics": metric_record, "deltay_i":delta_yi_np_arrrec, "deltaci_plus":delta_ci_plus_np_arrrec})
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
