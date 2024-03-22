import torch
import os

from torch import nn
from torch.utils.data import DataLoader

from utils import load_data
from models import NeuralNetwork, NeuralNetwork2

def run(train_config: dict):
    training_data, test_data = load_data(path_input=train_config["path"])
    train_dataloader = DataLoader(training_data,
                                  batch_size=train_config["batch_size"],
                                  **train_config["dataloader_args"]["train"],)
    test_dataloader = DataLoader(test_data,
                                 batch_size=train_config["batch_size"],
                                 **train_config["dataloader_args"]["val"],)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    if torch.backends.mps.is_available():
        device = torch.device("cpu")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = 1
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if "model2" in train_config and train_config["model2"]:
        model = NeuralNetwork2(input_size=train_config["model2"]["input_size"],
                               hidden_sizes=train_config["model2"]["hidden_sizes"],
                               output_size=train_config["model2"]["output_size"]).to(device)
    else:
        model = NeuralNetwork().to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"]["rate"])

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(train_config["num_epochs"]):
        # Training
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Validation
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    print("Done!")


    model_name = "model.pth"
    path_output = "./data/output"
    torch.save(model.state_dict(), os.path.join(path_output, model_name))
    print("Saved PyTorch Model State")