"""Training and evaluation."""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from common import ARTIFACT_NAME
from model_wrapper import ModelWrapper
from neural_network import NeuralNetwork
from utils_train_nn import evaluate, fit

DATA_DIR = "aml_online_endpoint/data"
PYTORCH_MODEL_DIR = "aml_online_endpoint/endpoint_2/pytorch_model"
PYFUNC_MODEL_DIR = "aml_online_endpoint/endpoint_2/pyfunc_model"


def load_train_val_data(
        data_dir: str, batch_size: int,
        training_fraction: float) -> Tuple[DataLoader, DataLoader]:
    """
    Returns two DataLoader objects that wrap training and validation data.
    Training and validation data are extracted from the full original training
    data, split according to training_fraction.
    """
    full_train_data = datasets.FashionMNIST(data_dir,
                                            train=True,
                                            download=True,
                                            transform=ToTensor())
    full_train_len = len(full_train_data)
    train_len = int(full_train_len * training_fraction)
    val_len = full_train_len - train_len
    (train_data, val_data) = random_split(dataset=full_train_data,
                                          lengths=[train_len, val_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return (train_loader, val_loader)


def save_model(pytorch_model_dir: str, pyfunc_model_dir: str,
               model: nn.Module) -> None:
    """
    Saves the trained model.
    """
    # Save PyTorch model.
    pytorch_input_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, 784)),
    ])
    pytorch_output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
    pytorch_signature = ModelSignature(inputs=pytorch_input_schema,
                                       outputs=pytorch_output_schema)

    pytorch_code_filenames = ["neural_network.py", "utils_train_nn.py"]
    pytorch_full_code_paths = [
        Path(Path(__file__).parent, code_path)
        for code_path in pytorch_code_filenames
    ]
    logging.info("Saving PyTorch model to %s", pytorch_model_dir)
    shutil.rmtree(pytorch_model_dir, ignore_errors=True)
    mlflow.pytorch.save_model(pytorch_model=model,
                              path=pytorch_model_dir,
                              code_paths=pytorch_full_code_paths,
                              signature=pytorch_signature)

    # Save PyFunc model that wraps the PyTorch model.
    pyfunc_input_schema = Schema(
        [ColSpec(type="double", name=f"col_{i}") for i in range(784)])
    pyfunc_output_schema = Schema([TensorSpec(np.dtype(np.int32), (-1, 1))])
    pyfunc_signature = ModelSignature(inputs=pyfunc_input_schema,
                                      outputs=pyfunc_output_schema)

    pyfunc_code_filenames = ["model_wrapper.py", "common.py"]
    pyfunc_full_code_paths = [
        Path(Path(__file__).parent, code_path)
        for code_path in pyfunc_code_filenames
    ]
    model = ModelWrapper()
    artifacts = {
        ARTIFACT_NAME: pytorch_model_dir,
    }
    logging.info("Saving PyFunc model to %s", pyfunc_model_dir)
    shutil.rmtree(pyfunc_model_dir, ignore_errors=True)
    mlflow.pyfunc.save_model(path=pyfunc_model_dir,
                             python_model=model,
                             artifacts=artifacts,
                             code_path=pyfunc_full_code_paths,
                             signature=pyfunc_signature)


def train(data_dir: str, pytorch_model_dir: str, pyfunc_model_dir: str,
          device: str) -> None:
    """
    Trains the model for a number of epochs, and saves it.
    """
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    (train_dataloader,
     val_dataloader) = load_train_val_data(data_dir, batch_size, 0.8)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        logging.info("Epoch %d", epoch + 1)
        (training_loss, training_accuracy) = fit(device, train_dataloader,
                                                 model, loss_fn, optimizer)
        (validation_loss,
         validation_accuracy) = evaluate(device, val_dataloader, model, loss_fn)

        metrics = {
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        }
        mlflow.log_metrics(metrics, step=epoch)

    save_model(pytorch_model_dir, pyfunc_model_dir, model)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--pytorch_model_dir",
                        dest="pytorch_model_dir",
                        default=PYTORCH_MODEL_DIR)
    parser.add_argument("--pyfunc_model_dir",
                        dest="pyfunc_model_dir",
                        default=PYFUNC_MODEL_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(**vars(args), device=device)


if __name__ == "__main__":
    main()
