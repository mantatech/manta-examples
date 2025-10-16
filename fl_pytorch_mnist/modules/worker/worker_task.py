import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from manta.light.task import Task
from manta.light.utils import bytes_to_numpy, numpy_to_bytes

from .model import MLP, device


class Worker(Task):
    def __init__(self):
        super().__init__()

        # Load MNIST dataset
        self.logger.warning(f"DEVICE : {device}")
        self.logger.info("Loading MNIST dataset from local context")

        raw_data = self.local.get_binary_data("mnist")
        self.data = np.load(raw_data)

        self.logger.info("MNIST dataset loaded")

        self.model = MLP()

    def run(self):
        self.logger.info("Starting training script")

        ## Get the hyperparameters from world
        hyperparameters = self.world.globals["hyperparameters"]
        self.logger.info(f"Hyperparameters received from world: {hyperparameters}")

        ## Get the model weights from world
        weights = self.world.globals["global_model_params"]
        self.model.set_weights(bytes_to_numpy(weights))
        self.logger.info("Model weights received from world")

        ## Train the model
        self.logger.info("Training the model")
        metrics = self.train_model(hyperparameters)
        self.logger.info("Training completed")

        # Save the metrics and the model weights to world
        self.logger.info(f"Saving metrics: {metrics} and model weights.")
        self.world.results.add("train_metrics", metrics)

        weights = self.model.get_weights()
        self.world.results.add("model_params", numpy_to_bytes(weights))

    def train_model(self, hyperparameters: dict):
        X_train, y_train = (
            self.data["x_train"],
            self.data["y_train"],
        )

        self.model.to(device)

        ## Define loss function and optimizer
        criterion = getattr(nn, hyperparameters["loss"])(
            **hyperparameters.get("loss_params", {})
        )
        optimizer = getattr(optim, hyperparameters["optimizer"])(
            self.model.parameters(), **hyperparameters.get("optimizer_params", {})
        )

        # Create a dictionary to store the metrics
        metrics = {
            "loss": [],
        }

        # Train the model
        for epoch in range(hyperparameters["epochs"]):
            self.model.train()
            for i in range(0, X_train.shape[0], hyperparameters["batch_size"]):
                # for i in range(
                #     0, 40 * hyperparameters["batch_size"], hyperparameters["batch_size"]
                # ):
                optimizer.zero_grad()

                inputs = torch.tensor(
                    X_train[i : i + hyperparameters["batch_size"]],
                    dtype=torch.float32,
                    device=device,
                )
                targets = torch.tensor(
                    y_train[i : i + hyperparameters["batch_size"]],
                    dtype=torch.long,
                    device=device,
                )

                output = self.model(inputs)
                loss = criterion(output, targets)

                loss.backward()
                optimizer.step()

            metrics["loss"].append(loss.item())

            self.logger.info(
                f"Epoch {epoch + 1}/{hyperparameters['epochs']} Loss: {loss.item()}"
            )

        return metrics
