import numpy as np
import torch
import torch.nn as nn
from manta.light.task import Task
from manta.light.utils import bytes_to_numpy

from .model import MLP, device


class Worker(Task):
    def __init__(self):
        super().__init__()

        # Load MNIST dataset
        self.logger.warning(f"DEVICE : {device}")
        self.logger.info("Loading MNIST dataset from local context")
        # raw_data = self.local.get_numpy_data("mnist_part.npz")
        raw_data = self.local.get_binary_data("mnist")
        self.data = np.load(raw_data)

        self.logger.info("MNIST dataset loaded")

        self.model = MLP()

    def run(self):
        self.logger.info("Starting training script")

        ## Get the model weights from world
        weights = self.world.globals["global_model_params"]
        self.model.set_weights(bytes_to_numpy(weights))
        self.logger.info("Model weights received from world")

        ## Get the hyperparameters from world
        hyperparameters = self.world.globals["hyperparameters"]
        self.logger.info(f"Hyperparameters received from world: {hyperparameters}")

        ## Train the model
        self.logger.info("Testing the model")
        metrics = self.test_model(hyperparameters)
        self.logger.info("Testing completed")

        # Save the metrics and the model weights to world
        self.logger.info(f"Saving metrics: {metrics}.")
        self.world.results.add("metrics", metrics)

    def test_model(self, hyperparameters: dict):
        X_test, y_test = (
            self.data["x_test"],
            self.data["y_test"],
        )

        self.model.to(device)

        ## Define loss function and optimizer
        criterion = getattr(nn, hyperparameters["loss"])(
            **hyperparameters.get("loss_params", {})
        )

        ## Validate the model
        metrics = {}
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(
                X_test,
                dtype=torch.float32,
                device=device,
            )
            targets = torch.tensor(
                y_test,
                dtype=torch.long,
                device=device,
            )

            output = self.model(inputs)
            val_loss = criterion(output, targets)
            val_acc = (output.argmax(1) == targets).float().mean()

            metrics["val_loss"] = val_loss.item()
            metrics["val_acc"] = val_acc.item()

            self.logger.info(
                f"Validation Loss: {val_loss.item()} Validation Accuracy: {val_acc.item()}"
            )

        return metrics
