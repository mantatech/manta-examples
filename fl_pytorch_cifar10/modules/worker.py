import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from manta.light.task import Task


# Build the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def set_weights(self, weights):
        for layer, w in weights.items():
            self.state_dict()[layer].copy_(torch.FloatTensor(w))

    def get_weights(self):
        weights = {}
        for layer, w in self.state_dict().items():
            weights[layer] = w.detach().tolist()
        return weights


class Worker(Task):
    def __init__(self):
        super().__init__()

        # Load CIFAR10 dataset
        self.logger.info("Loading CIFAR10 dataset from local context")
        raw_data = self.local.get_numpy_data("cifar10.npz")
        self.data = np.load(raw_data)

        self.logger.info("CIFAR10 dataset loaded")

        self.model = CNN()

    def run(self):
        self.logger.info("Starting training script")

        ## Get the model weights from world
        weights = self.world.globals["global_model_params"]
        self.model.set_weights(weights)
        self.logger.info("Model weights received from world")

        ## Get the hyperparameters from world
        hyperparameters = self.world.globals["hyperparameters"]
        self.logger.info(f"Hyperparameters received from world: {hyperparameters}")

        ## Train the model
        self.logger.info("Training the model")
        metrics = self.train_model(hyperparameters)
        self.logger.info("Training completed")

        # Save the metrics and the model weights to world
        self.logger.info(f"Saving metrics: {metrics} and model weights.")
        self.world.results.add("metrics", metrics)
        self.world.results.add("model_params", self.model.get_weights())

    def train_model(self, hyperparameters: dict):
        X_train, y_train, X_test, y_test = (
            self.data["x_train"],
            self.data["y_train"],
            self.data["x_test"],
            self.data["y_test"],
        )

        ## Define loss function and optimizer
        criterion = getattr(nn, hyperparameters["loss"])(
            **hyperparameters.get("loss_params", {})
        )
        optimizer = getattr(optim, hyperparameters["optimizer"])(
            self.model.parameters(), **hyperparameters.get("optimizer_params", {})
        )

        # Create a dictionary to store the metrics
        metrics = {}

        # Train the model
        for epoch in range(hyperparameters["epochs"]):
            self.model.train()
            for i in range(0, X_train.shape[0], hyperparameters["batch_size"]):
                # for i in range(
                #     0, 40 * hyperparameters["batch_size"], hyperparameters["batch_size"]
                # ):
                optimizer.zero_grad()
                output = self.model(
                    torch.tensor(X_train[i : i + hyperparameters["batch_size"]]).float()
                )
                loss = criterion(
                    output, torch.tensor(y_train[i : i + hyperparameters["batch_size"]])
                )
                loss.backward()
                optimizer.step()

            ## Validate the model
            self.model.eval()
            with torch.no_grad():
                output = self.model(torch.tensor(X_test).float())
                val_loss = criterion(output, torch.tensor(y_test))
                val_acc = (output.argmax(1) == torch.tensor(y_test)).float().mean()

            metrics["loss"] = loss.item()
            metrics["val_loss"] = val_loss.item()
            metrics["val_acc"] = val_acc.item()

            self.logger.info(
                f"Epoch {epoch + 1}/{hyperparameters['epochs']} Loss: {loss.item()} Validation Loss: {val_loss.item()} Validation Accuracy: {val_acc.item()}"
            )

        return metrics


def main():
    Worker().run()
