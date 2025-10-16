# Import CNN class from worker module for initial model weights
from pathlib import Path

from manta.apis import Module, Swarm, Task
from modules.worker import CNN


class FLSwarm(Swarm):
    def __init__(self, image: str = "manta_light:pytorch", gpu: bool = False):
        # TODO : find why it disables logging colors

        super().__init__()

        root_path = Path(__file__).resolve().parent
        self.aggregator = Task(
            Module(
                root_path / "modules" / "aggregator.py",
                image,  # Use tag pytorch_gpu to use GPU
                datasets=[],
            ),
            method="any",
            fixed=False,
            maximum=1,
            gpu=False,
        )
        self.worker = Task(
            Module(
                root_path / "modules" / "worker.py",
                image,  # Use tag pytorch_gpu to use GPU
                datasets=["cifar10"],
            ),
            method="all",
            fixed=False,
            maximum=-1,
            gpu=gpu,
        )
        self.scheduler = Task(
            Module(
                root_path / "modules" / "scheduler.py",
                image,  # Use tag pytorch_gpu to use GPU
                datasets=[],
            ),
            method="any",
            fixed=False,
            maximum=1,
            gpu=False,
        )

        # Set hyperparameters
        self.set_global(
            "hyperparameters",
            {
                "epochs": 1,
                "batch_size": 64,
                "loss": "CrossEntropyLoss",
                "loss_params": {},
                "optimizer": "SGD",
                "optimizer_params": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
                "val_acc_threshold": 0.80,
            },
        )
        # Set global model parameters
        self.set_global("global_model_params", CNN().get_weights())

    def execute(self):
        """
        Generation of the task graph

        +--------+     +------------+     +-----------+ if has_converged
        | Worker | --> | Aggregator | --> | Scheduler | ----------------> END PROGRAM
        +--------+     +------------+     +-----------+
            |                                   | else
            +--<<<----------<<<----------<<<----+
        """
        m = self.worker()
        m = self.aggregator(m)
        return self.scheduler(m)
