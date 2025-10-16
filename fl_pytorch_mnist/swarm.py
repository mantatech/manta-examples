from pathlib import Path

from manta.apis import Module, Swarm, Task
from manta.light.utils import numpy_to_bytes

from .modules.worker.model import MLP


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
        self.worker_train = Task(
            Module(
                root_path / "modules" / "worker",
                image,  # Use tag pytorch_gpu to use GPU
                datasets=["mnist"],
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
        self.worker_test = Task(
            Module(
                root_path / "modules" / "worker_test",
                image,  # Use tag pytorch_gpu to use GPU
                datasets=["mnist"],
            ),
            method="all",
            fixed=False,
            maximum=-1,
            gpu=gpu,
        )

        # Set hyperparameters
        self.set_global(
            "hyperparameters",
            {
                "epochs": 1,
                "batch_size": 32,
                "loss": "CrossEntropyLoss",
                "loss_params": {},
                "optimizer": "SGD",
                "optimizer_params": {"lr": 0.01, "momentum": 0.9},
                "val_acc_threshold": 0.99,
            },
        )
        # Set global model parameters
        self.set_global("global_model_params", numpy_to_bytes(MLP().get_weights()))

    def execute(self):
        """
        Generation of the task graph

        +--------+     +------------+     +------+     +-----------+ if has_converged
        | Worker | --> | Aggregator | --> | Test | --> | Scheduler | ----------------> END PROGRAM
        +--------+     +------------+     +------+     +-----------+
            |                                                       | else
            +--<<<----------<<<-------------<<<------------<<<------+
        """
        m = self.worker_train()
        m = self.aggregator(m)
        m = self.worker_test(m)
        return self.scheduler(m)
