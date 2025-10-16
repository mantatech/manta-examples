import numpy as np
from manta.light.task import Task


class Aggregator(Task):
    def run(self):
        self.logger.info("Starting aggregation script")

        # Aggregate the models
        self.logger.info("Aggregating models")
        models = list(self.world.results.select("model_params").values())
        aggregated_model = self.aggregate_models(models)
        self.logger.info("Models aggregated")

        # Set the aggregated model in the database
        self.world.globals["global_model_params"] = aggregated_model
        self.logger.info("Aggregated model sent to world")

    def aggregate_models(self, models):
        """
        Aggregate the models

        Parameters
        ----------
        models : list
            The list of models to aggregate

        Returns
        -------
        dict
            The aggregated model
        """
        # Initialize the aggregated model
        aggregated_model = {}

        # Aggregate the models
        for layer in models[0]:
            aggregated_model[layer] = np.mean(
                [model[layer] for model in models], axis=0
            ).tolist()

        return aggregated_model


def main():
    Aggregator().run()
