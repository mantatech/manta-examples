from manta.light.task import Task


class Scheduler(Task):
    def run(self):
        self.logger.info("Starting scheduler script")

        self.hp = self.world.globals["hyperparameters"]

        # Select the node ids for which the model has the lowest validation accuracy
        self.logger.info("Selecting nodes")
        metrics = self.world.results.select("metrics")
        selected_nodes = self.select_nodes(metrics)
        if len(selected_nodes) == 0:
            return

        # Set the selected nodes in the database for the next iteration
        self.logger.info(f"Nodes selected: {selected_nodes}")
        self.world.schedule_next_iter(
            node_ids=selected_nodes, task_to_schedule_alias="worker"
        )

    def select_nodes(self, metrics: dict):
        """
        Select the nodes

        Parameters
        ----------
        metrics : dict
            The list of metrics for each node

        Returns
        -------
        list
            The selected nodes
        """
        # Initialize the selected nodes
        selected_nodes = []

        val_acc_threshold = self.hp.get("val_acc_threshold", 0.9)

        # Select the nodes
        for node_id, metr in metrics.items():
            self.logger.info(
                f"Node {node_id} has validation accuracy {metr['val_acc']}"
            )
            if metr["val_acc"] < val_acc_threshold:
                selected_nodes.append(node_id)

        # If all the val_acc are above val_acc_threshold, stop the swarm
        if len(selected_nodes) == 0:
            self.logger.info(
                f"All nodes have validation accuracy above {val_acc_threshold}. Stopping the swarm"
            )
            self.world.stop_swarm()
        return selected_nodes


def main():
    Scheduler().run()
