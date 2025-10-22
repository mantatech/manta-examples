import argparse
import os
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as tf
from torch.utils.data import DataLoader


# Adapted from lazymanta/datasets.py
def load_cifar10(normalize=True, download_dir="data"):
    """
    Download and load the CIFAR-10 dataset.

    Parameters
    ----------
    normalize : bool, optional
        Whether to normalize the data (default is True).
    download_dir : str, optional
        Directory to download the data (default is "data").

    Returns
    -------
    tuple of numpy.ndarray
        Training and testing data in the format ((x_train, y_train), (x_test, y_test)).
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print(f"Downloading/loading CIFAR-10 dataset to {download_dir}...")

    # Load raw data to compute normalization parameters
    cifar_trainset = torchvision.datasets.CIFAR10(
        root=download_dir, train=True, download=True, transform=tf.ToTensor()
    )
    data = cifar_trainset.data / 255

    means = data.mean(axis=(0, 1, 2))
    stds = data.std(axis=(0, 1, 2))

    if normalize:
        transforms = tf.Compose(
            [tf.ToTensor(), tf.Normalize(means, stds, inplace=True)]
        )
    else:
        transforms = tf.ToTensor()

    # Load training set
    train_set = torchvision.datasets.CIFAR10(
        root=download_dir, train=True, download=True, transform=transforms
    )
    train_loader = DataLoader(train_set, batch_size=len(train_set))

    # Load validation set
    validation_set = torchvision.datasets.CIFAR10(
        root=download_dir, train=False, download=True, transform=transforms
    )
    validation_loader = DataLoader(validation_set, batch_size=len(validation_set))

    print("CIFAR-10 dataset loaded.")

    return (
        next(iter(train_loader))[0].cpu().numpy(),
        next(iter(train_loader))[1].cpu().numpy(),
    ), (
        next(iter(validation_loader))[0].cpu().numpy(),
        next(iter(validation_loader))[1].cpu().numpy(),
    )


# Adapted from lazymanta/datasets.py
default_rng = np.random.default_rng()


def generate_IID_parties(dataset, nb_per_node, rng=None):
    """
    Generate IID data (random shuffle) for each node.
    """
    if rng is None:
        rng = default_rng

    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]

    num_train = len(y_train)
    num_test = len(y_test)

    train_indices = np.arange(num_train)
    rng.shuffle(train_indices)
    test_indices = np.arange(num_test)
    rng.shuffle(test_indices)

    if sum(nb_per_node) > num_train:
        raise ValueError(
            f"The sum of samples per party exceeds the total number of samples ({sum(nb_per_node)} > {num_train})"
        )

    party_datas = {}
    current_train_idx = 0
    current_test_idx = 0
    for i in range(len(nb_per_node)):
        node_train_count = nb_per_node[i]
        node_train_indices = train_indices[
            current_train_idx : current_train_idx + node_train_count
        ]
        current_train_idx += node_train_count

        # Proportionally divide test set based on training set size for this node
        node_test_count = int(num_test * node_train_count / num_train)
        node_test_indices = test_indices[
            current_test_idx : current_test_idx + node_test_count
        ]
        current_test_idx += node_test_count

        party_datas[i] = {  # Changed from i + 1 to i for 0-based indexing
            "x_train": x_train[node_train_indices],
            "y_train": y_train[node_train_indices],
            "x_test": x_test[node_test_indices],
            "y_test": y_test[node_test_indices],
        }
    return party_datas


def generate_non_IID_label_parties(dataset, nb_per_node, labels_per_client=1, rng=None):
    """
    Generate non-IID data: each node receives samples from a specified number of different labels.
    """
    if rng is None:
        rng = default_rng

    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]

    num_train = len(y_train)
    num_test = len(y_test)
    unique_labels = np.unique(y_train)

    if sum(nb_per_node) > num_train:
        raise ValueError(
            f"The sum of samples per party exceeds the total number of samples ({sum(nb_per_node)} > {num_train})"
        )

    indices_per_label_train = {l: np.where(y_train == l)[0] for l in unique_labels}
    indices_per_label_test = {l: np.where(y_test == l)[0] for l in unique_labels}
    for l in unique_labels:
        rng.shuffle(indices_per_label_train[l])
        rng.shuffle(indices_per_label_test[l])

    nb_clusters = int(len(unique_labels) / labels_per_client) + (
        len(unique_labels) % labels_per_client > 0
    )
    shuffled_unique_labels = list(unique_labels)  # Create a mutable copy
    rng.shuffle(shuffled_unique_labels)  # Shuffle the copy
    labels_per_cluster = {
        c: shuffled_unique_labels[c * labels_per_client : (c + 1) * labels_per_client]
        for c in range(nb_clusters)
    }

    cluster_per_client = rng.choice(nb_clusters, len(nb_per_node))

    party_datas = {}
    for i in range(len(nb_per_node)):
        node_labels = labels_per_cluster[cluster_per_client[i]]

        # Ensure each client gets at least one sample if nb_per_node[i] > 0
        # Distribute nb_per_node[i] samples among the node_labels
        samples_per_label_in_node = [0] * len(node_labels)
        if nb_per_node[i] > 0:
            # Distribute samples somewhat evenly, ensuring each assigned label gets at least one if possible
            base_samples = nb_per_node[i] // len(node_labels)
            remainder = nb_per_node[i] % len(node_labels)
            samples_per_label_in_node = [base_samples] * len(node_labels)
            for k in range(remainder):
                samples_per_label_in_node[k] += 1

            # If some labels got 0 due to too few samples, try to give them at least 1
            # by taking from labels that got more than 1 (this is a simplification)
            for k_idx in range(len(samples_per_label_in_node)):
                if samples_per_label_in_node[k_idx] == 0 and nb_per_node[i] >= len(
                    node_labels
                ):
                    # try to re-allocate to ensure each label gets at least one sample
                    # This part can be complex, for now, we rely on the initial distribution
                    # or accept that some labels might not be present if nb_per_node[i] is too small.
                    pass  # For simplicity, we'll assume initial distribution is okay or nb_per_node[i] is sufficient

        node_train_indices_list = []
        for j, l in enumerate(node_labels):
            count = samples_per_label_in_node[j]
            available_indices = indices_per_label_train[l]
            actual_count = min(count, len(available_indices))  # Take what's available
            node_train_indices_list.append(available_indices[:actual_count])
            indices_per_label_train[l] = available_indices[
                actual_count:
            ]  # Remove used indices

        node_train_indices = (
            np.concatenate(node_train_indices_list)
            if node_train_indices_list
            else np.array([], dtype=int)
        )

        # Proportionally allocate test data based on actual training samples for this node
        len(node_train_indices)
        node_test_indices_list = []
        if num_train > 0:  # Avoid division by zero
            for j, l in enumerate(node_labels):
                # Calc proportion of this label's training data in current node relative to this label's total training data
                # This logic for test set distribution in non-IID needs careful handling.
                # A simpler approach: distribute test data for these labels proportionally to the node's overall training data size

                # Simplified proportional allocation for test data based on training samples for that label in the node
                label_train_count_in_node = samples_per_label_in_node[
                    j
                ]  # Use the target count for proportion

                # Number of test samples for label l, proportional to train samples for label l in this node
                test_count_for_label = (
                    int(num_test * label_train_count_in_node / num_train)
                    if num_train > 0
                    else 0
                )

                available_test_indices = indices_per_label_test[l]
                actual_test_count = min(
                    test_count_for_label, len(available_test_indices)
                )
                node_test_indices_list.append(
                    available_test_indices[:actual_test_count]
                )
                indices_per_label_test[l] = available_test_indices[actual_test_count:]

        node_test_indices = (
            np.concatenate(node_test_indices_list)
            if node_test_indices_list
            else np.array([], dtype=int)
        )

        party_datas[i] = {  # Changed from i + 1 to i for 0-based indexing
            "x_train": x_train[node_train_indices],
            "y_train": y_train[node_train_indices],
            "x_test": x_test[node_test_indices],
            "y_test": y_test[node_test_indices],
        }
    return party_datas


# Adapted from lazymanta/datasets.py
def generate_data(
    dataset_name: str,
    n_workers: int,
    data_folder: Path,
    part: str,
    raw_data_parent_dir: Path,
):
    """
    Generate partitioned data based on the specified arguments.
    """
    if dataset_name.lower() == "cifar10":
        data = load_cifar10(download_dir=str(raw_data_parent_dir))
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Only 'cifar10' is supported by this script."
        )

    num_train = len(data[0][1])

    # Ensure total samples can be somewhat evenly distributed
    if num_train < n_workers:
        raise ValueError(
            f"Number of workers ({n_workers}) cannot be greater than total training samples ({num_train})."
        )

    base_ppp = num_train // n_workers
    remainder = num_train % n_workers

    ppp = [base_ppp] * n_workers
    for i in range(remainder):
        ppp[i] += 1

    dict_dataset = {
        "x_train": data[0][0],
        "y_train": data[0][1],
        "x_test": data[1][0],
        "y_test": data[1][1],
    }

    print(f"Generating {part} partitions for {n_workers} workers...")
    if part == "iid":
        party_datas = generate_IID_parties(dict_dataset, ppp)
    elif part.startswith("non-iid-"):
        try:
            labels_per_client = int(part.split("-")[-1])
            if not (1 <= labels_per_client <= 10):  # CIFAR-10 has 10 labels
                raise ValueError(
                    "Labels per client for non-IID CIFAR-10 must be between 1 and 10."
                )
        except ValueError:
            raise ValueError(
                "Invalid non-IID format. Use 'non-iid-X' where X is number of labels (e.g., non-iid-2)."
            )
        party_datas = generate_non_IID_label_parties(
            dict_dataset, ppp, labels_per_client
        )
    else:
        raise ValueError(
            "Unsupported partition type. Use 'iid' or 'non-iid-X' (e.g., 'non-iid-1', 'non-iid-2')."
        )

    data_folder.mkdir(parents=True, exist_ok=True)  # Ensure data_folder exists

    for i, node_data in party_datas.items():
        # Save for worker `i` (0-indexed)
        node_specific_folder = (
            data_folder / f"node_{i}"
        )  # Save under data_folder/node_i
        node_specific_folder.mkdir(exist_ok=True)
        file_path = node_specific_folder / "cifar10.npz"
        np.savez(file_path, **node_data)
        print(f"Data for worker {i} (node_{i}) saved at {file_path}")
        print(
            f"  Training samples: {len(node_data['y_train'])}, Test samples: {len(node_data['y_test'])}"
        )


def main(
    n_workers: int,
    partitioned_data_folder: Path = Path(__file__).parent.parent / "partitioned",
    raw_data_folder: Path = Path(__file__).parent.parent / "raw" / "cifar10",
    partition_type: str = "iid",
    seed: int = None,
):
    if seed is not None:
        global default_rng
        default_rng = np.random.default_rng(seed)
        print(f"Using random seed: {seed}")

    print(f"Partitioned data will be saved in: {partitioned_data_folder}")
    print(
        f"Raw data will be downloaded to/checked in: {raw_data_folder}"
    )  # load_cifar10 specific path

    # Create directories if they don't exist
    partitioned_data_folder.mkdir(parents=True, exist_ok=True)
    raw_data_folder.mkdir(parents=True, exist_ok=True)

    generate_data(
        dataset_name="cifar10",
        n_workers=n_workers,
        data_folder=partitioned_data_folder,  # This is where node_X subfolders will be created
        part=partition_type,
        raw_data_parent_dir=raw_data_folder,  # Pass base so generate_data can construct .../raw_data/cifar10
    )
    print("Data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and partition CIFAR-10 dataset for federated learning."
    )
    parser.add_argument(
        "-n",
        "--n_workers",
        type=int,
        required=True,
        help="Number of worker nodes to partition the data for.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="temp/partitioned",  # Default output folder within the example directory
        help="Folder to save the partitioned data and raw data. Partitions will be in <output_folder>/partitioned, raw in <output_folder>/raw_data.",
    )
    parser.add_argument(
        "-r",
        "--raw_data_folder",
        type=str,
        default="temp/raw/cifar10",  # Default output folder within the example directory
        help="Folder to save the raw data.",
    )
    parser.add_argument(
        "-p",
        "--partition_type",
        type=str,
        default="iid",
        help="Type of data partitioning. 'iid' or 'non-iid-X' (e.g., 'non-iid-1', 'non-iid-2'). Default is 'iid'.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    main(
        n_workers=args.n_workers,
        partitioned_data_folder=Path(args.output_folder),
        raw_data_folder=Path(args.raw_data_folder),
        partition_type=args.partition_type,
        seed=args.seed,
    )
