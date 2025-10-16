# Federated Learning - MNIST Example

This example demonstrates how to train a Multi-Layer Perceptron (MLP) on the MNIST dataset using federated learning with the Manta platform.

## Overview

**Dataset**: MNIST handwritten digits (28x28 grayscale images, 10 classes)
**Model**: Multi-Layer Perceptron (MLP) with 2 hidden layers
**Federated Learning Algorithm**: FedAvg (Federated Averaging)
**Estimated Training Time**: 10-15 minutes per round with 2-4 nodes

## What You'll Learn

- How to partition datasets for federated learning
- How to configure and start Manta nodes
- How to define and deploy a federated learning swarm
- How to monitor training progress and results

## Prerequisites

### 1. Manta Platform Account
- Create an account at [dashboard.manta-tech.io](https://dashboard.manta-tech.io)
- Create a cluster and start it (status should be RUNNING)

### 2. Python Environment
- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- At least 4GB RAM available

### 3. Multiple Devices (or virtual environments)
- Minimum 2 devices/nodes for federated learning
- Can be run on a single machine using multiple manta-node instances

## Step-by-Step Guide

### Step 1: Install Dependencies

Install both the Manta SDK (for client) and manta-node (for edge nodes):

```bash
# In your notebook environment
pip install manta-sdk

# On each node device
pip install manta-node
```

Install example-specific dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Partition the Dataset

Run the data preparation script to split the MNIST dataset across your nodes:

```bash
python prepare_data.py -n <number_of_nodes>
```

**Example for 3 nodes:**
```bash
python prepare_data.py -n 3
```

This will create:
```
temp/
├── partitioned/
│   ├── node_0/
│   │   └── mnist.npz
│   ├── node_1/
│   │   └── mnist.npz
│   └── node_2/
│       └── mnist.npz
└── raw/
    └── mnist.npz
```

**Advanced Options:**

For non-IID data distribution (2 labels per client):
```bash
python prepare_data.py -n 3 -p non-iid-2
```

With a specific random seed for reproducibility:
```bash
python prepare_data.py -n 3 --seed 42
```

### Step 3: Configure Manta Nodes

For each node you want to connect:

1. **Download Node Configuration**
   - Go to your cluster page on [dashboard.manta-tech.io](https://dashboard.manta-tech.io)
   - Click "Configure New Node"
   - Follow the wizard and download the `.toml` file
   - Save it to `~/.manta/nodes/<node_name>.toml`

2. **Update Dataset Path**
   Edit the downloaded `.toml` file and add the dataset mapping:
   ```toml
   [datasets.mappings]
   mnist = "/absolute/path/to/temp/partitioned/node_0/mnist.npz"
   ```

   **Important**: Use the absolute path and ensure each node points to its specific partition:
   - Node 0: `/path/to/temp/partitioned/node_0/mnist.npz`
   - Node 1: `/path/to/temp/partitioned/node_1/mnist.npz`
   - etc.

### Step 4: Start Manta Nodes

On each device, start the Manta node:

```bash
manta node start <node_name>
```

**Verify nodes are connected:**
```bash
manta node status
```

Or check your cluster dashboard - connected nodes should appear in the "Connected Nodes" section.

**View node logs if needed:**
```bash
manta node logs <node_name>
```

### Step 5: Run the Jupyter Notebook

1. Open `swarm.ipynb` in Jupyter
2. **Replace credentials** in the first code cell:
   ```python
   USERNAME = "your-email@example.com"  # Your actual email
   PASSWORD = "your-password"           # Your actual password
   ```
3. Execute cells sequentially
4. Monitor training progress and results

## Understanding the Code

### Module Structure

The example consists of 4 main modules:

- **worker/**: Trains the local model on each node
  - `worker_task.py`: Training logic
  - `model.py`: MLP architecture

- **worker_test/**: Evaluates the global model
  - `worker_task.py`: Evaluation logic
  - Uses the same model architecture

- **aggregator.py**: Aggregates model weights using FedAvg

- **scheduler.py**: Coordinates training rounds and checks convergence

### Federated Learning Workflow

```
1. Worker Nodes → Train locally on partitioned data
2. Aggregator → Combines weights using FedAvg
3. Test Workers → Evaluate global model
4. Scheduler → Check convergence, decide to continue or stop
   ↓ (if not converged)
   Loop back to step 1
```

### Hyperparameters

Default configuration in the notebook:
```python
{
    "epochs": 1,              # Local epochs per round
    "batch_size": 32,         # Training batch size
    "optimizer": "SGD",       # Optimization algorithm
    "lr": 0.01,              # Learning rate
    "momentum": 0.9,         # SGD momentum
    "val_acc_threshold": 0.99 # Target accuracy for convergence
}
```

You can modify these in the notebook to experiment with different settings.

## Expected Results

With default hyperparameters and 2-3 nodes:
- **Round 1**: ~70-80% accuracy
- **Round 5**: ~90-95% accuracy
- **Round 10**: ~95-98% accuracy
- **Convergence**: Usually achieved within 10-15 rounds

## Troubleshooting

### No Nodes Connected
- Check node status: `manta node status`
- View logs: `manta node logs <node_name>`
- Verify cluster is RUNNING on dashboard
- Check network connectivity

### Dataset Not Found
- Verify you ran `prepare_data.py`
- Check dataset paths in node `.toml` files
- Ensure paths are absolute, not relative
- Verify each node has access to its partition

### Authentication Failed
- Double-check your username and password
- Ensure you're registered at dashboard.manta-tech.io
- Try logging into the dashboard to verify credentials

### Swarm Deployment Failed
- Verify all nodes are connected
- Check that Docker image `manta_light:pytorch` exists on nodes
- Review cluster logs on the dashboard

### Training Stuck or Slow
- Check node logs for errors
- Verify network connectivity between nodes
- Monitor resource usage (CPU/RAM) on nodes
- Consider reducing batch_size if out of memory

## Customization Ideas

### Change the Model
Edit `modules/worker/model.py` to use a different architecture:
- Add more layers
- Change activation functions
- Add dropout for regularization

### Modify Data Distribution
Use non-IID partitioning for more realistic scenarios:
```bash
python prepare_data.py -n 3 -p non-iid-1  # 1 label per client
python prepare_data.py -n 3 -p non-iid-2  # 2 labels per client
```

### Experiment with Hyperparameters
In the notebook, modify:
- Learning rate: `"lr": 0.001` (lower) or `"lr": 0.1` (higher)
- Batch size: `"batch_size": 64` or `"batch_size": 16`
- Local epochs: `"epochs": 2` or `"epochs": 5`

### Add Privacy Mechanisms
Explore differential privacy by adding noise to gradients (advanced).

## Next Steps

After completing this example:

1. **Try CIFAR10**: Move to the more advanced [CIFAR10 example](../fl_pytorch_cifar10/)
2. **Scale Up**: Add more nodes to see how federated learning scales
3. **Custom Dataset**: Adapt the code for your own dataset
4. **Advanced FL**: Explore FedProx, FedAdam, or other FL algorithms

## Resources

- **Manta Documentation**: [docs.manta-tech.io](https://docs.manta-tech.io/)
- **MNIST Dataset**: [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/)
- **Federated Learning Paper**: [FedAvg (McMahan et al., 2017)](https://arxiv.org/abs/1602.05629)

## Need Help?

- **Documentation**: [docs.manta-tech.io](https://docs.manta-tech.io/)
- **GitHub Issues**: [github.com/mantatech/manta-examples/issues](https://github.com/mantatech/manta-examples/issues)
- **Email Support**: support@manta-tech.io
