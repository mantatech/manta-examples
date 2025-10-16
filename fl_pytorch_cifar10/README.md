# Federated Learning - CIFAR10 Example

This example demonstrates how to train a Convolutional Neural Network (CNN) on the CIFAR10 dataset using federated learning with the Manta platform.

## Overview

**Dataset**: CIFAR10 (32x32 RGB images, 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
**Model**: Convolutional Neural Network (CNN) with 3 conv layers
**Federated Learning Algorithm**: FedAvg (Federated Averaging)
**Estimated Training Time**: 15-20 minutes per round with 2-4 nodes

## What You'll Learn

- Training CNNs in a federated setting
- Handling more complex image data (RGB vs grayscale)
- Optimizing hyperparameters for image classification
- Monitoring convergence for more challenging datasets

## Prerequisites

### 1. Manta Platform Account
- Create an account at [dashboard.manta-tech.io](https://dashboard.manta-tech.io)
- Create a cluster and start it (status should be RUNNING)

### 2. Python Environment
- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- At least 8GB RAM recommended (4GB minimum)

### 3. Multiple Devices
- Minimum 2 devices/nodes for federated learning
- GPU recommended for faster training (optional)

## Step-by-Step Guide

### Step 1: Install Dependencies

Install Manta components:

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

Run the data preparation script to split the CIFAR10 dataset:

```bash
python prepare_data.py -n <number_of_nodes>
```

**Example for 2 nodes:**
```bash
python prepare_data.py -n 2
```

This creates:
```
temp/
├── partitioned/
│   ├── node_0/
│   │   └── cifar10.npz
│   └── node_1/
│       └── cifar10.npz
└── raw/
    └── cifar10.pickle
```

**Advanced Options:**

For non-IID data distribution:
```bash
python prepare_data.py -n 2 -p non-iid-2  # 2 classes per client
python prepare_data.py -n 2 -p non-iid-5  # 5 classes per client
```

With reproducible randomization:
```bash
python prepare_data.py -n 2 --seed 42
```

### Step 3: Configure Manta Nodes

For each node:

1. **Download Node Configuration**
   - Go to your cluster on [dashboard.manta-tech.io](https://dashboard.manta-tech.io)
   - Click "Configure New Node"
   - Download the `.toml` configuration
   - Save to `~/.manta/nodes/<node_name>.toml`

2. **Add Dataset Path**
   Edit the `.toml` file:
   ```toml
   [datasets.mappings]
   cifar10 = "/absolute/path/to/temp/partitioned/node_0/cifar10.npz"
   ```

   **Critical**: Use absolute paths and ensure each node has its own partition:
   - Node 0: `/path/to/temp/partitioned/node_0/cifar10.npz`
   - Node 1: `/path/to/temp/partitioned/node_1/cifar10.npz`

### Step 4: Start Manta Nodes

Start each node:

```bash
manta node start <node_name>
```

**Verify connectivity:**
```bash
manta node status
```

Check your dashboard - nodes should appear as connected.

**Troubleshoot with logs:**
```bash
manta node logs <node_name>
```

### Step 5: Run the Jupyter Notebook

1. Open `swarm.ipynb`
2. **Update credentials** in the first cell:
   ```python
   USERNAME = "your-email@example.com"
   PASSWORD = "your-password"
   ```
3. Execute cells in order
4. Monitor training metrics and results

## Understanding the Code

### Module Structure

- **worker.py**: CNN training on each node
  - Implements CNN architecture
  - Local training with SGD + momentum
  - Data augmentation (optional)

- **aggregator.py**: FedAvg weight aggregation

- **scheduler.py**: Training coordination and convergence checking

### CNN Architecture

```python
Conv2D(32 filters, 3x3) → ReLU → MaxPool(2x2)
Conv2D(64 filters, 3x3) → ReLU → MaxPool(2x2)
Conv2D(64 filters, 3x3) → ReLU → MaxPool(2x2)
Flatten → Dense(64) → ReLU → Dropout(0.5)
Dense(10) → Softmax
```

### Federated Learning Flow

```
1. Workers → Train CNN locally
2. Aggregator → Average model weights
3. Scheduler → Check convergence (target: 80% accuracy)
   ↓ (if not converged)
   Loop back to step 1
```

### Default Hyperparameters

```python
{
    "epochs": 1,              # Local epochs per round
    "batch_size": 64,         # Larger batch for CNN
    "optimizer": "SGD",
    "lr": 0.001,             # Lower lr for stability
    "momentum": 0.9,
    "weight_decay": 5e-4,    # L2 regularization
    "val_acc_threshold": 0.80 # Target accuracy
}
```

## Expected Results

With 2-3 nodes and default settings:
- **Round 1**: ~35-45% accuracy (random baseline: 10%)
- **Round 10**: ~60-70% accuracy
- **Round 20**: ~75-80% accuracy
- **Convergence**: Typically 20-30 rounds to reach 80%

**Note**: CIFAR10 is more challenging than MNIST due to:
- Higher image complexity
- More varied classes
- Color information (3 channels vs 1)

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce `batch_size` to 32 or 16
- Close other applications
- Use fewer local epochs
- Consider using CPU if GPU memory is limited

**Slow Training**
- Enable GPU in the notebook: `gpu = True`
- Reduce image size (requires model changes)
- Use fewer nodes for faster coordination

**Low Accuracy**
- Increase learning rate: `"lr": 0.01`
- Train for more rounds
- Check data distribution (use IID partitioning first)
- Verify dataset paths are correct

**Connection Issues**
- Verify cluster status (must be RUNNING)
- Check node logs for errors
- Ensure network stability
- Confirm nodes appear on dashboard

### Dataset Issues

**Download Fails**
- Check internet connection
- Manually download CIFAR10 from [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html)
- Place in `temp/raw/` directory

**Partition Not Found**
- Re-run `prepare_data.py`
- Verify absolute paths in `.toml` files
- Check file permissions

## Customization Ideas

### Improve Model Performance

1. **Add Data Augmentation**
   In `modules/worker.py`, add transforms:
   ```python
   transforms = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ])
   ```

2. **Deeper Network**
   Add more convolutional layers for better feature extraction

3. **Residual Connections**
   Implement ResNet-style skip connections

### Experiment with FL Algorithms

- **FedProx**: Add proximal term to handle heterogeneity
- **FedAdam**: Use adaptive learning rates
- **Scaffold**: Implement variance reduction

### Non-IID Challenges

Test robustness with extreme non-IID:
```bash
python prepare_data.py -n 5 -p non-iid-1  # 1 class per client
```

This simulates highly heterogeneous data distributions.

## Comparison with MNIST

| Aspect | MNIST | CIFAR10 |
|--------|-------|---------|
| **Complexity** | Simple | Moderate |
| **Image Size** | 28x28x1 | 32x32x3 |
| **Classes** | 10 digits | 10 objects |
| **Baseline** | ~90% | ~40% |
| **Target** | ~99% | ~80-85% |
| **Training Time** | Faster | Slower |
| **Memory** | Low | Higher |

## Next Steps

1. **Optimize**: Tune hyperparameters for better accuracy
2. **Scale**: Add more nodes and observe communication overhead
3. **Advanced**: Implement differential privacy or secure aggregation
4. **Custom**: Adapt for your own image dataset

## Resources

- **Manta Docs**: [docs.manta-tech.io](https://docs.manta-tech.io/)
- **CIFAR10 Dataset**: [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **FedAvg Paper**: [arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629)
- **CNN Tutorial**: [pytorch.org/tutorials](https://pytorch.org/tutorials/)

## Support

- **Documentation**: [docs.manta-tech.io](https://docs.manta-tech.io/)
- **Issues**: [github.com/mantatech/manta-examples](https://github.com/mantatech/manta-examples)
- **Email**: support@manta-tech.io
