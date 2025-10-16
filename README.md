# Manta Examples

Welcome to the Manta Examples repository! This repository contains ready-to-run examples demonstrating federated learning and distributed computing on the Manta platform.

## Getting Started

### 1. Create an Account

Visit [dashboard.manta-tech.io](https://dashboard.manta-tech.io) to:

- Create your Manta account
- Set up your first cluster
- Get your authentication credentials

### 2. Install Manta Components

#### For Client/Notebook Environment

Install the Manta SDK to interact with the platform:

```bash
pip install manta-sdk
```

#### For Edge Nodes

Install manta-node on each device that will participate in distributed training:

```bash
pip install manta-node
```

## Available Examples

### [Federated Learning - MNIST](./fl_pytorch_mnist/)

Train a neural network on the MNIST digit classification dataset using federated learning.

- **Dataset**: MNIST (28x28 grayscale handwritten digits)
- **Model**: Multi-layer Perceptron (MLP)
- **Complexity**: Beginner-friendly
- **Estimated time**: 10-15 minutes per round

[View MNIST Example →](./fl_pytorch_mnist/README.md)

### [Federated Learning - CIFAR10](./fl_pytorch_cifar10/)

Train a convolutional neural network on the CIFAR10 image classification dataset.

- **Dataset**: CIFAR10 (32x32 RGB images, 10 classes)
- **Model**: Convolutional Neural Network (CNN)
- **Complexity**: Intermediate
- **Estimated time**: 15-20 minutes per round

[View CIFAR10 Example →](./fl_pytorch_cifar10/README.md)

## Quick Start Workflow

All examples follow a similar workflow:

1. **Install Dependencies**

   ```bash
   pip install manta-sdk manta-node
   ```

2. **Partition Your Dataset**

   ```bash
   cd <example_directory>
   python prepare_data.py -n <number_of_nodes>
   ```

3. **Configure Nodes**
   - Download node configurations from the dashboard
   - Save them to `~/.manta/nodes/<node_name>.toml`
   - Update dataset paths in each configuration

4. **Start Nodes**

   ```bash
   manta node start <node_name>
   ```

5. **Run the Notebook**
   - Open the `swarm.ipynb` notebook
   - Replace credentials with your own
   - Execute cells sequentially

## Installing from Wheel Files (Coming Soon)

Pre-built wheel files for manta-sdk and manta-node will be available in the [`wheels/`](./wheels/) directory for offline installation or specific versions.

## Documentation

- **Platform Documentation**: [docs.manta-tech.io](https://docs.manta-tech.io/)
- **API Reference**: [docs.manta-tech.io/api](https://docs.manta-tech.io/api)
- **GitHub**: [github.com/mantatech](https://github.com/mantatech)

## System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB (8GB recommended for CIFAR10)
- **Storage**: 2GB free space per example
- **Network**: Stable internet connection for communication with Manta platform

### Recommended Requirements

- **Python**: 3.10 or 3.11
- **RAM**: 8GB or more
- **GPU**: CUDA-compatible GPU for faster training (optional)
- **Storage**: 5GB free space

## Troubleshooting

### Common Issues

**Connection Failed**

- Verify your credentials are correct
- Check that your cluster is in RUNNING status
- Ensure network connectivity to api.manta-tech.io

**No Nodes Connected**

- Verify nodes are started: `manta node status`
- Check node logs: `manta node logs <node_name>`
- Verify dataset paths in node configurations

**Import Errors**

- Ensure manta-sdk is installed: `pip install manta-sdk`
- Verify Python version: `python --version` (should be 3.9+)

**Dataset Not Found**

- Run the prepare_data.py script first
- Verify dataset paths match node configurations
- Check that partitioned data exists in expected directories

### Getting Help

- **Documentation**: [docs.manta-tech.io](https://docs.manta-tech.io/)
- **Community**: [GitHub Discussions](https://github.com/mantatech/manta-examples/discussions)
- **Issues**: [GitHub Issues](https://github.com/mantatech/manta-examples/issues)
- **Email**: <support@manta-tech.io>

## Contributing

We welcome contributions! If you have an example you'd like to share:

1. Fork this repository
2. Create a new example directory
3. Include a README.md, notebook, and data preparation script
4. Submit a pull request

## License

This repository is provided as-is for educational and demonstration purposes. See individual example directories for specific licensing information.

## About Manta

Manta is a universal orchestrator for decentralized and federated AI. It enables you to deploy distributed machine learning workloads across edge devices, cloud servers, and hybrid infrastructures with enterprise-grade security and scalability.

Learn more at [manta-tech.io](https://manta-tech.io)
