# Manta Examples

Ready-to-run federated learning examples for the Manta platform. This is a public repository for end users learning to develop and deploy distributed ML workloads.

## Documentation

- `README.md` - Getting started guide, quick start workflow, system requirements
- `fl_pytorch_mnist/README.md` - MNIST example: step-by-step, module structure, hyperparameters
- `fl_pytorch_cifar10/README.md` - CIFAR10 example: CNN architecture, advanced usage
- **Platform docs**: https://mantatech.github.io/manta-docs/

## Quick Reference

- **Stack**: Python 3.9+, PyTorch, Jupyter, manta-sdk, manta-node
- **Examples**: `fl_pytorch_mnist/` (beginner), `fl_pytorch_cifar10/` (intermediate)
- **Workflow**: prepare_data.py → configure nodes → start nodes → run swarm.ipynb
- **Modules pattern**: Each example has `modules/` with worker, aggregator, scheduler
- **Data prep**: `python prepare_data.py -n <nodes>` (supports IID and non-IID partitioning)
- **Wheels**: `wheels/` directory for offline SDK/node installation
- **Dashboard**: https://dashboard.manta-tech.io
- **Rule**: Notebooks must be self-contained and reproducible without internal platform access
