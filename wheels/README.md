# Manta Wheel Files

This directory will contain pre-built wheel files for Manta components, enabling offline installation or specific version pinning.

## Current Installation (PyPI)

For now, install the latest versions directly from PyPI:

```bash
# Install Manta SDK (for client/notebook environments)
pip install manta-sdk

# Install Manta Node (for edge devices)
pip install manta-node
```

## Coming Soon: Offline Installation

Pre-built wheel files will be provided here for:

- `manta-sdk` - Client SDK for interacting with the Manta platform
- `manta-node` - Node agent for edge devices
- Specific version releases for reproducibility

## Why Use Wheel Files?

- **Offline Installation**: Install without internet access
- **Version Pinning**: Lock to specific tested versions
- **Faster Installation**: Pre-built binaries skip compilation
- **Air-Gapped Environments**: Deploy in isolated networks

## How to Use (When Available)

```bash
# Download the wheel file to your device
# Then install locally:
pip install manta_sdk-<version>-py3-none-any.whl
pip install manta_node-<version>-py3-none-any.whl
```

## Get Latest Versions

Until wheel files are available here, get the latest from:

- **PyPI**: [pypi.org/project/manta-sdk](https://pypi.org/project/manta-sdk/)
- **GitHub Releases**: [github.com/mantatech](https://github.com/mantatech)

## Questions?

Contact <support@manta-tech.io> or visit [docs.manta-tech.io](https://docs.manta-tech.io/)
