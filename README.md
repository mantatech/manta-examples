# Manta Examples — Moved

> **This repository is in archive-prep mode.** The federated learning examples
> previously hosted here have moved into [`mantatech/manta-sdk`](https://github.com/mantatech/manta-sdk)
> per [ADR-0014 (Examples Organization)](https://github.com/mantatech/manta-deploy/blob/main/docs/adr/0014-examples-organization.md).

## Where to find the examples now

| Example | New location |
|---------|--------------|
| `fl_pytorch_mnist` | [`mantatech/manta-sdk/examples/fl_pytorch_mnist/`](https://github.com/mantatech/manta-sdk/tree/main/examples/fl_pytorch_mnist) |
| `fl_pytorch_cifar10` | [`mantatech/manta-sdk/examples/fl_pytorch_cifar10/`](https://github.com/mantatech/manta-sdk/tree/main/examples/fl_pytorch_cifar10) |

## New install command

The example dependencies are now an SDK extra:

```bash
pip install "manta-sdk[examples]"
```

This pulls `torch`, `torchvision`, and `jupyter` alongside the SDK. Then run
the example notebooks directly out of the SDK install:

```bash
git clone https://github.com/mantatech/manta-sdk.git
cd manta-sdk/examples/fl_pytorch_mnist
python prepare_data.py -n 3
jupyter notebook swarm.ipynb
```

## Why the move

Co-locating the examples with the SDK makes them installable, versioned with
the SDK release, and discoverable from a single repo. ADR-0014 has the full
rationale.

## Status of this repository

- **Live but read-only candidate.** No new content lands here.
- This repository will be **archived on GitHub** once the migrated examples
  are available on a stable manta-sdk PyPI release. Existing links (blog
  posts, conference talks, tutorial sites) keep working — GitHub keeps
  archived public repos browsable.
- The historical `wheels/` placeholder has been removed (it was a stub for
  offline SDK/node installs that the SDK distributes natively now).

## Need help

- SDK docs: <https://docs.manta-tech.io/>
- SDK source: <https://github.com/mantatech/manta-sdk>
- Platform dashboard: <https://dashboard.manta-tech.io>

_Migration ADR: [`mantatech/manta-deploy` ADR-0014](https://github.com/mantatech/manta-deploy/blob/main/docs/adr/0014-examples-organization.md)_
