# Installation Guide

This guide will help you install PyAOgmaNeo and its dependencies.

## Prerequisites

- OpenMP (likely already installed on your system)
- pybind11 (will automatically install if not present)
- cmake

## Installation Methods

### From PyPI (Recommended)

The simplest way to install PyAOgmaNeo is through pip:

```bash
pip install pyaogmaneo
```

This will automatically:
- Download the required AOgmaNeo library
- Compile the necessary components
- Install all dependencies

### From Source

To install from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/ogmacorp/PyAOgmaNeo.git
   cd PyAOgmaNeo
   ```

2. Install:
   ```bash
   pip install .
   ```

#### Branch Selection

The AOgmaNeo library branch used for building is based on the current branch of PyAOgmaNeo. The build system automatically downloads the AOgmaNeo branch matching your checked-out PyAOgmaNeo branch.

## Using System AOgmaNeo

If you want to use an existing system installation of AOgmaNeo:

```bash
export USE_SYSTEM_AOGMANEO
pip install .
```

## Verification

TODO

## Troubleshooting

TODO

Common installation issues and their solutions will be documented here.

## Next Steps

- Check out our [Quickstart Guide](quickstart.md)
- Learn about [Basic Concepts](basic_concepts.md)
- Try the [First Hierarchy Tutorial](../tutorials/basic/first_hierarchy.md) 