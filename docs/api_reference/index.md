# PyAOgmaNeo API Reference

PyAOgmaNeo is a Python binding for AOgmaNeo (Adaptive Ogma Neo), a library implementing Sparse Predictive Hierarchies (SPH). SPH is a neuromorphic machine learning architecture that combines hierarchical processing with sparse coding and temporal prediction, inspired by the structure and function of biological neural networks. It creates adaptive, general-purpose learning systems that process information efficiently.

### Key Features
- Neuromorphic hierarchical sparse predictive coding
- Online learning with no separate training phase
- Temporal sequence prediction and learning
- Reinforcement learning capabilities
- Efficient sparse representations

### Quick Start

```python
import pyaogmaneo as neo
import numpy as np

# Set number of threads for parallel processing
neo.set_num_threads(4)

# Create a simple prediction hierarchy
h = neo.Hierarchy([
    neo.IODesc((8, 8, 16), io_type=neo.none),      # Input layer
    neo.IODesc((1, 1, 10), io_type=neo.prediction) # Output layer
], [
    neo.LayerDesc((6, 6, 32))  # Hidden layer
])

# Training loop
for _ in range(1000):
    input_data = prepare_input()
    h.step([input_data], learn_enabled=True)
    prediction = h.get_prediction_cis(1)
```

> **Complete Tutorial:** For step-by-step examples with real datasets, see the [Quickstart Guide](../getting_started/quickstart.md).

## Architecture Overview

PyAOgmaNeo hierarchies consist of **IO layers** and **hidden layers**:

- **IO Layers**: Interface with external data
  - `none`: Input-only layers (sensors, data sources)
  - `prediction`: Output layers for predictions/classifications  
  - `action`: Output layers for reinforcement learning
- **Hidden Layers**: Process and connect IO layers internally

**Data Flow**: Input data flows through `none` layers → hidden layers → `prediction`/`action` layers → outputs. All IO layers receive input during `step()`, but only `prediction`/`action` layers generate outputs via `get_prediction_cis()`.


## Core Components

### [Hierarchy](hierarchy.md)
The main class for creating and managing SPH networks. Handles:
- Network creation and configuration
- Training and inference
- State management
- Model persistence

### [Layer Configuration](layer_config.md)
Documentation for layer configuration including:
- LayerDesc: Hidden layer configuration
- IODesc: Input/output layer configuration
- Parameter tuning guidelines

### [Image Processing](image_encoder.md)
Specialized components for image processing:
- ImageEncoder class
- Image layer configuration
- Visual processing patterns

### [State Management](state_management.md)
Guide to managing model state:
- State operations
- Persistence patterns
- Runtime management
- Error handling

### [Model Persistence](model_persistence.md)
Guide to saving and loading models:
- File-based storage
- Buffer serialization
- Best practices
- Version management

### [Common Patterns](common_patterns.md)
Implementation patterns for common tasks:
- Reinforcement learning
- Sequence prediction
- Classification
- Time series forecasting

### [Parameters and Tuning](parameters.md)
Detailed parameter documentation:
- EncoderParams
- DecoderParams
- ActorParams
- LayerParams
- IOParams

## Global Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `set_num_threads(n: int)` | Set thread count | n ≥ 1 |
| `get_num_threads() -> int` | Get thread count | - |
| `set_global_state(state: int)` | Set random state | Any integer |
| `get_global_state() -> int` | Get random state | - |

## Examples

- [Fashion MNIST Classification](../examples/fashion_mnist/fashion_mnist.md)
- [Text Prediction](../examples/nlp/text_prediction.md)
- [Wave Prediction](../examples/wave_prediction.md)
- [Reinforcement Learning](../examples/rl/cartpole.md)

## See Also

- [Installation Guide](../getting_started/installation.md)
- [Tutorial](../getting_started/tutorial.md)
- [FAQ](../getting_started/faq.md)
- [Contributing](../development/contributing.md)

