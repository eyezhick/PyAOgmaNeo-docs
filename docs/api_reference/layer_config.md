# Layer Configuration

This document covers the configuration of layers in PyAOgmaNeo's Sparse Predictive Hierarchies.

## Overview

PyAOgmaNeo uses two main classes for layer configuration:
- `IODesc`: Configures input/output layers
- `LayerDesc`: Configures hidden layers

## IODesc (Input/Output Descriptor)

### Constructor

```python
def __init__(self,
             size: Tuple[int, int, int] = (5, 5, 16),  # Layer dimensions
             io_type: IOType = IOType.prediction,       # Layer type
             num_dendrites_per_cell: int = 4,          # Connection density
             value_num_dendrites_per_cell: int = 8,    # Value connections
             up_radius: int = 2,                       # Upward connectivity
             down_radius: int = 2,                     # Downward connectivity
             history_capacity: int = 512)              # Temporal memory
```

### Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| size | Tuple[int, int, int] | Layer dimensions (width, height, features) | All values ≥ 1 |
| io_type | IOType | Layer functionality | none/prediction/action |
| num_dendrites_per_cell | int | Number of dendrites per cell | ≥ 1 |
| value_num_dendrites_per_cell | int | Number of value dendrites | ≥ 1 |
| up_radius | int | Upward connection radius | ≥ 0 |
| down_radius | int | Downward connection radius | ≥ 0 |
| history_capacity | int | Size of history buffer | ≥ 2 |

### IO Types

```python
class IOType:
    none = 0        # Input-only layer
    prediction = 1  # Prediction layer
    action = 2      # Action layer for RL
```

IO layers can function in two configurations:

**Single Layer Configuration**: The layer serves as both input and output, enabling autoregressive learning for sequence modeling tasks.

**Multi-Layer Configuration**: Separate layers handle input and output, providing clear separation for classification and encoder-decoder architectures.

#### Layer Type Behaviors

| Type | Single Layer | Multi-Layer | Methods Available | Primary Use Cases |
|------|-------------|-------------|-------------------|-------------------|
| `none` | Not supported* | Input only | None | Raw data input (images, sensors) |
| `prediction` | Input + Output | Output only | `get_prediction_cis()`, `get_prediction_acts()` | Sequence prediction, classification |
| `action` | Input + Output | Output only | `get_prediction_cis()` (with exploration) | Reinforcement learning |

*`none` layers cannot exist alone as they provide no output mechanism.

#### Data Flow Architecture

**Multi-Layer Data Flow:**
```
Input Data → none layers ↘
                         Hidden Layers → prediction/action layers → Outputs
Previous Outputs → prediction/action layers ↗
```

**Key Principles:**
- **All IO layers receive input** during `step()` - one array per layer in order
- **Only `prediction`/`action` layers generate outputs** via `get_prediction_cis()`
- **Hidden layers connect inputs to outputs** and process temporal patterns
- **Previous outputs typically feed back** as inputs to maintain temporal consistency

**Multi-Input Example:**
```python
# 2 inputs + 1 output
h = neo.Hierarchy([
    neo.IODesc((16, 16, 8), io_type=neo.none),     # Visual input
    neo.IODesc((1, 1, 100), io_type=neo.none),     # Text input
    neo.IODesc((1, 1, 10), io_type=neo.prediction) # Combined output
], [hidden_layers])

# Usage: provide input for ALL layers
h.step([visual_data, text_data, prev_output], learn_enabled=True)
prediction = h.get_prediction_cis(2)[0]  # Only from prediction layer
```

### Example Configurations

#### Single Layer Examples

**1. Sequence Prediction (Single `prediction` layer)**
```python
# Word/character prediction - layer does both input and output
h = neo.Hierarchy([
    neo.IODesc((1, 1, vocab_size), 
              io_type=neo.prediction,
              history_capacity=1024)  # Large history for sequences
], [hidden_layers])

# Usage pattern:
for item in sequence:
    prediction = h.get_prediction_cis(0)[0]  # What's next?
    h.step([[item]], learn_enabled=True)     # Here's what actually came next
```

**2. Time Series Prediction**
```python
# Single layer predicts next value in series
h = neo.Hierarchy([
    neo.IODesc((1, 1, num_features),
              io_type=neo.prediction,
              num_dendrites_per_cell=8)
], [hidden_layers])
```

#### Multi-Layer Examples

**1. Classification (Input + Output layers)**
```python
h = neo.Hierarchy([
    # Input layer - receives images/data
    neo.IODesc((28, 28, 4),
              io_type=neo.none,      # Input only
              up_radius=8),
    
    # Output layer - makes classifications  
    neo.IODesc((1, 1, 10),
              io_type=neo.prediction, # Output only
              num_dendrites_per_cell=64)
], [hidden_layers])

# Usage:
h.step([image_data, prev_label], learn_enabled=True)
prediction = h.get_prediction_cis(1)[0]  # From output layer
```

**2. Reinforcement Learning (State + Action layers)**
```python
h = neo.Hierarchy([
    # State input layer
    neo.IODesc((8, 8, 16), 
              io_type=neo.none),     # Input only
    
    # Action output layer
    neo.IODesc((1, 1, num_actions), 
              io_type=neo.action,    # Output with RL features
              value_num_dendrites_per_cell=16)
], [hidden_layers])

# Usage:
h.step([state, prev_action], learn_enabled=True, reward=reward)
action = h.get_prediction_cis(1)[0]  # From action layer
```

**3. Mixed Input Types**
```python
h = neo.Hierarchy([
    # Image input
    neo.IODesc((32, 32, 4), io_type=neo.none),
    
    # Text input  
    neo.IODesc((1, 1, vocab_size), io_type=neo.none),
    
    # Combined prediction output
    neo.IODesc((1, 1, num_classes), io_type=neo.prediction)
], [hidden_layers])
```

#### Advanced Examples

**1. Multi-Modal Prediction**
```python
# Multiple inputs, single predictive output
h = neo.Hierarchy([
    neo.IODesc((16, 16, 8), io_type=neo.none),        # Visual input
    neo.IODesc((1, 1, 100), io_type=neo.none),        # Text input  
    neo.IODesc((1, 1, action_space), io_type=neo.action)  # Action output
], [hidden_layers])
```

**2. Hierarchical RL**
```python
# High-level and low-level actions
h = neo.Hierarchy([
    neo.IODesc((8, 8, 16), io_type=neo.none),         # State
    neo.IODesc((1, 1, high_actions), io_type=neo.action),  # High-level actions
    neo.IODesc((1, 1, low_actions), io_type=neo.action)    # Low-level actions
], [hidden_layers])
```

## LayerDesc (Hidden Layer Descriptor)

### Constructor

```python
def __init__(self,
             hidden_size: Tuple[int, int, int] = (5, 5, 16),  # Layer dimensions
             num_dendrites_per_cell: int = 4,                 # Connection density
             up_radius: int = 2,                              # Upward field
             recurrent_radius: int = 0,                       # Recurrent field
             down_radius: int = 2)                            # Downward field
```

### Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| hidden_size | Tuple[int, int, int] | Layer dimensions | All values ≥ 1 |
| num_dendrites_per_cell | int | Connection density | ≥ 1 |
| up_radius | int | Upward connection radius | ≥ 0 |
| recurrent_radius | int | Recurrent connection radius | ≥ -1 |
| down_radius | int | Downward connection radius | ≥ 0 |

### Example Configurations

1. **Standard Hidden Layer**
```python
# Basic processing layer
hidden_desc = neo.LayerDesc(
    hidden_size=(8, 8, 32),
    num_dendrites_per_cell=4
)
```

2. **Temporal Processing**
```python
# Sequence learning layer
temporal_desc = neo.LayerDesc(
    hidden_size=(6, 6, 64),
    recurrent_radius=2,
    num_dendrites_per_cell=8
)
```

## Configuration Guidelines

### Choosing Single vs Multi-Layer

**Use Single Layer When:**
- Sequence modeling (language, time series)
- Autoregressive tasks (predict next in sequence)
- Self-supervised learning
- Simple RL with action history

**Use Multi-Layer When:**
- Classification tasks
- Encoder-decoder architectures  
- Complex RL (separate state and action spaces)
- Multi-modal inputs
- Clear input/output separation needed

### Common Patterns

| Task Type | Configuration | Example |
|-----------|---------------|---------|
| Language Modeling | Single `prediction` | Word/character prediction |
| Image Classification | `none` + `prediction` | MNIST, CIFAR |
| Reinforcement Learning | `none` + `action` | CartPole, Atari |
| Time Series | Single `prediction` | Stock prices, sensor data |
| Translation | `none` + `prediction` | Seq2seq tasks |

## See Also

- [Hierarchy Class](hierarchy.md) - Using configured layers
- [State Management](state_management.md) - Managing layer states
- [Image Processing](image_encoder.md) - Specialized image handling 