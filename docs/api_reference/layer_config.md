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
    none = 0        # Regular input layer
    prediction = 1  # Layer that makes predictions
    action = 2      # Layer that generates actions
```

### Example Configurations

1. **Input Layer**
```python
# Regular input (e.g., sensor data)
input_desc = neo.IODesc(
    size=(10, 10, 16),
    io_type=neo.none,
    up_radius=4  # Larger radius for feature detection
)

# Image input
image_input = neo.IODesc(
    size=(28, 28, 4),  # Image dimensions with quantization
    io_type=neo.none,
    up_radius=8  # Large field for spatial patterns
)
```

2. **Prediction Layer**
```python
# Sequence prediction
pred_desc = neo.IODesc(
    size=(1, 1, vocab_size),
    io_type=neo.prediction,
    num_dendrites_per_cell=8,  # More connections for complex patterns
    history_capacity=1024  # Larger history for long sequences
)

# Classification output
class_desc = neo.IODesc(
    size=(1, 1, num_classes),
    io_type=neo.prediction,
    num_dendrites_per_cell=64  # Many dendrites for robust classification
)
```

3. **Action Layer**
```python
# Discrete actions
action_desc = neo.IODesc(
    size=(1, 1, num_actions),
    io_type=neo.action,
    value_num_dendrites_per_cell=16  
)
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

## See Also

- [Hierarchy Class](hierarchy.md)
- [Image Processing](image_encoder.md) 