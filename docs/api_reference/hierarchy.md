# Hierarchy Class

The Hierarchy class is the core component of PyAOgmaNeo, implementing Sparse Predictive Hierarchies (SPH). It manages the network structure, learning process, and predictions.

## Constructor

```python
def __init__(self,
             io_descs: List[IODesc] = [],           # Input/output layer descriptors
             layer_descs: List[LayerDesc] = [],     # Hidden layer descriptors
             file_name: str = "",                   # Load from file (optional)
             buffer: bytes = b"")                   # Load from buffer (optional)
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|----------|
| io_descs | List[IODesc] | Input/output layer configurations | [] |
| layer_descs | List[LayerDesc] | Hidden layer configurations | [] |
| file_name | str | Path to load saved model | "" |
| buffer | bytes | Binary buffer containing model | b"" |

## Core Methods

### Training and Inference

```python
def step(input_cis: List[np.ndarray],              # Input arrays
         learn_enabled: bool = True,                # Enable learning
         reward: float = 0.0,                       # Reward signal
         mimic: float = 0.0)                        # Imitation signal
```

Example:
```python
# Basic prediction
h.step([input_data], learn_enabled=True)

# Reinforcement learning
h.step([state], learn_enabled=True, reward=reward)

# Imitation learning
h.step([observation], learn_enabled=True, mimic=target)
```

#### How `step()` Works

The `step()` method is the core of PyAOgmaNeo - it processes inputs, updates internal state, and optionally learns from prediction errors.

**Core Process:**
1. **Input Processing**: Accepts one array per IO layer in order
2. **State Update**: Updates network's temporal context
3. **Learning**: If enabled, adjusts weights based on prediction errors
4. **Signal Processing**: Applies reward/imitation signals for specialized learning

**Input Requirements:**
- **Exact Count**: Must provide `len(input_cis) == h.get_num_io()` arrays
- **Correct Shape**: Each array shape must match `(layer.x * layer.y,)`
- **Valid Range**: Values must be in `[0, layer.z-1]` (discrete indices)

**Usage Patterns:**

| Pattern | Configuration | Usage | Purpose |
|---------|---------------|-------|---------|
| **Autoregressive** | Single `prediction` layer | `h.step([[item]], True)` | Sequence modeling (same data type in/out) |
| **Input-Output Separation** | `none` + `prediction` | `h.step([input, prev_output], True)` | Different input/output types |
| **Reinforcement Learning** | `none` + `action` | `h.step([state, prev_action], True, reward=r)` | State-action learning |
| **Inference** | Any configuration | `h.step(inputs, False)` | Deployment/evaluation |

**Key Principles:**
- **Prediction First**: Always get predictions before stepping (especially in RL)
- **Previous Outputs as Inputs**: Feed previous predictions back as inputs for multi-layer configs
- **Temporal Order**: Process sequences in chronological order
- **State Management**: Use `clear_state()` between unrelated sequences/episodes

### Predictions

```python
# Get deterministic predictions
predictions = h.get_prediction_cis(layer_idx)

# Get prediction probabilities
probs = h.get_prediction_acts(layer_idx)

# Sample from prediction distribution
sampled = h.sample_prediction(layer_idx, temperature=1.0)

# Get hidden layer predictions
hidden_preds = h.get_layer_prediction_cis(layer_idx)
```

#### Understanding get_prediction_cis Return Value

`get_prediction_cis(layer_idx)` returns a list of integers with the predictions for the specified IO layer. The number of integers corresponds to the spatial structure of the IO layer - for a layer with dimensions `(x, y, z)`, you get `x * y` prediction integers (one per spatial location/CSDR). The layer_idx parameter refers to the index of the IO layer (not counting hidden layers).

```python
# For a single-location output (1x1 spatial):
h = neo.Hierarchy([
    neo.IODesc((1, 1, 10), io_type=neo.prediction)   # IO layer 0: 1 spatial location
], [
    neo.LayerDesc((16, 16, 64))
])

predictions = h.get_prediction_cis(0)[0]  # [0] gets the first (and only) CSDR

# For image output (28x28 spatial):
h = neo.Hierarchy([
    neo.IODesc((28, 28, 4), io_type=neo.prediction)  # IO layer 0: 784 spatial locations
], [
    neo.LayerDesc((16, 16, 64))
])

predictions = h.get_prediction_cis(0)     # Returns list of 784 integers (28*28)
first_pixel = predictions[0]              # Prediction for pixel (0,0)  
pixel_100 = predictions[99]               # Prediction for pixel at index 99
all_pixels = predictions                  # All 784 pixel predictions

# For multi-layer with different spatial structures:
h = neo.Hierarchy([
    neo.IODesc((28, 28, 4), io_type=neo.none),       # IO layer 0: 784 locations
    neo.IODesc((1, 1, 10), io_type=neo.prediction)   # IO layer 1: 1 location
], [
    neo.LayerDesc((16, 16, 64))
])

# Get predictions from the classification layer (single location)
class_prediction = h.get_prediction_cis(1)[0]        # [0] gets the only CSDR
# Get predictions from the image layer (multiple locations) 
image_predictions = h.get_prediction_cis(0)          # List of 784 integers
```

The indexing `[0]` gets the first CSDR (Column State Data Representation) prediction as an integer class index, but there may be many more depending on the spatial dimensions of the IO layer.

### State Access

```python
# Get hidden layer states
states = h.get_hidden_cis(layer_idx)

# Get layer information
num_layers = h.get_num_layers()
num_io = h.get_num_io()
io_size = h.get_io_size(layer_idx)
io_type = h.get_io_type(layer_idx)

# Get connectivity information
up_radius = h.get_up_radius(layer_idx)
down_radius = h.get_down_radius(layer_idx)
```

### Advanced Operations

```python
# Get receptive field
field, size = h.get_encoder_receptive_field(
    layer_idx,                  # Layer index
    visible_layer_idx,          # Visible layer index
    position                    # (x, y, z) position
)

# Merge hierarchies
h.merge(hierarchies, mode=neo.MergeMode.merge_average)
```

## Example Usage

### Classification

```python
# Create classification hierarchy
h = neo.Hierarchy([
    # Input layer
    neo.IODesc((28, 28, 4),
              io_type=neo.none,
              up_radius=8),
    # Output layer (10 classes)
    neo.IODesc((1, 1, 10),
              io_type=neo.prediction,
              num_dendrites_per_cell=64)
], [
    neo.LayerDesc((16, 16, 64))
])

# Training
for epoch in range(num_epochs):
    for img, label in dataset:
        # Process image
        h.step([preprocess(img), prev_label], True)
        prediction = h.get_prediction_cis(1)[0]  # Get prediction from second IO layer
        prev_label = label
```

### Sequence Prediction

```python
# Create sequence prediction hierarchy
h = neo.Hierarchy([
    neo.IODesc((1, 1, input_size), neo.prediction)
], [
    neo.LayerDesc((4, 4, 32), recurrent_radius=2)
])

# Training
for sequence in sequences:
    h.clear_state()  # Reset for new sequence
    for item in sequence:
        h.step([item], True)
        prediction = h.get_prediction_cis(0)[0]  # Get prediction from only IO layer
```

### Reinforcement Learning

```python
# Create RL hierarchy
h = neo.Hierarchy([
    neo.IODesc((8, 8, 16), neo.none),      # State
    neo.IODesc((1, 1, num_actions), neo.action)  # Actions
], [
    neo.LayerDesc((6, 6, 32))
])

# Training loop
while training:
    state = env.get_state()
    action = h.get_prediction_cis(1)[0]  # Get prediction from second IO layer
    reward = env.step(action)
    h.step([state], True, reward=reward)
```

## See Also

- [Layer Configuration](layer_config.md)
- [State Management](state_management.md)
- [Image Processing](image_encoder.md)
