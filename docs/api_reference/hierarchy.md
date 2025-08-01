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

## Usage Patterns and Examples

### Training vs Inference Modes

PyAOgmaNeo hierarchies can operate in two distinct modes, each serving different purposes in your machine learning workflow.

> **Performance Note:** Unlike traditional neural networks, PyAOgmaNeo has only a ~2x performance difference between training and inference modes, making it highly efficient for online learning and real-time applications.

#### Why Training is Only ~2x Slower Than Inference

This efficiency stems from PyAOgmaNeo's neuromorphic architecture:

**Shared Forward Pass:** Both training and inference execute identical forward computations:
- Sparse activations through winner-take-all mechanisms
- Receptive field calculations 
- Activation computations and softmax normalization
- Prediction generation

**Lightweight Learning:** Training only adds simple weight updates:
- Local Hebbian-style learning rules (no complex backpropagation)
- Direct weight adjustments based on prediction errors
- No gradient chains or complex optimization algorithms
- Quantized 8-bit weights for efficient memory access

**Contrast with Traditional Neural Networks:**
- **Traditional:** 10x-100x slower training due to backpropagation, gradient computation, and complex optimizers
- **PyAOgmaNeo:** ~2x slower training due to simple local weight updates added to the same forward pass

This efficiency enables continuous online learning without the typical performance penalties of traditional deep learning systems.

#### Training Mode

During training, the network learns from the data and updates its weights:

```python
# Basic training - network learns from input patterns
h.step(inputs, learn_enabled=True)

# With reward signal (for RL) - network learns from environmental feedback
h.step([state, prev_action], learn_enabled=True, reward=reward)

# With imitation - network learns to mimic target behavior
h.step(observation, learn_enabled=True, mimic=target)
```

**Key points about training mode:**
- `learn_enabled=True` allows weight updates
- Rewards guide learning in reinforcement learning scenarios
- Imitation learning helps bootstrap from expert demonstrations

#### Inference Mode

During inference, you get predictions without modifying the learned weights:

```python
# Disable learning, just get predictions
h.step(inputs, learn_enabled=False)

# Get predictions
prediction = h.get_prediction_cis(layer_idx)[0]  # [0] gets the first prediction array
probabilities = h.get_prediction_acts(layer_idx)  # Get raw prediction probabilities
```

**Key points about inference mode:**
- `learn_enabled=False` prevents weight changes
- State still updates to maintain temporal context
- Multiple prediction formats available for different use cases

### Sequence Processing

Proper sequence handling is crucial for temporal learning. PyAOgmaNeo excels at learning temporal patterns when sequences are processed correctly.

#### Basic Sequence Processing

```python
# Reset state for new sequence
h.clear_state()  # Important: breaks temporal connection to previous sequence

# Process sequence
for item in sequence:
    h.step([item], learn_enabled=True)
    prediction = h.get_prediction_cis(0)[0]  # Get prediction for current item
    
# Generate continuation (creative mode)
for _ in range(length):
    pred = h.get_prediction_cis(0)[0]
    h.step([pred], learn_enabled=False)  # Use own predictions as input
```

#### Sequence Processing Best Practices

**Always clear state between unrelated sequences:**
- Prevents the model from treating separate sequences as continuous
- Ensures each sequence starts with a clean temporal context
- Critical for proper learning of sequence boundaries

**Process sequences in temporal order:**
- PyAOgmaNeo learns temporal dependencies between consecutive items
- Out-of-order processing breaks the temporal learning mechanism

**Use predictions as input for generation tasks:**
- The network can generate novel sequences by feeding its own predictions back as input
- Consider whether to enable learning during generation based on your use case

**When to clear state:**
- Starting new sequences during training
- Beginning inference on unrelated data
- Switching between different types of sequences
- Resetting temporal context for debugging

**When NOT to clear state:**
- During continuous sequences where temporal context is important
- When you want the model to maintain memory across related inputs
- During inference where context should be preserved

### Application Examples

#### Classification

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

#### Sequence Prediction

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

#### Reinforcement Learning Workflows

Reinforcement learning with PyAOgmaNeo requires careful coordination between the hierarchy and the environment. Here's a complete workflow:

##### Environment Setup

```python
# Import environment library (e.g., Gymnasium, formerly OpenAI Gym)
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')  # Or any other RL environment

# Create hierarchy matching environment specifications
h = neo.Hierarchy([
    # State input layer - matches environment observation space
    neo.IODesc(size=(1, 1, env.observation_space.shape[0]),
              io_type=neo.none),
    # Action output layer - matches environment action space
    neo.IODesc(size=(1, 1, env.action_space.n),
              io_type=neo.action)
], [
    neo.LayerDesc((4, 4, 32))  # Hidden layer for processing
])
```

##### Training Loop

```python
# Episode start
h.clear_state()  # Each episode starts fresh
state, _ = env.reset()  # Get initial environment state
prev_action = [0]  # Initial action (no action taken yet)

# Training loop
while not done:
    # Get action from hierarchy
    action = h.get_prediction_cis(1)[0]  # Action layer is at index 1
    
    # Environment step
    next_state, reward, done, truncated, info = env.step(action)
    
    # Update hierarchy with reward feedback
    h.step([state, prev_action], learn_enabled=True, reward=reward)
    
    # Update for next iteration
    state = next_state
    prev_action = action

env.close()
```

##### RL-Specific Considerations

**State and Action Representation:**
- Both current state and previous action must be provided to the hierarchy
- This allows the network to learn state-action-reward relationships
- Action layer uses `io_type=neo.action` for proper RL handling

**Reward Integration:**
- Rewards guide the learning process by reinforcing successful behaviors
- The hierarchy learns to predict actions that lead to higher rewards
- Reward signals should be consistent and meaningful

**Episode Management:**
- Each episode should start with cleared state using `h.clear_state()`
- This prevents temporal bleeding between different episodes
- Maintains proper episode boundaries for learning

**Exploration vs Exploitation:**
- The hierarchy naturally balances exploration and exploitation
- Early in training, actions may be more random (exploration)
- As learning progresses, actions become more deterministic (exploitation)

### Advanced Patterns

#### Transfer Learning

Load pre-trained weights into a new model for transfer learning:

```python
# Load weights from a pre-trained model
with open("pretrained_weights.bin", 'rb') as f:
    weights_buffer = f.read()

# Create new hierarchy and load weights
h_new = neo.Hierarchy(io_descs, layer_descs)
h_new.set_weights_from_buffer(weights_buffer)

# Continue training on new task
h_new.step(new_task_inputs, learn_enabled=True)
```

#### Multi-Task Learning

Handle multiple related tasks by managing state appropriately:

```python
# Clear state when switching between tasks
for task in tasks:
    h.clear_state()  # Reset context for new task
    
    for data in task.data:
        h.step(data, learn_enabled=True)
        
    # Evaluate on task
    task.evaluate(h)
```

#### Debugging and Analysis

Save intermediate states for debugging and analysis:

```python
# Save state at critical points
debug_states = []
for i, item in enumerate(sequence):
    h.step([item], learn_enabled=True)
    
    if i % debug_interval == 0:
        state_buffer = h.serialize_state_to_buffer()
        debug_states.append((i, state_buffer))

# Later: load specific state for analysis
h.set_state_from_buffer(debug_states[checkpoint_idx][1])
```

## See Also

- [Layer Configuration](layer_config.md)
- [State Management](state_management.md)
- [Image Processing](image_encoder.md)
