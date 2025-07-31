# State Management

PyAOgmaNeo's Sparse Predictive Hierarchies maintain internal state that can be managed through various operations. Understanding state management is crucial for effective use of the library, especially for sequence learning and reinforcement learning applications.

## Understanding State Components

PyAOgmaNeo's state consists of two main components that serve different purposes:

### 1. Internal State
The internal state represents the network's current context and temporal memory:
- **Current activation patterns**: Which neurons are currently active in each layer
- **Temporal memory (history buffer)**: Past activations stored for temporal learning
- **Current context for predictions**: The network's current "understanding" of the sequence
- **Cleared by `clear_state()`**: Resets temporal context but preserves learned knowledge
- **Saved/loaded via state buffers**: Can be serialized for checkpointing

### 2. Weights
The weights contain the network's learned knowledge:
- **Learned parameters**: Connection strengths between neurons
- **Connection strengths between layers**: How layers communicate with each other
- **Preserved during `clear_state()`**: Learning is retained when state is cleared
- **Can be saved/loaded independently**: Allows for transfer learning and model deployment


### State Size Information
These methods help you understand memory requirements and optimize storage:

```python
# Get component sizes (in bytes)
state_size = h.get_state_size()     # Size of internal state
weights_size = h.get_weights_size()  # Size of learned weights
total_size = h.get_size()           # Total size (state + weights)

print(f"Internal state: {state_size / 1024:.1f} KB")
print(f"Weights: {weights_size / 1024:.1f} KB")
print(f"Total model: {total_size / 1024:.1f} KB")
```

## State Operations

### Basic State Management

The `clear_state()` method resets the network's temporal context while preserving learned knowledge. This is essential for proper sequence handling:

```python
# Clear internal state (preserves weights)
h.clear_state()
```

**When to use `clear_state()`:**
- **Starting new sequences**: Prevents the model from treating separate sequences as continuous
- **Beginning new episodes in RL**: Each episode should start with a clean slate
- **Resetting temporal context**: When you want to break temporal dependencies
- **Between different tasks**: When switching between unrelated prediction tasks

**When NOT to use `clear_state()`:**
- **During continuous sequences**: Would break temporal learning
- **When you want temporal context**: For tasks requiring memory of previous inputs
- **During inference on related data**: When context should be maintained

### State Persistence

PyAOgmaNeo provides flexible options for saving and loading models, allowing you to choose between complete model persistence or component-wise operations.

#### Complete Model Save/Load

The simplest approach saves everything together in a single file:

```python
# Save complete model (state + weights)
h.save_to_file("model.ohr")

# Load complete model
h = neo.Hierarchy(file_name="model.ohr")
```

**Use complete model save/load when:**
- Deploying trained models
- Creating backups during training
- Sharing models with others
- You need both learned knowledge and current state

#### Component-wise Save/Load

For more control, you can save and load components separately:

```python
# Save components separately
state_buffer = h.serialize_state_to_buffer()    # Just internal state
weights_buffer = h.serialize_weights_to_buffer() # Just weights
full_buffer = h.serialize_to_buffer()           # Everything (state + weights)

# Load components
h.set_state_from_buffer(state_buffer)     # Load just internal state
h.set_weights_from_buffer(weights_buffer)  # Load just weights
```

**Use component-wise operations when:**
- **Transfer learning**: Load weights into a new model architecture
- **Debugging**: Save state at specific points for analysis
- **Memory optimization**: Only save what you need
- **Experimental workflows**: Mix and match different components

## Usage Patterns

### Training Mode

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

### Inference Mode

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

Proper sequence handling is crucial for temporal learning:

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

**Sequence processing best practices:**
- Always clear state between unrelated sequences
- Process sequences in temporal order
- Use predictions as input for generation tasks
- Consider whether to enable learning during generation

### Reinforcement Learning

RL requires careful coordination between the hierarchy and environment:

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

**RL-specific considerations:**
- State and action must be provided to the hierarchy
- Rewards guide the learning process
- Each episode should start with cleared state
- Action layer uses `io_type=neo.action` for proper RL handling

## State Checkpointing

### Regular Saving During Training

Implement periodic saving to prevent loss of training progress:

```python
# Save periodically during training
if (epoch + 1) % save_interval == 0:
    # Save full model - includes both state and weights
    h.save_to_file(f"model_epoch_{epoch+1}.ohr")
    
    # Or save components separately for flexibility
    with open(f"state_{epoch+1}.bin", 'wb') as f:
        f.write(h.serialize_state_to_buffer())
    with open(f"weights_{epoch+1}.bin", 'wb') as f:
        f.write(h.serialize_weights_to_buffer())
```

### Advanced Checkpointing with Metadata

For research and production workflows, save additional information with your models:

```python
def save_checkpoint(h, epoch, metrics, base_path):
    """
    Save checkpoint with comprehensive metadata for experiment tracking.
    
    Args:
        h: Hierarchy instance to save
        epoch: Current training epoch
        metrics: Dictionary of training metrics
        base_path: Base directory for checkpoints
    """
    # Create checkpoint directory
    checkpoint_dir = Path(base_path) / f"checkpoint_{epoch}"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save model components
    h.save_to_file(str(checkpoint_dir / "model.ohr"))
    
    # Save comprehensive metadata
    metadata = {
        'epoch': epoch,
        'metrics': metrics,
        'state_size': h.get_state_size(),
        'weights_size': h.get_weights_size(),
        'timestamp': time.time(),
        'model_architecture': {
            'num_layers': h.get_num_layers(),
            'num_io': h.get_num_io()
        }
    }
    with open(checkpoint_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
```

### State Recovery and Resuming Training

Implement robust checkpoint loading for resuming interrupted training:

```python
def load_latest_checkpoint(base_path):
    """
    Load the most recent checkpoint with full metadata.
    
    Args:
        base_path: Directory containing checkpoints
        
    Returns:
        tuple: (hierarchy, metadata) or (None, None) if no checkpoints found
    """
    # Find latest checkpoint
    checkpoint_dirs = sorted(Path(base_path).glob("checkpoint_*"))
    if not checkpoint_dirs:
        return None, None
        
    latest = checkpoint_dirs[-1]
    
    # Load model
    h = neo.Hierarchy(file_name=str(latest / "model.ohr"))
    
    # Load metadata
    with open(latest / "metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
        
    print(f"Loaded checkpoint from epoch {metadata['epoch']}")
    print(f"Model size: {metadata['weights_size'] / 1024:.1f} KB")
        
    return h, metadata
```

## Best Practices

### State Management Guidelines
- **Clear state appropriately**: Use `clear_state()` between unrelated sequences or episodes, but not during continuous processing
- **Understand the distinction**: Internal state is temporal context; weights are learned knowledge
- **Plan your persistence strategy**: Save weights frequently during training, full models for deployment
- **Use component-wise loading**: When you only need specific parts or for transfer learning scenarios

### Memory Efficiency Considerations
- **Monitor sizes**: Check `get_state_size()` and `get_weights_size()` to understand memory usage
- **Component-wise saving**: Save only what you need to reduce storage requirements
- **Compression**: Consider compressing state buffers for long-term storage
- **Cleanup**: Remove old checkpoints to manage disk space

### Checkpointing Strategies
- **Include metadata**: Save training metrics, timestamps, and model architecture information
- **Version your checkpoints**: Use epoch numbers or timestamps in checkpoint names
- **Implement rotation**: Keep only the N most recent checkpoints to manage storage
- **Test recovery**: Regularly verify that your checkpoints can be loaded successfully


## See Also

- [Hierarchy Class](hierarchy.md) - Core hierarchy functionality and methods
- [Layer Configuration](layer_config.md) - Setting up layers and their parameters
- [Image Processing](image_encoder.md) - Specialized processing for image data 