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

## Usage Examples

> **Complete Workflows:** For comprehensive usage patterns including training workflows, sequence processing, and reinforcement learning, see the [Hierarchy Usage Patterns](hierarchy.md#usage-patterns-and-examples).

### Basic State Management Example

```python
import pyaogmaneo as neo

# Create hierarchy
h = neo.Hierarchy(io_descs, layer_descs)

# Check memory usage
print(f"State size: {h.get_state_size()} bytes")
print(f"Weights size: {h.get_weights_size()} bytes")

# Process some data
h.step(inputs, learn_enabled=True)

# Save current state
state_buffer = h.serialize_state_to_buffer()

# Clear state for new sequence
h.clear_state()

# Process new sequence
h.step(new_inputs, learn_enabled=True)

# Restore previous state if needed
h.set_state_from_buffer(state_buffer)

# Save complete model
h.save_to_file("model.ohr")
```

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

## See Also

- [Hierarchy Usage Patterns](hierarchy.md#usage-patterns-and-examples) - Common workflows and usage patterns
- [Hierarchy Class](hierarchy.md) - Core hierarchy functionality and methods
- [Layer Configuration](layer_config.md) - Setting up layers and their parameters
- [Image Processing](image_encoder.md) - Specialized processing for image data 