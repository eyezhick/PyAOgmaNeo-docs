# State Management

PyAOgmaNeo's Sparse Predictive Hierarchies maintain internal state that can be managed through various operations.

## State Operations

### Basic State Management

```python
# Clear all internal state
h.clear_state()

# Get state information
state_size = h.get_state_size()
weights_size = h.get_weights_size()
total_size = h.get_size()
```

### State Persistence

```python
# Save complete state
h.save_to_file("model.ohr")

# Save state components
state_buffer = h.serialize_state_to_buffer()
weights_buffer = h.serialize_weights_to_buffer()
full_buffer = h.serialize_to_buffer()

# Load state
h.set_state_from_buffer(state_buffer)
h.set_weights_from_buffer(weights_buffer)
```

## Usage Patterns

### Training Mode

```python
# Enable learning
h.step(inputs, learn_enabled=True)

# With reward signal
h.step(state, learn_enabled=True, reward=reward)

# With imitation
h.step(observation, learn_enabled=True, mimic=target)
```

### Inference Mode

```python
# Disable learning
h.step(inputs, learn_enabled=False)

# Get predictions
prediction = h.get_prediction_cis(layer_idx)
probabilities = h.get_prediction_acts(layer_idx)
```

### Sequence Processing

```python
# Reset state for new sequence
h.clear_state()

# Process sequence
for item in sequence:
    h.step([item], learn_enabled=True)
    
# Generate continuation
for _ in range(length):
    pred = h.get_prediction_cis(0)
    h.step([pred], learn_enabled=False)
```

### Reinforcement Learning

```python
# Episode start
h.clear_state()

# Training loop
while not done:
    # Get action
    action = h.get_prediction_cis(1)[0]
    
    # Environment step
    next_state, reward, done, _ = env.step(action)
    
    # Update with reward
    h.step([state], learn_enabled=True, reward=reward)
    
    state = next_state
```

## State Checkpointing

### Regular Saving

```python
# Save periodically
if (epoch + 1) % save_interval == 0:
    h.save_to_file(f"model_epoch_{epoch+1}.ohr")
    
    # Optional: Save just the weights
    with open(f"weights_epoch_{epoch+1}.bin", 'wb') as f:
        f.write(h.serialize_weights_to_buffer())
```

### Versioned Checkpoints

```python
# Save with metadata
import pickle
from pathlib import Path

def save_checkpoint(h, epoch, metrics, base_path):
    # Create checkpoint directory
    checkpoint_dir = Path(base_path) / f"checkpoint_{epoch}"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save model
    h.save_to_file(str(checkpoint_dir / "model.ohr"))
    
    # Save metadata
    metadata = {
        'epoch': epoch,
        'metrics': metrics,
        'timestamp': time.time()
    }
    with open(checkpoint_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
```

### State Recovery

```python
def load_latest_checkpoint(base_path):
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
        
    return h, metadata
```

## See Also

- [Hierarchy Class](hierarchy.md)
- [Layer Configuration](layer_config.md)
- [Image Processing](image_encoder.md) 