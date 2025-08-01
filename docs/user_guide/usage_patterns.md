# Usage Patterns

This guide covers common usage patterns and workflows when working with PyAOgmaNeo's Sparse Predictive Hierarchies. Understanding these patterns will help you implement effective learning systems for various applications.

> **API Reference:** For detailed information about state management methods, see [State Management API](../api_reference/state_management.md).

## Training vs Inference Modes

PyAOgmaNeo hierarchies can operate in two distinct modes, each serving different purposes in your machine learning workflow.

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

## Sequence Processing

Proper sequence handling is crucial for temporal learning. PyAOgmaNeo excels at learning temporal patterns when sequences are processed correctly.

### Basic Sequence Processing

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

## Reinforcement Learning Workflows

Reinforcement learning with PyAOgmaNeo requires careful coordination between the hierarchy and the environment. Here's a complete workflow:

### Environment Setup

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

### Training Loop

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

## See Also

- [State Management API](../api_reference/state_management.md) - Detailed API documentation for state operations
- [Hierarchy Class](../api_reference/hierarchy.md) - Core hierarchy functionality and methods
- [Quickstart Guide](../getting_started/quickstart.md) - Get started with basic examples
- [Examples](../examples/) - Complete working examples for various use cases