# Quickstart Guide

Welcome to PyAOgmaNeo! This guide will get you up and running with PyAOgmaNeo in under 10 minutes. You'll learn how to create your first neural hierarchy and see it learn patterns in real-time.

## What is PyAOgmaNeo?

PyAOgmaNeo implements **Sparse Predictive Hierarchies (SPH)** - a neuromorphic machine learning architecture inspired by biological neural networks. Unlike traditional neural networks, SPH:

- **Learns online** with no separate training phase
- **Predicts temporal sequences** naturally 
- **Uses sparse representations** for efficiency
- **Processes hierarchical patterns** resembling the brain

## Prerequisites

Make sure you have PyAOgmaNeo installed:

```bash
pip install pyaogmaneo scikit-learn
```

> **Need help with installation?** Check our detailed [Installation Guide](installation.md) for troubleshooting and advanced options.

Verify your installation:

```python
import pyaogmaneo as neo
print(neo.__version__)  # Should print the current version
```

## 5-Minute Example: Word Prediction

Let's start with a simple **word prediction** task using a small English text dataset. The model will learn to predict the next word based on context.

```python
import pyaogmaneo as neo

# Sample text for demonstration - about SPH and this example
text = """Sparse Predictive Hierarchies are a neuromorphic machine learning architecture inspired by biological neural networks. 
Unlike traditional neural networks SPH learns online with no separate training phase. 
The system predicts temporal sequences naturally using sparse representations for efficiency. 

This example demonstrates word prediction using PyAOgmaNeo. 
The model learns to predict the current word based on its context. 
The hierarchy maintains an internal state that updates as it processes each word. 

Before processing each word the model makes a prediction. 
The prediction is compared with the actual current word. 
The model is then updated with the actual word. 
This process repeats for each word in the sequence. 

The network uses sparse predictive coding with configurable hidden layer size. 
Online learning means the system adapts continuously without batch training. 
Hierarchical processing resembles biological neural networks with multiple layers. 
Sparse activations mean only some cells are active like real neurons. 

The learning loop follows a simple pattern of predict then update. 
This approach enables real-time adaptation as new data arrives. 
Memory efficiency comes from not storing large training sets. 
The system processes streaming data one sample at a time. 
. 
"""

words = text.split()
word_to_idx = {word: i for i, word in enumerate(set(words))}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(word_to_idx)

print(f"Vocabulary: {vocab_size} words")

# Create model with longer memory
h = neo.Hierarchy([
    neo.IODesc(
        (1, 1, vocab_size), 
        io_type=neo.prediction,
        history_capacity=512,        # Longer history buffer
        num_dendrites_per_cell=16    # More connections per cell
    )
], [
    neo.LayerDesc((8, 8, 64)),      # Larger hidden layer
    neo.LayerDesc((6, 6, 48))       # Additional layer for longer dependencies
])

# Train: predict current word, then update
print("Training...")
correct = 0
for i, word in enumerate(words):
    word_idx = word_to_idx[word] 
    # Update with actual word
    h.step([[word_idx]], True)

# Generate text
h.clear_state()
generated = ["Sparse"]

# Feed starting word and generate next 10 words
h.step([[word_to_idx["Sparse"]]], True)
for _ in range(50):
    pred_idx = h.get_prediction_cis(0)[0]
    word = idx_to_word[pred_idx]
    generated.append(word)
    h.step([[pred_idx]], True)

print(f"Generated: {' '.join(generated)}")
```

**Expected Results**: You should see accuracy improve as the model learns word patterns and generates coherent text.

> **Want more details?** Check out our comprehensive [Word Prediction Notebook](../examples/nlp/word_prediction_v2.ipynb) with detailed analysis, visualization, and advanced techniques.

## Core Concepts

> **Want to understand the theory?** Dive deeper with our [Basic Concepts](basic_concepts.md) guide.

### 1. Hierarchy Structure

PyAOgmaNeo networks consist of **IO layers** and **hidden layers**:

```python
h = neo.Hierarchy([
    # IO Layers - interface with external data
    neo.IODesc((width, height, features), io_type=neo.none),      # Input-only
    neo.IODesc((1, 1, classes), io_type=neo.prediction)          # Output layer
], [
    # Hidden Layers - process and connect IO layers
    neo.LayerDesc((hidden_width, hidden_height, hidden_features))
])
```

### 2. IO Layer Types

- **`neo.none`**: Input-only layers (for images, sensors, data)
- **`neo.prediction`**: Output layers for predictions/classifications  
- **`neo.action`**: Output layers for reinforcement learning

> **Learn about layer configuration:** Check out our comprehensive [Layer Configuration Guide](../api_reference/layer_config.md).

### 3. Learning Loop

The fundamental PyAOgmaNeo pattern:

```python
for data_point in training_data:
    # 1. Get prediction (what the network thinks comes next)
    prediction = h.get_prediction_cis(layer_index)[0]
    
    # 2. Update network with actual data (learning happens here)
    h.step([input_data], learn_enabled=True)
```

## 10-Minute Example: Online Classification on Wine Dataset

Now let's tackle another classic ML dataset - the **Wine dataset**. We'll show how PyAOgmaNeo handles multi-feature classification with online learning on a different domain.

```python
import pyaogmaneo as neo
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import KBinsDiscretizer

# Set up the network
neo.set_num_threads(4)

# Load the Wine dataset - chemical analysis of wines
print("Loading Wine dataset from scikit-learn...")
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
class_names = wine.target_names

print(f"Dataset: {len(X)} wine samples, {len(feature_names)} chemical features")
print(f"Features: {feature_names[:6]}... (showing first 6)")
print(f"Wine classes: {list(class_names)}")
print()

# Discretize continuous chemical measurements
n_bins = 6  # 6 discrete levels per feature
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
X_discrete = discretizer.fit_transform(X).astype(int)

print(f"Chemical features discretized to {n_bins} levels each")
print(f"Sample wine: {X[0][:4]} → {X_discrete[0][:4]} (first 4 features)")
print()

# Create multi-input wine classification hierarchy
# We'll use the first 6 most important features for simplicity
important_features = [0, 1, 6, 9, 11, 12]  # alcohol, malic_acid, flavanoids, etc.
n_features = len(important_features)

# Build IO layers for each chemical feature
io_layers = []
for i in range(n_features):
    io_layers.append(
        neo.IODesc((1, 1, n_bins),
                  io_type=neo.none,
                  up_radius=1)
    )

# Add classification output layer
io_layers.append(
    neo.IODesc((1, 1, 3),  # 3 wine classes
              io_type=neo.prediction,
              num_dendrites_per_cell=20,
              down_radius=3)
)

h = neo.Hierarchy(io_layers, [
    # Hidden layer for learning chemical feature combinations
    neo.LayerDesc((5, 5, 48))
])

print("Training online wine classifier...")
print("Learning to distinguish wine classes from chemical analysis")
print()

# Create shuffled indices for realistic online learning
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

correct = 0
total = 0

print("Chemical Features | Predicted | Actual | ✓/✗ | Accuracy")
print("-" * 60)

# Online training
for i, idx in enumerate(train_indices):
    features = X_discrete[idx]
    selected_features = features[important_features]
    true_class = y[idx]
    
    # Get prediction
    pred_class = h.get_prediction_cis(n_features)[0]
    
    # Check accuracy
    is_correct = (pred_class == true_class)
    correct += is_correct
    total += 1
    
    # Show progress
    if i % 15 == 0:
        accuracy = correct / total if total > 0 else 0
        pred_name = class_names[pred_class] if pred_class < 3 else "Unknown"
        true_name = class_names[true_class]
        
        feature_str = f"[{','.join(map(str, selected_features[:3]))}...]"
        print(f"{feature_str:17} | {pred_name:9} | {true_name:9} | {'✓' if is_correct else '✗'} | {accuracy:.3f}")
    
    # Update network with current wine's features
    inputs = [[int(f)] for f in selected_features] + [[true_class]]
    h.step(inputs, learn_enabled=True)

train_accuracy = correct / total
print(f"\nTraining accuracy: {train_accuracy:.3f}")
print()

# Test on held-out wines
print("Testing on unseen wines...")
test_correct = 0

print("Chemical Features | Predicted | Actual | Result")
print("-" * 50)

for idx in test_indices[:15]:  # Show first 15 test cases
    features = X_discrete[idx]
    selected_features = features[important_features]
    true_class = y[idx]
    
    # Get prediction (no learning)
    inputs = [[int(f)] for f in selected_features] + [[0]]
    h.step(inputs, learn_enabled=False)
    pred_class = h.get_prediction_cis(n_features)[0]
    
    is_correct = (pred_class == true_class)
    test_correct += is_correct
    
    # Show results
    pred_name = class_names[pred_class] if pred_class < 3 else "Unknown"
    true_name = class_names[true_class]
    feature_str = f"[{','.join(map(str, selected_features[:3]))}...]"
    
    print(f"{feature_str:17} | {pred_name:9} | {true_name:9} | {'✓' if is_correct else '✗'}")

test_accuracy = test_correct / len(test_indices)
print(f"\nTest accuracy: {test_accuracy:.3f}")
print("The network learned to classify wines from chemical measurements.")
```

**What's remarkable:** The network learned to distinguish wine classes based on chemical analysis - the same problem wine experts solve, but using only discrete measurements and online learning.

> **Curious about the brain inspiration?** Learn more in our [Core Concepts Guide](../user_guide/core_concepts.md).

## Next Steps

**Immediate Next Steps:**
- Try your own text data
- Experiment with different sklearn datasets (digits, breast_cancer, etc.)
- Adjust model parameters and architecture

**Learn More:**
- [Basic Concepts](basic_concepts.md) - Deeper understanding of SPH principles
- [API Reference](../api_reference/index.md) - Complete method documentation
- [Fashion MNIST Example](../examples/fashion_mnist/fashion_mnist.md) - Real-world image classification
- [Parameter Tuning Guide](../technical_guide/parameter_tuning.md) - Optimize your networks

**Advanced Usage:**
- [Custom Environments](../user_guide/custom_environments.md) - Build your own applications
- [First Hierarchy Tutorial](../tutorials/basic/first_hierarchy.md) - Step-by-step deep dive
- [State Management](../api_reference/state_management.md) - Save and load trained models


## Real-World Applications

The examples above demonstrate PyAOgmaNeo's applicability for:

- **NLP**: Real-time text generation, language modeling, adaptive chatbots
- **Scientific Analysis**: Chemical analysis, sensor data interpretation, quality control
- **Data Streams**: Online classification of streaming data, adaptive filtering
- **Personalization**: Adaptive recommendations, user behavior modeling
- **Industrial AI**: Predictive maintenance, process optimization, anomaly detection

> **See more examples:** Browse our [Examples Collection](../examples) for real-world applications.

---

**Congratulations!** You've successfully built online learning systems on real benchmark datasets. PyAOgmaNeo's neuromorphic approach enables continuous learning from streaming data without the complexity of traditional batch training.

Ready to dive deeper? Check out our [complete examples](../examples) and [detailed tutorials](../tutorials/basic/first_hierarchy.md).
