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
with open("sample_text_sph.txt", "r", encoding="utf-8") as f:
    text = f.read()

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
        num_dendrites_per_cell=16    # More connections per cell
    )
], [
    neo.LayerDesc((8, 8, 64))
])

# predict current word, then update
correct = 0
for epoch in range(4):
    for i, word in enumerate(words):
        pred_idx = h.get_prediction_cis(0)[0]
        predicted_word = idx_to_word[pred_idx]
        word_idx = word_to_idx[word]
        h.step([[word_idx]], learn_enabled=True)

        # Check correctness
        if predicted_word == word:
            color = "\033[32m"  # Green
        else:
            color = "\033[31m"  # Red
        reset = "\033[0m"

        print(f"{color}{predicted_word}{reset}", end=" ")
        if i % 25 == 24:
            print()
    print("\n")

h.clear_state()
```


**Expected Results**: You should see the training accuracy improve over time. Initally it gets every word wrong, but by 4th run the model effectively memorizes the whole text.

> **Want more details?** Check out our comprehensive [Word Prediction Notebook](../examples/nlp/word_prediction_v2.ipynb) with detailed analysis and visualization.

## Core Concepts

> **Want to understand the theory?** Dive deeper with our [Basic Concepts](basic_concepts.md) guide.

### 1. Hierarchy Structure

PyAOgmaNeo networks consist of **IO layers** and **hidden layers**:

```python
h = neo.Hierarchy([
    # IO Layers - interface with external data
    neo.IODesc((width, height, features), io_type=neo.none),    # Input-only
    neo.IODesc((1, 1, classes), io_type=neo.prediction)     # Input and Output
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
print("Green = correct prediction, Red = incorrect")
print()

# Create shuffled indices for realistic online learning
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Online training - show predictions in real-time
for epoch in range(3):
    print(f"Epoch {epoch + 1}: ", end="")
    for i, idx in enumerate(train_indices[:50]):  # Show first 50 for clarity
        features = X_discrete[idx]
        selected_features = features[important_features]
        true_class = y[idx]
        
        # Get prediction
        pred_class = h.get_prediction_cis(n_features)[0]
        
        # Update network with current wine's features
        inputs = [[int(f)] for f in selected_features] + [[true_class]]
        h.step(inputs, learn_enabled=True)
        
        # Color code prediction
        if pred_class == true_class:
            color = "\033[32m"  # Green
        else:
            color = "\033[31m"  # Red
        reset = "\033[0m"
        
        pred_name = class_names[pred_class] if pred_class < 3 else "?"
        print(f"{color}{pred_name[0]}{reset}", end="")  # Just first letter
    print()  # New line after each epoch

print("\nTraining complete! (C=class_0, P=class_1, B=class_2)")
print()

```

Now let's test the trained model on new wines:

```python
# Test on held-out wines
print("Testing on unseen wines...")
print("Test results: ", end="")

test_correct = 0
for idx in test_indices[:30]:  # Test on 30 unseen wines
    features = X_discrete[idx]
    selected_features = features[important_features]
    true_class = y[idx]
    
    # Get prediction (no learning)
    inputs = [[int(f)] for f in selected_features] + [[0]]
    h.step(inputs, learn_enabled=False)
    pred_class = h.get_prediction_cis(n_features)[0]
    
    is_correct = (pred_class == true_class)
    test_correct += is_correct
    
    # Color code results
    if is_correct:
        color = "\033[32m"  # Green
    else:
        color = "\033[31m"  # Red
    reset = "\033[0m"
    
    pred_name = class_names[pred_class] if pred_class < 3 else "?"
    print(f"{color}{pred_name[0]}{reset}", end="")

test_accuracy = test_correct / 30
print(f"\n\nTest accuracy: {test_accuracy:.3f}")
print("The network learned to classify wines from chemical measurements!")
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
