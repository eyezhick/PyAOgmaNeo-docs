# Quickstart Guide

TODO: finish pre-Encoding section. better structure the quickstart guide. Brainstorm what should be in here

---


Welcome to PyAOgmaNeo! This guide will get you up and running with PyAOgmaNeo in under 10 minutes. You'll learn how to create your first neural hierarchy and see it learn patterns in real-time.

## What is PyAOgmaNeo?

PyAOgmaNeo implements **Sparse Predictive Hierarchies (SPH)** - a neuromorphic machine learning architecture inspired by biological neural networks. Unlike traditional neural networks, SPH:

- **Learns online** with no separate training phase
- **Predicts temporal sequences** naturally 
- **Uses sparse representations** for efficiency
- **Processes hierarchical patterns** resembling the brain
- **Minimal training overhead** - only ~2x slower than inference (vs 10x-100x for traditional networks)

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

> **Sample Data:** You can use the [sample_text_sph.txt](./sample_text_sph.txt) file provided in our example for this tutorial.

```python
import pyaogmaneo as neo

# Sample text for demonstration - about SPH and this example
with open("sample_text_sph.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create mapping between words and indices
words = text.split()
word_to_idx = {word: i for i, word in enumerate(set(words))}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(word_to_idx)

print(f"Vocabulary: {vocab_size} words")

# Create model
h = neo.Hierarchy([
    neo.IODesc(
        (1, 1, vocab_size),
        io_type=neo.prediction,
        num_dendrites_per_cell=16    # More connections per cell
    )
], [
    # Hidden Layer
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

From the examples above, you've seen the key concepts in action. Here's what makes PyAOgmaNeo unique:

### 1. CSDR Format (Column-Sparse Distributed Representation)

PyAOgmaNeo requires specific input format representing sparse patterns:

```python
# Text example: words → integer indices
word_to_idx = {"hello": 0, "world": 1, "prediction": 2}
inputs = [[2]]  # "prediction" as index 2

# Create model
h = neo.Hierarchy([
    neo.IODesc(
        (1, 1, vocab_size),
        io_type=neo.prediction
    )
])
```

Notice how the input is formatted as (1, 1, vocab_size), where vocab_size represent one hot vector.

### 2. Pre-Encoding is Essential

Your data needs conversion to CSDR format before PyAOgmaNeo can use it:

TODO: talk about pre-encoding, how to convert, and link to md that has details

### 3. Hierarchy Structure

```python
h = neo.Hierarchy([
    # IO Layers - your data interfaces
    neo.IODesc((1, 1, vocab_size), io_type=neo.prediction),  # Output layer
    neo.IODesc((1, 1, n_bins), io_type=neo.none)            # Input layer
], [
    # Hidden Layer - learns patterns between IO layers
    neo.LayerDesc((8, 8, 64))  # (width, height, features)
])
```

### 4. Online Learning Loop

```python
# 1. Get prediction first
prediction = h.get_prediction_cis(layer_index)[0]

# 2. Then update with actual data
h.step([actual_input], learn_enabled=True)
```

**Key insight**: Predict first, then learn from the difference.

> **Need more detail?** See our [Basic Concepts](basic_concepts.md) guide and [Layer Configuration](../api_reference/layer_config.md) reference.

## 10-Minute Example: Fashion MNIST Image Classification

Now let's tackle a classic computer vision task - **Fashion MNIST classification**. We'll show how PyAOgmaNeo handles image data with online learning, achieving competitive accuracy without traditional batch training.

```python
import pyaogmaneo as neo
import numpy as np
from datasets import load_dataset

# Set the number of computation threads for faster training (optional)
neo.set_num_threads(4)

# Load Fashion MNIST dataset from Hugging Face
print("Loading Fashion MNIST dataset...")
dataset = load_dataset("zalando-datasets/fashion_mnist")

# List of class names (for display)
class_names = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
]

# Hierarchy input resolution (quantize grayscale to 4 levels: 0,1,2,3)
input_res = 4

# Build the AoGMaNeo hierarchy for classification
h = neo.Hierarchy([
    # Image input: 28x28 pixels, quantized to 4 levels
    neo.IODesc((28, 28, input_res), io_type=neo.none, up_radius=8),
    # Class label prediction output: 10 classes
    neo.IODesc((1, 1, 10), io_type=neo.prediction, num_dendrites_per_cell=64, down_radius=8)
], [
    # Hidden layer: feature maps for pattern extraction
    neo.LayerDesc((16, 16, 64))
])

print("Training...")

# For sequential prediction, start with previous label 0
label_prev = 0
correct_count = 0
total_count = 0
train_iterations = 20000  # Adjust this for more/less training

for i in range(train_iterations):
    # Select a random training example
    idx = np.random.randint(0, dataset["train"].num_rows)
    img = np.array(dataset["train"][idx]["image"], dtype=np.float32) / 255.0
    true_label = int(dataset["train"][idx]["label"])
    
    # Quantize the image (required by AoGMaNeo)
    quantized_img = (img * (input_res - 1) + 0.5).astype(np.int32)

    # ---- TRAIN NETWORK FIRST ----
    h.step([quantized_img.ravel().tolist(), [label_prev]], learn_enabled=True)
    # ---- THEN GET PREDICTION ----
    prediction = h.get_prediction_cis(1)[0]

    # Update previous label for sequence input
    label_prev = true_label

    # Track accuracy
    is_correct = (prediction == true_label)
    correct_count += is_correct
    total_count += 1

    # Show progress every 500 iterations
    if (i + 1) % 500 == 0:
        accuracy = correct_count / total_count
        correct_name = class_names[true_label]
        pred_name = class_names[prediction] if prediction < 10 else "?"
        print(f"Step {i+1:4d}: Acc={accuracy:.3f} | "
              f"True: {correct_name:9s} | Pred: {pred_name:9s}")

print(f"\nFinal training accuracy: {correct_count / total_count:.3f}\n")
```

Now let's test the trained model:

```python
# Test on Fashion MNIST test set
print("Testing on Fashion MNIST test set...")
test_correct = 0
test_samples = 1000  # Test on 1000 samples for speed

for i in range(test_samples):
    # Get test sample
    img = np.array(dataset["test"][i]["image"], dtype=np.float32) / 255.0
    true_label = int(dataset["test"][i]["label"])
    
    # Quantize image
    quantized_img = (img * (input_res - 1) + 0.5).astype(np.int32)
    
    # Get prediction (no learning)
    h.step([quantized_img.ravel().tolist(), [0]], learn_enabled=False)
    prediction = h.get_prediction_cis(1)[0]
    
    # Check correctness
    if prediction == true_label:
        test_correct += 1
    
    # Progress indicator
    if (i + 1) % 200 == 0:
        current_acc = test_correct / (i + 1)
        print(f"Tested {i+1:4d} samples, accuracy: {current_acc:.3f}")

final_accuracy = test_correct / test_samples
print(f"\nFinal test accuracy: {final_accuracy:.3f}")
```

**Key insights:** 
- Images were quantized to 4 discrete levels 
- The network achieved good accuracy with **no batch training** - just online learning
- **Spatial patterns** were captured through the large upward radius (8) in the image layer

> **Want more details?** For a more thorough explanation and comprehensive notebook with detailed analysis and visualization, check out:
> - [Fashion MNIST Tutorial](../examples/fashion_mnist/fashion_mnist.md) - Complete guide with theory and implementation details
> - [Fashion MNIST Jupyter Notebook](../examples/fashion_mnist/fashion_mnist_tutorial.ipynb) - Interactive notebook with visualizations and step-by-step analysis

> **Curious about the brain inspiration?** Learn more in our [Core Concepts Guide](../user_guide/core_concepts.md).

## Next Steps

**Learn More:**
- [Basic Concepts](basic_concepts.md) - Deeper understanding of SPH principles
- [API Reference](../api_reference/index.md) - Complete method documentation
- [Fashion MNIST Example](../examples/fashion_mnist/fashion_mnist.md) - Real-world image classification
- [Parameter Tuning Guide](../technical_guide/parameter_tuning.md) - Optimize your networks

**Advanced Usage:**
- [Custom Environments](../user_guide/custom_environments.md) - Build your own applications
- [First Hierarchy Tutorial](../tutorials/basic/first_hierarchy.md) - Step-by-step deep dive
- [State Management](../api_reference/state_management.md) - Save and load trained models


> **See more examples:** Browse our [Examples Collection](../examples) for real-world applications.

---

 You've successfully built online learning systems on real benchmark datasets. PyAOgmaNeo's neuromorphic approach enables continuous learning from streaming data without the complexity of traditional batch training.

Ready to dive deeper? Check out our [complete examples](../examples) and [detailed tutorials](../tutorials/basic/first_hierarchy.md).
