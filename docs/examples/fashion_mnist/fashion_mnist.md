# Fashion MNIST Classification with PyAOgmaNeo

This tutorial demonstrates one method of using PyAOgmaNeo for classification tasks, using Fashion MNIST as an example. By leveraging PyAOgmaNeo's temporal prediction capabilities, we can achieve >90% test accuracy using just a simple binning encoding approach without complex image encoders.

## A Prediction-Based Approach to Classification

While PyAOgmaNeo is primarily designed for temporal prediction tasks, this example shows how we can adapt it for classification by treating class prediction as a temporal sequence. Here's how this approach works:

### How it Works

1. **Temporal Context**
   - Instead of direct classification, PyAOgmaNeo treats it as a prediction task
   - Each prediction uses both the current image AND the previous label as context
   - Example sequence:
     ```
     Step 1: Image(T-shirt) + PrevLabel(0) → Predict("T-shirt")
     Step 2: Image(Sneaker) + PrevLabel("T-shirt") → Predict("Sneaker")
     Step 3: Image(Dress) + PrevLabel("Sneaker") → Predict("Dress")
     ```

2. **Unordered Learning**
   - Images are shown in random order during training
   - The randomization ensures the network doesn't learn meaningless temporal patterns
   - Only the image-to-label relationship remains consistent

3. **System Design Considerations**
   - PyAOgmaNeo is fundamentally a prediction-based system
   - We adapt classification to fit this paradigm
   - Some computational overhead exists but is neutralized through randomization

## Implementation Details

### Network Configuration

```python
h = neo.Hierarchy([
    # Image input layer (28x28 pixels quantized to 4 values)
    neo.IODesc((28, 28, 4),
              io_type=neo.none,  # Regular input
              up_radius=8),  # Large radius for spatial patterns
    
    # Label prediction layer (10 classes)
    neo.IODesc((1, 1, 10),
              io_type=neo.prediction,
              num_dendrites_per_cell=64,  # Many dendrites for robust prediction
              up_radius=0,
              down_radius=8)
    ],
    # Hidden layer configuration
    1 * [ neo.LayerDesc((16, 16, 64),
                      recurrent_radius=-1) ]
)
```

### Key Features

1. **Input Processing**
   - Images are quantized to 4 levels
   - Simple binning encoding instead of complex image encoders
   - Maintains good performance while being computationally efficient

2. **Network Architecture**
   - Input layer: Captures spatial patterns with large upward radius
   - Hidden layer: 16x16x64 for sufficient processing capacity
   - Output layer: Uses many dendrites for robust classification

3. **Training Process**
   - Random sampling from training set
   - Learning through temporal prediction
   - Previous label used as context

## Performance

The network results:

- Overall Test Accuracy: 86%
- Per-class Performance:
  ```
  T-shirt/top: 82.5%
  Trouser: 95.9%
  Pullover: 78.3%
  Dress: 86.7%
  Coat: 76.8%
  Sandal: 94.0%
  Shirt: 58.9%
  Sneaker: 95.8%
  Bag: 96.1%
  Ankle boot: 94.7%
  ```

## Complete Example

You can find the complete implementation in the [Fashion MNIST Tutorial Notebook](fashion_mnist_tutorial.ipynb).

