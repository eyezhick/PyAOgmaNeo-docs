# Image Processing

PyAOgmaNeo provides specialized components for processing image data through the `ImageEncoder` class and related configurations.

## ImageEncoder Class

### Constructor

```python
def __init__(self,
             hidden_size: Tuple[int, int, int] = (5, 5, 16),  # Hidden layer size
             visible_layer_descs: List[ImageVisibleLayerDesc] = [],  # Input configs
             file_name: str = "",  # Load from file
             buffer: bytes = b"")  # Load from buffer
```

### Core Methods

#### Processing

```python
def step(inputs: List[np.ndarray],                 # Process inputs
         learn_enabled: bool = True,               # Enable learning
         learn_recon: bool = False)                # Learn reconstruction

def reconstruct(recon_cis: np.ndarray)             # Reconstruct from hidden state
```

#### State Access

```python
def get_hidden_cis() -> np.ndarray                # Get hidden state
def get_reconstruction(i: int) -> np.ndarray      # Get reconstruction
def get_hidden_size() -> Tuple[int, int, int]     # Get hidden dimensions
def get_visible_size(i: int) -> Tuple[int, int, int] # Get input dimensions
def get_num_visible_layers() -> int               # Number of input layers
```

#### Persistence

```python
def save_to_file(file_name: str)                  # Save to file
def serialize_to_buffer() -> bytes                # Save to buffer
def set_state_from_buffer(buffer: bytes)          # Load state
def set_weights_from_buffer(buffer: bytes)        # Load weights
```

## Image Layer Configuration

### ImageVisibleLayerDesc

```python
class ImageVisibleLayerDesc:
    def __init__(self,
                 size: Tuple[int, int, int],  # Input dimensions
                 radius: int = 4)             # Receptive field radius
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|----------|
| size | Tuple[int, int, int] | Input dimensions | (5, 5, 16) |
| radius | int | Receptive field radius | 4 |

### ImageEncoderParams

```python
class ImageEncoderParams:
    falloff: float      # Activation falloff
    lr: float          # Learning rate
    scale: float       # Output scaling
    rr: float          # Response ratio
    n_radius: int      # Neighborhood radius
```

## Example Usage

### Basic Image Processing

```python
# Create encoder for 32x32 RGB images
encoder = neo.ImageEncoder(
    hidden_size=(16, 16, 64),
    visible_layer_descs=[
        neo.ImageVisibleLayerDesc(
            size=(32, 32, 3),  # RGB input
            radius=4
        )
    ]
)

# Process image
encoder.step([image], learn_enabled=True)

# Get encoded representation
features = encoder.get_hidden_cis()
```

### Multi-Scale Processing

```python
# Create encoder with multiple scales
encoder = neo.ImageEncoder(
    hidden_size=(8, 8, 128),
    visible_layer_descs=[
        # Original scale
        neo.ImageVisibleLayerDesc(
            size=(32, 32, 3),
            radius=4
        ),
        # Downsampled scale
        neo.ImageVisibleLayerDesc(
            size=(16, 16, 3),
            radius=2
        )
    ]
)

# Process both scales
encoder.step([original_image, downsampled_image], True)
```

### Image Reconstruction

```python
# Create autoencoder-style encoder
encoder = neo.ImageEncoder(
    hidden_size=(8, 8, 256),
    visible_layer_descs=[
        neo.ImageVisibleLayerDesc(
            size=(64, 64, 3),
            radius=8
        )
    ]
)

# Train with reconstruction
encoder.step([image], learn_enabled=True, learn_recon=True)

# Get reconstruction
recon = encoder.get_reconstruction(0)
```

## Integration with Hierarchy

### Classification Example

```python
# Create image encoder
img_encoder = neo.ImageEncoder(
    hidden_size=(16, 16, 64),
    visible_layer_descs=[
        neo.ImageVisibleLayerDesc(
            size=(28, 28, 1),
            radius=4
        )
    ]
)

# Create classifier hierarchy
h = neo.Hierarchy([
    neo.IODesc((16, 16, 64), neo.none),      # Encoder output
    neo.IODesc((1, 1, 10), neo.prediction)   # Class prediction
], [
    neo.LayerDesc((12, 12, 128))
])

# Training loop
for image, label in dataset:
    # Encode image
    img_encoder.step([image], True)
    features = img_encoder.get_hidden_cis()
    
    # Classify
    h.step([features, prev_label], True)
    prediction = h.get_prediction_cis(1)[0]
    prev_label = label
```

## See Also

- [Hierarchy Class](hierarchy.md)
- [Layer Configuration](layer_config.md) 