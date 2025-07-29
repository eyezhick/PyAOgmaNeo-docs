# Parameter Tuning Guide

## Structural Parameters

Structural parameters are set in the descriptors upon the creation of a hierarchy. There are 2 descriptors, `IODesc` and `LayerDesc`.

> **Note**: Radius parameters are used to determine receptive field sizes. Receptive fields are squares, and the diameter is 2 * radius + 1. The area is diameter² columns.

> **Note**: In C++ we use Pascal case for structures and snake case for everything else. In Python, we use camel case for structures and snake case for everything else (PEP8). C++ Example structure: `Layer_Desc`. Python: `LayerDesc`.

Default parameter values for your particular build can be seen in the headers of the respective parts.

### IODesc
Describes an input/output

| Parameter | Description |
|-----------|-------------|
| `size` | Size (Int3) of the input/output layer (width, height, column_size) |
| `type` | Enum type of the IO layer. Can be none (input only), prediction, and actor (RL) |
| `num_dendrites_per_cell` | Number of dendrites attached to each output cell. Applies to decoders and actor policies |
| `value_num_dendrites_per_cell` | Number of dendrites attached to each output cell specifically for actor value functions |
| `up_radius` | Receptive field radius from this IO layer to the first layer (encoder radius) |
| `down_radius` | Receptive field radius from the first layer to this IO layer (decoder/actor radius) |
| `history_capacity` | (RL only) the credit assignment horizon |

### LayerDesc
Describes a higher (not IO) layer

| Parameter | Description |
|-----------|-------------|
| `hidden_size` | Size (Int3) of the encoder's hidden state for that layer (width, height, column_size) |
| `num_dendrites_per_cell` | Number of dendrites attached to each output cell in the decoder |
| `up_radius` | Receptive field radius from previous (lower) layer to this layer (encoder radius) |
| `down_radius` | Receptive field radius from this layer to previous (lower) layer (decoder/actor radius) |
| `ticks_per_update` | Exponential Memory stride size |
| `temporal_horizon` | Memory window size of the layer, must be >= ticks_per_update |

## Runtime-adjustable Parameters

Runtime adjustable parameters are accessed via the hierarchy's `params` member. `params` contains `ios` (IO layer parameters) and `layers` (higher layer parameters) members.

### Example Access (Python)
```python
h.params.ios[2].decoder.lr = 0.1
h.params.ios[1].actor.plr = 0.01

h.params.layers[3].encoder.lr = 0.05
h.params.layers[4].decoder.scale = 64.0
```

### IOParams
| Parameter | Description |
|-----------|-------------|
| `decoder` | (DecoderParams) Decoder parameters |
| `actor` | (ActorParams) Actor parameters |
| `importance` | Importance scaling of this IO layer's input. Defaults to 1. Can be used to tweak the relative influences of the inputs, which can accelerate learning if properly adjusted |

### LayerParams
| Parameter | Description |
|-----------|-------------|
| `decoder` | (DecoderParams) Decoder parameters |
| `encoder` | (EncoderParams) Encoder parameters |

### DecoderParams
| Parameter | Description |
|-----------|-------------|
| `scale` | Range scaling for byte-weights. Unlikely you need to change this, best left as-is |
| `lr` | Learning rate, in [0.0, 1.0] range |
| `leak` | Leaky ReLU amount, for dendrites |

### EncoderParams
Same as decoder, plus:

| Parameter | Description |
|-----------|-------------|
| `choice` | ART choice parameter (alpha). Usually a small value. Must be > 0.0 |
| `vigilance` | ART vigilance parameter in [0.0, 1.0] range. The closer to 1, the pickier it gets about updating clusters |
| `lr` | Learning rate, in [0.0, 1.0] range |
| `active_ratio` | Activity ratio in the second stage of inhibition [0.0, 1.0] range |
| `l_radius` | Lateral inhibition radius. Must be large enough to fit the active_ratio ((l_radius + 1)^-2 <= active_ratio) |

### ActorParams
| Parameter | Description |
|-----------|-------------|
| `vlr` | Value function learning rate. Around 0.01, cannot be > 1! |
| `plr` | Actor learning rate. Around 0.01, can be > 1 (but this is very rare) |
| `leak` | Leaky ReLU amount, for dendrites |
| `discount` | Reinforcement learning discounting factor (lambda). Around 0.99. Must be in [0.0, 1.0) |
| `policy_clip` | Gradient clipping on the policy. Must be > 0.0 |
| `value_clip` | Gradient clipping on the value function. Must be > 0.0 |
| `trace_decay` | Decay multiplier for eligibility traces in [0.0, 1.0) |


## See Also
- [Variable Naming Reference](../appendix/naming_reference.md)
- [Core Concepts](../user_guide/core_concepts.md)
- [Advanced Optimization](../tutorials/advanced/optimization.md)
- [Custom Encoders Tutorial](../tutorials/advanced/custom_encoders.md) 