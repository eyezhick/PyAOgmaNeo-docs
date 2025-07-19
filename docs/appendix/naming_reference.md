# Variable Naming Reference

This document provides a comprehensive reference for variable naming conventions used throughout PyAOgmaNeo and AOgmaNeo codebases.

## Naming Conventions

### Language-Specific Conventions
- **C++**: 
  - Structures use Pascal case (e.g., `Layer_Desc`)
  - Other identifiers use snake case
- **Python**: 
  - Classes use Camel case (e.g., `LayerDesc`)
  - Other identifiers use snake case (PEP8)

## Common Variable Names

### Encoder/Decoder Variables
| Variable | Description | Range/Notes |
|----------|-------------|-------------|
| `vl` | Visible layer | - |
| `vld` | Visible layer descriptor | - |
| `hc` | Hidden column index | [0, hidden_size.z) |
| `vc` | Visible column index | [0, vld.size.z) |
| `hidden_cell_index` | 3D cell position index | - |
| `count` | Input counter for normalization | - |
| `radius` | Receptive field radius | - |
| `diam` | Receptive field diameter | 2 * radius + 1 |
| `area` | Receptive field area | diam² |

### Projection Variables
| Variable | Description | Notes |
|----------|-------------|-------|
| `h_to_v` | Hidden to visible projection factors | Used for scaling |
| `project` | Position scaling function | - |
| `visible_center` | Receptive field center | - |
| `field_lower_bound` | Receptive field lower bound | Before edge clamping |
| `iter_lower_bound` | Clamped field lower bound | After edge clamping |
| `iter_upper_bound` | Clamped field upper bound | After edge clamping |

### Index Variables
| Variable | Description | Range/Notes |
|----------|-------------|-------------|
| `hidden_stride` | Weight indices stride | - |
| `in_ci` | Input column index | - |
| `offset` | Receptive field position | [0, diam) |
| `wi_offset` | Partial weight index | Missing hidden cell influence |
| `wi` | Final weight index | Complete index |
| `wi_start` | Partial weight index | Missing visible cell influence |

### State Variables
| Variable | Description | Notes |
|----------|-------------|-------|
| `max_index` | Highest activation index | - |
| `max_activation` | Highest activation value | - |
| `total` | Softmax accumulator | For normalization |
| `delta` | Weight increment | For learning |
| `state` | State variable | RNG or working memory |

### Hierarchy Variables
| Variable | Description | Notes |
|----------|-------------|-------|
| `ticks` | Exponential memory clocks | - |
| `ticks_per_update` | Layer activation interval | - |
| `i_indices` | First layer decoder to input mapping | - |
| `d_indices` | IO to decoder mapping | -1 if no decoder |
| `updates` | Layer update flags | Whether layer "ticked" |
| `io_types` | IO layer types | - |
| `io_sizes` | IO layer sizes | - |

### Common Prefixes and Suffixes
| Prefix/Suffix | Meaning | Example |
|---------------|---------|---------|
| `prev_` | Previous timestep value | `prev_state` |
| `next_` | Next timestep value | `next_state` |
| `base_` | Cached partial computation | `base_seed` |
| `acts_` | Activations | `hidden_acts` |
| `cis_` | Column indices | `input_cis` |
| `visible_` | Visible/input related | `visible_layer` |
| `input_` | Input related | `input_size` |
| `hidden_` | Hidden/output related | `hidden_state` |
| `probs_` | Probabilities | `action_probs` |
| `size_` | Byte size for serialization | `encoder_size` |
| `e_` | Encoder | `e_params` |
| `d_` | Decoder | `d_params` |
| `a_` | Actor | `a_params` |

## See Also
- [Parameter Tuning Guide](../technical_guide/parameter_tuning.md)
- [Core Concepts](../user_guide/core_concepts.md)
- [API Reference](../api_reference/index.md) 