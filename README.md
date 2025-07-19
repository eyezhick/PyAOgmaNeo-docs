<!---
  PyAOgmaNeo Documentation
  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
--->

# PyAOgmaNeo Documentation

This repository contains the comprehensive documentation for PyAOgmaNeo, Python bindings for the AOgmaNeo library.

## Status Legend
- ✅ Complete (exists in this repository with full documentation)
- 🔄 In Progress
- ⚠️ Exists in PyAOgmaNeo (needs to be transferred and expanded)
- ⏳ Planned
- ❌ Blocked

## Documentation Structure

```
docs/
├── getting_started/
│   ├── installation.md         # Installation instructions
│   ├── quickstart.md          # Quick introduction and first steps
│   └── basic_concepts.md      # Essential concepts to get started
│
├── user_guide/
│   ├── overview.md            # High-level overview of PyAOgmaNeo
│   ├── core_concepts.md       # Detailed explanation of core concepts
│   ├── env_runner.md         # Guide to using EnvRunner
│   └── custom_environments.md # Creating custom environments
│
├── api_reference/
│   ├── index.md              # API documentation overview
│   ├── hierarchy.md          # Hierarchy class documentation
│   ├── layer_desc.md         # LayerDesc class documentation
│   └── io_desc.md           # IODesc class documentation
│
├── tutorials/
│   ├── basic/
│   │   ├── first_hierarchy.md     # Creating your first hierarchy
│   │   └── simple_prediction.md   # Basic prediction example
│   └── advanced/
│       ├── custom_encoders.md     # Working with custom encoders
│       └── optimization.md        # Performance optimization
│
├── examples/
│   ├── cartpole/
│   │   ├── manual_implementation.md
│   │   └── env_runner_implementation.md
│   ├── nlp/
│   │   └── text_prediction.md    # Text prediction example
│   ├── wave_prediction.md
│   └── lunar_lander.md
│
└── contributing/
    ├── guidelines.md         # Contribution guidelines
    ├── development_setup.md  # Setting up development environment
    └── documentation.md      # Documentation contribution guide
```

## Documentation Contents

### Getting Started ⚠️
- [Installation](docs/getting_started/installation.md) ⚠️ (Basic guide exists in PyAOgmaNeo)
- [Quickstart](docs/getting_started/quickstart.md) ⏳
- [Basic Concepts](docs/getting_started/basic_concepts.md) ⏳

### User Guide ⏳
- [Overview](docs/user_guide/overview.md) ⏳
- [Core Concepts](docs/user_guide/core_concepts.md) ⏳
- [EnvRunner Guide](docs/user_guide/env_runner.md) ⚠️ (Basic explanation exists in PyAOgmaNeo)
- [Custom Environments](docs/user_guide/custom_environments.md) ⏳

### API Reference ⏳
- [Overview](docs/api_reference/index.md) ⏳
- [Hierarchy Class](docs/api_reference/hierarchy.md) ⏳
- [LayerDesc Class](docs/api_reference/layer_desc.md) ⏳
- [IODesc Class](docs/api_reference/io_desc.md) ⏳

### Tutorials 🔄
#### Basic
- [First Hierarchy](docs/tutorials/basic/first_hierarchy.md) ⏳
- [Simple Prediction](docs/tutorials/basic/simple_prediction.md) ⏳

#### Advanced
- [Custom Encoders](docs/tutorials/advanced/custom_encoders.md) ⏳
- [Optimization](docs/tutorials/advanced/optimization.md) ⏳

### Examples ⚠️
#### CartPole (exists in PyAOgmaNeo)
- [Manual Implementation](docs/examples/cartpole/manual_implementation.md) ⚠️
- [EnvRunner Implementation](docs/examples/cartpole/env_runner_implementation.md) ⚠️

#### Natural Language Processing
- [Text Prediction](docs/examples/nlp/text_prediction.md) ⏳

#### Other Examples
- [Wave Prediction](docs/examples/wave_prediction.md) ⚠️
- [Lunar Lander](docs/examples/lunar_lander.md) ⚠️

### Contributing 🔄
- [Guidelines](docs/contributing/guidelines.md) ✅
- [Development Setup](docs/contributing/development_setup.md) ⏳
- [Documentation](docs/contributing/documentation.md) ⏳

## Next Steps

More or a TODO list: will keep it updated

1. Transfer existing content from PyAOgmaNeo:
   - Installation guide
   - EnvRunner documentation
   - Example code and explanations
   
2. Begin with high-priority sections:
   - Complete quickstart guide
   - Basic concepts documentation
   - API reference overview

3. Expand tutorials and examples:
   - First hierarchy tutorial
   - Text prediction example
   - Additional use cases

## Contributing

Please see our [contributing guide](docs/contributing/guidelines.md) for details.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

This documentation is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

PyAOgmaNeo Copyright (c) 2020-2025 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved. 