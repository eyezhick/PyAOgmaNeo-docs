<!---
  PyAOgmaNeo Documentation
  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
--->

<div align="center">

# PyAOgmaNeo Documentation

Comprehensive documentation for PyAOgmaNeo, Python bindings for the AOgmaNeo library.

</div>

---

### Documentation Includes

- **`API Reference`** &nbsp;&nbsp; Detailed documentation of classes, methods, and parameters
- **`Tutorials`** &nbsp;&nbsp; Step-by-step guides from basic to advanced usage
- **`Examples`** &nbsp;&nbsp; Real-world implementations and use cases
- **`Implementation Guides`** &nbsp;&nbsp; Best practices and technical details
- **`User Guides`** &nbsp;&nbsp; Conceptual overviews and usage patterns

---

### Documentation Status

| Symbol | Meaning |
|:------:|---------|
| ✅ | Complete (exists in this repository with full documentation) |
| 🔄 | In Progress |
| 📝 | Draft Available |
| ⚠️ | Exists in PyAOgmaNeo (needs to be transferred and expanded) |
| ⏳ | Not Started |

---

### Contents

<details open>
<summary><b>Getting Started</b> ⚠️</summary>

- [Installation](docs/getting_started/installation.md) ⚠️ _(Basic guide exists in PyAOgmaNeo)_
- [Quickstart](docs/getting_started/quickstart.md) ⏳
- [Basic Concepts](docs/getting_started/basic_concepts.md) ⏳
</details>

<details open>
<summary><b>User Guide</b> ⏳</summary>

- [Overview](docs/user_guide/overview.md) ⏳
- [Core Concepts](docs/user_guide/core_concepts.md) ⏳
- [EnvRunner Guide](docs/user_guide/env_runner.md) ⚠️ _(Basic explanation exists in PyAOgmaNeo)_
- [Custom Environments](docs/user_guide/custom_environments.md) ⏳
</details>

<details open>
<summary><b>Technical Guide</b> 🔄</summary>

- [Parameter Tuning](docs/technical_guide/parameter_tuning.md) 🔄 _(Transferred from AOgmaNeo)_
</details>

<details open>
<summary><b>API Reference</b> ⏳</summary>

- [Overview](docs/api_reference/index.md) ⏳
- [Hierarchy Class](docs/api_reference/hierarchy.md) ⏳
- [LayerDesc Class](docs/api_reference/layer_desc.md) ⏳
- [IODesc Class](docs/api_reference/io_desc.md) ⏳
</details>

<details open>
<summary><b>Tutorials</b> 🔄</summary>

**Basic**
- [First Hierarchy](docs/tutorials/basic/first_hierarchy.md) ⏳
- [Simple Prediction](docs/tutorials/basic/simple_prediction.md) ⏳

**Advanced**
- [Custom Encoders](docs/tutorials/advanced/custom_encoders.md) ⏳
- [Optimization](docs/tutorials/advanced/optimization.md) ⏳
</details>

<details open>
<summary><b>Examples</b> ⚠️</summary>

**CartPole** _(exists in PyAOgmaNeo)_
- [Manual Implementation](docs/examples/cartpole/manual_implementation.md) ⚠️
- [EnvRunner Implementation](docs/examples/cartpole/env_runner_implementation.md) ⚠️

**Natural Language Processing**
- [Text Prediction](docs/examples/nlp/text_prediction.md) ⏳

**Other Examples**
- [Wave Prediction](docs/examples/wave_prediction.md) ⚠️
- [Lunar Lander](docs/examples/lunar_lander.md) ⚠️
</details>

<details open>
<summary><b>Contributing</b> 🔄</summary>

- [Guidelines](docs/contributing/guidelines.md) ✅
- [Development Setup](docs/contributing/development_setup.md) ⏳
- [Documentation](docs/contributing/documentation.md) ⏳
</details>

<details open>
<summary><b>Appendix</b> 🔄</summary>

- [Naming Reference](docs/appendix/naming_reference.md) 🔄 _(Transferred from AOgmaNeo)_
</details>

---

### Next Steps

> More of a TODO list: will keep it updated

1. **Transfer existing content from PyAOgmaNeo:**
   - Installation guide
   - EnvRunner documentation
   - Example code and explanations
    
2. **Begin with high-priority sections:**
   - Complete quickstart guide
   - Basic concepts documentation
   - API reference overview

3. **Expand tutorials and examples:**
   - First hierarchy tutorial
   - Text prediction example
   - Additional use cases

---

### Contributing

Please see [contributing guide](docs/contributing/guidelines.md) for details.

---

### Repository Structure

<details>
<summary>Click to expand full directory structure</summary>

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
├── technical_guide/
│   └── parameter_tuning.md    # Comprehensive parameter tuning guide
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
├── appendix/
│   └── naming_reference.md    # Variable naming conventions and reference
│
└── contributing/
    ├── guidelines.md         # Contribution guidelines
    ├── development_setup.md  # Setting up development environment
    └── documentation.md      # Documentation contribution guide
```
</details>

---

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

This documentation is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

PyAOgmaNeo Copyright (c) 2020-2025 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved. 