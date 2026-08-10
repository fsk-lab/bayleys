# Domain-Adapted Molecular Language Models for Efficient Search in Make-on-Demand Libraries

This repository contains all code for the paper *Domain-Adapted Molecular Language Models for Efficient Search in Make-on-Demand Libraries*. 

The central component of the codebase is the Python library `BAYLEYS` (Bayesian Library Exploration and Virtual Screening). `BAYLEYS` enables benchmarking arbitrary combinations of fixed and learned molecular representations, surrogate models, acquisition functions, and batch acquisition strategies for sample-efficient optimization in large virtual molecular libraries. An optimization campaign in `BAYLEYS` is defined by a *search space* (the virtual library), a *target property* (the objective function), and a *budget* (the number of molecules that can be evaluated experimentally), and a *batch size* (the number of molecules that can be evaluated in parallel). Within these constraints, `BAYLEYS` follows the philosophy of *Active Learning* and *Bayesian Optimization*: Using all molecules that have been evaluated experimentally, a Bayesian surrogate model is trained to learn the structure–property / structure–activity relationships. Based on this surrogate, new candidate(s) for evaluation are proposed. BAYLEYS provides the infrastructure to perform such iterative optimizations, and to benchmark different methodological choices. 

## Getting Started

For installing BAYLEYS, use the following steps: 
1. Clone the repository, e.g. via `git clone https://github.com/fsk-lab/bayleys/`
2. `cd bayleys`
3. Install *via* pip: `pip install .` (or editable: `pip install -e .`).

> :grey_exclamation: **Note**: In some environments, installation with the `--no-build-isolation` flag may be necessary due to possible incompatibilities with `pytorch-fast-transformers`.


## Structure of the Repository

Under `src/bayleys`, the code is organized into the following submodules:
- `molecule_library`: Data structures for storing and accessing large virtual molecular libraries.
- `encoders`: Molecular representations, including fixed (fingerprints, descriptors) and learned representations (molecular language models), including fine-tuning and domain adaptation strategies.
- `surrogate`: Surrogate models for structure–property / structure–activity relationships.
- `acquisition`: Acquisition functions and batch acquisition strategies.
- `campaign`: Infrastructure for running optimization campaigns, including statistic repeats, logging, and data storage.
- `utils`: Utility functions for data handling, logging, etc.

The `experiments` folder contains scripts, configurations and utility functions for reproducing the experiments in the paper.


## Results and Data

All configurations and results are provided on Zenodo at the following link:

--- TODO ---