# Affect-Dynamics

## Project Overview
This project analyzes dyadic affect dynamics using Markov models. It is designed for reproducibility and ease of use, even for non-programmers.

## Quick Start 

### 1. Setup Python Environment
- Open a terminal in the project folder.
- Create a virtual environment (if not already present):
  ```sh
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -e .
  ```

### 2. Prepare Your Data
- Place your raw SPAFF CSV files in the `data/raw/` directory.
- Ensure your code legend is at `spaff_codes.csv` in the project root.
- Adjust any config values in `configs/ingest_and_preprocess.yaml` and `configs/analysis.yaml` as needed (see comments in those files).

### 3. Run the Analysis Pipeline
- To run data ingest and preprocessing:
  ```sh
  make ingest-preprocess
  ```
- To run the coupling analysis:
  ```sh
  make analysis
  ```
- To generate all figures and stats in the notebook:
  ```sh
  make notebooks
  ```

### 4. Reproducibility Notes
- All configurable options (file paths, toggles, parameters) are in the `configs/` directory.
- The Makefile ensures each step is run in the correct order.
- If you move the project, update paths in the config files.
- If you add new scripts, add them to the Makefile for easy access.

---

For more details, see comments in the config files and notebooks.