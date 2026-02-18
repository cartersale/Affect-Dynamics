# Affect-Dynamics

This repository contains a complete pipeline for analyzing dyadic affect dynamics using Hidden Markov Models (HMMs), Recurrence Quantification Analysis (RQA), and motif/grammar analysis. 

## Project Structure

- **`scripts/`**: The core analysis pipeline. Scripts are numbered `01` through `05` to indicate execution order.
- **`src/`**: The `affectdynamics` Python package containing reusable logic for models, preprocessing, and metrics.
- **`notebooks/`**: Jupyter notebooks for interactive analysis and visualization of results.
- **`configs/`**: YAML configuration files controlling parameters for all scripts.
- **`data/`**: Directory for raw input data and processed outputs.
- **`artifacts/`**: Directory where analysis results (CSVs, models, plots) are saved.

## Setup

### Option A: Local Python Environment (Recommended)

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd Affect-Dynamics
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -U pip
    pip install -e .
    ```

### Option B: Docker

1.  **Build the Docker image**:
    ```bash
    docker build -t affect-dynamics .
    ```

2.  **Run the container**:
    ```bash
    docker run -it -v $(pwd):/app affect-dynamics
    ```
    This mounts your current directory to `/app` inside the container, so any files generated (in `artifacts/` or `data/processed/`) will persist on your host machine.

## Workflow

The analysis pipeline consists of five main steps. You can run them individually or use the `Makefile` shortcuts.

### 1. Data Ingestion & Preprocessing
**Script**: `scripts/01_ingest_and_preprocess.py`  
**Description**: Reads raw SPAFF CSV files, maps codes to a simplified valence space (e.g., Positive, Negative, Neutral), applies debouncing (smoothing), and optionally segments sessions into sliding windows.  
**Input**: CSV files in `data/raw/` (configured in `configs/ingest_and_preprocess.yaml`).  
**Output**: Parquet files in `data/processed/`.

### 2. Coupling & Directionality Analysis
**Script**: `scripts/02_coupling_test.py`  
**Description**: Fits coupled Markov models to test for interpersonal influence.
-   **Coupling**: Does the partner's previous state predict my current state better than my own history alone?
-   **Directionality**: Is the influence stronger from Therapist -> Client or Client -> Therapist?
**Output**: `artifacts/coupling_results.csv`, `artifacts/directionality_results.csv`.

### 3. Hidden Markov Model (HMM) Fitting
**Script**: `scripts/03_fit_shared_hmm.py`  
**Description**: Fits a valid Shared-Emission HMM to learn latent "regimes" of interaction (e.g., "Mutual Positivity", "Disengagement"). Performs a grid search for optimal `K` (number of states) and saves the Viterbi-decoded paths for every session.
**Note**: This step involves multiple model fits and cross-validation, so it is computationally intensive and may take a long time to run.
**Output**: `artifacts/chmm/decoded_sessions.csv`, model parameters, and Viterbi paths.

### 4. Categorical RQA (catRQA)
**Script**: `scripts/04_catRQA.py`  
**Description**: Performs Categorical Recurrence Quantification Analysis within the identified HMM regimes. Metrics include Recurrence Rate (RR), Determinism (DET), and Laminarity (LAM), compared against shuffled null surrogates.
**Output**: `artifacts/catrqa_episode_results.csv` and example recurrence plots.

### 5. Grammar/Motif Analysis
**Script**: `scripts/05_grammar.py`  
**Description**: Extracts recurrent sequences (k-grams/motifs) of joint affect within HMM regimes. Computes entropy, diversity, and top motifs to characterize the texture of interaction in each state.
**Output**: `artifacts/grammar_episode_results.csv`.

## Using the Makefile

The `Makefile` provides convenient shortcuts for running the pipeline:

-   `make ingest-preprocess`: Runs step 1 (Ingest).
-   `make coupling`: Runs step 2 (Coupling).
-   `make hmm`: Runs step 3 (HMM Fitting). **Warning: This step takes a long time.**
-   `make catrqa`: Runs step 4 (catRQA).
-   `make grammar`: Runs step 5 (Grammar).
-   `make analysis`: **Runs steps 2 through 5 sequentially.** Because this includes the HMM fitting step, `make analysis` will take a significant amount of time to complete. It is often better to run steps individually for debugging or if you only need specific results. 
-   `make all`: Runs the entire pipeline from ingestion to final analysis.
-   `make clean`: Removes python cache files.

## Notebooks

After running the pipeline, you can explore the results using the provided notebooks:
-   `notebooks/coupling_test.ipynb`: Visualize coupling and directionality results (lagged influence metrics, partner→self vs self→self comparisons, summary tables and session examples).
-   `notebooks/hmm_outputs.ipynb`: Visualize HMM states and decoded sessions.
-   `notebooks/catrqa_analysis.ipynb`: Analyze recurrence metrics.
-   `notebooks/grammar_analysis.ipynb`: Explore affect motifs and grammar statistics.
-   `notebooks/regime_geometry.ipynb`: Explore geometric structure of HMM regimes (state embeddings, transition graphs, PCA/UMAP of emissions, and trajectory visualizations).


## Configuration

All parameters (paths, hyperparameters, toggles) are defined in:
-   `configs/ingest_and_preprocess.yaml`: For step 1.
-   `configs/analysis.yaml`: For steps 2-5.

Modify these files to change input directories, model parameters (e.g., number of HMM states range), or analysis settings.
