# GW-MF-PINN

**Version**: 1.0.2

**Last generated**: 2025-10-10 18:54:05

A pipeline for gravitational-wave (GW) data analysis, focused on the data acquisition and evaluation modules. This project provides tools for GWOSC data ingestion, preprocessing, and evaluation using Machine Learning and Matched Filtering baselines.

## Overview

This repository is organized into two main modules:

1.  **`gwdata`**: Responsible for data acquisition and preparation.
    * **Acquisition (GWOSC)**: Discovery of public events and download of strain files (preferably HDF5, 4 kHz, 4096 s).
    * **Preprocessing**: Application of a band-pass filter, PSD estimation, and whitening.
    * **Windowing/SNR**: Generation of sliding windows over the whitened strain and calculation of SNR metrics.

2.  **`eval`**: Responsible for building datasets and running baseline models.
    * **Dataset Building**: Creation of `dataset.parquet` with labels based on the coalescence time (`tc`) of GWOSC events.
    * **Baselines**:
        * **Machine Learning (`baseline_ml.py`)**: Reference pipeline with classical ML.
        * **Matched Filtering (`mf_baseline.py`)**: Matched filtering using an IMRPhenomD template bank.

## Theoretical Background

The detection and analysis of gravitational waves are rooted in Einstein's theory of General Relativity. The gravitational field is described by the curvature of spacetime, governed by the Einstein Field Equations:

$G_{\mu
u} = rac{8\pi G}{c^4} T_{\mu
u}$

In the weak-field limit, where spacetime is nearly flat, the metric $g_{\mu
u}$ can be described as a small perturbation $h_{\mu
u}$ from the Minkowski metric $\eta_{\mu
u}$. This leads to the linearized wave equation:

$\Box h_{\mu
u} = 0$

Solving these equations for astrophysical phenomena like black hole mergers is computationally intensive and prone to discretization errors. **Physics-Informed Neural Networks (PINNs)** offer a modern solution. Instead of relying on traditional numerical solvers, a PINN is a neural network trained to minimize a loss function that enforces the validity of the underlying physical laws.

The total loss function combines a data term and a physics term:

$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{phys}$

The goal of this project is to leverage a **Multi-Fidelity PINN (MF-PINN)** architecture. This approach integrates data from multiple levels of fidelity—analytical approximations, numerical simulations, and real observational data from LIGO/Virgo—to build a hierarchical learning system. This allows the model to capture the complex dynamics of General Relativity while being refined by real-world measurements, bridging the gap between theory and observation.

## Project Structure

The structure has been simplified to focus on the main modules:

```
gw-mf-pinn/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/         # Raw GWOSC data (HDF5)
│  ├─ interim/     # Preprocessed (whitened) data
│  └─ processed/   # Windows, dataset.parquet, metadata
├─ src/
│  ├─ gwdata/
│  │  ├─ gwosc_client.py
│  │  ├─ preprocess.py
│  │  └─ windows.py
│  └─ eval/
│     ├─ dataset_builder.py
│     ├─ baseline_ml.py
│     └─ mf_baseline.py
└─ venv/
```

## Requirements

* Python 3.12+
* NumPy, SciPy, h5py, pandas, pyarrow
* Virtual environment (recommended)

### Quick Setup

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install the dependencies
pip install -r requirements.txt
```

## Usage

The `gwdata` and `eval` modules are designed to be imported and used in orchestration scripts or analysis notebooks.

**Example workflow:**

1.  **Data Step (using `gwdata`):**
    * Use `gwdata.gwosc_client` to download event data.
    * Apply `gwdata.preprocess` to clean and whiten the signals.
    * Generate analysis windows with `gwdata.windows`.

2.  **Evaluation Step (using `eval`):**
    * Build a cohesive dataset with `eval.dataset_builder`.
    * Run the `eval.baseline_ml` and `eval.mf_baseline` reference models on the dataset.

The results, such as logs, reports, and data artifacts, will be saved in the `logs/`, `reports/`, and `data/processed/` directories, respectively.

## License

Check the `LICENSE` file if applicable. GWOSC data is generally distributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license, following GWOSC acknowledgment guidelines.
