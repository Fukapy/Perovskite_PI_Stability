# Standardised Process Descriptors for Interpretable Regression of Perovskite Solar-Cell Stability

**Ryo Fukasawa, Toru Asahi, Takuya Taniguchi**

This repository contains the analysis code accompanying the paper:

> *Standardised process descriptors for interpretable regression of perovskite solar-cell stability*

---

## Overview

Fabrication processes strongly influence perovskite solar-cell stability, yet their heterogeneous nature has hindered unified data-driven analysis. This work proposes a standardised descriptor framework that converts diverse process information recorded in [The Perovskite Database](https://www.perovskitedatabase.com/) into a unified numerical feature matrix for interpretable regression.

5,800 curated entries from The Perovskite Database are used to train a random-forest model predicting ln(TS80m). SHAP-based interpretation identifies annealing conditions, solvent type, and layer-specific process parameters as dominant factors.

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── main.py                   ← Pipeline runner (executes all steps)
├── data_curation.py          ← Step 1: Load, filter, compute ln(TS80m)
├── vectorization.py          ← Step 2: Build CSR feature matrices
├── reanalysis.py             ← Step 3: 5-fold CV + FULL model training
├── interpretation.py         ← Step 4: Feature importance (Lv1–Lv3)
├── shap_analysis.py          ← Step 5: SHAP values and bar chart
├── DB_k_plot.py              ← Step 6: Apparent rate constant k surface
│
├── pipeline_config.py        ← Shared config (target selection: stability / PCE)
├── process_stab.py           ← Shared utilities (encoding functions)
├── compute_fti_shap_sp2dummy.py  ← Standalone FTI+SHAP for primary model
│
├── revised_CBFV/             ← Modified CBFV package for composition vectorisation
│   ├── __init__.py
│   ├── composition.py
│   └── element_properties/
│
├── data/                     ← Input data (tracked by Git)
│   └── ZhangZ_PD.csv        ← The Perovskite Database
│
└── outputs/                  ← Generated outputs (excluded from Git)
    ├── curated/              ← Curated CSVs from Step 1
    ├── csr/                  ← CSR feature matrices from Steps 1–2
    ├── model/
    │   ├── cv5/              ← Trained models, CV results, FTI CSVs
    │   ├── cv5_per_only/     ← Perovskite-only CV results
    │   └── shap/             ← SHAP outputs
    └── figures/              ← Publication figures
```

> **Working directory**: all scripts must be run from the **repository root** (the directory containing this README).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (stability target: ln(TS80m), default)
python main.py

# Run the full pipeline with PCE as the regression target
python main.py --target PCE

# Run only specific steps (e.g. interpretation and SHAP)
python main.py --steps 4 5

# Skip steps that are already done
python main.py --skip 1 2

# Combine target selection and step control
python main.py --target PCE --skip 1 2
```

The CBFV vectorisation additionally requires the [`CBFV`](https://github.com/kaaiian/CBFV) package and element-property tables (oliynyk, magpie, mat2vec).

---

## Pipeline Steps

### Step 1: `data_curation.py`

**Purpose**: Load and curate the raw Perovskite Database, apply filtering criteria, and build the initial feature matrix.

**Input**: `data/ZhangZ_PD.csv` — download from [perovskitedatabase.com](https://www.perovskitedatabase.com/)

**Filtering steps**:
1. Remove entries with missing stability records or non-ASCII characters
2. Remove stoichiometrically inconsistent compositions
3. Remove entries producing zero-vector CBFV representations (magpie check)
4. Compute ln(TS80m) as the regression target

**Output**: `outputs/curated/Perovskite_5800data_addln.csv`, `outputs/csr/all_sp1_oliynyk_zero_{csr.npz,_columns.npy}`

---

### Step 2: `vectorization.py`

**Purpose**: Build all feature matrix variants across encoding schemes (sp0–sp3) and CBFV representations (oliynyk, magpie, mat2vec, dummy/one-hot).

**Encoding schemes**:
| Tag | Categorical encoding |
|-----|---------------------|
| sp0 | one-hot |
| sp1 | multi-hot variant 1 |
| sp2 | multi-hot variant 2 |
| sp3 | multi-hot variant 3 |

**Output**: CSR matrices saved to `outputs/csr/`

---

### Step 3: `reanalysis.py`

**Purpose**: Rigorous 5-fold cross-validation of the random-forest model for ln(TS80m) across all encoding variants. Saves per-fold models, out-of-fold (OOF) predictions, and a final FULL model trained on the complete dataset.

**Key setting** (top of file): `RUN_TS80M = False` — set to `True` to additionally run regression on raw TS80m

**Output** (written to `outputs/model/cv5/`):
| File | Description |
|------|-------------|
| `rf_model_{tag}_fold{k}.pkl` | Per-fold trained models |
| `cv5_oof_{tag}.csv` | Out-of-fold predictions |
| `cv5_metrics_{tag}.csv` / `cv5_summary_{tag}.csv` | CV R² statistics |
| `rf_model_{tag}_FULL.pkl` | Full-data model (used for interpretation) |
| `CV_summary_clean_table.csv` | Collated summary across all variants |

---

### Step 4: `interpretation.py`

**Purpose**: Compute feature importances from the FULL models at three levels of aggregation.

**Three-level aggregation**:
| Level | Output file | Description |
|-------|-------------|-------------|
| Lv1 | `fti_{tag}.csv` | Raw expanded features (thousands of columns) |
| Lv2 | `fti_sum_{tag}.csv` | Aggregated by original DB column name |
| Lv3 | `fti_layer_{tag}.csv` | Aggregated by functional layer / process category |

**Layer categories** (editable via `LAYER_MAP` at the top of the file):
`Perovskite (composition)`, `Perovskite (deposition)`, `ETL`, `HTL`, `Back contact`, `Substrate`, `Cell architecture`, `Encapsulation`, `Stability test conditions`

---

### Step 5: `shap_analysis.py`

**Purpose**: Compute SHAP values for the FULL models and generate the mean SHAP bar chart, showing the top stabilising and destabilising qualitative variables.

**Output** (written to `outputs/model/shap/`):
| File | Description |
|------|-------------|
| `shap_mean_bar_{tag}.png` | Mean SHAP bar chart (top-5 positive / top-5 negative) |
| `shap_mean_all_{tag}.csv` | Mean SHAP values for all features |
| `shap_mean_grouped_{tag}.csv` | \|mean SHAP\| aggregated by DB column group |

---

### Step 6: `DB_k_plot.py`

**Purpose**: Filter the database to DMF-only, unencapsulated devices and visualise the apparent first-order decomposition rate constant *k* as a 3D response surface over annealing conditions.

**Physics**: Assuming exponential decay `PCE(t)/PCE₀ = exp(−k·t)`, at `t = TS80m`: `k [day⁻¹] = −ln(0.80) / TS80m × 24`

**Key settings** (top of file): `K_APP_MAX`, `TEMP_MIN`, `TEMP_MAX`, `TIME_MAX`

**Output**: `outputs/figures/DB_k_response_surface.png`

---

## Citation

If you use this code, please cite:

```
Fukasawa R., Asahi T., Taniguchi T.
"Standardised process descriptors for interpretable regression of perovskite solar-cell stability"
(in preparation / journal TBD)
```

---

## License

MIT License
