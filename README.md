# Standardised Process Descriptors for Interpretable Regression of Perovskite Solar-Cell Stability

**Ryo Fukasawa, Toru Asahi, Takuya Taniguchi**

This repository contains the analysis code accompanying the paper:

> *Standardised process descriptors for interpretable regression of perovskite solar-cell stability*

---

## Overview

Fabrication processes strongly influence perovskite solar-cell stability, yet their heterogeneous nature has hindered unified data-driven analysis. This work proposes a standardised descriptor framework that converts diverse process information recorded in [The Perovskite Database](https://www.perovskitedatabase.com/) into a unified numerical feature matrix for interpretable regression.

The analysis proceeds in two parallel streams:

- **Stream A — Database regression**: 5,800 curated entries from The Perovskite Database are used to train a random-forest model predicting ln(TS80m). SHAP-based interpretation identifies annealing conditions, solvent type, and layer-specific process parameters as dominant factors.
- **Stream B — Thin-film experiments**: Controlled MAPbI₃ thin-film degradation experiments independently validate the trends extracted from the database model, using XRD-monitored PbI₂ formation as a degradation index.

---

## Repository Structure

```
.
├── README.md
│
├── # ── Stream A: Perovskite Database ──────────────────────────────
├── data_curation.py
├── vectorization.py
├── reanalysis.py
├── interpretation.py
├── shap_analysis.py
├── DB_k_plot.py
│
├── # ── Stream B: Thin-film XRD experiments ────────────────────────
├── XRD_regression.py
├── XRD_rf_shap.py
├── XRD_response_surface.py
│
├── # ── Data ────────────────────────────────────────────────────────
├── data/
│   ├── 20250624_5800/
│   │   ├── raw/                        ← curated CSVs (input + output of Stream A)
│   │   ├── csr/                        ← CSR feature matrices (.npz / .npy)
│   │   └── model/
│   │       ├── cv5/                    ← 5-fold CV models, OOF predictions, FTI CSVs
│   │       └── shap/                   ← SHAP output (created on first run)
│   └── XRD/
│       ├── PbI2_ratio_20241114_successed_3.csv   ← raw XRD measurements
│       ├── PbI2_ratio_summary_by_sample_20251010.csv  ← per-sample slopes
│       └── (output CSVs and figures written here by Stream B scripts)
│
└── figures/                            ← publication figures (created on first run)
```

> **Working directory**: all scripts must be run from the **repository root** (the directory containing this README). All paths are relative to that location.

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib shap
```

The CBFV vectorisation additionally requires the [`CBFV`](https://github.com/kaaiian/CBFV) package and element-property tables (oliynyk, magpie, mat2vec).

---

## Stream A — Perovskite Database Analysis

Run scripts in the following order.

### 1. `data_curation.py`

**Purpose**: Load and curate the raw Perovskite Database, apply filtering criteria, and build the initial feature matrix.

**Input**:
- `data/20250624_5800/raw/ZhangZ_PD.csv` — The Perovskite Database (download from [perovskitedatabase.com](https://www.perovskitedatabase.com/))

**Filtering steps**:
1. Remove entries with missing stability records or non-ASCII characters
2. Remove stoichiometrically inconsistent compositions
3. Remove entries producing zero-vector CBFV representations (magpie check)
4. Compute ln(TS80m) as the regression target

**Output** (written to `data/20250624_5800/raw/` and `data/20250624_5800/csr/`):
- `Perovskite_5800data_addln.csv` — curated dataset (5,800 entries)
- `all_sp1_oliynyk_zero_{csr.npz,_columns.npy}` — baseline CSR feature matrix

```bash
python data_curation.py
```

---

### 2. `vectorization.py`

**Purpose**: Build all feature matrix variants across encoding schemes (sp0–sp3) and CBFV representations (oliynyk, magpie, mat2vec, dummy/one-hot).

**Input**: `data/20250624_5800/raw/Perovskite_5800data_addln.csv`

**Encoding schemes**:
| Tag | Categorical encoding |
|-----|---------------------|
| sp0 | one-hot |
| sp1 | multi-hot variant 1 |
| sp2 | multi-hot variant 2 |
| sp3 | multi-hot variant 3 |

**Output**: CSR matrices saved to `data/20250624_5800/csr/`
(e.g. `all_sp3_dummy_zero_{csr.npz,_columns.npy}`, `per_sp0_oliynyk_dummy_{csr.npz,_columns.npy}`, …)

```bash
python vectorization.py
```

---

### 3. `reanalysis.py`

**Purpose**: Rigorous 5-fold cross-validation of the random-forest model for ln(TS80m) across all encoding variants. Saves per-fold models, out-of-fold (OOF) predictions, and a final FULL model trained on the complete dataset.

**Input**: `data/20250624_5800/raw/Perovskite_5800data_addln.csv` + CSR matrices from step 2

**Key setting** (top of file):
- `RUN_TS80M = False` — set to `True` to additionally run regression on raw TS80m (not used in the paper)

**Output** (written to `data/20250624_5800/model/cv5/`):
| File | Description |
|------|-------------|
| `rf_model_{tag}_fold{k}.pkl` | Per-fold trained models |
| `cv5_oof_{tag}.csv` | Out-of-fold predictions |
| `cv5_metrics_{tag}.csv` / `cv5_summary_{tag}.csv` | CV R² statistics |
| `rf_model_{tag}_FULL.pkl` | Full-data model (used for interpretation) |
| `CV_summary_clean_table.csv` | Collated summary across all variants |

```bash
python reanalysis.py
```

---

### 4. `interpretation.py`

**Purpose**: Compute feature importances from the FULL models at three levels of aggregation.

**Input**: `rf_model_{tag}_FULL.pkl` and `{tag}_columns.npy` from step 3

**Three-level aggregation**:
| Level | Output file | Description |
|-------|-------------|-------------|
| Lv1 | `fti_{tag}.csv` | Raw expanded features (thousands of columns) |
| Lv2 | `fti_sum_{tag}.csv` | Aggregated by original DB column name |
| Lv3 | `fti_layer_{tag}.csv` | Aggregated by functional layer / process category |

**Layer categories** (editable via `LAYER_MAP` at the top of the file):
`Perovskite (composition)`, `Perovskite (deposition)`, `ETL`, `HTL`, `Back contact`, `Substrate`, `Cell architecture`, `Encapsulation`, `Stability test conditions`

**Output**: Written to `data/20250624_5800/model/cv5/`

```bash
python interpretation.py
```

---

### 5. `shap_analysis.py`

**Purpose**: Compute SHAP values for the FULL models and generate the mean SHAP bar chart (Fig. 4e in the paper), showing the top stabilising and destabilising qualitative variables.

**Input**: `rf_model_{tag}_FULL.pkl` + CSR matrices from step 3

**Output** (written to `data/20250624_5800/model/shap/`):
| File | Description |
|------|-------------|
| `shap_mean_bar_{tag}.png` | Mean SHAP bar chart (top-5 positive / top-5 negative) |
| `shap_mean_all_{tag}.csv` | Mean SHAP values for all features |
| `shap_mean_grouped_{tag}.csv` | \|mean SHAP\| aggregated by DB column group |

```bash
python shap_analysis.py
```

---

### 6. `DB_k_plot.py`

**Purpose**: Filter the database to DMF-only, unencapsulated devices and visualise the apparent first-order decomposition rate constant *k* as a 3D response surface over annealing conditions (Fig. 6d in the paper).

**Physics**:
Assuming exponential decay: `PCE(t)/PCE₀ = exp(−k·t)`
At `t = TS80m`, `PCE/PCE₀ = 0.80` → `k [day⁻¹] = −ln(0.80) / TS80m × 24`

**Input**: `data/20250624_5800/raw/Perovskite_5800data_addln.csv`

**Key settings** (top of file):
- `K_APP_MAX = 0.05` — outlier threshold for *k* (day⁻¹); adjust to explore sensitivity
- `TEMP_MIN / TEMP_MAX / TIME_MAX` — annealing condition filter range

**Output**: `figures/DB_k_response_surface.png`

```bash
python DB_k_plot.py
```

---

## Stream B — Thin-Film XRD Experiments

Run scripts in the following order.

### 1. `XRD_regression.py`

**Purpose**: For each MAPbI₃ thin-film sample, fit a linear regression of the XRD intensity ratio I(PbI₂ 001)/I(MAPbI₃ 110) vs storage day to obtain the degradation slope. Apply humidity correction and generate grouped regression plots (Figs. 5c–e).

**Input**: `data/XRD/PbI2_ratio_20241114_successed_3.csv`
(raw XRD measurements: one row per measurement per sample)

**Humidity correction**:
`slope_corr = slope_per_day / Storage_RH × 20`
(normalises to 20 % RH reference)

**Output** (written to `data/XRD/`):
- `PbI2_ratio_linear_regression_by_sample_{DATE}.csv` — slope, intercept, R² per sample
- `Sample_conditions_by_sample_{DATE}.csv` — annealing conditions per sample
- `PbI2_ratio_summary_by_sample_{DATE}.csv` — merged summary with corrected slopes

**Output** (written to `figures/`):
- `XRD_regression_Heat_minute.png` — degradation rate vs days, grouped by annealing time
- `XRD_regression_Heat_temperature.png` — grouped by annealing temperature
- `XRD_regression_DMF_ratio.png` — grouped by solvent (DMF vs DMSO)

```bash
python XRD_regression.py
```

---

### 2. `XRD_rf_shap.py`

**Purpose**: Train a random-forest model on the per-sample humidity-corrected degradation rate, compute Gini and permutation feature importances, and run SHAP analysis.

**Input**: `data/XRD/PbI2_ratio_summary_by_sample_20251010.csv`

**Features**: `Heat_temperature`, `Heat_minute`, `DMF_ratio`
**Target**: `corr_slope_per_day`

**Validation**: Leave-one-out cross-validation (LOO-CV) is used given the small dataset size (n ≈ 45).

**Output** (written to `data/XRD/`):
| File | Description |
|------|-------------|
| `rf_feature_importances_all.csv` / `.png` | Gini feature importances |
| `rf_permutation_importances_all.csv` / `.png` | Permutation importances |
| `rf_in_sample_r2.csv` | In-sample and LOO R² |
| `shap_summary_bar_all.png` | SHAP summary bar chart |
| `shap_beeswarm_all.png` | SHAP beeswarm plot |
| `shap_dependence_std_{feature}.png` | SHAP dependence plots per feature |

```bash
python XRD_rf_shap.py
```

---

### 3. `XRD_response_surface.py`

**Purpose**: Fit a second-order polynomial to the humidity-corrected degradation rate as a function of annealing temperature and time, and visualise the result as a 3D response surface with negative-Z regions masked (Figs. 6b–c).

**Polynomial model**:
`S = c₀ + c₁T + c₂t + c₃Tt + c₄T² + c₅t²`

**Input**: `data/XRD/PbI2_ratio_summary_by_sample_20251010.csv`

**Key settings** (top of file):
- `ELEV`, `AZIM` — 3D view angles (default: 30°, −30°)
- `GRID_N` — surface mesh resolution (default: 200)

**Output** (written to `figures/`):
- `thinfilm_DMF1.png` — response surface for DMF solvent (DMF_ratio = 1)
- `thinfilm_DMF0.png` — response surface for DMSO solvent (DMF_ratio = 0)

```bash
python XRD_response_surface.py
```

---

## Data Files

| Path | Description | Provided |
|------|-------------|----------|
| `data/20250624_5800/raw/ZhangZ_PD.csv` | The Perovskite Database (original) | Download from [perovskitedatabase.com](https://www.perovskitedatabase.com/) |
| `data/20250624_5800/raw/Perovskite_5800data_addln.csv` | Curated dataset with ln(TS80m) | Generated by `data_curation.py` |
| `data/20250624_5800/csr/` | CSR feature matrices | Generated by `vectorization.py` |
| `data/20250624_5800/model/cv5/` | Trained models and CV results | Generated by `reanalysis.py` |
| `data/XRD/PbI2_ratio_20241114_successed_3.csv` | Raw XRD intensity ratio measurements | ✓ Included |
| `data/XRD/PbI2_ratio_summary_by_sample_20251010.csv` | Per-sample degradation slopes and conditions | ✓ Included |

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
