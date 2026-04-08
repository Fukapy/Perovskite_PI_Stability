"""
FTI + SHAP for 5-Fold CV models: lnTS80m / sp2 / dummy
========================================================
Standalone script for the primary model configuration.
Run from the repository root.

Changes from the original (single FULL model) version:
- Loads fold1-fold5 models instead of a single FULL model
- FTI: averages feature_importances_ across 5 folds (+ std, per-fold)
- SHAP: uses feature_perturbation='interventional' (numerically stable)
- SHAP: parallel computation across folds via joblib

Outputs
-------
outputs/model/cv5/
  fti_lnTS80m_all_sp2_dummy_zero_CV5mean.csv        — Lv1 raw FTI
  fti_sum_lnTS80m_all_sp2_dummy_zero_CV5mean.csv    — Lv2 DB-column FTI
  fti_layer_lnTS80m_all_sp2_dummy_zero_CV5mean.csv  — Lv3 layer/process FTI

outputs/model/shap/
  shap_mean_bar_lnTS80m_all_sp2_dummy_zero_CV5mean.png
  shap_mean_all_lnTS80m_all_sp2_dummy_zero_CV5mean.csv
  shap_mean_grouped_lnTS80m_all_sp2_dummy_zero_CV5mean.csv

Usage
-----
    python compute_fti_shap_sp2dummy.py
"""

# =============================================================================
# 1. Imports
# =============================================================================
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.sparse import load_npz

warnings.filterwarnings("ignore")

# =============================================================================
# 2. Paths
# =============================================================================
BASE_DIR     = "outputs"
MODEL_DIR    = f"{BASE_DIR}/model/cv5"
MODEL_PREFIX = "rf_model_lnTS80m_all_sp2_dummy_zero"
N_FOLDS      = 5

CSR_PATH     = f"{BASE_DIR}/csr/all_sp2_dummy_zero_csr.npz"
COL_PATH     = f"{BASE_DIR}/csr/all_sp2_dummy_zero_columns.npy"
DATA_PATH    = f"{BASE_DIR}/curated/Perovskite_5800data_addln.csv"
FTI_OUT_DIR  = f"{BASE_DIR}/model/cv5"
SHAP_OUT_DIR = f"{BASE_DIR}/model/shap"

TAG          = "lnTS80m_all_sp2_dummy_zero_CV5mean"
TARGET_LABEL = "ln(TS80m)"
N_TOP        = 5          # top pos / neg features in SHAP bar chart
N_SAMPLES    = None       # set to e.g. 2000 for faster SHAP; None = full dataset

# =============================================================================
# 3. Matplotlib configuration
# =============================================================================
plt.rcParams["font.family"]   = "Arial"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"]     = 14

# =============================================================================
# 4. Layer / process category map  (Lv3 aggregation)
# =============================================================================
LAYER_MAP = {
    "Perovskite_composition"  : "Perovskite (composition)",
    "Perovskite_deposition"   : "Perovskite (deposition)",
    "Perovskite"              : "Perovskite (other)",
    "ETL"                     : "Electron transport layer",
    "HTL"                     : "Hole transport layer",
    "Backcontact"             : "Back contact",
    "Substrate"               : "Substrate",
    "Cell"                    : "Cell architecture",
    "Encapsulation"           : "Encapsulation",
    "Stability"               : "Stability test conditions",
    "JV"                      : "JV characteristics",
}


# =============================================================================
# 5. Utilities
# =============================================================================
def classify_columns(df_5, exclude=None):
    if exclude is None:
        exclude = {"Original_index", "TS80", "TS80m", "lnTS80m", "JV_default_PCE"}
    num_cols, obj_cols = [], []
    for col in df_5.columns:
        if col in exclude:
            continue
        if pd.api.types.is_bool_dtype(df_5[col]):
            obj_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df_5[col]):
            num_cols.append(col)
        else:
            obj_cols.append(col)
    num_set    = set(str(c) for c in num_cols)
    obj_sorted = sorted((str(c) for c in obj_cols), key=len, reverse=True)
    return num_set, obj_sorted


def csr2df(csr_path, col_path):
    return pd.DataFrame(
        load_npz(csr_path).toarray(),
        columns=np.load(col_path, allow_pickle=True),
    )


# =============================================================================
# 6. Feature importance (3 levels) — 5-fold average
# =============================================================================
def compute_fti_cv(models, columns, num_set, obj_sorted, fti_out_dir, tag):
    """Compute and save Lv1, Lv2, Lv3 feature importances (5-fold mean)."""
    os.makedirs(fti_out_dir, exist_ok=True)

    imp_array = np.array([m.feature_importances_ for m in models])
    mean_imp = imp_array.mean(axis=0)
    std_imp  = imp_array.std(axis=0)

    # Lv1 — raw expanded features (mean + std + per-fold)
    fti_lv1 = pd.DataFrame({
        "importance_mean": mean_imp,
        "importance_std" : std_imp,
    }, index=list(columns))
    for fold in range(1, len(models) + 1):
        fti_lv1[f"fold{fold}"] = imp_array[fold - 1]
    fti_lv1 = fti_lv1.sort_values("importance_mean", ascending=False)
    fti_lv1.index.name = "feature"
    lv1_path = os.path.join(fti_out_dir, f"fti_{tag}.csv")
    fti_lv1.to_csv(lv1_path)
    print(f"  [Lv1] Raw FTI  → {lv1_path}  ({len(fti_lv1)} features)")

    # Lv2 — aggregate to DB column groups (using mean_imp)
    fti_mean_series = pd.Series(mean_imp, index=list(columns))
    group_sum = {c: 0.0 for c in list(num_set) + obj_sorted}
    group_sum["UNMATCHED"] = 0.0
    for fn in fti_mean_series.index.astype(str):
        val = float(fti_mean_series.loc[fn])
        if fn in num_set:
            group_sum[fn] += val
            continue
        matched = False
        for col in obj_sorted:
            if fn.startswith(col + "_"):
                group_sum[col] += val
                matched = True
                break
        if not matched:
            group_sum["UNMATCHED"] += val
    fti_sum = (pd.Series(group_sum, name="feature_importance_sum")
               .sort_values(ascending=False).to_frame())
    lv2_path = os.path.join(fti_out_dir, f"fti_sum_{tag}.csv")
    fti_sum.to_csv(lv2_path)
    print(f"  [Lv2] Column-level FTI  → {lv2_path}  ({len(fti_sum)} groups)")
    print(f"        Total={mean_imp.sum():.4f}  "
          f"UNMATCHED={fti_sum.loc['UNMATCHED','feature_importance_sum']:.4f}")
    print("        Top 10:")
    print(fti_sum.head(10).to_string())

    # Lv3 — aggregate to layer/process categories
    sorted_pfx = sorted(LAYER_MAP.keys(), key=len, reverse=True)
    layer_sum = {}
    for param, row in fti_sum.iterrows():
        imp = float(row["feature_importance_sum"])
        assigned = False
        for pfx in sorted_pfx:
            if str(param).startswith(pfx):
                lbl = LAYER_MAP[pfx]
                layer_sum[lbl] = layer_sum.get(lbl, 0.0) + imp
                assigned = True
                break
        if not assigned:
            layer_sum["Other / CBFV"] = layer_sum.get("Other / CBFV", 0.0) + imp
    fti_layer = (pd.Series(layer_sum, name="feature_importance_sum")
                 .sort_values(ascending=False).to_frame())
    lv3_path = os.path.join(fti_out_dir, f"fti_layer_{tag}.csv")
    fti_layer.to_csv(lv3_path)
    print(f"\n  [Lv3] Layer-level FTI  → {lv3_path}")
    print(fti_layer.to_string())

    return fti_lv1, fti_sum, fti_layer


# =============================================================================
# 7. SHAP analysis — 5-fold parallel with interventional perturbation
# =============================================================================
def _shap_one_fold(fold_i, model, X_data):
    """Compute SHAP values for a single fold (called in parallel)."""
    explainer = shap.TreeExplainer(
        model,
        data=X_data,
        feature_perturbation="interventional",
    )
    sv = explainer.shap_values(X_data, check_additivity=False)
    print(f"  fold {fold_i} done — shape={np.array(sv).shape}")
    return sv


def compute_shap_cv(models, X, n_samples=None, seed=0):
    """Compute SHAP values averaged over all CV folds (parallel)."""
    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_used = X.iloc[idx]
        print(f"  Subsampled {n_samples}/{len(X)} rows for SHAP")
    else:
        X_used = X

    n_jobs = min(len(models), os.cpu_count())
    print(f"  Parallel SHAP: {len(models)} folds × {n_jobs} jobs "
          f"(CPUs={os.cpu_count()})")
    print(f"  feature_perturbation='interventional'")

    t0 = time.time()
    shap_all_folds = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_shap_one_fold)(i, m, X_used)
        for i, m in enumerate(models, 1)
    )
    elapsed = time.time() - t0

    shap_values_mean = np.mean(shap_all_folds, axis=0)
    print(f"  Averaged SHAP shape: {shap_values_mean.shape}")
    print(f"  Elapsed: {elapsed:.1f} sec")
    print(f"  min={shap_values_mean.min():.6e}  "
          f"max={shap_values_mean.max():.6e}  "
          f"median={np.median(shap_values_mean):.6e}")

    return shap_values_mean, X_used


def save_shap_outputs(shap_values, columns, num_set, obj_sorted,
                      shap_out_dir, tag, target_label, n_top=5):
    os.makedirs(shap_out_dir, exist_ok=True)
    mean_shap = pd.Series(np.mean(shap_values, axis=0), index=list(columns))

    # --- bar chart (qualitative features only) ---
    qual_mask = []
    for fn in columns:
        fn_str = str(fn)
        if fn_str in num_set:
            qual_mask.append(False)
        else:
            qual_mask.append(any(fn_str.startswith(c + "_") for c in obj_sorted))
    qual_mask = np.array(qual_mask, dtype=bool)

    qual_cols = set(str(c) for c, m in zip(columns, qual_mask) if m)
    mean_shap_qual = mean_shap[[c for c in mean_shap.index if str(c) in qual_cols]]

    top_pos = mean_shap_qual.nlargest(n_top)
    top_neg = mean_shap_qual.nsmallest(n_top)
    selected = pd.concat([top_neg, top_pos]).sort_values()
    bar_colors = ["tab:orange" if v > 0 else "tab:blue" for v in selected.values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(selected)), selected.values, color=bar_colors, edgecolor="none")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected.index, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Mean SHAP value for {target_label}", fontsize=13)
    ax.set_title("Representative qualitative variables affecting device stability\n"
                 "(5-Fold CV average)", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    bar_path = os.path.join(shap_out_dir, f"shap_mean_bar_{tag}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP bar chart  → {bar_path}")

    # --- mean SHAP CSV ---
    mean_shap_sorted = mean_shap.reindex(
        mean_shap.abs().sort_values(ascending=False).index)
    all_path = os.path.join(shap_out_dir, f"shap_mean_all_{tag}.csv")
    pd.DataFrame({"mean_shap": mean_shap_sorted}).to_csv(all_path)
    print(f"  Mean SHAP (all) → {all_path}")

    # --- grouped |mean SHAP| CSV ---
    group_sum = {c: 0.0 for c in list(num_set) + obj_sorted}
    group_sum["UNMATCHED"] = 0.0
    for fn in mean_shap.index:
        fn_str = str(fn)
        val = abs(float(mean_shap.loc[fn]))
        if fn_str in num_set:
            group_sum[fn_str] += val
            continue
        matched = False
        for col in obj_sorted:
            if fn_str.startswith(col + "_"):
                group_sum[col] += val
                matched = True
                break
        if not matched:
            group_sum["UNMATCHED"] += val
    grp_path = os.path.join(shap_out_dir, f"shap_mean_grouped_{tag}.csv")
    (pd.Series(group_sum, name="mean_abs_shap_sum")
       .sort_values(ascending=False)
       .to_frame()
       .to_csv(grp_path))
    print(f"  Grouped SHAP    → {grp_path}")


# =============================================================================
# 8. Main
# =============================================================================
def main():
    print("=" * 60)
    print(f"5-Fold CV model: {TAG}")
    print("=" * 60)

    # Load artefacts
    print("\n[Step 1] Loading models and feature matrix...")
    models = []
    for fold in range(1, N_FOLDS + 1):
        path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_fold{fold}.pkl")
        models.append(joblib.load(path))
        print(f"  Loaded fold {fold}: {path}")

    columns = np.load(COL_PATH, allow_pickle=True)
    X       = csr2df(CSR_PATH, COL_PATH)
    df_5    = pd.read_csv(DATA_PATH)
    print(f"  Matrix : {X.shape}")
    print(f"  Data   : {len(df_5)} rows")

    num_set, obj_sorted = classify_columns(df_5)
    print(f"  num={len(num_set)}  cat={len(obj_sorted)}")

    # FTI (3 levels, 5-fold mean)
    print("\n[Step 2] Feature importance (5-fold mean, 3-level aggregation)...")
    compute_fti_cv(models, columns, num_set, obj_sorted, FTI_OUT_DIR, TAG)

    # SHAP (5-fold mean, interventional, parallel)
    print("\n[Step 3] SHAP analysis (5-fold mean, interventional, parallel)...")
    shap_values, _ = compute_shap_cv(models, X, n_samples=N_SAMPLES)
    save_shap_outputs(
        shap_values, columns, num_set, obj_sorted,
        SHAP_OUT_DIR, TAG, TARGET_LABEL, n_top=N_TOP,
    )

    print("\n=== Done ===")
    print(f"FTI outputs : {FTI_OUT_DIR}/")
    print(f"SHAP outputs: {SHAP_OUT_DIR}/")


if __name__ == "__main__":
    main()
