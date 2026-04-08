"""
SHAP Analysis for Perovskite Stability Models (5-Fold CV)
=========================================================
Computes SHAP values using TreeExplainer with **interventional**
perturbation, averaged across 5-fold CV models (not a single FULL
model).  Uses joblib for parallel computation across folds.

Outputs per configuration:

  1. Mean SHAP bar chart — top positive (orange) / negative (blue)
     qualitative variables  (5-fold CV average)

  2. Full mean SHAP CSV — mean SHAP value for every expanded feature

  3. Group-aggregated SHAP CSV — summed |mean SHAP| per original
     DB column group

Prerequisites:
  - Trained fold models (.pkl) from regression.py
  - The corresponding CSR feature matrix
  - outputs/curated/Perovskite_5800data_addln.csv  (for column classification)

Configuration is via the MODEL_CONFIGS lists at the bottom of the file.
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
# 2. Matplotlib configuration
# =============================================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 14

# =============================================================================
# 3. Number of CV folds
# =============================================================================
N_FOLDS = 5


# =============================================================================
# 4. CSR I/O utility
# =============================================================================
def csr2vec(csr_file_name, columns_file_name=None):
    """Load a compressed sparse row matrix and reconstruct a DataFrame."""
    if columns_file_name is None:
        return load_npz(csr_file_name).toarray()
    return pd.DataFrame(
        load_npz(csr_file_name).toarray(),
        columns=np.load(columns_file_name, allow_pickle=True),
    )


# =============================================================================
# 5. Column classification
# =============================================================================
def classify_columns(df_5, exclude=None):
    """Return (num_set, obj_sorted) for feature importance grouping."""
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


# =============================================================================
# 6. Identify qualitative (categorical) expanded features
# =============================================================================
def get_qualitative_feature_names(columns, num_set, obj_sorted):
    """Return a boolean mask: True if the feature is categorical."""
    mask = []
    for fn in columns:
        fn_str = str(fn)
        if fn_str in num_set:
            mask.append(False)
            continue
        matched = any(fn_str.startswith(col + "_") for col in obj_sorted)
        mask.append(matched)
    return np.array(mask, dtype=bool)


# =============================================================================
# 7. SHAP computation — 5-fold CV parallel with interventional
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


def compute_shap_values_cv(models, X, n_samples=None, seed=0):
    """Compute SHAP values averaged over all CV folds (parallel).

    Parameters
    ----------
    models    : list of trained RandomForestRegressor (one per fold)
    X         : feature DataFrame (or array)
    n_samples : if not None, subsample X for faster computation
    seed      : random seed for subsampling

    Returns
    -------
    shap_values_mean : ndarray (n_samples, n_features) — 5-fold average
    X_used           : DataFrame actually used for SHAP
    """
    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_used = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        print(f"  Subsampled {n_samples}/{len(X)} rows for SHAP computation")
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

    # Sanity check
    print(f"  min={shap_values_mean.min():.6e}  "
          f"max={shap_values_mean.max():.6e}  "
          f"median={np.median(shap_values_mean):.6e}")

    return shap_values_mean, X_used


# =============================================================================
# 8. Mean SHAP bar chart (qualitative features)
# =============================================================================
def plot_mean_shap_bar(
    shap_values,
    columns,
    qual_mask,
    target_label,
    n_top=5,
    save_path=None,
    figsize=(10, 6),
):
    """Draw a horizontal bar chart of mean SHAP values.

    Shows the top `n_top` positive and top `n_top` negative features among
    qualitative (categorical) expanded features.
    """
    mean_shap = pd.Series(
        np.mean(shap_values, axis=0),
        index=list(columns),
    )

    qual_cols = [str(c) for c, m in zip(columns, qual_mask) if m]
    mean_shap_qual = mean_shap[[c for c in mean_shap.index if str(c) in set(qual_cols)]]

    top_pos = mean_shap_qual.nlargest(n_top)
    top_neg = mean_shap_qual.nsmallest(n_top)
    selected = pd.concat([top_neg, top_pos]).sort_values()

    colors = ["tab:orange" if v > 0 else "tab:blue" for v in selected.values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(selected)), selected.values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected.index, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Mean SHAP value for {target_label}", fontsize=13)
    ax.set_title("Representative qualitative variables affecting device stability\n"
                 "(5-Fold CV average)", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    else:
        plt.show()
    plt.close()

    return selected


# =============================================================================
# 9. Save mean SHAP CSV (all features)
# =============================================================================
def save_mean_shap_csv(shap_values, columns, out_path):
    """Save mean SHAP value (signed) for every expanded feature."""
    mean_shap = pd.Series(np.mean(shap_values, axis=0), index=list(columns))
    mean_shap_sorted = mean_shap.reindex(mean_shap.abs().sort_values(ascending=False).index)
    pd.DataFrame({"mean_shap": mean_shap_sorted}).to_csv(out_path, index=True)
    print(f"  Mean SHAP (all features) saved: {out_path}")


# =============================================================================
# 10. Group-aggregated SHAP CSV
# =============================================================================
def save_grouped_shap_csv(shap_values, columns, num_set, obj_sorted,
                          cbfv_block_label, out_path):
    """Aggregate |mean SHAP| by original DB column group and save to CSV."""
    mean_shap = pd.Series(np.mean(shap_values, axis=0), index=list(columns))

    group_sum = {c: 0.0 for c in list(num_set) + obj_sorted}
    group_sum[cbfv_block_label] = 0.0
    group_sum["UNMATCHED"] = 0.0

    for fn in mean_shap.index:
        fn_str = str(fn)
        val = abs(float(mean_shap.loc[fn]))

        if fn_str in num_set:
            group_sum[fn_str] += val
            continue

        assigned = False
        for col in obj_sorted:
            if fn_str.startswith(col + "_"):
                group_sum[col] += val
                assigned = True
                break

        if not assigned:
            if cbfv_block_label in group_sum:
                group_sum[cbfv_block_label] += val
            else:
                group_sum["UNMATCHED"] += val

    summary = (
        pd.Series(group_sum, name="mean_abs_shap_sum")
        .sort_values(ascending=False)
        .to_frame()
    )
    summary.to_csv(out_path, index=True)
    print(f"  Grouped |mean SHAP| saved: {out_path}")
    return summary


# =============================================================================
# 11. Per-model analysis pipeline (5-fold CV)
# =============================================================================
def run_shap_analysis(
    model_dir,
    model_prefix,
    csr_path,
    col_path,
    df_5,
    out_dir,
    tag,
    target_label,
    n_top=5,
    n_samples=None,
    n_folds=N_FOLDS,
    cbfv_block_label="UNMATCHED",
):
    """Load 5-fold models + features, compute SHAP (parallel), save outputs.

    Parameters
    ----------
    model_dir        : directory containing rf_model_*_fold{k}.pkl
    model_prefix     : tag for model files (e.g. "lnTS80m_all_sp2_dummy_zero")
    csr_path         : path to .npz feature matrix
    col_path         : path to .npy column names
    df_5             : curated DataFrame (for column classification)
    out_dir          : output directory
    tag              : string tag for output filenames
    target_label     : axis label string, e.g. "ln(TS80m)"
    n_top            : number of top pos/neg features in bar chart
    n_samples        : subsample size for SHAP (None = use full dataset)
    n_folds          : number of CV folds
    cbfv_block_label : label for unmatched (CBFV) features in grouping
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"SHAP analysis (5-fold CV): {tag}")
    print(f"{'='*60}")

    # Load fold models
    models = []
    for k in range(1, n_folds + 1):
        path = os.path.join(model_dir, f"rf_model_{model_prefix}_fold{k}.pkl")
        models.append(joblib.load(path))
        print(f"  Loaded fold {k}: {path}")

    X = csr2vec(csr_path, col_path)
    columns = np.load(col_path, allow_pickle=True)
    print(f"  Feature matrix: {X.shape}")

    num_set, obj_sorted = classify_columns(df_5)
    qual_mask = get_qualitative_feature_names(columns, num_set, obj_sorted)
    print(f"  Qualitative features: {qual_mask.sum()}/{len(columns)}")

    # SHAP computation (5-fold parallel, interventional)
    shap_values, X_used = compute_shap_values_cv(
        models, X, n_samples=n_samples,
    )

    # Bar chart
    selected = plot_mean_shap_bar(
        shap_values=shap_values,
        columns=columns,
        qual_mask=qual_mask,
        target_label=target_label,
        n_top=n_top,
        save_path=os.path.join(out_dir, f"shap_mean_bar_{tag}.png"),
    )
    print(f"\n  Top positive features:")
    for fn, v in selected[selected > 0].items():
        print(f"    {fn:60s}  {v:+.5f}")
    print(f"  Top negative features:")
    for fn, v in selected[selected < 0].items():
        print(f"    {fn:60s}  {v:+.5f}")

    # CSVs
    save_mean_shap_csv(
        shap_values, columns,
        out_path=os.path.join(out_dir, f"shap_mean_all_{tag}.csv"),
    )
    save_grouped_shap_csv(
        shap_values, columns, num_set, obj_sorted,
        cbfv_block_label=cbfv_block_label,
        out_path=os.path.join(out_dir, f"shap_mean_grouped_{tag}.csv"),
    )


# =============================================================================
# 12. Model configurations per target
# =============================================================================
# Each entry now uses model_prefix (base tag) instead of a single FULL model path.
# The fold models are: rf_model_{model_prefix}_fold{1..5}.pkl

STABILITY_SHAP_CONFIGS = [
    dict(
        model_prefix ="lnTS80m_all_sp2_dummy_zero",
        csr_path     ="outputs/csr/all_sp2_dummy_zero_csr.npz",
        col_path     ="outputs/csr/all_sp2_dummy_zero_columns.npy",
        tag          ="lnTS80m_all_sp2_dummy_zero_CV5mean",
        target_label ="ln(TS80m)",
        n_samples    =None,
        cbfv_block_label="UNMATCHED",
    ),
    dict(
        model_prefix ="lnTS80m_all_sp3_dummy_zero",
        csr_path     ="outputs/csr/all_sp3_dummy_zero_csr.npz",
        col_path     ="outputs/csr/all_sp3_dummy_zero_columns.npy",
        tag          ="lnTS80m_all_sp3_dummy_zero_CV5mean",
        target_label ="ln(TS80m)",
        n_samples    =None,
        cbfv_block_label="UNMATCHED",
    ),
    dict(
        model_prefix ="TS80m_all_sp0_oliynyk_zero",
        csr_path     ="outputs/csr/all_sp0_oliynyk_zero_csr.npz",
        col_path     ="outputs/csr/all_sp0_oliynyk_zero_columns.npy",
        tag          ="TS80m_all_sp0_oliynyk_zero_CV5mean",
        target_label ="TS80m",
        n_samples    =None,
        cbfv_block_label="Perovskite_Oliynyk_features",
    ),
]

PCE_SHAP_CONFIGS = [
    dict(
        model_prefix ="JV_default_PCE_all_sp2_dummy_zero",
        csr_path     ="outputs/csr/all_sp2_dummy_zero_csr.npz",
        col_path     ="outputs/csr/all_sp2_dummy_zero_columns.npy",
        tag          ="JV_default_PCE_all_sp2_dummy_zero_CV5mean",
        target_label ="PCE (%)",
        n_samples    =None,
        cbfv_block_label="UNMATCHED",
    ),
    dict(
        model_prefix ="JV_default_PCE_all_sp2_oliynyk_zero",
        csr_path     ="outputs/csr/all_sp2_oliynyk_zero_csr.npz",
        col_path     ="outputs/csr/all_sp2_oliynyk_zero_columns.npy",
        tag          ="JV_default_PCE_all_sp2_oliynyk_zero_CV5mean",
        target_label ="PCE (%)",
        n_samples    =None,
        cbfv_block_label="Perovskite_Oliynyk_features",
    ),
]


# =============================================================================
# 13. Main
# =============================================================================
def main():
    import pipeline_config

    data_path = "outputs/curated/Perovskite_5800data_addln.csv"
    model_dir = "outputs/model/cv5"
    out_dir   = "outputs/model/shap"

    df_5 = pd.read_csv(data_path)
    print(f"Loaded: {len(df_5)} rows from {data_path}")

    # Select configurations based on target
    if pipeline_config.TARGET_MODE == "PCE":
        configs = PCE_SHAP_CONFIGS
    else:
        configs = STABILITY_SHAP_CONFIGS

    for cfg in configs:
        run_shap_analysis(
            model_dir       =model_dir,
            model_prefix    =cfg["model_prefix"],
            csr_path        =cfg["csr_path"],
            col_path        =cfg["col_path"],
            df_5            =df_5,
            out_dir         =out_dir,
            tag             =cfg["tag"],
            target_label    =cfg["target_label"],
            n_top           =5,
            n_samples       =cfg.get("n_samples"),
            cbfv_block_label=cfg.get("cbfv_block_label", "UNMATCHED"),
        )

    print("\n=== SHAP analysis complete ===")
    print(f"Outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
