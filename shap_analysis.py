"""
SHAP Analysis for Perovskite Stability Models
==============================================
Computes SHAP values using TreeExplainer and generates:

  1. Mean SHAP bar chart — "Representative qualitative variables affecting
     device stability"
       - Top N features with highest positive mean SHAP value  (orange)
       - Top N features with lowest  negative mean SHAP value  (blue)
       - Categorical (qualitative) features only by default

  2. Full mean SHAP CSV — mean SHAP value for every expanded feature

  3. Group-aggregated SHAP CSV — summed absolute mean SHAP per original
     DB column group (numerical / categorical / Oliynyk CBFV block)

Prerequisites:
  - A trained FULL model (.pkl) from 20260214_reanalysis.py
  - The corresponding CSR feature matrix
  - outputs/raw/Perovskite_5800data_addln.csv  (for column classification)

Configuration is via the MODEL_CONFIGS list at the bottom of the file.
Add entries to run analysis for multiple models in one pass.
"""

# =============================================================================
# 1. Imports
# =============================================================================
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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
# 3. CSR I/O utility
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
# 4. Column classification
# =============================================================================
def classify_columns(df_5, exclude=None):
    """Return (num_set, obj_sorted) for feature importance grouping.

    num_set    : set of numerical column name strings
    obj_sorted : list of categorical column name strings, sorted by length
                 descending (prevents partial prefix shadowing)
    """
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
# 5. Identify qualitative (categorical) expanded features
# =============================================================================
def get_qualitative_feature_names(columns, num_set, obj_sorted):
    """Return a boolean mask: True if the expanded feature name belongs to a
    categorical (qualitative) DB column.

    Numerical features and CBFV features are excluded.
    """
    mask = []
    for fn in columns:
        fn_str = str(fn)
        # Numerical: exact match
        if fn_str in num_set:
            mask.append(False)
            continue
        # Categorical: prefix match
        matched = any(fn_str.startswith(col + "_") for col in obj_sorted)
        mask.append(matched)
    return np.array(mask, dtype=bool)


# =============================================================================
# 6. SHAP computation
# =============================================================================
def compute_shap_values(model, X, n_samples=None, seed=0):
    """Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model     : trained RandomForestRegressor
    X         : feature DataFrame (or array)
    n_samples : if not None, subsample X for faster computation
    seed      : random seed for subsampling

    Returns
    -------
    shap_values : ndarray (n_samples, n_features)
    X_used      : DataFrame actually used for SHAP
    """
    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_used = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        print(f"  Subsampled {n_samples}/{len(X)} rows for SHAP computation")
    else:
        X_used = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_used)
    print(f"  SHAP values shape: {np.array(shap_values).shape}")
    return shap_values, X_used


# =============================================================================
# 7. Mean SHAP bar chart (qualitative features)
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

    Orange bars : positive mean SHAP (increase stability)
    Blue bars   : negative mean SHAP (decrease stability)

    Parameters
    ----------
    shap_values  : ndarray (n_samples, n_features)
    columns      : array-like of expanded feature names
    qual_mask    : boolean array — True for qualitative features
    target_label : str — label for x-axis (e.g. "ln(TS80m)")
    n_top        : number of top positive / negative features to show
    save_path    : file path to save figure (None = display only)
    figsize      : tuple
    """
    mean_shap = pd.Series(
        np.mean(shap_values, axis=0),
        index=list(columns),
    )

    # Restrict to qualitative features
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
    ax.set_title("Representative qualitative variables affecting device stability",
                 fontsize=13, pad=10)
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
# 8. Save mean SHAP CSV (all features)
# =============================================================================
def save_mean_shap_csv(shap_values, columns, out_path):
    """Save mean SHAP value (signed) for every expanded feature."""
    mean_shap = pd.Series(np.mean(shap_values, axis=0), index=list(columns))
    mean_shap_sorted = mean_shap.reindex(mean_shap.abs().sort_values(ascending=False).index)
    pd.DataFrame({"mean_shap": mean_shap_sorted}).to_csv(out_path, index=True)
    print(f"  Mean SHAP (all features) saved: {out_path}")


# =============================================================================
# 9. Group-aggregated SHAP CSV
# =============================================================================
def save_grouped_shap_csv(shap_values, columns, num_set, obj_sorted,
                          cbfv_block_label, out_path):
    """Aggregate |mean SHAP| by original DB column group and save to CSV.

    Aggregation rules (same as 20260215_interpretation.py):
      - Numerical   : exact feature name match
      - Categorical : prefix match "col_value" → col
      - Remainder   : assigned to cbfv_block_label (or "UNMATCHED")
    """
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
# 10. Per-model analysis pipeline
# =============================================================================
def run_shap_analysis(
    model_path,
    csr_path,
    col_path,
    df_5,
    out_dir,
    tag,
    target_label,
    n_top=5,
    n_samples=None,
    cbfv_block_label="Perovskite_CBFV_features",
):
    """Load model + features, compute SHAP, and save all outputs.

    Parameters
    ----------
    model_path       : path to .pkl FULL model
    csr_path         : path to .npz feature matrix
    col_path         : path to .npy column names
    df_5             : curated DataFrame (for column classification)
    out_dir          : output directory
    tag              : string tag for output filenames
    target_label     : axis label string, e.g. "ln(TS80m)"
    n_top            : number of top pos/neg features in bar chart
    n_samples        : subsample size for SHAP (None = use full dataset)
    cbfv_block_label : label for unmatched (CBFV) features in grouping
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"SHAP analysis: {tag}")
    print(f"{'='*60}")

    model = joblib.load(model_path)
    X = csr2vec(csr_path, col_path)
    columns = np.load(col_path, allow_pickle=True)
    print(f"  Feature matrix: {X.shape}")

    num_set, obj_sorted = classify_columns(df_5)
    qual_mask = get_qualitative_feature_names(columns, num_set, obj_sorted)
    print(f"  Qualitative features: {qual_mask.sum()}/{len(columns)}")

    # SHAP computation
    shap_values, X_used = compute_shap_values(model, X, n_samples=n_samples)

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
# 11. Model configurations per target
# =============================================================================
STABILITY_SHAP_CONFIGS = [
    dict(
        model_path="outputs/model/cv5/rf_model_lnTS80m_all_sp2_dummy_zero_FULL.pkl",
        csr_path  ="outputs/csr/all_sp2_dummy_zero_csr.npz",
        col_path  ="outputs/csr/all_sp2_dummy_zero_columns.npy",
        tag           ="lnTS80m_all_sp2_dummy_zero_FULL",
        target_label  ="ln(TS80m)",
        n_samples     =None,
        cbfv_block_label="UNMATCHED",
    ),
    dict(
        model_path="outputs/model/cv5/rf_model_lnTS80m_all_sp3_dummy_zero_FULL.pkl",
        csr_path  ="outputs/csr/all_sp3_dummy_zero_csr.npz",
        col_path  ="outputs/csr/all_sp3_dummy_zero_columns.npy",
        tag           ="lnTS80m_all_sp3_dummy_zero_FULL",
        target_label  ="ln(TS80m)",
        n_samples     =None,
        cbfv_block_label="UNMATCHED",
    ),
    dict(
        model_path="outputs/model/cv5/rf_model_TS80m_all_sp0_oliynyk_zero_FULL.pkl",
        csr_path  ="outputs/csr/all_sp0_oliynyk_zero_csr.npz",
        col_path  ="outputs/csr/all_sp0_oliynyk_zero_columns.npy",
        tag           ="TS80m_all_sp0_oliynyk_zero_FULL",
        target_label  ="TS80m",
        n_samples     =None,
        cbfv_block_label="Perovskite_Oliynyk_features",
    ),
]

PCE_SHAP_CONFIGS = [
    dict(
        model_path="outputs/model/cv5/rf_model_JV_default_PCE_all_sp2_dummy_zero_FULL.pkl",
        csr_path  ="outputs/csr/all_sp2_dummy_zero_csr.npz",
        col_path  ="outputs/csr/all_sp2_dummy_zero_columns.npy",
        tag           ="JV_default_PCE_all_sp2_dummy_zero_FULL",
        target_label  ="PCE (%)",
        n_samples     =None,
        cbfv_block_label="UNMATCHED",
    ),
    dict(
        model_path="outputs/model/cv5/rf_model_JV_default_PCE_all_sp2_oliynyk_zero_FULL.pkl",
        csr_path  ="outputs/csr/all_sp2_oliynyk_zero_csr.npz",
        col_path  ="outputs/csr/all_sp2_oliynyk_zero_columns.npy",
        tag           ="JV_default_PCE_all_sp2_oliynyk_zero_FULL",
        target_label  ="PCE (%)",
        n_samples     =None,
        cbfv_block_label="Perovskite_Oliynyk_features",
    ),
]


# =============================================================================
# 12. Main
# =============================================================================
def main():
    import pipeline_config

    data_path = "outputs/curated/Perovskite_5800data_addln.csv"
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
            model_path      =cfg["model_path"],
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
