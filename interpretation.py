"""
Feature Importance Interpretation for Perovskite Stability Models
=================================================================
Computes feature importances as the **mean across 5-fold CV models**
(not from a single FULL model).  For each configuration, the five
fold models (rf_model_{tag}_fold1.pkl … fold5.pkl) are loaded, their
Gini importances are extracted, and the per-feature mean ± std are
reported.

Three levels of aggregation:

  Level 1 — raw expanded features
      → fti_{tag}_cv5mean.csv

  Level 2 — original DB column groups
      → fti_sum_{tag}_cv5mean.csv

  Level 3 — layer / process categories
      → fti_layer_{tag}_cv5mean.csv

Input directory: outputs/model/cv5/
                 outputs/csr/
"""

# =============================================================================
# 1. Imports
# =============================================================================
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 2. Column classification utility
# =============================================================================
def classify_columns(df_5, exclude=None):
    """Separate numerical and categorical columns in df_5.

    Parameters
    ----------
    df_5    : curated DataFrame (Perovskite_5800data_addln.csv)
    exclude : list of column names to ignore (default: index and target columns)

    Returns
    -------
    num_list : pd.Index of numerical column names
    obj_list : pd.Index of categorical column names
    """
    if exclude is None:
        exclude = ["Original_index", "TS80", "TS80m", "lnTS80m", "JV_default_PCE"]

    num_cols = []
    obj_cols = []
    for col in df_5.columns:
        if col in exclude:
            continue
        if pd.api.types.is_bool_dtype(df_5[col]):
            obj_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df_5[col]):
            num_cols.append(col)
        else:
            obj_cols.append(col)

    return pd.Index(num_cols), pd.Index(obj_cols)


# =============================================================================
# 3. Feature importance extraction (5-fold CV mean)
# =============================================================================
N_FOLDS = 5


def extract_cv_mean_fti(model_paths, columns):
    """Compute the mean and std of Gini feature importances across CV folds.

    Parameters
    ----------
    model_paths : list of str — paths to fold model .pkl files
    columns     : array-like of expanded feature names

    Returns
    -------
    fti_mean : pd.Series — mean importance per feature (sorted descending)
    fti_std  : pd.Series — std of importance per feature (same order)
    """
    col_index = pd.Index(columns, dtype="object")
    fti_matrix = np.zeros((len(model_paths), len(columns)))

    for i, path in enumerate(model_paths):
        model = joblib.load(path)
        fti_matrix[i, :] = model.feature_importances_

    fti_mean = pd.Series(fti_matrix.mean(axis=0), index=col_index)
    fti_std  = pd.Series(fti_matrix.std(axis=0, ddof=1), index=col_index)

    # Sort by mean descending
    order = fti_mean.sort_values(ascending=False).index
    return fti_mean.reindex(order), fti_std.reindex(order)


# =============================================================================
# 4. Layer / process category map  (Level 3 aggregation)
# =============================================================================
# Maps DB column name prefix → layer/process label.
# Checked longest-prefix-first to avoid partial matches
# (e.g. "Perovskite_composition" is matched before "Perovskite").
#
# Adjust this dict to match your paper's layer grouping convention.
LAYER_MAP = {
    # ---- Perovskite layer (split by composition vs. deposition) ----
    "Perovskite_composition"     : "Perovskite (composition)",
    "Perovskite_deposition"      : "Perovskite (deposition)",
    "Perovskite"                 : "Perovskite (other)",
    # ---- Adjacent layers ----
    "ETL"                        : "Electron transport layer",
    "HTL"                        : "Hole transport layer",
    "Backcontact"                : "Back contact",
    "Substrate"                  : "Substrate",
    # ---- Device-level ----
    "Cell"                       : "Cell architecture",
    "Encapsulation"              : "Encapsulation",
    # ---- Measurement / test conditions ----
    "Stability"                  : "Stability test conditions",
    "JV"                         : "JV characteristics",
}


# =============================================================================
# 5. Level 3 aggregation: DB columns → layer/process categories
# =============================================================================
def aggregate_fti_by_layer(fti_summary, layer_map=None):
    """Aggregate parameter-level importances into layer/process categories.

    Takes the output of aggregate_fti() (Level 2) and further groups it by
    the layer or process category each DB column belongs to.

    Matching uses the longest-prefix-first rule:
      "Perovskite_composition_*" → "Perovskite (composition)"
      "Perovskite_deposition_*"  → "Perovskite (deposition)"
      "Perovskite_*"             → "Perovskite (other)"
      (and so on for ETL, HTL, Backcontact, …)

    Parameters
    ----------
    fti_summary : pd.DataFrame  (output of aggregate_fti)
                  index = DB column name or CBFV block label
                  column = 'feature_importance_sum'
    layer_map   : dict {prefix: label} — defaults to LAYER_MAP above

    Returns
    -------
    pd.DataFrame sorted descending, column = 'feature_importance_sum'
    """
    if layer_map is None:
        layer_map = LAYER_MAP

    # Sort prefixes longest-first to avoid partial shadowing
    sorted_prefixes = sorted(layer_map.keys(), key=len, reverse=True)

    layer_sum = {}
    for param_name, row in fti_summary.iterrows():
        importance = float(row["feature_importance_sum"])
        assigned = False
        for prefix in sorted_prefixes:
            if str(param_name).startswith(prefix):
                label = layer_map[prefix]
                layer_sum[label] = layer_sum.get(label, 0.0) + importance
                assigned = True
                break
        if not assigned:
            # Covers CBFV block label and any truly unmatched columns
            layer_sum["Other / CBFV"] = layer_sum.get("Other / CBFV", 0.0) + importance

    return (
        pd.Series(layer_sum, name="feature_importance_sum")
        .sort_values(ascending=False)
        .to_frame()
    )


# =============================================================================
# 6. Feature importance aggregation  (Level 2: expanded → DB column)
# =============================================================================
def aggregate_fti(fti, num_list, obj_list, cbfv_block_label=None):
    """Aggregate expanded feature importances back to original DB column groups.

    Strategy:
      1. Numerical columns — exact match on feature name
      2. Categorical columns — prefix match: feature starts with "col_"
         (sorted by length descending to avoid partial shadowing)
      3. Remaining features:
         - If cbfv_block_label is given → assigned to that label (e.g. Oliynyk block)
         - Otherwise → assigned to "UNMATCHED"

    Parameters
    ----------
    fti              : pd.Series of feature importances (expanded feature names)
    num_list         : pd.Index of numerical column names
    obj_list         : pd.Index of categorical column names
    cbfv_block_label : str or None — label for unmatched CBFV features

    Returns
    -------
    fti_summary : pd.DataFrame sorted descending, column = 'feature_importance_sum'
    """
    num_set = set(num_list.astype(str))
    obj_sorted = sorted(obj_list.astype(str), key=len, reverse=True)

    group_sum = {c: 0.0 for c in list(num_set) + obj_sorted}
    if cbfv_block_label:
        group_sum[cbfv_block_label] = 0.0
    group_sum["UNMATCHED"] = 0.0

    for fn in fti.index.astype(str).to_list():
        # Case 1: numerical (exact match)
        if fn in num_set:
            group_sum[fn] += float(fti.loc[fn])
            continue

        # Case 2: categorical (prefix match)
        assigned = False
        for col in obj_sorted:
            if fn.startswith(col + "_"):
                group_sum[col] += float(fti.loc[fn])
                assigned = True
                break

        if assigned:
            continue

        # Case 3: CBFV block or unmatched
        if cbfv_block_label:
            group_sum[cbfv_block_label] += float(fti.loc[fn])
        else:
            group_sum["UNMATCHED"] += float(fti.loc[fn])

    fti_summary = (
        pd.Series(group_sum, name="feature_importance_sum")
        .sort_values(ascending=False)
        .to_frame()
    )
    return fti_summary


# =============================================================================
# 7. Per-configuration analysis pipeline (5-fold CV mean)
# =============================================================================
def analyze_cv_models(
    model_dir,
    base_tag,
    columns_path,
    df_5,
    out_dir,
    cbfv_block_label=None,
    layer_map=None,
    n_folds=N_FOLDS,
):
    """Load 5-fold CV models and run 3-level feature importance analysis
    using the mean importance across folds.

    Parameters
    ----------
    model_dir        : directory containing rf_model_*_fold{k}.pkl
    base_tag         : tag WITHOUT _FULL suffix
                       e.g. "lnTS80m_all_sp2_dummy_zero"
    columns_path     : path to .npy expanded feature names
    df_5             : curated DataFrame for column classification
    out_dir          : output directory for CSVs
    cbfv_block_label : label for the CBFV feature block (None for dummy runs)
    layer_map        : dict for Level-3 aggregation (defaults to LAYER_MAP)

    Returns
    -------
    fti_mean    : pd.Series  — Level 1 (mean across folds)
    fti_summary : pd.DataFrame — Level 2 (DB column groups)
    fti_layer   : pd.DataFrame — Level 3 (layer/process categories)
    """
    # Locate fold models
    model_paths = []
    for k in range(1, n_folds + 1):
        p = os.path.join(model_dir, f"rf_model_{base_tag}_fold{k}.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fold model not found: {p}")
        model_paths.append(p)

    print(f"\n  Config: {base_tag}  ({n_folds} folds)")
    columns = np.load(columns_path, allow_pickle=True)

    # --- Level 1: mean feature importances across folds ---
    fti_mean, fti_std = extract_cv_mean_fti(model_paths, columns)

    out_tag = f"{base_tag}_cv5mean"
    raw_out = os.path.join(out_dir, f"fti_{out_tag}.csv")
    pd.DataFrame({
        "feature_importance_mean": fti_mean,
        "feature_importance_std":  fti_std,
    }).to_csv(raw_out, index=True)
    print(f"    [Lv1] CV-mean FTI saved  : {raw_out}  ({len(fti_mean)} features)")

    # --- Level 2: aggregate to DB column groups ---
    num_list, obj_list = classify_columns(df_5)
    print(f"    n_numeric={len(num_list)}, n_categorical={len(obj_list)}")

    fti_summary = aggregate_fti(fti_mean, num_list, obj_list, cbfv_block_label)
    agg_out = os.path.join(out_dir, f"fti_sum_{out_tag}.csv")
    fti_summary.to_csv(agg_out, index=True)
    print(f"    [Lv2] Column-level FTI   : {agg_out}  ({len(fti_summary)} groups)")

    # Sanity check
    total         = float(fti_mean.sum())
    grouped_total = float(fti_summary["feature_importance_sum"].sum())
    unmatched     = float(
        fti_summary.loc["UNMATCHED", "feature_importance_sum"]
        if "UNMATCHED" in fti_summary.index else 0.0
    )
    print(f"    Total={total:.4f}  Grouped={grouped_total:.4f}  Unmatched={unmatched:.4f}")
    print("    Top 10 (column level):")
    print(fti_summary.head(10).to_string())

    # --- Level 3: aggregate to layer/process categories ---
    fti_layer = aggregate_fti_by_layer(fti_summary, layer_map=layer_map)
    layer_out = os.path.join(out_dir, f"fti_layer_{out_tag}.csv")
    fti_layer.to_csv(layer_out, index=True)
    print(f"\n    [Lv3] Layer-level FTI    : {layer_out}")
    print(fti_layer.to_string())

    return fti_mean, fti_summary, fti_layer


# =============================================================================
# 8. Main
# =============================================================================
# =============================================================================
# Model configurations per target
# =============================================================================
# Each entry: (base_tag, csr_columns_tag, cbfv_block_label)
#   base_tag is the model tag WITHOUT "_FULL" — fold files are named
#   rf_model_{base_tag}_fold{k}.pkl
#   cbfv_block_label=None  → remaining features go to "UNMATCHED" (dummy runs)
#   cbfv_block_label=str   → remaining features go to that label (oliynyk runs)

STABILITY_CONFIGS = [
    # --- lnTS80m, sp2, dummy (primary model — best CV R²) ---
    ("lnTS80m_all_sp2_dummy_zero", "all_sp2_dummy_zero", None),
    # --- lnTS80m, sp3, dummy ---
    ("lnTS80m_all_sp3_dummy_zero", "all_sp3_dummy_zero", None),
    # --- TS80m, sp0, oliynyk ---
    ("TS80m_all_sp0_oliynyk_zero", "all_sp0_oliynyk_zero",
     "Perovskite_Oliynyk_features"),
]

PCE_CONFIGS = [
    # --- PCE, sp2, dummy ---
    ("JV_default_PCE_all_sp2_dummy_zero", "all_sp2_dummy_zero", None),
    # --- PCE, sp2, oliynyk ---
    ("JV_default_PCE_all_sp2_oliynyk_zero", "all_sp2_oliynyk_zero",
     "Perovskite_Oliynyk_features"),
]


def main():
    import pipeline_config

    base_dir = "outputs"
    csr_dir = os.path.join(base_dir, "csr")
    model_dir = os.path.join(base_dir, "model", "cv5")
    data_path = os.path.join(base_dir, "curated", "Perovskite_5800data_addln.csv")

    df_5 = pd.read_csv(data_path)
    print(f"Loaded: {len(df_5)} rows from {data_path}")

    # Select model configurations based on target
    if pipeline_config.TARGET_MODE == "PCE":
        model_configs = PCE_CONFIGS
    else:
        model_configs = STABILITY_CONFIGS

    print("\n" + "=" * 60)
    print(f"Feature importance interpretation  (target: {pipeline_config.TARGET_MODE})")
    print(f"Method: mean across {N_FOLDS}-fold CV models")
    print("=" * 60)

    for base_tag, col_tag, cbfv_label in model_configs:
        columns_path = os.path.join(csr_dir, f"{col_tag}_columns.npy")

        analyze_cv_models(
            model_dir=model_dir,
            base_tag=base_tag,
            columns_path=columns_path,
            df_5=df_5,
            out_dir=model_dir,
            cbfv_block_label=cbfv_label,
        )

    print("\n=== Interpretation complete ===")


if __name__ == "__main__":
    main()
