"""
Feature Importance Interpretation for Perovskite Stability Models
=================================================================
Date: 2026/02/15

Loads trained FULL models (from reanalysis.py) and computes feature
importances at three levels of aggregation:

  Level 1 — raw expanded features
      e.g. "Perovskite_deposition_thermal_annealing_temperature_100"
      → fti_{model_tag}.csv

  Level 2 — original DB column groups
      e.g. "Perovskite_deposition_thermal_annealing_temperature"
      Numerical: exact match | Categorical: prefix match col_value→col
      → fti_sum_{model_tag}.csv

  Level 3 — layer / process categories  (new)
      e.g. "Perovskite (deposition)", "ETL", "HTL", …
      Prefix-based grouping of Level-2 results; longest prefix wins.
      → fti_layer_{model_tag}.csv

Runs the analysis for a configurable list of (model_tag, csr_columns_tag) pairs.

Input directory: data/20250624_5800/model/cv5/
                 data/20250624_5800/csr/
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
        exclude = ["Original_index", "TS80", "TS80m", "lnTS80m"]

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
# 3. Feature importance extraction
# =============================================================================
def extract_raw_fti(model, columns):
    """Return a sorted pd.Series of feature importances.

    Parameters
    ----------
    model   : fitted RandomForestRegressor
    columns : array-like of expanded feature names

    Returns
    -------
    fti_sort : pd.Series sorted descending
    """
    fti = pd.Series(
        model.feature_importances_,
        index=pd.Index(columns, dtype="object"),
    )
    return fti.sort_values(ascending=False)


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
# 7. Per-model analysis pipeline
# =============================================================================
def analyze_model(
    model_path,
    columns_path,
    df_5,
    out_dir,
    tag,
    cbfv_block_label=None,
    layer_map=None,
):
    """Load a model and run 3-level feature importance analysis.

    Parameters
    ----------
    model_path       : path to .pkl model file
    columns_path     : path to .npy expanded feature names
    df_5             : curated DataFrame for column classification
    out_dir          : output directory
    tag              : string tag used in output filenames
    cbfv_block_label : label for the CBFV feature block (None for dummy runs)
    layer_map        : dict for Level-3 aggregation (defaults to LAYER_MAP)

    Returns
    -------
    fti_sort     : pd.Series  — Level 1 (raw expanded features)
    fti_summary  : pd.DataFrame — Level 2 (DB column groups)
    fti_layer    : pd.DataFrame — Level 3 (layer/process categories)
    """
    print(f"\n  Model : {os.path.basename(model_path)}")

    model = joblib.load(model_path)
    columns = np.load(columns_path, allow_pickle=True)

    # --- Level 1: raw feature importances ---
    fti_sort = extract_raw_fti(model, columns)
    raw_out = os.path.join(out_dir, f"fti_{tag}.csv")
    pd.DataFrame(fti_sort, columns=["feature_importance"]).to_csv(raw_out, index=True)
    print(f"    [Lv1] Raw FTI saved      : {raw_out}  ({len(fti_sort)} features)")

    # --- Level 2: aggregate to DB column groups ---
    num_list, obj_list = classify_columns(df_5)
    print(f"    n_numeric={len(num_list)}, n_categorical={len(obj_list)}")

    fti_summary = aggregate_fti(fti_sort, num_list, obj_list, cbfv_block_label)
    agg_out = os.path.join(out_dir, f"fti_sum_{tag}.csv")
    fti_summary.to_csv(agg_out, index=True)
    print(f"    [Lv2] Column-level FTI   : {agg_out}  ({len(fti_summary)} groups)")

    # Sanity check
    total         = float(fti_sort.sum())
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
    layer_out = os.path.join(out_dir, f"fti_layer_{tag}.csv")
    fti_layer.to_csv(layer_out, index=True)
    print(f"\n    [Lv3] Layer-level FTI    : {layer_out}")
    print(fti_layer.to_string())

    return fti_sort, fti_summary, fti_layer


# =============================================================================
# 8. Main
# =============================================================================
def main():
    base_dir = "data/20250624_5800"
    csr_dir = os.path.join(base_dir, "csr")
    model_dir = os.path.join(base_dir, "model", "cv5")
    data_path = os.path.join(base_dir, "raw", "Perovskite_5800data_addln.csv")

    df_5 = pd.read_csv(data_path)
    print(f"Loaded: {len(df_5)} rows from {data_path}")

    # -------------------------------------------------------------------------
    # Define which models to interpret.
    # Each entry: (model_tag, csr_columns_tag, cbfv_block_label)
    #   cbfv_block_label=None  → remaining features go to "UNMATCHED" (dummy runs)
    #   cbfv_block_label=str   → remaining features go to that label (oliynyk runs)
    # -------------------------------------------------------------------------
    model_configs = [
        # --- lnTS80m, sp3, dummy (primary model) ---
        (
            "lnTS80m_all_sp3_dummy_zero_FULL",
            "all_sp3_dummy_zero",
            None,                          # dummy: no CBFV block
        ),
        # --- lnTS80m, sp2, dummy ---
        (
            "lnTS80m_all_sp2_dummy_zero_FULL",
            "all_sp2_dummy_zero",
            None,
        ),
        # --- TS80m, sp0, oliynyk ---
        (
            "TS80m_all_sp0_oliynyk_zero_FULL",
            "all_sp0_oliynyk_zero",
            "Perovskite_Oliynyk_features",  # oliynyk: remaining → CBFV block
        ),
    ]

    print("\n" + "=" * 60)
    print("Feature importance interpretation")
    print("=" * 60)

    for model_tag, col_tag, cbfv_label in model_configs:
        model_path = os.path.join(model_dir, f"rf_model_{model_tag}.pkl")
        columns_path = os.path.join(csr_dir, f"{col_tag}_columns.npy")

        analyze_model(
            model_path=model_path,
            columns_path=columns_path,
            df_5=df_5,
            out_dir=model_dir,
            tag=model_tag,
            cbfv_block_label=cbfv_label,
            # layer_map=LAYER_MAP,  # default; customise here if needed
        )

    print("\n=== Interpretation complete ===")


if __name__ == "__main__":
    main()
