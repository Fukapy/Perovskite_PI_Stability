"""
XRD PbI2 Ratio — Random Forest Regression & SHAP Analysis
==========================================================
Date: 2025/08/18

Reads the per-sample summary CSV (output of 20250818_XRD_regression.py) and:

  1. Trains a RandomForestRegressor to predict corr_slope_per_day from
     Heat_temperature, Heat_minute, DMF_ratio  (features standardised)

  2. Evaluates model quality:
       - In-sample R²
       - Leave-One-Out (LOO) cross-validation R²

  3. Computes feature importances:
       - Gini importance
       - Permutation importance (30 repeats)

  4. SHAP analysis via TreeExplainer:
       - Summary bar chart
       - Beeswarm plot
       - Dependence plot per feature

Outputs (data/XRD/):
  rf_feature_importances_all.csv / .png
  rf_permutation_importances_all.csv / .png
  shap_summary_bar_all.png
  shap_beeswarm_all.png
  shap_dependence_std_{feature}.png

Input:
  data/XRD/PbI2_ratio_summary_by_sample_{DATE_TAG}.csv
  (or set SUMMARY_PATH directly)
"""

# =============================================================================
# 1. Imports
# =============================================================================
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 2. Settings
# =============================================================================
SUMMARY_PATH = Path("data/XRD/PbI2_ratio_summary_by_sample_20251010.csv")
OUT_DIR      = Path("data/XRD")

FEAT_NAMES   = ["Heat_temperature", "Heat_minute", "DMF_ratio"]
TARGET_COL   = "corr_slope_per_day"

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

PERM_N_REPEATS = 30
PERM_SEED      = 42

# =============================================================================
# 3. Matplotlib configuration
# =============================================================================
plt.rcParams["font.family"]              = "Arial"
plt.rcParams["xtick.direction"]          = "in"
plt.rcParams["ytick.direction"]          = "in"
plt.rcParams["font.size"]                = 14
plt.rcParams["figure.subplot.bottom"]   = 0.2
plt.rcParams["figure.subplot.left"]     = 0.2


# =============================================================================
# 4. Data loading & preprocessing
# =============================================================================
def load_data(summary_path: Path, feat_names: list, target_col: str):
    """Load summary CSV, apply humidity correction if needed, return X_raw, y, work."""
    df = pd.read_csv(summary_path)

    required = feat_names + [target_col]
    for col in required + ["slope_per_day", "Storage_RH"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If corr_slope_per_day is not present, recompute from slope_per_day / Storage_RH * 20
    if target_col not in df.columns or df[target_col].isna().all():
        print(f"  '{target_col}' not found; computing from slope_per_day / Storage_RH * 20")
        mask = df["slope_per_day"].notna() & df["Storage_RH"].notna() & (df["Storage_RH"] > 0)
        df.loc[mask,  target_col] = df.loc[mask, "slope_per_day"] / df.loc[mask, "Storage_RH"] * 20.0
        df.loc[~mask, target_col] = np.nan

    # Drop rows where any required column is missing
    valid_mask = df[required].notna().all(axis=1)
    work = df.loc[valid_mask].copy().reset_index(drop=True)
    print(f"  Loaded: {len(df)} rows → valid for analysis: {len(work)}")

    X_raw = work[feat_names].copy()
    y = work[target_col].values
    return X_raw, y, work


# =============================================================================
# 5. Model training
# =============================================================================
def train_random_forest(X_scaled: np.ndarray, y: np.ndarray,
                        rf_params: dict) -> RandomForestRegressor:
    """Fit RandomForestRegressor and report in-sample and LOO R²."""
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_scaled, y)

    y_pred_train = rf.predict(X_scaled)
    r2_train = r2_score(y, y_pred_train)
    print(f"  In-sample R²  : {r2_train:.4f}")

    loo_scores = cross_val_score(rf, X_scaled, y, cv=LeaveOneOut(), scoring="r2")
    r2_loo = loo_scores.mean()
    print(f"  LOO R² (mean) : {r2_loo:.4f}  (std={loo_scores.std():.4f})")

    return rf, r2_train, r2_loo


# =============================================================================
# 6. Feature importance (Gini + Permutation)
# =============================================================================
def compute_feature_importance(rf, X_scaled, y, feat_names, n_repeats, seed):
    """Return Gini importance DataFrame and permutation importance DataFrame."""
    # Gini
    fi_df = (
        pd.DataFrame({"feature": feat_names, "gini_importance": rf.feature_importances_})
        .sort_values("gini_importance", ascending=False)
        .reset_index(drop=True)
    )

    # Permutation
    perm = permutation_importance(rf, X_scaled, y, n_repeats=n_repeats,
                                  random_state=seed, n_jobs=-1)
    perm_df = (
        pd.DataFrame({
            "feature": feat_names,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std":  perm.importances_std,
        })
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return fi_df, perm_df


def plot_feature_importance(fi_df, perm_df, out_dir):
    """Save Gini and Permutation importance bar charts."""
    plt.figure(figsize=(6, 4))
    plt.bar(fi_df["feature"], fi_df["gini_importance"])
    plt.title("RandomForest Feature Importances (Gini)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_feature_importances_all.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(perm_df["feature"], perm_df["perm_importance_mean"],
            yerr=perm_df["perm_importance_std"], capsize=4)
    plt.title(f"Permutation Importances ({PERM_N_REPEATS} repeats)")
    plt.xlabel("Feature")
    plt.ylabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_permutation_importances_all.png", dpi=300)
    plt.close()
    print(f"  Feature importance plots saved to {out_dir}")


# =============================================================================
# 7. SHAP analysis
# =============================================================================
def run_shap_analysis(rf, X_scaled, feat_names, out_dir):
    """Compute SHAP values and save summary bar, beeswarm, and dependence plots."""
    std_names = [f"std_{c}" for c in feat_names]
    X_df = pd.DataFrame(X_scaled, columns=std_names)

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_df)

    # Summary bar
    plt.figure()
    shap.summary_plot(shap_values, features=X_df, feature_names=std_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar_all.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_values, features=X_df, feature_names=std_names,
                      show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm_all.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Dependence plots
    for col in std_names:
        plt.figure()
        try:
            shap.dependence_plot(col, shap_values, X_df, show=False)
            plt.tight_layout()
            plt.savefig(out_dir / f"shap_dependence_{col}.png", dpi=200)
        except Exception as e:
            print(f"  Dependence plot skipped for {col}: {e}")
        plt.close()

    print(f"  SHAP plots saved to {out_dir}")


# =============================================================================
# 8. Main
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Input: {SUMMARY_PATH}")

    # Load
    X_raw, y, work = load_data(SUMMARY_PATH, FEAT_NAMES, TARGET_COL)
    print(f"\nTarget ({TARGET_COL}) stats:")
    print(pd.Series(y, name=TARGET_COL).describe().to_string())

    # Standardise
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Train
    print("\n[Step 1] Training RandomForestRegressor...")
    rf, r2_train, r2_loo = train_random_forest(X_scaled, y, RF_PARAMS)

    # Feature importance
    print("\n[Step 2] Computing feature importances...")
    fi_df, perm_df = compute_feature_importance(
        rf, X_scaled, y, FEAT_NAMES, PERM_N_REPEATS, PERM_SEED
    )
    print("\nGini importance:\n", fi_df.to_string(index=False))
    print("\nPermutation importance:\n", perm_df.to_string(index=False))
    plot_feature_importance(fi_df, perm_df, OUT_DIR)

    # SHAP
    print("\n[Step 3] Running SHAP analysis...")
    run_shap_analysis(rf, X_scaled, FEAT_NAMES, OUT_DIR)

    # Save CSVs
    fi_df.to_csv(OUT_DIR / "rf_feature_importances_all.csv", index=False)
    perm_df.to_csv(OUT_DIR / "rf_permutation_importances_all.csv", index=False)
    pd.DataFrame({
        "metric": ["in_sample_R2", "LOO_R2"],
        "value":  [r2_train, r2_loo],
    }).to_csv(OUT_DIR / "rf_model_scores.csv", index=False)

    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")
    print("\n=== RF + SHAP analysis complete ===")
    print("Next step: run 20250818_XRD_response_surface.py")


if __name__ == "__main__":
    main()
