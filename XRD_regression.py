"""
XRD PbI2 Ratio — Linear Regression per Sample & Humidity Correction
=====================================================================
Date: 2025/08/18

Reads raw XRD peak intensity data and computes, for each sample:
  - Linear regression of PbI2_ratio vs Storage_day
      → slope_per_day, intercept, r_squared, n_points
  - Humidity-corrected slope
      → corr_slope_per_day = slope_per_day / Storage_RH * 20
  - Experimental conditions (Heat_temperature, Heat_minute, DMF_ratio, Storage_RH)

Also generates panel plots of PbI2_ratio vs Days since synthesis grouped by each
experimental condition (Heat_minute, Heat_temperature, DMF_ratio), with one linear
regression line fitted per condition group.

Outputs (data/XRD/):
  PbI2_ratio_linear_regression_by_sample_{DATE}.csv   — regression results
  Sample_conditions_by_sample_{DATE}.csv              — conditions per sample
  PbI2_ratio_summary_by_sample_{DATE}.csv             — merged summary

Outputs (figures/):
  XRD_regression_Heat_minute.png        — panel (c): grouped by heating time
  XRD_regression_Heat_temperature.png   — panel (d): grouped by temperature
  XRD_regression_DMF_ratio.png          — panel (e): grouped by solvent

Input:
  data/XRD/PbI2_ratio_20241114_successed_3.csv
"""

# =============================================================================
# 1. Imports
# =============================================================================
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# 2. Settings
# =============================================================================
DATA_PATH   = Path("data/XRD/PbI2_ratio_20241114_successed_3.csv")
OUT_DIR     = Path("data/XRD")
FIGURES_DIR = Path("figures")
DATE_TAG    = "20250823"         # output filename suffix

# Humidity correction reference: corr_slope = slope_per_day / Storage_RH * RH_REF
RH_REF = 20.0


# =============================================================================
# 3. Helper: mode or NaN
# =============================================================================
def mode_or_nan(series: pd.Series):
    """Return the most frequent value, or NaN if the series is all-NaN."""
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]


# =============================================================================
# 4. Linear regression per sample
# =============================================================================
def compute_regression(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute linear regression (Storage_day → PbI2_ratio) for each sample.

    Returns a DataFrame with columns:
      Sample No., slope_per_day, intercept, r_squared, n_points,
      Storage_day_min, Storage_day_max
    """
    records = []
    for sample_no, g in df_clean.groupby("Sample No."):
        g = g.sort_values("Storage_day")
        unique_days = np.unique(g["Storage_day"].dropna().values)

        if unique_days.size < 2 or g["PbI2_ratio"].dropna().size < 2:
            records.append({
                "Sample No.":      sample_no,
                "slope_per_day":   np.nan,
                "intercept":       np.nan,
                "r_squared":       np.nan,
                "n_points":        len(g),
                "Storage_day_min": np.nanmin(unique_days) if unique_days.size else np.nan,
                "Storage_day_max": np.nanmax(unique_days) if unique_days.size else np.nan,
            })
            continue

        X = g[["Storage_day"]].values
        y = g["PbI2_ratio"].values
        mask = ~np.isnan(X).ravel() & ~np.isnan(y)
        X, y = X[mask].reshape(-1, 1), y[mask]

        if len(np.unique(X)) >= 2 and len(y) >= 2:
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            try:
                r2 = float(r2_score(y, y_pred))
            except Exception:
                r2 = np.nan
            records.append({
                "Sample No.":      sample_no,
                "slope_per_day":   float(lr.coef_[0]),
                "intercept":       float(lr.intercept_),
                "r_squared":       r2,
                "n_points":        int(len(y)),
                "Storage_day_min": float(np.nanmin(unique_days)),
                "Storage_day_max": float(np.nanmax(unique_days)),
            })
        else:
            records.append({
                "Sample No.":      sample_no,
                "slope_per_day":   np.nan,
                "intercept":       np.nan,
                "r_squared":       np.nan,
                "n_points":        int(len(y)),
                "Storage_day_min": float(np.nanmin(unique_days)) if unique_days.size else np.nan,
                "Storage_day_max": float(np.nanmax(unique_days)) if unique_days.size else np.nan,
            })

    return pd.DataFrame.from_records(records).sort_values("Sample No.").reset_index(drop=True)


# =============================================================================
# 5. Condition table per sample
# =============================================================================
def compute_conditions(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Collect experimental conditions for each sample (mode if multiple values)."""
    rows = []
    for sample_no, g in df_clean.groupby("Sample No."):
        rows.append({
            "Sample No.":      sample_no,
            "Heat_temperature": mode_or_nan(g["Heat_temperature"]),
            "Heat_minute":      mode_or_nan(g["Heat_minute"]),
            "DMF_ratio":        mode_or_nan(g["DMF_ratio"]),
            "Storage_RH":       mode_or_nan(g["Storage_RH"]),
            "Storage_day_list": (
                ",".join(map(str, sorted(g["Storage_day"].dropna().unique().astype(int))))
                if g["Storage_day"].notna().any() else ""
            ),
        })
    return pd.DataFrame(rows).sort_values("Sample No.").reset_index(drop=True)


# =============================================================================
# 6. Humidity correction
# =============================================================================
def apply_humidity_correction(summary_df: pd.DataFrame, rh_ref: float = RH_REF) -> pd.DataFrame:
    """Add corr_slope_per_day column: slope_per_day / Storage_RH * rh_ref."""
    df = summary_df.copy()
    for col in ["slope_per_day", "Storage_RH"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df["slope_per_day"].notna() & df["Storage_RH"].notna() & (df["Storage_RH"] > 0)
    df.loc[valid,  "corr_slope_per_day"] = df.loc[valid, "slope_per_day"] / df.loc[valid, "Storage_RH"] * rh_ref
    df.loc[~valid, "corr_slope_per_day"] = np.nan

    print(f"  Humidity correction: slope_per_day / Storage_RH × {rh_ref}")
    print(f"  Valid samples: {valid.sum()} / {len(df)}")
    return df


# =============================================================================
# 7. Matplotlib configuration
# =============================================================================
plt.rcParams["font.family"]            = "Arial"
plt.rcParams["xtick.direction"]        = "in"
plt.rcParams["ytick.direction"]        = "in"
plt.rcParams["font.size"]              = 14
plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.left"]   = 0.18


# =============================================================================
# 8. Grouped regression plot  (panels c / d / e)
# =============================================================================
def plot_regression_by_condition(
    df_raw: pd.DataFrame,
    group_col: str,
    group_labels: dict = None,
    cmap: str = "Blues",
    colors_override: dict = None,
    xlabel: str = "Days since synthesis",
    ylabel: str = r"$I(\mathrm{PbI_2})\ /\ I(\mathrm{MAPbI_3})$ ratio",
    save_path=None,
) -> None:
    """Plot PbI2_ratio vs Storage_day with one regression line per condition group.

    For each unique value in *group_col*, all raw data points sharing that value
    are pooled, a single linear regression is fitted (Storage_day → PbI2_ratio),
    and the scatter points + dashed regression line are drawn together.

    Parameters
    ----------
    df_raw         : raw long-format DataFrame (one row per measurement)
    group_col      : column to group by, e.g. "Heat_minute", "Heat_temperature",
                     or "DMF_ratio"
    group_labels   : optional dict mapping group value → legend label
                     e.g. {0: "DMSO", 1: "DMF"}
    cmap           : matplotlib colormap name for sequential colouring of groups
                     (ignored when colors_override is supplied)
    colors_override: dict mapping group value → matplotlib colour string
                     e.g. {0: "tab:blue", 1: "tab:orange"}
    xlabel         : x-axis label
    ylabel         : y-axis label
    save_path      : file path to save the figure (None = display only)
    """
    needed = ["Storage_day", "PbI2_ratio", group_col]
    df = df_raw.dropna(subset=needed).copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)

    unique_vals = sorted(df[group_col].unique())
    n = len(unique_vals)
    if n == 0:
        print(f"  No valid data for group_col='{group_col}'")
        return

    # --- Colour assignment ---
    if colors_override:
        color_map = colors_override
    else:
        cmap_fn = plt.get_cmap(cmap)
        # spread colours across the usable range of the colormap (avoid very pale)
        color_map = {v: cmap_fn(0.35 + 0.6 * i / max(n - 1, 1))
                     for i, v in enumerate(unique_vals)}

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    x_global_max = float(df["Storage_day"].max())

    for val in unique_vals:
        grp = df[df[group_col] == val]
        x_arr = grp["Storage_day"].to_numpy().reshape(-1, 1)
        y_arr = grp["PbI2_ratio"].to_numpy()

        col = color_map.get(val, "gray")
        label = group_labels.get(val, str(val)) if group_labels else str(val)

        # Scatter
        ax.scatter(x_arr, y_arr, color=col, s=40, edgecolor="black",
                   linewidth=0.5, zorder=3, label=label)

        # Regression line (need at least 2 unique x values)
        if len(np.unique(x_arr)) >= 2:
            lr = LinearRegression()
            lr.fit(x_arr, y_arr)
            x_line = np.array([[0.0], [x_global_max]])
            y_line = lr.predict(x_line)
            ax.plot(x_line.ravel(), y_line, color=col,
                    linestyle="--", linewidth=1.4, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=10, frameon=True, loc="upper left",
              title=group_col.replace("_", " "))
    ax.set_xlim(left=0)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close(fig)


# =============================================================================
# 9. Main
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    required = ["Sample No.", "Storage_day", "PbI2_ratio",
                 "Heat_temperature", "Heat_minute", "DMF_ratio", "Storage_RH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns not found: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_clean = df.dropna(subset=["Sample No.", "Storage_day", "PbI2_ratio"]).copy()
    print(f"Loaded: {len(df)} rows → after dropna: {len(df_clean)} rows")
    print(f"Unique samples: {df_clean['Sample No.'].nunique()}")

    # --- Regression ---
    print("\n[Step 1] Linear regression per sample...")
    regression_df = compute_regression(df_clean)

    # --- Conditions ---
    print("[Step 2] Collecting conditions per sample...")
    conditions_df = compute_conditions(df_clean)

    # --- Merge ---
    summary_df = regression_df.merge(conditions_df, on="Sample No.", how="left")

    # --- Humidity correction ---
    print("[Step 3] Applying humidity correction...")
    summary_df = apply_humidity_correction(summary_df)

    # Reorder columns for readability
    col_order = [
        "Sample No.",
        "slope_per_day", "intercept", "r_squared", "corr_slope_per_day",
        "n_points", "Storage_day_min", "Storage_day_max",
        "Heat_temperature", "Heat_minute", "DMF_ratio",
        "Storage_RH", "Storage_day_list",
    ]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    # --- Save ---
    out_reg  = OUT_DIR / f"PbI2_ratio_linear_regression_by_sample_{DATE_TAG}.csv"
    out_cond = OUT_DIR / f"Sample_conditions_by_sample_{DATE_TAG}.csv"
    out_sum  = OUT_DIR / f"PbI2_ratio_summary_by_sample_{DATE_TAG}.csv"

    regression_df.to_csv(out_reg,  index=False)
    conditions_df.to_csv(out_cond, index=False)
    summary_df.to_csv(out_sum,     index=False)

    print(f"\nSaved:")
    print(f"  {out_reg}")
    print(f"  {out_cond}")
    print(f"  {out_sum}")
    print(f"\nSummary statistics (slope_per_day):")
    print(summary_df["slope_per_day"].describe().to_string())
    print(f"\nSummary statistics (corr_slope_per_day):")
    print(summary_df["corr_slope_per_day"].describe().to_string())

    # --- Plots: panels (c) / (d) / (e) ---
    print("\n[Step 4] Generating grouped regression plots...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # (c) Grouped by heating time (Heat_minute)
    plot_regression_by_condition(
        df_clean,
        group_col="Heat_minute",
        cmap="Blues",
        save_path=FIGURES_DIR / "XRD_regression_Heat_minute.png",
    )

    # (d) Grouped by heating temperature (Heat_temperature)
    plot_regression_by_condition(
        df_clean,
        group_col="Heat_temperature",
        cmap="Purples",
        save_path=FIGURES_DIR / "XRD_regression_Heat_temperature.png",
    )

    # (e) Grouped by solvent (DMF_ratio: 0=DMSO, 1=DMF)
    plot_regression_by_condition(
        df_clean,
        group_col="DMF_ratio",
        group_labels={0: "DMSO", 1: "DMF"},
        colors_override={0: "tab:blue", 1: "tab:orange"},
        save_path=FIGURES_DIR / "XRD_regression_DMF_ratio.png",
    )

    print("\n=== Regression complete ===")
    print("Next step: run 20250818_XRD_rf_shap.py")


if __name__ == "__main__":
    main()
