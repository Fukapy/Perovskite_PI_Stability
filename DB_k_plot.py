"""
Perovskite Database — Apparent Decomposition Rate Constant k (Figure 6d)
=========================================================================
Filters The Perovskite Database to DMF-only, unencapsulated MAPbI3-based
devices, computes the apparent first-order decomposition rate constant k
from TS80m, then plots a 3D response surface of k vs annealing conditions.

Physics
-------
Assuming exponential decay: PCE(t) / PCE(0) = exp(-k * t)

At t = TS80m, PCE / PCE0 = 0.80:
    0.80 = exp(-k * TS80m)
    k = -ln(0.80) / TS80m                      [per hour]
    k_app = k * 24 = -ln(0.80) / TS80m * 24    [per day]

Inputs
------
data/20250624_5800/raw/Perovskite_5800data_addln.csv
  (or the _extract variant — column names are detected automatically)

Outputs (figures/)
------------------
  DB_k_response_surface.png
"""

# =============================================================================
# 1. Imports
# =============================================================================
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# =============================================================================
# 2. Settings — adjust these to tune the plot
# =============================================================================
CSV_PATH    = Path("data/20250624_5800/raw/Perovskite_5800data_addln.csv")
FIGURES_DIR = Path("figures")

# Annealing condition filters
TEMP_MIN    = 55    # °C  (lower bound for anneal temperature)
TEMP_MAX    = 145   # °C  (upper bound)
TIME_MAX    = 55    # min (upper bound for anneal time)

# Outlier filter: drop rows where k_app (per day) exceeds this threshold
# The notebooks explored 1.0, 0.5, 0.05 — use the tightest value for the paper
K_APP_MAX   = 0.05

# Surface mesh resolution
GRID_N      = 140

# 3D view angles
ELEV        = 30
AZIM        = -30

# Colormap
CMAP        = "Spectral_r"

# =============================================================================
# 3. Matplotlib configuration
# =============================================================================
plt.rcParams["font.family"]            = "Arial"
plt.rcParams["xtick.direction"]        = "in"
plt.rcParams["ytick.direction"]        = "in"
plt.rcParams["font.size"]              = 14


# =============================================================================
# 4. Data loading & filtering
# =============================================================================
def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from *candidates* that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_solvents(s) -> set:
    """Parse a solvent field (semicolon/comma/plus/slash separated) to a set."""
    if pd.isna(s):
        return set()
    toks = re.split(r"[;,+/]", str(s).upper())
    out = []
    for t in toks:
        t = t.strip()
        t = t.replace("N,N-DIMETHYLFORMAMIDE", "DMF").replace("DIMETHYLFORMAMIDE", "DMF")
        if t:
            out.append(t)
    return set(out)


def _is_unencapsulated(v) -> bool:
    """Return True if the Encapsulation field indicates no encapsulation."""
    if pd.isna(v):
        return False
    s = str(v).strip().upper()
    if s in {"FALSE", "0", "NO", "N"}:
        return True
    if s in {"TRUE", "1", "YES", "Y"}:
        return False
    return False


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    """Load the database CSV and return a filtered DataFrame with columns:
    anneal_temp_C, anneal_time_min, lnTS80m, TS80m, k_app.
    """
    df_raw = pd.read_csv(csv_path)

    # Detect column names (handles both raw DB names and pre-renamed variants)
    TEMP  = _pick(df_raw, ["Perovskite_deposition_thermal_annealing_temperature", "anneal_temp_C"])
    TIME  = _pick(df_raw, ["Perovskite_deposition_thermal_annealing_time",        "anneal_time_min"])
    SOLV  = _pick(df_raw, ["Perovskite_deposition_solvents"])
    ENCAP = _pick(df_raw, ["Encapsulation", "encapsulation"])
    LNTS  = _pick(df_raw, ["lnTS80m", "ln_ts80m"])

    missing = [name for name, col in
               [("temperature", TEMP), ("time", TIME),
                ("solvents", SOLV), ("encapsulation", ENCAP), ("lnTS80m", LNTS)]
               if col is None]
    if missing:
        raise ValueError(f"Required columns not found: {missing}")

    # Build boolean masks
    solv_sets    = df_raw[SOLV].apply(_parse_solvents)
    mask_dmf     = solv_sets.apply(lambda s: len(s) == 1 and "DMF" in s)
    mask_unenc   = df_raw[ENCAP].apply(_is_unencapsulated)

    df = df_raw.loc[mask_dmf & mask_unenc, [TEMP, TIME, LNTS]].copy()
    df.columns = ["anneal_temp_C", "anneal_time_min", "lnTS80m"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.query(
        "anneal_temp_C >= @TEMP_MIN and anneal_temp_C <= @TEMP_MAX"
        " and anneal_time_min <= @TIME_MAX"
        " and lnTS80m > 0"
    )

    # Compute apparent decomposition rate constant
    # 0.80 = exp(-k * TS80m)  →  k [/h]  →  k_app [/day]
    df["TS80m"]  = np.exp(df["lnTS80m"])
    df["k_app"]  = -np.log(0.8) / df["TS80m"] * 24

    # Outlier removal
    n_before = len(df)
    df = df[df["k_app"] <= K_APP_MAX].copy()
    print(f"  Outlier filter k_app <= {K_APP_MAX}: "
          f"{n_before} → {len(df)} rows "
          f"(removed {n_before - len(df)})")

    return df


# =============================================================================
# 5. Aggregation & polynomial surface fit
# =============================================================================
def aggregate_and_fit(df: pd.DataFrame):
    """Compute median k_app per (temp, time) group and fit a 2nd-order surface.

    Returns
    -------
    agg      : DataFrame with columns anneal_temp_C, anneal_time_min,
               k_app_median, n
    Ti_grid, ti_grid, Zi : meshgrid arrays for the fitted surface
    coef     : polynomial coefficients [c_T, c_t, c_T2, c_t2, c_Tt, c0]
    """
    agg = (
        df.groupby(["anneal_temp_C", "anneal_time_min"], as_index=False)
          .agg(k_app_median=("k_app", "median"), n=("k_app", "size"))
    )

    T = agg["anneal_temp_C"].to_numpy()
    t = agg["anneal_time_min"].to_numpy()
    z = agg["k_app_median"].to_numpy()

    # 2nd-order polynomial: k = c0 + c_T*T + c_t*t + c_T2*T^2 + c_t2*t^2 + c_Tt*T*t
    A    = np.column_stack([T, t, T**2, t**2, T * t, np.ones_like(T)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)

    Ti       = np.linspace(T.min(), T.max(), GRID_N)
    ti       = np.linspace(t.min(), t.max(), GRID_N)
    Ti_grid, ti_grid = np.meshgrid(Ti, ti)
    Zi = (
        coef[0] * Ti_grid
        + coef[1] * ti_grid
        + coef[2] * Ti_grid**2
        + coef[3] * ti_grid**2
        + coef[4] * Ti_grid * ti_grid
        + coef[5]
    )

    print(f"  Unique (temp, time) groups: {len(agg)}")
    print(f"  k_app median range: [{z.min():.4f}, {z.max():.4f}] per day")

    return agg, Ti_grid, ti_grid, Zi, coef


# =============================================================================
# 6. Plotting
# =============================================================================
def plot_k_response_surface(
    agg: pd.DataFrame,
    Ti_grid: np.ndarray,
    ti_grid: np.ndarray,
    Zi: np.ndarray,
    save_path=None,
) -> None:
    """Draw 3D scatter + polynomial response surface coloured by k_app."""
    T = agg["anneal_temp_C"].to_numpy()
    t = agg["anneal_time_min"].to_numpy()
    z = agg["k_app_median"].to_numpy()

    fig = plt.figure(figsize=(7, 5.8))
    ax  = fig.add_subplot(111, projection="3d")

    # Scatter: observed group medians
    sc = ax.scatter(
        T, t, z,
        c=z,
        cmap=CMAP,
        s=34,
        edgecolor="k",
        linewidth=0.3,
        alpha=1.0,
        marker="o",
    )

    # Surface: fitted 2nd-order polynomial
    ax.plot_surface(
        Ti_grid, ti_grid, Zi,
        cmap=CMAP,
        linewidth=0,
        antialiased=True,
        alpha=0.6,
    )

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("Annealing temperature (°C)")
    ax.set_ylabel("Annealing time (min)")
    ax.set_zlabel("Apparent decomposition\nrate constant $k$ (day$^{-1}$)")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, aspect=12, pad=0.12)
    cbar.set_label("$k$ (day$^{-1}$)")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close(fig)


# =============================================================================
# 7. Main
# =============================================================================
def main():
    print(f"Input: {CSV_PATH}")

    print("\n[Step 1] Loading and filtering data...")
    df = load_and_filter(CSV_PATH)
    print(f"  Final dataset: {len(df)} rows")

    print("\n[Step 2] Aggregating and fitting response surface...")
    agg, Ti_grid, ti_grid, Zi, coef = aggregate_and_fit(df)

    print("\n[Step 3] Plotting...")
    plot_k_response_surface(
        agg, Ti_grid, ti_grid, Zi,
        save_path=FIGURES_DIR / "DB_k_response_surface.png",
    )

    print("\n=== DB_k_plot complete ===")
    print(f"Tip: adjust K_APP_MAX (currently {K_APP_MAX}) in Settings to change")
    print("     the outlier threshold.  TEMP_MIN/MAX and TIME_MAX control filtering.")


if __name__ == "__main__":
    main()
