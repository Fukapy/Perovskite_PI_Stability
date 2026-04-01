"""
XRD PbI2 Ratio — 3D Response Surface Plots
===========================================
Date: 2025/08/18

Reads the per-sample summary CSV (output of 20250818_XRD_regression.py) and:

  1. Fits a 2nd-order polynomial surface to corr_slope_per_day as a function of
     Heat_temperature and Heat_minute, separately for each DMF_ratio condition.

       S = c0 + c1*T + c2*t + c3*T*t + c4*T^2 + c5*t^2

  2. Prints the fitted equation coefficients for each DMF_ratio.

  3. Saves 3D response surface plots (Z < 0 hidden) to the figures/ directory.

Outputs (figures/):
  thinfilm_DMF1.png  — DMF_ratio = 1
  thinfilm_DMF0.png  — DMF_ratio = 0

Input:
  data/XRD/PbI2_ratio_summary_by_sample_20251010.csv
  (or any summary CSV produced by 20250818_XRD_regression.py)
"""

# =============================================================================
# 1. Imports
# =============================================================================
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

warnings.filterwarnings("ignore")

# =============================================================================
# 2. Settings
# =============================================================================
SUMMARY_PATH = Path("data/XRD/PbI2_ratio_summary_by_sample_20251010.csv")
FIGURES_DIR  = Path("figures")

# View angles for 3D plot
ELEV = 30
AZIM = -30

# =============================================================================
# 3. Matplotlib configuration
# =============================================================================
plt.rcParams["font.family"]            = "Arial"
plt.rcParams["xtick.direction"]        = "in"
plt.rcParams["ytick.direction"]        = "in"
plt.rcParams["font.size"]              = 16
plt.rcParams["figure.subplot.bottom"] = 0.2
plt.rcParams["figure.subplot.left"]   = 0.2


# =============================================================================
# 4. Polynomial fitting & equation reporting
# =============================================================================
def get_surface_equation(df: pd.DataFrame, dmf_value: float) -> np.ndarray:
    """Fit a 2nd-order polynomial to corr_slope_per_day(T, t) and print
    the equation coefficients.

    Model:
        S = c0 + c1*T + c2*t + c3*T*t + c4*T^2 + c5*t^2

    Parameters
    ----------
    df        : summary DataFrame (must contain Heat_temperature,
                Heat_minute, corr_slope_per_day, DMF_ratio)
    dmf_value : DMF_ratio value to filter (typically 0 or 1)

    Returns
    -------
    coeff : ndarray of shape (6,) with [c0, c1, c2, c3, c4, c5]
    """
    filtered = df[df["DMF_ratio"] == dmf_value].copy()
    filtered = filtered.dropna(
        subset=["Heat_temperature", "Heat_minute", "corr_slope_per_day"]
    )

    X = filtered["Heat_temperature"].to_numpy()
    Y = filtered["Heat_minute"].to_numpy()
    Z = filtered["corr_slope_per_day"].to_numpy()

    A = np.c_[np.ones_like(X), X, Y, X * Y, X**2, Y**2]
    coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

    print(f"=== DMF_ratio = {dmf_value} ===")
    print(
        f"S = {coeff[0]:.4f}"
        f" + ({coeff[1]:.4f})*T"
        f" + ({coeff[2]:.4f})*t"
        f" + ({coeff[3]:.4f})*T*t"
        f" + ({coeff[4]:.4f})*T^2"
        f" + ({coeff[5]:.4f})*t^2"
    )
    return coeff


# =============================================================================
# 5. 3D response surface plot (Z < 0 masked)
# =============================================================================
def plot_surface_mask_negative(
    df: pd.DataFrame,
    dmf_value: float,
    elev: float = 30,
    azim: float = -30,
    cmap: str = "Spectral_r",
    grid_n: int = 200,
    save_path=None,
) -> None:
    """Draw a 3D response surface for the given DMF_ratio condition.

    Regions where the fitted surface falls below zero are masked (hidden).
    The colour scale is normalised over the positive-Z region only.

    Parameters
    ----------
    df         : summary DataFrame
    dmf_value  : DMF_ratio value to filter (0 or 1)
    elev       : elevation angle for 3D view
    azim       : azimuth angle for 3D view
    cmap       : matplotlib colormap name
    grid_n     : resolution of the surface mesh (grid_n × grid_n)
    save_path  : file path to save the figure (None = display only)
    """
    # --- Filter & extract data ---
    f = df[df["DMF_ratio"] == dmf_value].dropna(
        subset=["Heat_temperature", "Heat_minute", "corr_slope_per_day"]
    )
    X = f["Heat_temperature"].to_numpy()
    Y = f["Heat_minute"].to_numpy()
    Z = f["corr_slope_per_day"].to_numpy()

    # --- 2nd-order polynomial fit ---
    A = np.c_[np.ones_like(X), X, Y, X * Y, X**2, Y**2]
    coef, *_ = np.linalg.lstsq(A, Z, rcond=None)

    # --- High-resolution surface grid ---
    xx, yy = np.meshgrid(
        np.linspace(X.min(), X.max(), grid_n),
        np.linspace(Y.min(), Y.max(), grid_n),
    )
    zz = (
        coef[0]
        + coef[1] * xx
        + coef[2] * yy
        + coef[3] * xx * yy
        + coef[4] * xx**2
        + coef[5] * yy**2
    )

    # --- Mask negative region ---
    zz_plot = zz.copy()
    neg_mask = zz_plot < 0
    zz_plot[neg_mask] = np.nan

    # --- Colour normalisation over positive values only ---
    pos_vals = zz[~neg_mask]
    vmin = float(np.nanmin(pos_vals))
    vmax = float(np.nanmax(pos_vals))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = plt.get_cmap(cmap)

    # Build per-cell RGBA array; transparent where Z < 0
    facecolors = cmap_fn(norm(np.nan_to_num(zz_plot, nan=vmin)))
    facecolors[..., 3] = np.where(np.isnan(zz_plot), 0.0, 0.55)

    # --- Figure ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(right=0.86)

    # Scatter: observed data points
    sc = ax.scatter(
        X, Y, Z,
        c=Z,
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolor="black",
        linewidth=0.7,
    )

    # Surface: fitted polynomial (Z < 0 invisible)
    ax.plot_surface(
        xx, yy, zz_plot,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Heating temperature (°C)")
    ax.set_ylabel("Heating time (min)")
    ax.set_zlabel("corr_slope")
    ax.set_title(f"Response surface (DMF_ratio={dmf_value})  [Z<0 hidden]")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, aspect=12, pad=0.12)
    cbar.set_label("corr slope")

    plt.tight_layout()

    # --- Save ---
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close(fig)


# =============================================================================
# 6. Main
# =============================================================================
def main():
    print(f"Input: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    for col in ["Heat_temperature", "Heat_minute", "DMF_ratio", "corr_slope_per_day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded: {len(df)} rows, "
          f"valid corr_slope: {df['corr_slope_per_day'].notna().sum()}")

    # --- Polynomial equation coefficients ---
    print("\n[Step 1] Fitting 2nd-order polynomial surfaces...")
    get_surface_equation(df, dmf_value=1)
    get_surface_equation(df, dmf_value=0)

    # --- 3D response surface plots ---
    print("\n[Step 2] Generating 3D response surface plots...")
    plot_surface_mask_negative(
        df, dmf_value=1, elev=ELEV, azim=AZIM,
        save_path=FIGURES_DIR / "thinfilm_DMF1.png",
    )
    plot_surface_mask_negative(
        df, dmf_value=0, elev=ELEV, azim=AZIM,
        save_path=FIGURES_DIR / "thinfilm_DMF0.png",
    )

    print("\n=== Response surface analysis complete ===")


if __name__ == "__main__":
    main()
