"""
Initial Random Forest Models for Perovskite Stability
======================================================
Simple train/test split (80/20) experiments performed as preliminary analysis
before the full 5-fold CV in 20260214_reanalysis.py.

Three model configurations:
  Model A: ln(TS80m) ~ all features   (n_estimators=270, min_samples_split=3)
  Model B: TS80m     ~ all features   (n_estimators=10,  max_depth=2)
  Model C: ln(TS80m) ~ perovskite composition only  (n_estimators=270)

Prerequisites:
  - data_stab/raw/Perovskite_5800data_addln.csv  (from 20250624_data_curation.py)
  - data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_{csr,columns}.*
    (from 20250624_data_curation.py or 20251115_vectorization.py)

Outputs (data_stab/20250624_5800/model/):
  rf_model_20250806.pkl              — Model A (lnTS80m, all)
  rf_model_20250823_TS80m.pkl        — Model B (TS80m, all)
  rf_model_only_per_lnTS80m.pkl      — Model C (lnTS80m, per-only)
  lnTS80m_all_{Train,Predict}_RF.csv
  TS80m_all_{Train,Predict}_RF.csv
  lnTS80m_per_{Train,Predict}_RF.csv
  fti_new.csv / fti_sum_new.csv      — Feature importance for Model A
  TS80m_fti_20250823.csv             — Feature importance for Model B
  jointplot_*.png                    — Parity plots
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
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# 2. CSR I/O utility
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
# 3. Parity plot helper
# =============================================================================
def _parity_plot(stack, x_col, y_col, hue_col, save_path):
    """Draw a seaborn jointplot parity diagram and save to file."""
    plt.rcParams.update({
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.size": 14,
    })
    fig = sns.jointplot(
        x=stack[x_col], y=stack[y_col], hue=stack[hue_col],
        joint_kws={"alpha": 0.3}, edgecolor="white",
    )
    handles, labels = fig.ax_joint.get_legend_handles_labels()
    fig.ax_joint.legend(handles, labels)
    lim_min = stack[x_col].min() - 2
    lim_max = stack[x_col].max() + 2
    plt.plot([lim_min, lim_max], [lim_min, lim_max], c="black")
    plt.locator_params(axis="x", nbins=5)
    plt.locator_params(axis="y", nbins=5)
    plt.xlim([lim_min, lim_max])
    plt.ylim([lim_min, lim_max])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Parity plot: {save_path}")


# =============================================================================
# 4. Feature importance helper
# =============================================================================
def _save_feature_importance(model, columns_file, df_X, out_fti, out_fti_sum):
    """Compute per-feature and per-column-group importances and save to CSV."""
    columns = np.load(columns_file, allow_pickle=True)
    fti = pd.Series(model.feature_importances_, index=list(columns))
    fti_sort = fti.sort_values(ascending=False)
    pd.DataFrame(fti_sort, columns=["feature_importance"]).to_csv(out_fti, index=True)

    x_col = [c for c in df_X.columns if c != "Perovskite_composition_long_form"]
    fti_agg = [
        sum(fti_sort[i] for i in range(len(fti_sort)) if x_name in fti_sort.index[i])
        for x_name in x_col
    ]
    fti_agg.append(1 - sum(fti_agg))
    fti_summary = pd.DataFrame(
        fti_agg, index=x_col + ["Perovskite_composition_long_form"]
    ).sort_values(by=0, ascending=False)
    fti_summary.to_csv(out_fti_sum, index=True)
    print(f"  Feature importance: {out_fti}")


# =============================================================================
# 5. Model A — ln(TS80m), all features
# =============================================================================
def train_model_A(df_5, X, model_dir):
    """RF for ln(TS80m) with all features.
    Hyperparams: n_estimators=270, min_samples_split=3, random_state=0 (split seed=0)
    """
    print("\n[Model A] ln(TS80m) ~ all features")
    y = df_5["lnTS80m"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rf = RandomForestRegressor(min_samples_split=3, n_estimators=270, random_state=0)
    rf.fit(X_train, y_train)
    y_tr_pred = rf.predict(X_train)
    y_te_pred = rf.predict(X_test)

    print(f"  R2 train={r2_score(y_train, y_tr_pred):.4f}  test={r2_score(y_test, y_te_pred):.4f}")

    joblib.dump(rf, f"{model_dir}/rf_model_20250806.pkl")

    # Save predictions
    train_df = pd.DataFrame({"y_train_real": y_train.values, "y_train_pred": y_tr_pred})
    test_df = pd.DataFrame({"y_test_real": y_test.values, "y_test_pred": y_te_pred})
    train_df.to_csv(f"{model_dir}/lnTS80m_all_Train_RF.csv", index=False)
    test_df.to_csv(f"{model_dir}/lnTS80m_all_Predict_RF.csv", index=False)

    # Parity plot
    tr = pd.read_csv(f"{model_dir}/lnTS80m_all_Train_RF.csv")
    te = pd.read_csv(f"{model_dir}/lnTS80m_all_Predict_RF.csv")
    tr[" "] = "train"; tr.rename(columns={"y_train_real": "Experimental ln(TS80m)", "y_train_pred": "Predicted ln(TS80m)"}, inplace=True)
    te[" "] = "test";  te.rename(columns={"y_test_real":  "Experimental ln(TS80m)", "y_test_pred":  "Predicted ln(TS80m)"}, inplace=True)
    stack = pd.concat([tr, te]).reset_index()
    _parity_plot(stack, "Experimental ln(TS80m)", "Predicted ln(TS80m)", " ",
                 f"{model_dir}/jointplot_lnTS80m_all_RF.png")

    # Feature importance
    df_X = df_5.iloc[:, 1:249]
    _save_feature_importance(
        rf, "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy",
        df_X, f"{model_dir}/fti_new.csv", f"{model_dir}/fti_sum_new.csv",
    )


# =============================================================================
# 6. Model B — TS80m (raw), all features
# =============================================================================
def train_model_B(df_5, X, model_dir):
    """RF for TS80m (raw) with all features.
    Hyperparams: n_estimators=10, max_depth=2, random_state=0 (split seed=42)
    """
    print("\n[Model B] TS80m ~ all features")
    y = df_5["TS80m"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)
    rf.fit(X_train, y_train)
    y_tr_pred = rf.predict(X_train)
    y_te_pred = rf.predict(X_test)

    print(f"  R2 train={r2_score(y_train, y_tr_pred):.4f}  test={r2_score(y_test, y_te_pred):.4f}")

    joblib.dump(rf, f"{model_dir}/rf_model_20250823_TS80m.pkl")

    # Save predictions
    train_df = pd.DataFrame({"y_train_real": y_train.values, "y_train_pred": y_tr_pred})
    test_df  = pd.DataFrame({"y_test_real":  y_test.values,  "y_test_pred":  y_te_pred})
    train_df.to_csv(f"{model_dir}/TS80m_all_Train_RF.csv", index=False)
    test_df.to_csv(f"{model_dir}/TS80m_all_Predict_RF.csv", index=False)

    # Parity plot (linear scale)
    tr = pd.read_csv(f"{model_dir}/TS80m_all_Train_RF.csv")
    te = pd.read_csv(f"{model_dir}/TS80m_all_Predict_RF.csv")
    tr[" "] = "train"; tr.rename(columns={"y_train_real": "Experimental TS80m (h)", "y_train_pred": "Predicted TS80m (h)"}, inplace=True)
    te[" "] = "test";  te.rename(columns={"y_test_real":  "Experimental TS80m (h)", "y_test_pred":  "Predicted TS80m (h)"}, inplace=True)
    stack = pd.concat([tr, te]).reset_index()
    _parity_plot(stack, "Experimental TS80m (h)", "Predicted TS80m (h)", " ",
                 f"{model_dir}/jointplot_TS80m_all_RF.png")

    # Parity plot (log10 scale)
    fig = sns.jointplot(
        x=np.log10(stack["Experimental TS80m (h)"] + 1),
        y=np.log10(stack["Predicted TS80m (h)"] + 1),
        hue=stack[" "], joint_kws={"alpha": 0.3}, edgecolor="white",
    )
    fig.ax_joint.legend()
    x_max = np.log10(stack["Experimental TS80m (h)"].max() + 1)
    plt.plot([0, x_max], [0, x_max], c="black")
    fmt = lambda x, _: r"$10^{{{:.0f}}}$".format(x)
    fig.ax_joint.xaxis.set_major_formatter(FuncFormatter(fmt))
    fig.ax_joint.yaxis.set_major_formatter(FuncFormatter(fmt))
    plt.savefig(f"{model_dir}/log_jointplot_TS80m_all_RF.png", dpi=300)
    plt.close()

    # Feature importance
    df_X = df_5.iloc[:, 1:249]
    _save_feature_importance(
        rf, "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy",
        df_X, f"{model_dir}/TS80m_fti_20250823.csv", f"{model_dir}/TS80m_fti_sum_20250823.csv",
    )


# =============================================================================
# 7. Model C — ln(TS80m), perovskite composition only
# =============================================================================
def train_model_C(df_5, X, model_dir):
    """RF for ln(TS80m) using only perovskite CBFV features (columns 2361–2624).
    Hyperparams: n_estimators=270, min_samples_split=3, random_state=0 (split seed=42)
    """
    print("\n[Model C] ln(TS80m) ~ perovskite composition only")
    y = df_5["lnTS80m"]
    X_per = X.iloc[:, 2361:2361 + 264]
    print(f"  Perovskite block shape: {X_per.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_per, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(min_samples_split=3, n_estimators=270, random_state=0)
    rf.fit(X_train, y_train)
    y_tr_pred = rf.predict(X_train)
    y_te_pred = rf.predict(X_test)

    print(f"  R2 train={r2_score(y_train, y_tr_pred):.4f}  test={r2_score(y_test, y_te_pred):.4f}")

    joblib.dump(rf, f"{model_dir}/rf_model_only_per_lnTS80m.pkl")

    train_df = pd.DataFrame({"y_train_real": y_train.values, "y_train_pred": y_tr_pred})
    test_df  = pd.DataFrame({"y_test_real":  y_test.values,  "y_test_pred":  y_te_pred})
    train_df.to_csv(f"{model_dir}/lnTS80m_per_Train_RF.csv", index=False)
    test_df.to_csv(f"{model_dir}/lnTS80m_per_Predict_RF.csv", index=False)

    tr = pd.read_csv(f"{model_dir}/lnTS80m_per_Train_RF.csv")
    te = pd.read_csv(f"{model_dir}/lnTS80m_per_Predict_RF.csv")
    tr[" "] = "train"; tr.rename(columns={"y_train_real": "Experimental ln(TS80m)", "y_train_pred": "Predicted ln(TS80m)"}, inplace=True)
    te[" "] = "test";  te.rename(columns={"y_test_real":  "Experimental ln(TS80m)", "y_test_pred":  "Predicted ln(TS80m)"}, inplace=True)
    stack = pd.concat([tr, te]).reset_index()
    _parity_plot(stack, "Experimental ln(TS80m)", "Predicted ln(TS80m)", " ",
                 f"{model_dir}/jointplot_lnTS80m_per_RF.png")


# =============================================================================
# 8. Main
# =============================================================================
def main():
    data_path = "data_stab/raw/Perovskite_5800data_addln.csv"
    csr_path  = "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_csr.npz"
    col_path  = "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy"
    model_dir = "data_stab/20250624_5800/model"
    os.makedirs(model_dir, exist_ok=True)

    df_5 = pd.read_csv(data_path)
    print(f"Loaded: {len(df_5)} rows")

    X = csr2vec(csr_path, col_path)
    print(f"Feature matrix: {X.shape}")

    train_model_A(df_5, X, model_dir)
    train_model_B(df_5, X, model_dir)
    train_model_C(df_5, X, model_dir)

    print("\n=== Initial RF models complete ===")
    print("Note: For rigorous evaluation, run 20260214_reanalysis.py (5-fold CV).")


if __name__ == "__main__":
    main()
