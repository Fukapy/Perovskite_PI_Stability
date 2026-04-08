"""
5-Fold Cross-Validation Regression for Perovskite Stability
============================================================
Date: 2026/02/14

Runs 5-fold CV with RandomForestRegressor across all combinations of:
  - Target       : ln(TS80m)  [default]
                   TS80m      [optional — set RUN_TS80M = True below]
  - CBFV scheme  : oliynyk, dummy
  - Categorical  : sp0 (one-hot), sp1–sp3 (multi-hot variants)

Also runs CV for perovskite-only representations (Oliynyk, Magpie, Mat2Vec,
dummy one-hot).

For each run:
  - Saves per-fold metrics  (cv5_metrics_{tag}.csv)
  - Saves fold mean/std     (cv5_summary_{tag}.csv)
  - Saves OOF predictions   (cv5_oof_{tag}.csv)
  - Saves each fold model   (rf_model_{tag}_fold{n}.pkl)
  - Trains a FULL-data model (rf_model_{tag}_FULL.pkl)

Collates all summary results into a single comparison table:
  outputs/model/cv5/CV_summary_clean_table.csv

Input files expected in: outputs/csr/
Output written to:       outputs/model/cv5/
"""

# =============================================================================
# 1. Imports
# =============================================================================
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.sparse import load_npz

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
# 3. Core utilities
# =============================================================================
def ensure_1d_y(y):
    """Convert y to a 1D numpy array."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            return y.iloc[:, 0].to_numpy()
        raise ValueError("y DataFrame must contain exactly one column.")
    if isinstance(y, pd.Series):
        return y.to_numpy()
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    if y.ndim != 1:
        raise ValueError("y must be 1D or (n, 1).")
    return y


def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return mean_squared_error(y_true, y_pred) ** 0.5


def row_slice(X, idx):
    """Slice rows of X by integer index array.
    Works for pandas DataFrame/Series, numpy array, and scipy sparse matrices.
    """
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


# =============================================================================
# 4. 5-Fold CV runner
# =============================================================================
def run_rf_5fold_cv(
    X,
    y,
    out_dir,
    tag,
    n_splits=5,
    kfold_seed=42,
    rf_seed=0,
    rf_params=None,
    save_models=True,
):
    """Perform 5-fold cross validation using RandomForestRegressor.

    For each fold:
      - Trains RF, records R2/MAE/RMSE for train and test
      - Stores OOF predictions
      - Optionally saves the fold model

    After all folds:
      - Saves per-fold metrics  → cv5_metrics_{tag}.csv
      - Saves mean/std summary  → cv5_summary_{tag}.csv
      - Saves OOF predictions   → cv5_oof_{tag}.csv
      - Trains a FULL-data model → rf_model_{tag}_FULL.pkl

    Returns (metrics_df, summary_df, oof_df)
    """
    os.makedirs(out_dir, exist_ok=True)

    y_vec = ensure_1d_y(y)
    if rf_params is None:
        rf_params = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=kfold_seed)

    fold_metrics = []
    oof_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        kf.split(np.arange(len(y_vec))), start=1
    ):
        X_train = row_slice(X, train_idx)
        X_test = row_slice(X, test_idx)
        y_train = y_vec[train_idx]
        y_test = y_vec[test_idx]

        rf = RandomForestRegressor(random_state=rf_seed, **rf_params)
        rf.fit(X_train, y_train)

        y_tr_pred = rf.predict(X_train)
        y_te_pred = rf.predict(X_test)

        fold_metrics.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "R2_train":   r2_score(y_train, y_tr_pred),
            "R2_test":    r2_score(y_test, y_te_pred),
            "MAE_train":  mean_absolute_error(y_train, y_tr_pred),
            "MAE_test":   mean_absolute_error(y_test, y_te_pred),
            "RMSE_train": rmse(y_train, y_tr_pred),
            "RMSE_test":  rmse(y_test, y_te_pred),
        })

        oof_rows.append(pd.DataFrame({
            "index": test_idx,
            "fold": fold,
            "y_true": y_test,
            "y_pred": y_te_pred,
        }))

        if save_models:
            joblib.dump(rf, os.path.join(out_dir, f"rf_model_{tag}_fold{fold}.pkl"))

    metrics_df = pd.DataFrame(fold_metrics)

    mean_row = metrics_df.drop(columns=["fold"]).mean(numeric_only=True)
    std_row = metrics_df.drop(columns=["fold"]).std(numeric_only=True, ddof=1)
    summary_df = (
        pd.DataFrame([mean_row, std_row], index=["mean", "std"])
        .reset_index()
        .rename(columns={"index": "stat"})
    )

    metrics_df.to_csv(os.path.join(out_dir, f"cv5_metrics_{tag}.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, f"cv5_summary_{tag}.csv"), index=False)

    oof_df = (
        pd.concat(oof_rows, axis=0)
        .sort_values("index")
        .reset_index(drop=True)
    )
    oof_df.to_csv(os.path.join(out_dir, f"cv5_oof_{tag}.csv"), index=False)

    # Full-data model
    rf_full = RandomForestRegressor(random_state=rf_seed, **rf_params)
    rf_full.fit(X, y_vec)
    joblib.dump(rf_full, os.path.join(out_dir, f"rf_model_{tag}_FULL.pkl"))

    return metrics_df, summary_df, oof_df


# =============================================================================
# 5. Perovskite-only 5-fold CV (CBFV comparison)
# =============================================================================
def run_cv_perovskite_only(df_5, base_dir):
    """Run 5-fold CV for perovskite-only representations (CBFV comparison).

    Representations: oliynyk, magpie, mat2vec, dummy
    Target: determined by pipeline_config

    Output: outputs/model/cv5_per_only/
    """
    import pipeline_config

    print("\n" + "=" * 60)
    print("5-fold CV — perovskite-only representations")
    print("=" * 60)

    target_col = pipeline_config.target_column()
    csr_dir = os.path.join(base_dir, "csr")
    out_dir = os.path.join(base_dir, "model", "cv5_per_only")
    os.makedirs(out_dir, exist_ok=True)

    y_vec = ensure_1d_y(df_5[target_col])

    rep_files = {
        "oliynyk": (
            os.path.join(csr_dir, "per_sp0_oliynyk_dummy_csr.npz"),
            os.path.join(csr_dir, "per_sp0_oliynyk_dummy_columns.npy"),
        ),
        "magpie": (
            os.path.join(csr_dir, "per_sp0_magpie_dummy_csr.npz"),
            os.path.join(csr_dir, "per_sp0_magpie_dummy_columns.npy"),
        ),
        "mat2vec": (
            os.path.join(csr_dir, "per_sp0_mat2vec_dummy_csr.npz"),
            os.path.join(csr_dir, "per_sp0_mat2vec_dummy_columns.npy"),
        ),
        "dummy": (
            os.path.join(csr_dir, "per_sp0_dummy_dummy_csr.npz"),
            os.path.join(csr_dir, "per_sp0_dummy_dummy_columns.npy"),
        ),
    }

    all_metrics = []

    for rep_name, (csr_path, col_path) in rep_files.items():
        print(f"\n  Representation: {rep_name}")
        X = csr2vec(csr_file_name=csr_path, columns_file_name=col_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rows = []
        for fold, (tr, te) in enumerate(kf.split(np.arange(len(y_vec))), start=1):
            X_tr = row_slice(X, tr)
            X_te = row_slice(X, te)
            y_tr, y_te = y_vec[tr], y_vec[te]

            rf = RandomForestRegressor(random_state=0)
            rf.fit(X_tr, y_tr)

            y_tr_pred = rf.predict(X_tr)
            y_te_pred = rf.predict(X_te)

            rows.append({
                "fold": fold,
                "n_train": len(tr), "n_test": len(te),
                "R2_train":   r2_score(y_tr, y_tr_pred),
                "R2_test":    r2_score(y_te, y_te_pred),
                "MAE_train":  mean_absolute_error(y_tr, y_tr_pred),
                "MAE_test":   mean_absolute_error(y_te, y_te_pred),
                "RMSE_train": rmse(y_tr, y_tr_pred),
                "RMSE_test":  rmse(y_te, y_te_pred),
            })

        metrics_df = pd.DataFrame(rows)
        metrics_df.insert(0, "representation", rep_name)
        all_metrics.append(metrics_df)

    all_metrics_df = pd.concat(all_metrics).reset_index(drop=True)

    metric_cols = ["R2_train", "R2_test", "MAE_train", "MAE_test", "RMSE_train", "RMSE_test"]
    summary_mean = all_metrics_df.groupby("representation")[metric_cols].mean().reset_index()
    summary_std = all_metrics_df.groupby("representation")[metric_cols].std(ddof=1).reset_index()
    summary_mean.insert(1, "stat", "mean")
    summary_std.insert(1, "stat", "std")
    summary_df = pd.concat([summary_mean, summary_std]).sort_values(
        ["representation", "stat"]
    ).reset_index(drop=True)

    tag = pipeline_config.target_file_tag()
    all_metrics_df.to_csv(os.path.join(out_dir, f"cv5_metrics_per_sp0_{tag}_RF.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, f"cv5_summary_per_sp0_{tag}_RF.csv"), index=False)
    print(f"\n  Saved results to: {out_dir}")
    print(summary_df.to_string(index=False))


# =============================================================================
# 6. All-feature 5-fold CV (main grid)
# =============================================================================
def run_cv_all_features(df_5, base_dir, targets=None):
    """Run 5-fold CV for all (target × CBFV × sp) combinations.

    Parameters
    ----------
    targets : list of str, or None
        Target column names to run CV for.
        Defaults to ["lnTS80m"].
        Pass ["lnTS80m", "TS80m"] to also include raw TS80m.

    Grid (per target):
      per : oliynyk, dummy
      sp  : 0, 1, 2, 3
    → 2 × 4 = 8 configurations per target

    Output: outputs/model/cv5/
    """
    if targets is None:
        targets = ["lnTS80m"]

    print("\n" + "=" * 60)
    print(f"5-fold CV — all features  targets={targets}")
    print("=" * 60)

    csr_dir = os.path.join(base_dir, "csr")
    model_dir = os.path.join(base_dir, "model", "cv5")
    os.makedirs(model_dir, exist_ok=True)

    target_dict = {t: df_5[t] for t in targets}

    for target_name, y_target in target_dict.items():
        for per in ["oliynyk", "dummy"]:
            for sp in ["0", "1", "2", "3"]:
                tag = f"{target_name}_all_sp{sp}_{per}_zero"
                print(f"\n  Target={target_name}  sp={sp}  per={per}")

                X = csr2vec(
                    csr_file_name=os.path.join(csr_dir, f"all_sp{sp}_{per}_zero_csr.npz"),
                    columns_file_name=os.path.join(csr_dir, f"all_sp{sp}_{per}_zero_columns.npy"),
                )

                _, summary_df, _ = run_rf_5fold_cv(
                    X=X,
                    y=y_target,
                    out_dir=model_dir,
                    tag=tag,
                    n_splits=5,
                    kfold_seed=42,
                    rf_seed=0,
                    rf_params=None,
                    save_models=True,
                )

                print(summary_df.to_string(index=False))


# =============================================================================
# 7. Collate all CV summaries into a single table
# =============================================================================
def collate_cv_results(base_dir, targets=None):
    """Read cv5_summary_{tag}.csv files and combine into one table.

    Output: outputs/model/cv5/CV_summary_clean_table.csv
    """
    if targets is None:
        targets = ["lnTS80m"]

    print("\n" + "=" * 60)
    print(f"Collating CV summary results  targets={targets}")
    print("=" * 60)

    model_dir = os.path.join(base_dir, "model", "cv5")
    records = []

    for target in targets:
        for per in ["oliynyk", "dummy"]:
            for sp in ["0", "1", "2", "3"]:
                fpath = os.path.join(
                    model_dir,
                    f"cv5_summary_{target}_all_sp{sp}_{per}_zero.csv",
                )
                df = pd.read_csv(fpath)
                mean_row = df[df["stat"] == "mean"].iloc[0]
                std_row = df[df["stat"] == "std"].iloc[0]
                records.append({
                    "target":         target,
                    "representation": per,
                    "sp":             int(sp),
                    "R2_test_mean":   mean_row["R2_test"],
                    "R2_test_std":    std_row["R2_test"],
                    "MAE_test_mean":  mean_row["MAE_test"],
                    "MAE_test_std":   std_row["MAE_test"],
                    "RMSE_test_mean": mean_row["RMSE_test"],
                    "RMSE_test_std":  std_row["RMSE_test"],
                })

    summary_all = (
        pd.DataFrame(records)
        .sort_values(["target", "representation", "sp"])
        .reset_index(drop=True)
    )

    out_path = os.path.join(model_dir, "CV_summary_clean_table.csv")
    summary_all.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(summary_all.to_string(index=False))

    return summary_all


# =============================================================================
# 8. Settings
# =============================================================================
# Set RUN_TS80M = True to additionally run raw TS80m regression.
# When False (default), only the primary target is analysed.
RUN_TS80M = False


# =============================================================================
# 9. Main
# =============================================================================
def main():
    import pipeline_config

    base_dir = "outputs"
    data_path = os.path.join(base_dir, "curated", "Perovskite_5800data_addln.csv")

    df_5 = pd.read_csv(data_path)
    print(f"Loaded: {len(df_5)} rows from {data_path}")

    # Determine active targets from pipeline_config
    primary_target = pipeline_config.target_column()  # "lnTS80m" or "JV_default_PCE"

    # For PCE mode, drop rows where JV_default_PCE is NaN
    if primary_target == "JV_default_PCE":
        n_before = len(df_5)
        df_5 = df_5.dropna(subset=["JV_default_PCE"]).reset_index(drop=True)
        print(f"  Dropped {n_before - len(df_5)} rows with missing PCE "
              f"→ {len(df_5)} rows remain")

    targets = [primary_target]
    if RUN_TS80M and primary_target != "TS80m":
        targets.append("TS80m")
    print(f"Active targets: {targets}")

    # Step 1: Perovskite-only CV (CBFV comparison)
    run_cv_perovskite_only(df_5, base_dir)

    # Step 2: All-feature CV (main grid)
    run_cv_all_features(df_5, base_dir, targets=targets)

    # Step 3: Collate results
    collate_cv_results(base_dir, targets=targets)

    print("\n=== Regression complete ===")


if __name__ == "__main__":
    main()
