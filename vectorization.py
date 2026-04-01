"""
Feature Matrix Vectorization for Perovskite Stability Dataset
=============================================================
Date: 2025/11/15

Builds and saves CSR feature matrices for all experimental configurations:

  Perovskite-only representations (sp0):
    per_sp0_oliynyk_dummy  — CBFV Oliynyk
    per_sp0_magpie_dummy   — CBFV Magpie
    per_sp0_mat2vec_dummy  — CBFV Mat2Vec
    per_sp0_dummy_dummy    — One-hot (composition dummy)

  All-feature matrices (perovskite CBFV = Oliynyk):
    all_sp0_oliynyk_zero   — categorical: one-hot
    all_sp1_oliynyk_zero   — categorical: multi-hot variant 1
    all_sp2_oliynyk_zero   — categorical: multi-hot variant 2
    all_sp3_oliynyk_zero   — categorical: multi-hot variant 3

  All-feature matrices (perovskite CBFV = dummy / one-hot):
    all_sp0_dummy_zero
    all_sp1_dummy_zero
    all_sp2_dummy_zero
    all_sp3_dummy_zero

After building the vectors, a quick train/test RF evaluation is run on each
perovskite-only representation to compare CBFV schemes.

Output directory: data/20250624_5800/csr/
"""

# =============================================================================
# 1. Imports
# =============================================================================
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from process_stab import (
    cbfv_table,
    multihot_vector_table_1,
    multihot_vector_table_2,
    multihot_vector_table_3,
    numerical_sum_table,
    onehot_vector_table,
)

warnings.filterwarnings("ignore")

# =============================================================================
# 2. Matplotlib configuration
# =============================================================================
plt.rcParams["figure.subplot.bottom"] = 0.2
plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["font.family"] = "Arial"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 16


# =============================================================================
# 3. CSR I/O utilities
# =============================================================================
def vec2csr(vec, csr_file_name, columns_file_name=None):
    """Save a DataFrame as a compressed sparse row matrix and optionally save
    the column index to a .npy file."""
    csr = csr_matrix(vec)
    save_npz(csr_file_name, csr)
    if columns_file_name is not None:
        np.save(columns_file_name, np.array(vec.columns))
    print(f"  Saved: {csr_file_name}  shape={vec.shape}")


def csr2vec(csr_file_name, columns_file_name=None):
    """Load a CSR matrix and reconstruct a DataFrame."""
    if columns_file_name is None:
        return load_npz(csr_file_name).toarray()
    return pd.DataFrame(
        load_npz(csr_file_name).toarray(),
        columns=np.load(columns_file_name, allow_pickle=True),
    )


# =============================================================================
# 4. Helper: build a feature vector from a single column list
# =============================================================================
def _build_X(df_X, num_list, cbfv_prop, cat_encoder):
    """Construct and return a feature DataFrame.

    Parameters
    ----------
    df_X        : DataFrame of feature columns
    num_list    : Index of numerical column names
    cbfv_prop   : str or None — CBFV property for Perovskite_composition_long_form
                  ('oliynyk', 'magpie', 'mat2vec', None)
    cat_encoder : callable — encoder for non-numerical, non-CBFV columns
                  (onehot_vector_table / multihot_vector_table_1 / etc.)
    """
    x_list = []
    for x_name in list(df_X.columns):
        if x_name in num_list:
            x = numerical_sum_table(x_name, df_X, "zero")
        elif x_name == "Perovskite_composition_long_form":
            if cbfv_prop is not None:
                x = cbfv_table(x_name, df_X, cbfv_prop)
            else:
                x = onehot_vector_table(x_name, df_X)
        else:
            x = cat_encoder(x_name, df_X)
        x_list.append(x)
    return pd.concat(x_list, axis=1).fillna(0)


# =============================================================================
# 5. Helper: quick RF evaluation (train/test split)
# =============================================================================
def _quick_rf_eval(X, y, label, random_state=0, test_size=0.2, split_seed=42):
    """Train RF and print R2/MAE/RMSE for train and test splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed
    )
    rf = RandomForestRegressor(random_state=random_state)
    rf.fit(X_train, y_train)

    y_tr_pred = rf.predict(X_train)
    y_te_pred = rf.predict(X_test)

    print(f"\n  [{label}]")
    print(f"    R2    train={r2_score(y_train, y_tr_pred):.4f}  test={r2_score(y_test, y_te_pred):.4f}")
    print(f"    MAE   train={mean_absolute_error(y_train, y_tr_pred):.4f}  "
          f"test={mean_absolute_error(y_test, y_te_pred):.4f}")
    rmse_tr = mean_squared_error(y_train, y_tr_pred) ** 0.5
    rmse_te = mean_squared_error(y_test, y_te_pred) ** 0.5
    print(f"    RMSE  train={rmse_tr:.4f}  test={rmse_te:.4f}")


# =============================================================================
# 6. Perovskite-only representation vectors
# =============================================================================
def build_perovskite_only_vectors(df_X, csr_base):
    """Build four perovskite-only feature matrices (CBFV comparison).

    Output files:
      per_sp0_oliynyk_dummy_{csr,columns}
      per_sp0_magpie_dummy_{csr,columns}
      per_sp0_mat2vec_dummy_{csr,columns}
      per_sp0_dummy_dummy_{csr,columns}
    """
    print("\n" + "=" * 60)
    print("Building perovskite-only feature matrices")
    print("=" * 60)

    configs = [
        ("oliynyk", "oliynyk"),
        ("magpie",  "magpie"),
        ("mat2vec", "mat2vec"),
        ("dummy",   None),       # None → one-hot (no CBFV)
    ]

    for rep_name, cbfv_prop in configs:
        x_list = []
        for x_name in list(df_X.columns):
            if x_name == "Perovskite_composition_long_form":
                if cbfv_prop is not None:
                    x = cbfv_table(x_name, df_X, cbfv_prop)
                else:
                    x = onehot_vector_table(x_name, df_X)
                x_list.append(x)

        X = pd.concat(x_list, axis=1).fillna(0)
        tag = f"per_sp0_{rep_name}_dummy"
        vec2csr(
            X,
            csr_file_name=f"{csr_base}/{tag}_csr.npz",
            columns_file_name=f"{csr_base}/{tag}_columns.npy",
        )


# =============================================================================
# 7. All-feature matrices — Oliynyk CBFV × four categorical encodings
# =============================================================================
def build_all_oliynyk_vectors(df_X, num_list, csr_base):
    """Build all-feature matrices using Oliynyk CBFV and varying categorical
    encoding (sp0–sp3).

    sp0: one-hot
    sp1: multi-hot variant 1
    sp2: multi-hot variant 2
    sp3: multi-hot variant 3
    """
    print("\n" + "=" * 60)
    print("Building all-feature matrices (CBFV=oliynyk, sp0-sp3)")
    print("=" * 60)

    sp_encoders = {
        "0": onehot_vector_table,
        "1": multihot_vector_table_1,
        "2": multihot_vector_table_2,
        "3": multihot_vector_table_3,
    }

    for sp, encoder in sp_encoders.items():
        print(f"\n  sp={sp}, cbfv=oliynyk")
        X = _build_X(df_X, num_list, cbfv_prop="oliynyk", cat_encoder=encoder)
        tag = f"all_sp{sp}_oliynyk_zero"
        vec2csr(
            X,
            csr_file_name=f"{csr_base}/{tag}_csr.npz",
            columns_file_name=f"{csr_base}/{tag}_columns.npy",
        )


# =============================================================================
# 8. All-feature matrices — dummy CBFV (one-hot composition) × four encodings
# =============================================================================
def build_all_dummy_vectors(df_X, num_list, csr_base):
    """Build all-feature matrices using dummy (one-hot) composition encoding
    and varying categorical encoding (sp0–sp3).
    """
    print("\n" + "=" * 60)
    print("Building all-feature matrices (CBFV=dummy, sp0-sp3)")
    print("=" * 60)

    sp_encoders = {
        "0": onehot_vector_table,
        "1": multihot_vector_table_1,
        "2": multihot_vector_table_2,
        "3": multihot_vector_table_3,
    }

    for sp, encoder in sp_encoders.items():
        print(f"\n  sp={sp}, cbfv=dummy")
        # dummy: treat Perovskite_composition_long_form like any other categorical col
        x_list = []
        for x_name in list(df_X.columns):
            if x_name in num_list:
                x = numerical_sum_table(x_name, df_X, "zero")
            else:
                x = encoder(x_name, df_X)
            x_list.append(x)
        X = pd.concat(x_list, axis=1).fillna(0)
        tag = f"all_sp{sp}_dummy_zero"
        vec2csr(
            X,
            csr_file_name=f"{csr_base}/{tag}_csr.npz",
            columns_file_name=f"{csr_base}/{tag}_columns.npy",
        )


# =============================================================================
# 9. Quick evaluation of perovskite-only representations
# =============================================================================
def evaluate_perovskite_representations(y, csr_base):
    """Run a quick train/test RF evaluation for each perovskite-only CSR."""
    print("\n" + "=" * 60)
    print("Quick RF evaluation — perovskite-only representations (lnTS80m)")
    print("=" * 60)

    reps = ["oliynyk", "magpie", "mat2vec", "dummy"]
    for rep in reps:
        tag = f"per_sp0_{rep}_dummy"
        X = csr2vec(
            csr_file_name=f"{csr_base}/{tag}_csr.npz",
            columns_file_name=f"{csr_base}/{tag}_columns.npy",
        )
        _quick_rf_eval(X, y, label=rep)


# =============================================================================
# 10. Main
# =============================================================================
def main():
    # ---- Paths ----
    data_path = "data/20250624_5800/raw/Perovskite_5800data_addln.csv"
    csr_base = "data/20250624_5800/csr"

    import os
    os.makedirs(csr_base, exist_ok=True)

    # ---- Load data ----
    df_5 = pd.read_csv(data_path)
    df_X = df_5.iloc[:, 1:249]

    # ---- Identify numerical and categorical columns ----
    exclude = {
        "Original_index", "TS80", "TS80m", "lnTS80m",
        "Perovskite_composition_long_form",
    }
    num_columns = [
        j for j in range(len(df_5.columns))
        if df_5.columns[j] not in exclude
        and (df_5.iloc[:, j].dtype == "float64" or df_5.iloc[:, j].dtype == "int64")
    ]
    num_list = df_5.columns[num_columns]
    print(f"Loaded: {len(df_5)} rows, {len(df_X.columns)} feature columns")
    print(f"  Numerical columns: {len(num_list)}")

    y = df_5["lnTS80m"]

    # ---- Build vectors ----
    build_perovskite_only_vectors(df_X, csr_base)
    build_all_oliynyk_vectors(df_X, num_list, csr_base)
    build_all_dummy_vectors(df_X, num_list, csr_base)

    # ---- Quick evaluation (perovskite-only) ----
    evaluate_perovskite_representations(y, csr_base)

    print("\n=== Vectorization complete ===")


if __name__ == "__main__":
    main()
