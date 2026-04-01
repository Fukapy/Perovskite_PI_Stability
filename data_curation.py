"""
Data Curation for Perovskite Solar Cell Stability Dataset
==========================================================
Source: The Perovskite Database (ZhangZ_PD.csv)
Output: data/20250624_5800/raw/Perovskite_5800data_addln.csv
        data/20250624_5800/csr/all_sp1_oliynyk_zero_{csr,columns}.*

Filtering steps:
  (a) Select feature columns (cols 11–258) and target columns (TS80, TS80m)
  (b) Remove rows with missing essential information
  (c) Replace common non-ASCII characters; remove rows still containing
      non-ASCII characters or commas
  (d) Retain only rows with integer-sum stoichiometry (A, B, C sites)
  (e) Remove rows for which CBFV (magpie) returns all-zero feature vectors

Distribution analysis:
  - Histograms of TS80m and ln(TS80m)
  - Q-Q plots
  - Kolmogorov-Smirnov normality test

Feature matrix construction:
  - Builds all_sp1_oliynyk_zero (multi-hot sp1 + Oliynyk CBFV) and saves as CSR.
  - Note: all feature matrix variants (sp0–3 × oliynyk/dummy) are generated in
    20251115_vectorization.py.  Run that script to obtain the full set.

History:
  2023/06/06  Initial version
  2023/07/14  Edit
  2023/07/26  Corrected Original_index bug
  2024/10/10  New data added
  2025/06/24  Additional data added
"""

# =============================================================================
# 1. Imports
# =============================================================================
import datetime
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, load_npz, save_npz

from process_stab import cbfv_table, multihot_vector_table_1, numerical_sum_table

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
    """Save a DataFrame as a compressed sparse row matrix."""
    csr = csr_matrix(vec)
    save_npz(csr_file_name, csr)
    if columns_file_name is not None:
        np.save(columns_file_name, np.array(vec.columns))


def csr2vec(csr_file_name, columns_file_name=None):
    """Load a compressed sparse row matrix and reconstruct a DataFrame."""
    if columns_file_name is None:
        return load_npz(csr_file_name).toarray()
    return pd.DataFrame(
        load_npz(csr_file_name).toarray(),
        columns=np.load(columns_file_name, allow_pickle=True),
    )


# =============================================================================
# 4. Data curation
# =============================================================================
def curate_data():
    """Load raw database, apply curation filters, and save the cleaned dataset.

    Returns
    -------
    df_5 : cleaned DataFrame with ln(TS80m) column added
    """
    # --- Load ---
    raw_df = pd.read_csv("data/20250624_5800/raw/ZhangZ_PD.csv")
    raw_df["Original_index"] = raw_df.index

    df_1 = pd.concat(
        [raw_df["Original_index"], raw_df.iloc[:, 11:259],
         raw_df["TS80"], raw_df["TS80m"]],
        axis=1,
    )
    print(f"[Step 0] Initial rows: {len(df_1)}")

    # --- (b) Remove rows with missing essential information ---
    essential_list = [
        "Substrate_stack_sequence", "ETL_stack_sequence",
        "HTL_stack_sequence", "Backcontact_stack_sequence",
        "Perovskite_composition_a_ions", "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions", "TS80", "TS80m",
    ]
    miss_index = set()
    for i in range(len(df_1)):
        for col in essential_list:
            val = df_1[col][i]
            if val == "Unknown" or val == "none" or pd.isna(val):
                miss_index.add(i)
    print(f"[Step 1] Missing rows: {len(miss_index)}")
    df_2 = df_1.drop(df_1.index[list(miss_index)]).reset_index(drop=True)
    print(f"         Remaining: {len(df_2)}")

    # --- (c) Non-ASCII / comma removal ---
    str_array = np.array(df_2.astype(str))
    translation_table = str.maketrans({
        "\u2011": "-", "\u2010": "-", "\u2013": "-",
        "\u2018": "'", "\u2032": "'", "\u00b4": "'",
        "\u2223": "|", "\u00b5": "u", "\u03b1": "a",
    })
    for i in range(len(str_array)):
        for j in range(len(str_array[0])):
            str_array[i][j] = str_array[i][j].translate(translation_table)

    error_index = set(
        i
        for i in range(len(str_array))
        for j in range(len(str_array[0]))
        if not str_array[i][j].isascii() or "," in str_array[i][j]
    )
    print(f"[Step 2] Non-ASCII/comma rows: {len(error_index)}")
    df_3 = df_2.drop(df_2.index[list(error_index)]).reset_index(drop=True)
    print(f"         Remaining: {len(df_3)}")

    # --- (d) Integer stoichiometry ---
    integer_rows = []
    for i in range(len(df_3)):
        ok = {}
        for site in ["a", "b", "c"]:
            col = f"Perovskite_composition_{site}_ions_coefficients"
            try:
                nums = []
                for section in df_3[col][i].split("|"):
                    nums.extend(section.split(";"))
                ok[site] = sum(float(n) for n in nums).is_integer()
            except Exception:
                ok[site] = False
        if all(ok.values()):
            integer_rows.append(i)
    df_4 = df_3.iloc[integer_rows].reset_index(drop=True)
    print(f"[Step 3] Integer-stoichiometry rows: {len(integer_rows)}")

    # --- (e) CBFV (magpie) zero-vector removal ---
    print("[Step 4] Computing CBFV (magpie)…")
    t1 = datetime.datetime.now()
    mag_df = cbfv_table("Perovskite_composition_long_form", df_4, elem_prop="magpie")
    print(f"         Elapsed: {datetime.datetime.now() - t1}")

    cbfv_err = [i for i in range(len(mag_df)) if sum(mag_df.iloc[i]) == 0]
    use_idx = [x for x in df_4.index if x not in cbfv_err]
    df_5 = df_4.iloc[use_idx].reset_index(drop=True)
    print(f"         CBFV-unsupported: {len(cbfv_err)}, final rows: {len(df_5)}")

    # --- Save ---
    df_5.to_csv("data/20250624_5800/raw/Perovskite_5800data.csv", index=False)
    df_5["lnTS80m"] = np.log(df_5["TS80m"])
    df_5.to_csv("data/20250624_5800/raw/Perovskite_5800data_addln.csv", index=False)
    print("Saved: data/20250624_5800/raw/Perovskite_5800data_addln.csv")
    return df_5


# =============================================================================
# 5. Distribution analysis
# =============================================================================
def analyze_distribution(df_5):
    """Save histograms, Q-Q plots, and print KS test result."""
    print("\n--- Distribution statistics ---")
    print("TS80m:\n", df_5["TS80m"].describe())
    print("\nln(TS80m):\n", df_5["lnTS80m"].describe())

    for col, fname in [("lnTS80m", "lnTS80m_hist_new.png"),
                        ("TS80m",   "TS80m_hist_new.png")]:
        plt.figure()
        plt.hist(df_5[col], bins=30)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

    # Tail-enlarged
    plt.figure()
    plt.hist(df_5["TS80m"], bins=30)
    plt.xlim(df_5["TS80m"].max() / 30, df_5["TS80m"].max())
    plt.ylim(0, 50)
    plt.xlabel("TS80m (h)")
    plt.tight_layout()
    plt.savefig("TS80m_hist_new_tail_enlarged.png", dpi=300)
    plt.close()

    # Q-Q plots
    for col, fname in [("lnTS80m", "qqplot_lnTS80m.png"),
                        ("TS80m",   "qqplot_TS80m.png")]:
        fig, ax = plt.subplots()
        stats.probplot(df_5[col], dist="norm", plot=ax)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

    ks = stats.kstest(df_5["lnTS80m"], "norm")
    print(f"\nKS test ln(TS80m): {ks}")


# =============================================================================
# 6. Feature matrix construction (all_sp1_oliynyk_zero)
# =============================================================================
def build_feature_matrix(df_5):
    """Build all_sp1_oliynyk_zero feature matrix and save as CSR.

    Encoding:
      Numerical columns               → numerical_sum_table (zero-fill)
      Perovskite_composition_long_form → cbfv_table (oliynyk)
      Categorical columns             → multihot_vector_table_1

    Note: the full set of feature matrix variants (sp0–3 × oliynyk/dummy)
    is generated by 20251115_vectorization.py.
    """
    csr_dir = "data/20250624_5800/csr"
    os.makedirs(csr_dir, exist_ok=True)

    df_X = df_5.iloc[:, 1:249]
    exclude = {
        "Original_index", "TS80", "TS80m", "lnTS80m",
        "Perovskite_composition_long_form",
    }
    num_columns = [
        j for j in range(len(df_5.columns))
        if df_5.columns[j] not in exclude
        and df_5.iloc[:, j].dtype in ("float64", "int64")
    ]
    num_list = df_5.columns[num_columns]
    print(f"\n[Feature matrix] Numerical columns: {len(num_list)}")

    x_list = []
    for x_name in list(df_X.columns):
        if x_name in num_list:
            x = numerical_sum_table(x_name, df_X, "zero")
        elif x_name == "Perovskite_composition_long_form":
            x = cbfv_table(x_name, df_X, "oliynyk")
        else:
            x = multihot_vector_table_1(x_name, df_X)
        x_list.append(x)

    X = pd.concat(x_list, axis=1).fillna(0)
    print(f"  Shape: {X.shape}")

    vec2csr(
        X,
        csr_file_name=f"{csr_dir}/all_sp1_oliynyk_zero_csr.npz",
        columns_file_name=f"{csr_dir}/all_sp1_oliynyk_zero_columns.npy",
    )
    print(f"  Saved: {csr_dir}/all_sp1_oliynyk_zero_csr.npz")
    return X


# =============================================================================
# 7. Main
# =============================================================================
def main():
    df_5 = curate_data()
    analyze_distribution(df_5)

    csr_path = "data/20250624_5800/csr/all_sp1_oliynyk_zero_csr.npz"
    if os.path.exists(csr_path):
        print(f"\n[Feature matrix] CSR already exists: {csr_path}  (skipping build)")
    else:
        build_feature_matrix(df_5)

    print("\n=== Data curation complete ===")
    print("Next step: run 20251115_vectorization.py to build all CSR variants.")


if __name__ == "__main__":
    main()
