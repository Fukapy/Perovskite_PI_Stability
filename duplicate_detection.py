"""
Duplicate Sample Detection in the Feature Matrix
=================================================
Identifies exact duplicate rows in the feature matrix X.

Two-stage algorithm:
  Stage 1 (fast)  : Group rows by their row-sum to find candidate groups.
  Stage 2 (exact) : Within each candidate group, verify element-wise equality.

The resulting duplicate index list is saved to:
  X1_dup_index_stab_5800.npy

Prerequisites:
  - data_stab/raw/Perovskite_5800data_addln.csv
  - data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_{csr,columns}.*
"""

# =============================================================================
# 1. Imports
# =============================================================================
import datetime
import warnings

import numpy as np
import pandas as pd
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
# 3. Helper: count elements in a nested list
# =============================================================================
def count_elements(nested_list):
    """Recursively count all non-list elements in a nested list."""
    count = 0
    for element in nested_list:
        if isinstance(element, list):
            count += count_elements(element)
        else:
            count += 1
    return count


# =============================================================================
# 4. Duplicate detection
# =============================================================================
def detect_duplicates(X, save_path="X1_dup_index_stab_5800.npy"):
    """Detect exact duplicate rows in feature matrix X.

    Parameters
    ----------
    X         : pandas DataFrame (feature matrix)
    save_path : path to save the resulting duplicate index list

    Returns
    -------
    dup : list of lists — each sub-list is a group of row indices that are
          exact duplicates of each other
    """
    print("=" * 60)
    print("Duplicate detection")
    print("=" * 60)

    # --- Stage 1: candidate groups by row sum ---
    sum_dup = []
    grouped = X.groupby(X.sum(axis=1))
    for _, indices in grouped.groups.items():
        if len(indices) > 1:
            sum_dup.append(list(X.index[indices]))
    sum_dup.sort()
    print(f"  Stage 1 candidate groups: {len(sum_dup)}")

    # --- Stage 2: element-wise verification ---
    print(f"  Stage 2 start: {datetime.datetime.now()}")
    t1 = datetime.datetime.now()
    genuine_dup = []

    for j, group_idx in enumerate(sum_dup):
        df_sub = pd.DataFrame(np.array(X)).iloc[group_idx]
        grp = df_sub.groupby(df_sub.columns.tolist(), as_index=False)
        duplicated = grp.filter(lambda x: len(x) > 1)
        grouped_index = [
            duplicated.index[
                duplicated[df_sub.columns.tolist()].eq(val).all(axis=1)
            ].tolist()
            for val in duplicated[df_sub.columns.tolist()].drop_duplicates().values
        ]
        genuine_dup.append(grouped_index)
        if j % 200 == 0:
            print(f"    {j}/{len(sum_dup)}  elapsed: {datetime.datetime.now() - t1}")

    t2 = datetime.datetime.now()
    print(f"  Stage 2 total time: {t2 - t1}")

    dup = [
        genuine_dup[j][k]
        for j in range(len(genuine_dup))
        for k in range(len(genuine_dup[j]))
    ]
    dup.sort()

    # --- Save and report ---
    np.save(save_path, np.array(dup, dtype=object))

    n_total = len(X)
    n_dup_samples = count_elements(dup)
    n_dup_groups  = len(dup)
    n_unique_est  = n_total - n_dup_samples + n_dup_groups

    print(f"\n  Total samples        : {n_total}")
    print(f"  Duplicate groups     : {n_dup_groups}")
    print(f"  Samples in dup groups: {n_dup_samples}  ({n_dup_samples / n_total * 100:.1f}%)")
    print(f"  Unique (estimate)    : {n_unique_est}")
    print(f"  Saved: {save_path}")

    return dup


# =============================================================================
# 5. Main
# =============================================================================
def main():
    csr_path = "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_csr.npz"
    col_path = "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy"

    X = csr2vec(csr_path, col_path)
    print(f"Loaded feature matrix: {X.shape}")

    dup = detect_duplicates(X, save_path="X1_dup_index_stab_5800.npy")
    print(f"\n=== Duplicate detection complete  ({len(dup)} duplicate groups) ===")


if __name__ == "__main__":
    main()
