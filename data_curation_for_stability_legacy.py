"""
Data Curation and Machine Learning for Perovskite Solar Cell Stability
=======================================================================
Source: The Perovskite Database (ZhangZ_PD.csv)
Target: TS80m (stability metric) and ln(TS80m)

Workflow:
  1. Data curation (filtering missing, non-ASCII, non-stoichiometric rows)
  2. CBFV-based feature matrix construction and CSR storage
  3. Random Forest regression for ln(TS80m) — all features
  4. Random Forest regression for TS80m — all features
  5. Random Forest regression for ln(TS80m) — perovskite composition only
  6. Duplicate sample detection

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
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

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
# 3. Utility functions for CSR sparse matrix I/O
# =============================================================================
def vec2csr(vec, csr_file_name, columns_file_name=None):
    """Save a DataFrame as a compressed sparse row matrix (.npz) and optionally
    save the column names to a .npy file."""
    csr = csr_matrix(vec)
    save_npz(csr_file_name, csr)
    if columns_file_name is not None:
        np.save(columns_file_name, np.array(vec.columns))


def csr2vec(csr_file_name, columns_file_name=None):
    """Load a compressed sparse row matrix and reconstruct a DataFrame."""
    if columns_file_name is None:
        vec = load_npz(csr_file_name).toarray()
    else:
        vec = pd.DataFrame(
            load_npz(csr_file_name).toarray(),
            columns=np.load(columns_file_name, allow_pickle=True),
        )
    return vec


# =============================================================================
# 4. Data curation
# =============================================================================
def curate_data():
    """Load raw database, apply curation filters, and save the cleaned dataset.

    Filtering steps:
      (a) Select feature columns (cols 11-258) and target columns (TS80, TS80m)
      (b) Remove rows with missing essential information
      (c) Remove rows containing non-ASCII characters or commas
      (d) Retain only rows with integer-sum stoichiometry coefficients (A, B, C sites)
      (e) Remove rows for which CBFV (magpie) returns all-zero vectors

    Outputs:
      data_stab/raw/Perovskite_5800data.csv
      data_stab/raw/Perovskite_5800data_addln.csv
    """
    # --- 4-a. Load raw database ---
    raw_df = pd.read_csv("data_stab/raw/ZhangZ_PD.csv")
    raw_df["Original_index"] = raw_df.index

    # Select X (cols 11-258) and y (TS80, TS80m) columns
    df_1 = pd.concat(
        [raw_df["Original_index"], raw_df.iloc[:, 11:259], raw_df["TS80"], raw_df["TS80m"]],
        axis=1,
    )
    print(f"[Step 0] Initial row count: {len(df_1)}")

    # --- 4-b. Remove rows with missing essential information ---
    essential_list = [
        "Substrate_stack_sequence",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions",
        "TS80",
        "TS80m",
    ]

    miss_index = []
    for i in range(len(df_1)):
        for col in essential_list:
            val = df_1[col][i]
            if val == "Unknown" or val == "none" or pd.isna(val):
                miss_index.append(i)

    print(f"[Step 1] Missing value entries: {len(miss_index)}, unique rows: {len(set(miss_index))}")
    df_2 = df_1.drop(df_1.index[list(set(miss_index))]).reset_index(drop=True)
    print(f"         Remaining rows: {len(df_2)}")

    # --- 4-c. Remove rows containing non-ASCII characters or commas ---
    # Convert entire DataFrame to str for character-level inspection
    str_array = np.array(df_2.astype(str))

    # Replace common non-ASCII variants with ASCII equivalents
    translation_table = str.maketrans({
        "\u2011": "-", "\u2010": "-", "\u2013": "-",
        "\u2018": "'", "\u2032": "'", "\u00b4": "'",
        "\u2223": "|", "\u00b5": "u", "\u03b1": "a",
    })
    for i in range(len(str_array)):
        for j in range(len(str_array[0])):
            str_array[i][j] = str_array[i][j].translate(translation_table)

    error_index = []
    for i in range(len(str_array)):
        for j in range(len(str_array[0])):
            if not str_array[i][j].isascii() or "," in str_array[i][j]:
                error_index.append(i)

    unique_error = list(set(error_index))
    print(f"[Step 2] Non-ASCII/comma entries: {len(error_index)}, unique rows: {len(unique_error)}")
    df_3 = df_2.drop(df_2.index[unique_error]).reset_index(drop=True)
    print(f"         Remaining rows: {len(df_3)}")

    # --- 4-d. Retain only rows with integer-sum stoichiometry ---
    integer_rows = []
    for i in range(len(df_3)):
        site_ok = {}
        for site in ["a", "b", "c"]:
            col = f"Perovskite_composition_{site}_ions_coefficients"
            try:
                numbers = df_3[col][i]
                number_list = []
                for section in numbers.split("|"):
                    number_list.extend(section.split(";"))
                total = sum(float(n) for n in number_list)
                site_ok[site] = total.is_integer()
            except Exception:
                site_ok[site] = False

        if all(site_ok.values()):
            integer_rows.append(i)

    df_4 = df_3.iloc[integer_rows].reset_index(drop=True)
    print(f"[Step 3] Integer-stoichiometry rows: {len(integer_rows)}")
    print(f"         Remaining rows: {len(df_4)}")

    # --- 4-e. Remove rows for which CBFV (magpie) returns all-zero vectors ---
    print("[Step 4] Computing CBFV (magpie) to identify unsupported compositions...")
    t1 = datetime.datetime.now()
    mag_df_4 = cbfv_table("Perovskite_composition_long_form", df_4, elem_prop="magpie")
    t2 = datetime.datetime.now()
    print(f"         Elapsed time: {t2 - t1}")

    cbfv_error_index = [i for i in range(len(mag_df_4)) if sum(mag_df_4.iloc[i]) == 0]
    use_index = [x for x in df_4.index if x not in cbfv_error_index]
    print(f"         CBFV-unsupported rows: {len(cbfv_error_index)}")

    df_5 = df_4.iloc[use_index].reset_index(drop=True)
    print(f"[Step 4] Final curated row count: {len(df_5)}")

    # --- Save intermediate dataset ---
    df_5.to_csv("data_stab/raw/Perovskite_5800data.csv", index=False)
    print("         Saved: data_stab/raw/Perovskite_5800data.csv")

    # --- Add ln(TS80m) and save final dataset ---
    df_5["lnTS80m"] = np.log(df_5["TS80m"])
    df_5.to_csv("data_stab/raw/Perovskite_5800data_addln.csv", index=False)
    print("         Saved: data_stab/raw/Perovskite_5800data_addln.csv")

    return df_5


# =============================================================================
# 5. Distribution analysis
# =============================================================================
def analyze_distribution(df_5):
    """Plot histograms and Q-Q plots for TS80m and ln(TS80m)."""
    print("\n--- Distribution statistics ---")
    print("TS80m:\n", df_5["TS80m"].describe())
    print("\nln(TS80m):\n", df_5["lnTS80m"].describe())

    # Histograms
    plt.figure()
    plt.hist(df_5["lnTS80m"], bins=30)
    plt.xlabel("ln(TS80m)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("lnTS80m_hist_new.png", dpi=300)
    plt.close()

    plt.figure()
    plt.hist(df_5["TS80m"], bins=30)
    plt.xlabel("TS80m (h)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("TS80m_hist_new.png", dpi=300)
    plt.close()

    # Tail-enlarged histogram
    plt.figure()
    plt.hist(df_5["TS80m"], bins=30)
    plt.xlim(df_5["TS80m"].max() / 30, df_5["TS80m"].max())
    plt.ylim(0, 50)
    plt.xlabel("TS80m (h)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("TS80m_hist_new_tail_enlarged.png", dpi=300)
    plt.close()

    # Q-Q plots
    fig, ax = plt.subplots()
    stats.probplot(df_5["lnTS80m"], dist="norm", plot=ax)
    plt.tight_layout()
    plt.savefig("qqplot_lnTS80m.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    stats.probplot(df_5["TS80m"], dist="norm", plot=ax)
    plt.tight_layout()
    plt.savefig("qqplot_TS80m.png", dpi=300)
    plt.close()

    # Kolmogorov-Smirnov test
    ks_result = stats.kstest(df_5["lnTS80m"], "norm")
    print(f"\nKS test for ln(TS80m): {ks_result}")


# =============================================================================
# 6. Feature matrix construction and CSR storage
# =============================================================================
def build_feature_matrix(df_5):
    """Construct the feature matrix X from the curated DataFrame and save as CSR.

    Numerical columns  -> numerical_sum_table (with zero-fill)
    Perovskite CBFV    -> cbfv_table (oliynyk)
    Categorical columns -> multihot_vector_table_1

    Output files:
      data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_csr.npz
      data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy
    """
    df_X = df_5.iloc[:, 1:249]

    # Identify numerical and categorical columns
    exclude = {"Original_index", "TS80", "TS80m", "lnTS80m", "Perovskite_composition_long_form"}
    num_columns = [
        j for j in range(len(df_5.columns))
        if df_5.columns[j] not in exclude
        and df_5.iloc[:, j].dtype in ("float64", "int64", int, float)
    ]
    num_list = df_5.columns[num_columns]

    print(f"\n[Feature matrix] Numerical column count: {len(num_columns)}")

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
    print(f"           Feature matrix shape: {X.shape}")

    vec2csr(
        vec=X,
        csr_file_name="data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_csr.npz",
        columns_file_name="data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy",
    )
    print("           Saved CSR matrix: data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_csr.npz")
    return X


# =============================================================================
# 7. Helper: parity plot (jointplot)
# =============================================================================
def _parity_plot(stack, x_col, y_col, hue_col, save_path):
    """Draw a seaborn jointplot parity diagram and save to file."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["font.size"] = 14

    fig = sns.jointplot(
        x=stack[x_col],
        y=stack[y_col],
        hue=stack[hue_col],
        joint_kws={"alpha": 0.3},
        edgecolor="white",
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
    print(f"           Parity plot saved: {save_path}")


# =============================================================================
# 8. Helper: feature importance aggregation
# =============================================================================
def _save_feature_importance(model, columns_file, df_X, output_fti, output_fti_sum):
    """Compute and save per-feature and per-column-group feature importances."""
    columns = np.load(columns_file, allow_pickle=True)
    fti_array = model.feature_importances_
    fti_series = pd.Series(fti_array, index=list(columns))
    fti_sort = fti_series.sort_values(ascending=False)
    pd.DataFrame(fti_sort, columns=["feature_importance"]).to_csv(output_fti, index=True)
    print(f"           Feature importance saved: {output_fti}")

    # Aggregate by original column name
    x_col = [c for c in df_X.columns if c != "Perovskite_composition_long_form"]
    fti_agg = []
    for x_name in x_col:
        fti_list = [fti_sort[i] for i in range(len(fti_sort)) if x_name in fti_sort.index[i]]
        fti_agg.append(sum(fti_list))
    per_fti = 1 - sum(fti_agg)
    fti_agg.append(per_fti)

    fti_summary = pd.DataFrame(
        fti_agg, index=x_col + ["Perovskite_composition_long_form"]
    ).sort_values(by=0, ascending=False)
    fti_summary.to_csv(output_fti_sum, index=True)
    print(f"           Aggregated feature importance saved: {output_fti_sum}")


# =============================================================================
# 9. Random Forest — ln(TS80m), all features
# =============================================================================
def train_rf_lnTS80m_all(df_5, X):
    """Train a Random Forest regressor for ln(TS80m) using all features.

    Hyperparameters: n_estimators=270, min_samples_split=3, random_state=0
    Train/test split: 80/20, random_state=0
    """
    print("\n" + "=" * 60)
    print("Model A: Random Forest — ln(TS80m), all features")
    print("=" * 60)

    y = df_5["lnTS80m"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rf = RandomForestRegressor(min_samples_split=3, n_estimators=270, random_state=0)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    R2_train = r2_score(y_train, y_train_pred)
    R2_test = r2_score(y_test, y_test_pred)
    print(f"  R2 (train): {R2_train:.4f}")
    print(f"  R2 (test) : {R2_test:.4f}")

    # Save model
    model_path = "data_stab/20250624_5800/model/rf_model_20250806.pkl"
    joblib.dump(rf, model_path)
    print(f"  Model saved: {model_path}")

    # Save predictions
    train_df = pd.concat(
        [pd.DataFrame(y_train).reset_index()["lnTS80m"], pd.DataFrame(y_train_pred)], axis=1
    )
    train_df.columns = ["y_train_real", "y_train_pred"]
    train_df.to_csv("data_stab/20250624_5800/model/lnTS80m_all_Train_RF.csv", index=False)

    test_df = pd.concat(
        [pd.DataFrame(y_test).reset_index()["lnTS80m"], pd.DataFrame(y_test_pred)], axis=1
    )
    test_df.columns = ["y_test_real", "y_test_pred"]
    test_df.to_csv("data_stab/20250624_5800/model/lnTS80m_all_Predict_RF.csv", index=False)

    # Parity plot
    train_plot = pd.read_csv("data_stab/20250624_5800/model/lnTS80m_all_Train_RF.csv")
    test_plot = pd.read_csv("data_stab/20250624_5800/model/lnTS80m_all_Predict_RF.csv")
    train_plot[" "] = "train"
    train_plot.rename(
        columns={"y_train_real": "Experimental ln(TS80m)", "y_train_pred": "Predicted ln(TS80m)"},
        inplace=True,
    )
    test_plot[" "] = "test"
    test_plot.rename(
        columns={"y_test_real": "Experimental ln(TS80m)", "y_test_pred": "Predicted ln(TS80m)"},
        inplace=True,
    )
    stack = pd.concat([train_plot, test_plot]).reset_index()
    _parity_plot(
        stack,
        x_col="Experimental ln(TS80m)",
        y_col="Predicted ln(TS80m)",
        hue_col=" ",
        save_path="data_stab/20250624_5800/model/jointplot_lnTS80m_all_RF.png",
    )

    # Feature importance
    df_X = df_5.iloc[:, 1:249]
    _save_feature_importance(
        model=rf,
        columns_file="data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy",
        df_X=df_X,
        output_fti="data_stab/20250624_5800/model/fti_new.csv",
        output_fti_sum="data_stab/20250624_5800/model/fti_sum_new.csv",
    )

    return rf


# =============================================================================
# 10. Random Forest — TS80m (raw), all features
# =============================================================================
def train_rf_TS80m_all(df_5, X):
    """Train a Random Forest regressor for TS80m (raw) using all features.

    Hyperparameters: n_estimators=10, max_depth=2, random_state=0
    Train/test split: 80/20, random_state=42
    """
    print("\n" + "=" * 60)
    print("Model B: Random Forest — TS80m (raw), all features")
    print("=" * 60)

    y = df_5["TS80m"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    R2_train = r2_score(y_train, y_train_pred)
    R2_test = r2_score(y_test, y_test_pred)
    print(f"  R2 (train): {R2_train:.4f}")
    print(f"  R2 (test) : {R2_test:.4f}")

    # Save model
    model_path = "data_stab/20250624_5800/model/rf_model_20250823_TS80m.pkl"
    joblib.dump(rf, model_path)
    print(f"  Model saved: {model_path}")

    # Save predictions
    train_df = pd.concat(
        [pd.DataFrame(y_train).reset_index()["TS80m"], pd.DataFrame(y_train_pred)], axis=1
    )
    train_df.columns = ["y_train_real", "y_train_pred"]
    train_df.to_csv("data_stab/20250624_5800/model/TS80m_all_Train_RF.csv", index=False)

    test_df = pd.concat(
        [pd.DataFrame(y_test).reset_index()["TS80m"], pd.DataFrame(y_test_pred)], axis=1
    )
    test_df.columns = ["y_test_real", "y_test_pred"]
    test_df.to_csv("data_stab/20250624_5800/model/TS80m_all_Predict_RF.csv", index=False)

    # Parity plot (linear scale)
    train_plot = pd.read_csv("data_stab/20250624_5800/model/TS80m_all_Train_RF.csv")
    test_plot = pd.read_csv("data_stab/20250624_5800/model/TS80m_all_Predict_RF.csv")
    train_plot[" "] = "train"
    train_plot.rename(
        columns={"y_train_real": "Experimental TS80m (h)", "y_train_pred": "Predicted TS80m (h)"},
        inplace=True,
    )
    test_plot[" "] = "test"
    test_plot.rename(
        columns={"y_test_real": "Experimental TS80m (h)", "y_test_pred": "Predicted TS80m (h)"},
        inplace=True,
    )
    stack = pd.concat([train_plot, test_plot]).reset_index()
    _parity_plot(
        stack,
        x_col="Experimental TS80m (h)",
        y_col="Predicted TS80m (h)",
        hue_col=" ",
        save_path="data_stab/20250624_5800/model/jointplot_TS80m_all_RF.png",
    )

    # Parity plot (log10 scale)
    def log_tick_format(x, pos):
        return r"$10^{{{:.0f}}}$".format(x)

    fig = sns.jointplot(
        x=np.log10(stack["Experimental TS80m (h)"] + 1),
        y=np.log10(stack["Predicted TS80m (h)"] + 1),
        hue=stack[" "],
        joint_kws={"alpha": 0.3},
        edgecolor="white",
    )
    fig.ax_joint.legend()
    x_max_log = np.log10(stack["Experimental TS80m (h)"].max() + 1)
    plt.plot([0, x_max_log], [0, x_max_log], c="black")
    fig.ax_joint.xaxis.set_major_formatter(FuncFormatter(log_tick_format))
    fig.ax_joint.yaxis.set_major_formatter(FuncFormatter(log_tick_format))
    plt.savefig("log_jointplot_TS80m_all_RF.png", dpi=300)
    plt.close()
    print("           Log-scale parity plot saved: log_jointplot_TS80m_all_RF.png")

    # Feature importance
    df_X = df_5.iloc[:, 1:249]
    _save_feature_importance(
        model=rf,
        columns_file="data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy",
        df_X=df_X,
        output_fti="data_stab/20250624_5800/model/TS80m_fti_20250823.csv",
        output_fti_sum="data_stab/20250624_5800/model/TS80m_fti_sum_20250823.csv",
    )

    return rf


# =============================================================================
# 11. Random Forest — ln(TS80m), perovskite composition only
# =============================================================================
def train_rf_lnTS80m_perovskite_only(df_5, X):
    """Train a Random Forest regressor for ln(TS80m) using only perovskite
    composition features (CBFV block, columns 2361 to 2624).

    Hyperparameters: n_estimators=270, min_samples_split=3, random_state=0
    Train/test split: 80/20, random_state=42
    """
    print("\n" + "=" * 60)
    print("Model C: Random Forest — ln(TS80m), perovskite composition only")
    print("=" * 60)

    y = df_5["lnTS80m"]
    X_per = X.iloc[:, 2361:2361 + 264]
    print(f"  Perovskite feature block shape: {X_per.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_per, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(min_samples_split=3, n_estimators=270, random_state=0)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    R2_train = r2_score(y_train, y_train_pred)
    R2_test = r2_score(y_test, y_test_pred)
    print(f"  R2 (train): {R2_train:.4f}")
    print(f"  R2 (test) : {R2_test:.4f}")

    # Save model
    model_path = "data_stab/20250624_5800/model/rf_model_only_per_lnTS80m.pkl"
    joblib.dump(rf, model_path)
    print(f"  Model saved: {model_path}")

    # Save predictions
    train_df = pd.concat(
        [pd.DataFrame(y_train).reset_index()["lnTS80m"], pd.DataFrame(y_train_pred)], axis=1
    )
    train_df.columns = ["y_train_real", "y_train_pred"]
    train_df.to_csv("data_stab/20250624_5800/model/lnTS80m_per_Train_RF.csv", index=False)

    test_df = pd.concat(
        [pd.DataFrame(y_test).reset_index()["lnTS80m"], pd.DataFrame(y_test_pred)], axis=1
    )
    test_df.columns = ["y_test_real", "y_test_pred"]
    test_df.to_csv("data_stab/20250624_5800/model/lnTS80m_per_Predict_RF.csv", index=False)

    # Parity plot
    train_plot = pd.read_csv("data_stab/20250624_5800/model/lnTS80m_per_Train_RF.csv")
    test_plot = pd.read_csv("data_stab/20250624_5800/model/lnTS80m_per_Predict_RF.csv")
    train_plot[" "] = "train"
    train_plot.rename(
        columns={"y_train_real": "Experimental ln(TS80m)", "y_train_pred": "Predicted ln(TS80m)"},
        inplace=True,
    )
    test_plot[" "] = "test"
    test_plot.rename(
        columns={"y_test_real": "Experimental ln(TS80m)", "y_test_pred": "Predicted ln(TS80m)"},
        inplace=True,
    )
    stack = pd.concat([train_plot, test_plot]).reset_index()
    _parity_plot(
        stack,
        x_col="Experimental ln(TS80m)",
        y_col="Predicted ln(TS80m)",
        hue_col=" ",
        save_path="data_stab/20250624_5800/model/jointplot_lnTS80m_per_RF.png",
    )

    return rf


# =============================================================================
# 12. Duplicate sample detection
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


def detect_duplicates(X):
    """Detect exact duplicate rows in the feature matrix X.

    Step 1: Group by row sum to find candidate duplicate groups (fast).
    Step 2: Verify element-wise equality within each candidate group.

    The result is saved to X1_dup_index_stab.npy.
    """
    print("\n" + "=" * 60)
    print("Duplicate detection")
    print("=" * 60)

    X_1 = X

    # Step 1: candidate duplicates (row-sum grouping)
    sum_dup = []
    df = X_1.copy()
    grouped = df.groupby(df.sum(axis=1))
    for key, indices in grouped.groups.items():
        if len(indices) > 1:
            sum_dup.append(list(df.index[indices]))
    sum_dup.sort()
    print(f"  Candidate duplicate groups (by row sum): {len(sum_dup)}")

    # Step 2: element-wise verification
    print(f"  Starting element-wise verification at {datetime.datetime.now()}")
    t1 = datetime.datetime.now()
    genuine_dup = []
    for j in range(len(sum_dup)):
        df_sub = pd.DataFrame(np.array(X_1)).iloc[sum_dup[j]]
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
            print(f"    Progress: {j}/{len(sum_dup)}, elapsed: {datetime.datetime.now() - t1}")

    t2 = datetime.datetime.now()
    print(f"  Processing time: {t2 - t1}")

    dup = [
        genuine_dup[j][k]
        for j in range(len(genuine_dup))
        for k in range(len(genuine_dup[j]))
    ]
    dup.sort()

    np.save("X1_dup_index_stab.npy", np.array(dup, dtype=object))
    print(f"  Duplicate groups: {len(dup)}")
    print(f"  Duplicate samples total: {count_elements(dup)}")
    print(f"  Unique samples after dedup (estimate): {len(X_1) - count_elements(dup) + len(dup)}")
    print(f"  Saved: X1_dup_index_stab.npy")

    return dup


# =============================================================================
# 13. Main
# =============================================================================
def main():
    # ---- Step 1: Data curation ----
    df_5 = curate_data()

    # ---- Step 2: Distribution analysis ----
    analyze_distribution(df_5)

    # ---- Step 3: Feature matrix construction (build once and cache as CSR) ----
    # If CSR files already exist, skip construction and load directly.
    import os
    csr_path = "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_csr.npz"
    col_path = "data_stab/20250624_5800/csr/all_sp1_oliynyk_zero_columns.npy"

    if os.path.exists(csr_path):
        print(f"\n[Feature matrix] Loading cached CSR from {csr_path}")
        df_5 = pd.read_csv("data_stab/raw/Perovskite_5800data_addln.csv")
        X = csr2vec(csr_file_name=csr_path, columns_file_name=col_path)
    else:
        X = build_feature_matrix(df_5)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target (lnTS80m) shape: {df_5['lnTS80m'].shape}")

    # ---- Step 4: Model A — RF for ln(TS80m), all features ----
    train_rf_lnTS80m_all(df_5, X)

    # ---- Step 5: Model B — RF for TS80m (raw), all features ----
    train_rf_TS80m_all(df_5, X)

    # ---- Step 6: Model C — RF for ln(TS80m), perovskite composition only ----
    train_rf_lnTS80m_perovskite_only(df_5, X)

    # ---- Step 7: Duplicate detection ----
    detect_duplicates(X)

    print("\n=== All steps completed ===")


if __name__ == "__main__":
    main()
