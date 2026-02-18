import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.ensemble import IsolationForest
from sklearn.utils import resample

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

try:
    import category_encoders as ce
    HAS_CATEGORY_ENCODERS = True
except Exception:
    HAS_CATEGORY_ENCODERS = False

st.set_page_config(page_title="NDEDC ML Workflow", layout="wide")
st.title("Workflow: Cleaning -> Eda_Data -> Model_Training")
st.sidebar.markdown("### Workflow Order")
st.sidebar.markdown("1. Cleaning Data")
st.sidebar.markdown("2. EDA_Data")
st.sidebar.markdown("3. Model_Training")


def convert_df(df_to_convert: pd.DataFrame) -> bytes:
    return df_to_convert.to_csv(index=False).encode("utf-8-sig")


def normalize_github_url(url: str) -> str:
    url = url.strip()
    if "github.com" in url and "/blob/" in url:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return url


def load_dataframe_from_source(uploaded_file, github_url: str):
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext in [".xlsx", ".xls"]:
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file, encoding="utf-8", encoding_errors="ignore")

    if github_url and github_url.strip():
        normalized_url = normalize_github_url(github_url)
        lower_url = normalized_url.lower()
        if lower_url.endswith((".xlsx", ".xls")):
            return pd.read_excel(normalized_url)
        return pd.read_csv(normalized_url, encoding="utf-8", encoding_errors="ignore")

    return None


def list_project_folders(base_path: str, max_depth: int = 3):
    folders = ["."]
    base_path = os.path.abspath(base_path)
    base_depth = base_path.rstrip(os.sep).count(os.sep)
    for root, dirnames, _ in os.walk(base_path):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in {"venv", "__pycache__", ".git"}
        ]
        for d in dirnames:
            rel = os.path.relpath(os.path.join(root, d), base_path)
            folders.append(rel)
    return sorted(set(folders))


def get_iqr_mask(series: pd.Series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index), q1, q3
    low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < low) | (series > up)
    return mask, low, up


def apply_outlier_strategy_until_stable(
    df: pd.DataFrame, column: str, strategy: str, max_iter: int = 50
):
    df_work = df.copy()
    iterations = 0

    while iterations < max_iter:
        mask, low, up = get_iqr_mask(df_work[column])
        out_count = int(mask.sum())
        if out_count == 0:
            break

        iterations += 1
        if strategy == "Mean Replace":
            df_work.loc[mask, column] = df_work[column].mean()
        elif strategy == "Drop Rows":
            df_work = df_work.loc[~mask].copy()
        elif strategy == "IQR Capping (Winsorize)":
            df_work[column] = df_work[column].clip(lower=low, upper=up)
        elif strategy == "Percentile Clipping (1%-99%)":
            lower_p = df_work[column].quantile(0.01)
            upper_p = df_work[column].quantile(0.99)
            df_work[column] = df_work[column].clip(lower=lower_p, upper=upper_p)
        elif strategy == "MAD Clipping (Robust)":
            s = df_work[column].astype(float)
            median = s.median()
            mad = (s - median).abs().median()
            if mad == 0 or pd.isna(mad):
                df_work[column] = df_work[column].clip(lower=low, upper=up)
            else:
                robust_sigma = mad / 0.6745
                k = 3.5
                lower_mad = median - k * robust_sigma
                upper_mad = median + k * robust_sigma
                df_work[column] = s.clip(lower=lower_mad, upper=upper_mad)
        elif strategy == "Log Transform + IQR Capping":
            s = df_work[column].astype(float)
            shift = 0.0
            min_val = float(s.min())
            if min_val <= -1.0:
                shift = abs(min_val) + 1.001
            transformed = np.log1p(s + shift)
            df_work[column] = transformed
        elif strategy == "Isolation Forest (Drop Anomalies)":
            s = df_work[[column]].astype(float)
            contamination = min(0.1, max(0.01, float(mask.mean())))
            iso = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
            )
            pred = iso.fit_predict(s)
            keep_mask = pred == 1
            df_work = df_work.loc[keep_mask].copy()
        else:
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(len(df_work), 100),
            )
            df_work[column] = qt.fit_transform(df_work[[column]].values).flatten()
            # Quantile transform can still leave IQR outliers; continue to re-check.

    # Final strict pass to guarantee no IQR outliers remain.
    while iterations < max_iter:
        mask, low, up = get_iqr_mask(df_work[column])
        if int(mask.sum()) == 0:
            break
        iterations += 1
        df_work[column] = df_work[column].clip(lower=low, upper=up)

    return df_work, iterations


def one_shot_checkbox(label: str, key: str) -> bool:
    fired = bool(st.session_state.get(key, False))
    if fired:
        st.session_state[key] = False
    st.checkbox(label, key=key)
    return fired


st.markdown("---")
st.header("Step 2: EDA_Data")
st.caption("Run this after finishing Step 1 (Cleaning Data).")


def render_stage_eda(df_in: pd.DataFrame, stage_key: str, stage_title: str):
    st.markdown(f"**Plots and Statistics: {stage_title}**")
    if df_in is None or df_in.empty:
        st.info("No data available for plots.")
        return

    numeric_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.info("No numeric columns available for plots in this stage.")
        return

    # 1) Correlation heatmap
    with st.expander(f"{stage_title} - Correlation Heatmap"):
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            corr_method = st.radio(
                "Correlation method:",
                ["pearson", "spearman", "kendall"],
                horizontal=True,
                key=f"{stage_key}_corr_method",
            )
            corr_matrix = df_in[numeric_cols].corr(method=corr_method)
            st.dataframe(corr_matrix, use_container_width=True, height=260)
            fig_hm, ax_hm = plt.subplots(figsize=(max(7, len(numeric_cols) * 0.55), max(5, len(numeric_cols) * 0.45)))
            sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.2, ax=ax_hm)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            st.pyplot(fig_hm)

    # 2) Outlier summary + boxplot
    with st.expander(f"{stage_title} - Outliers"):
        out_stats = []
        for col in numeric_cols:
            q1, q3 = df_in[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df_in[col] < low) | (df_in[col] > up)
            out_stats.append({"column": col, "outlier_count": int(mask.sum()), "lower": float(low), "upper": float(up)})
        out_df = pd.DataFrame(out_stats).sort_values("outlier_count", ascending=False)
        st.dataframe(out_df, use_container_width=True, height=min(320, 36 * (len(out_df) + 1)))
        out_col = st.selectbox("Column for outlier boxplot:", options=numeric_cols, key=f"{stage_key}_outlier_col")
        fig_box, ax_box = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=df_in[out_col], ax=ax_box, color="#8ecae6")
        st.pyplot(fig_box)

    # 3) Histogram
    with st.expander(f"{stage_title} - Histogram"):
        hist_col = st.selectbox("Column for histogram:", options=numeric_cols, key=f"{stage_key}_hist_col")
        bins = st.slider("Bins:", 10, 80, 30, 5, key=f"{stage_key}_hist_bins")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3.5))
        sns.histplot(df_in[hist_col].dropna(), bins=bins, kde=True, ax=ax_hist, color="#219ebc")
        st.pyplot(fig_hist)

    # 4) Dispersion / معدل الانتشار
    with st.expander(f"{stage_title} - Dispersion (معدل الانتشار)"):
        disp_rows = []
        for col in numeric_cols:
            s = df_in[col].dropna()
            mean_val = float(s.mean()) if len(s) else 0.0
            std_val = float(s.std()) if len(s) else 0.0
            iqr_val = float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else 0.0
            cv = float(std_val / mean_val) if mean_val not in [0, 0.0] else np.nan
            disp_rows.append(
                {
                    "column": col,
                    "std": std_val,
                    "variance": float(s.var()) if len(s) else 0.0,
                    "IQR": iqr_val,
                    "dispersion_rate_CV": cv,
                }
            )
        disp_df = pd.DataFrame(disp_rows).sort_values("std", ascending=False)
        st.dataframe(disp_df, use_container_width=True, height=min(320, 36 * (len(disp_df) + 1)))

        st.markdown("**Scatter Plot Matrix**")
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for scatter matrix.")
        else:
            scatter_mode = st.radio(
                "Scatter matrix mode:",
                ["All numeric features", "Selected features", "One feature vs others"],
                horizontal=True,
                key=f"{stage_key}_disp_scatter_mode",
            )
            scatter_cols = []

            if scatter_mode == "All numeric features":
                max_cols = min(8, len(numeric_cols))
                if max_cols <= 2:
                    use_cols_count = 2
                    st.caption("Using 2 features (available numeric features).")
                else:
                    use_cols_count = st.slider(
                        "Number of features",
                        min_value=2,
                        max_value=max_cols,
                        value=min(5, max_cols),
                        key=f"{stage_key}_disp_scatter_all_count",
                    )
                scatter_cols = numeric_cols[:use_cols_count]
            elif scatter_mode == "Selected features":
                scatter_cols = st.multiselect(
                    "Select features (min 2):",
                    options=numeric_cols,
                    default=numeric_cols[: min(4, len(numeric_cols))],
                    key=f"{stage_key}_disp_scatter_selected",
                )
            else:
                focus_col = st.selectbox(
                    "Focus feature:",
                    options=numeric_cols,
                    key=f"{stage_key}_disp_scatter_focus",
                )
                compare_opts = [c for c in numeric_cols if c != focus_col]
                compare_cols = st.multiselect(
                    "Compare with:",
                    options=compare_opts,
                    default=compare_opts[: min(3, len(compare_opts))],
                    key=f"{stage_key}_disp_scatter_compare",
                )
                scatter_cols = [focus_col] + compare_cols

            if len(scatter_cols) < 2:
                st.info("Please select at least 2 features.")
            else:
                max_rows = min(3000, len(df_in))
                min_rows = min(200, len(df_in))
                if max_rows <= min_rows:
                    sample_rows = max_rows
                    st.caption(f"Using all available rows: {sample_rows}")
                else:
                    sample_rows = st.slider(
                        "Rows for scatter matrix (sampled):",
                        min_value=min_rows,
                        max_value=max_rows,
                        value=min(1000, max_rows),
                        key=f"{stage_key}_disp_scatter_rows",
                    )
                scatter_df = df_in[scatter_cols].dropna()
                if len(scatter_df) > sample_rows:
                    scatter_df = scatter_df.sample(sample_rows, random_state=42)
                if scatter_df.empty:
                    st.info("No rows left after dropping missing values.")
                else:
                    g = sns.pairplot(scatter_df, corner=True, diag_kind="hist")
                    st.pyplot(g.fig)
                    plt.close(g.fig)

    # 5) Skewness
    with st.expander(f"{stage_title} - Skewness"):
        skew_df = pd.DataFrame(
            {
                "column": numeric_cols,
                "skewness": [float(df_in[c].skew()) for c in numeric_cols],
            }
        )
        skew_df["abs_skewness"] = skew_df["skewness"].abs()
        skew_df["skew_direction"] = skew_df["skewness"].apply(
            lambda x: "right" if x > 0.5 else ("left" if x < -0.5 else "near-symmetric")
        )
        skew_df = skew_df.sort_values("abs_skewness", ascending=False)
        st.dataframe(skew_df, use_container_width=True, height=min(320, 36 * (len(skew_df) + 1)))
        fig_sk, ax_sk = plt.subplots(figsize=(8, max(3, len(skew_df) * 0.22)))
        sns.barplot(data=skew_df, x="skewness", y="column", ax=ax_sk, color="#ffb703")
        ax_sk.axvline(0, color="black", linewidth=1)
        st.pyplot(fig_sk)


# ------------------------------------------------------------------
# 1. Data Source (Upload or GitHub)
# ------------------------------------------------------------------
st.markdown("---")
st.header("Step 1: Cleaning Data")
st.caption("Start here: load, clean, encode, transform, and balance your dataset.")
if "df" not in st.session_state:
    st.info("Please upload a dataset file to get started.")
    upload_col, github_col = st.columns(2)

    with upload_col:
        uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"])

    with github_col:
        github_url = st.text_input(
            "Or paste a direct/Raw GitHub file URL",
            placeholder="https://github.com/user/repo/blob/main/data.csv",
        )
        load_from_github = one_shot_checkbox("Load from GitHub", "load_from_github_chk")

    if uploaded_file is not None:
        loaded_df = load_dataframe_from_source(uploaded_file, "")
        st.session_state.df_original = loaded_df.copy()
        st.session_state.df = loaded_df.copy()
        st.rerun()

    if load_from_github:
        try:
            df = load_dataframe_from_source(None, github_url)
            if df is None:
                st.error("Please enter a valid GitHub URL.")
            else:
                st.session_state.df_original = df.copy()
                st.session_state.df = df.copy()
                st.rerun()
        except Exception as e:
            st.error(f"Failed to load from GitHub: {e}")
else:
    upload_different_file_trigger = bool(st.session_state.get("upload_different_file_chk", False))
    if upload_different_file_trigger:
        st.session_state.upload_different_file_chk = False
    st.sidebar.checkbox("Upload Different File", key="upload_different_file_chk")
    if upload_different_file_trigger:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ------------------------------------------------------------------
# 2. Cleaning and Preview Tools
# ------------------------------------------------------------------
if "df" in st.session_state:
    if "df_original" not in st.session_state:
        st.session_state.df_original = st.session_state.df.copy()

    st.markdown("---")
    st.subheader("Original Data Preview")
    st.dataframe(st.session_state.df_original, height=600, use_container_width=True)
    all_cols = st.session_state.df.columns.tolist()
    if all_cols:
        default_output_idx = len(all_cols) - 1
        if st.session_state.get("model_output_col") in all_cols:
            default_output_idx = all_cols.index(st.session_state.get("model_output_col"))
        output_col = st.selectbox(
            "Select Output/Target column:",
            options=all_cols,
            index=default_output_idx,
            key="model_output_col",
        )
        prev_output_col = st.session_state.get("model_output_col_prev")
        output_changed = prev_output_col != output_col
        st.session_state.model_output_col_prev = output_col

        allowed_inputs = [c for c in all_cols if c != output_col]
        remembered_inputs = st.session_state.get("model_input_cols", [])
        checkbox_grid_cols = st.columns(3)
        for idx, col_name in enumerate(allowed_inputs):
            chk_key = f"model_input_chk_{col_name}"
            if output_changed:
                st.session_state[chk_key] = True
            elif chk_key not in st.session_state:
                st.session_state[chk_key] = col_name in remembered_inputs if remembered_inputs else True
            with checkbox_grid_cols[idx % 3]:
                st.checkbox(col_name, key=chk_key)

        input_cols = [c for c in allowed_inputs if st.session_state.get(f"model_input_chk_{c}", False)]
        st.session_state.model_input_cols = input_cols
        st.caption(f"Current Output: `{output_col}`")
        st.caption(f"Current Inputs count: `{len(input_cols)}`")
    with st.expander("Inspect Data Types"):
        st.write(st.session_state.df_original.dtypes)

    st.markdown("---")
    st.subheader("Processing Tools")
    with st.expander("Process Guide (Step-by-Step)"):
        process_guide_df = pd.DataFrame(
            [
                {"step": "1) Drop Columns", "what_it_does": "Removes irrelevant features.", "why_use_it": "Reduce noise and simplify modeling."},
                {"step": "2) Encode Text Columns", "what_it_does": "Converts categorical text to numeric values.", "why_use_it": "Most ML models require numeric inputs."},
                {"step": "2.1) Correlation Check", "what_it_does": "Ranks relationships to selected output/feature.", "why_use_it": "Find strongest predictors and redundancy."},
                {"step": "3) Feature Transformation", "what_it_does": "Rescales/reshapes numeric features.", "why_use_it": "Improve model stability and skewness behavior."},
                {"step": "4) Outlier Preview", "what_it_does": "Shows where extreme values exist.", "why_use_it": "Decide whether transformation is enough."},
                {"step": "5) Class Balancing", "what_it_does": "Makes class counts equal for classification.", "why_use_it": "Reduce target-class bias in training."},
            ]
        )
        st.dataframe(process_guide_df, use_container_width=True, height=280)
        st.caption("Run steps in order: Drop -> Encode -> Correlation -> Transform -> Outlier Check -> Balance (if classification).")

    # --- 1. Drop Columns ---
    with st.expander("1) Drop Columns"):
        st.caption("Use this when columns are IDs, duplicates, leakage, or irrelevant for modeling.")
        all_columns = st.session_state.df.columns.tolist()
        cols_to_drop = st.multiselect("Select columns to drop:", options=all_columns)
        st.markdown("**Preview Before Drop**")
        st.caption(f"Current shape: {st.session_state.df.shape[0]} rows x {st.session_state.df.shape[1]} columns")
        st.dataframe(st.session_state.df.head(20), use_container_width=True, height=260)

        if cols_to_drop:
            preview_after_drop = st.session_state.df.drop(columns=cols_to_drop, errors="ignore")
            st.markdown("**Preview After Drop (Before Apply)**")
            st.caption(
                f"After dropping {len(cols_to_drop)} column(s): "
                f"{preview_after_drop.shape[0]} rows x {preview_after_drop.shape[1]} columns"
            )
            st.dataframe(preview_after_drop.head(20), use_container_width=True, height=260)

        if one_shot_checkbox("Confirm Drop", "confirm_drop_chk"):
            if cols_to_drop:
                before_df = st.session_state.df.copy()
                st.session_state.df.drop(columns=cols_to_drop, inplace=True)
                st.session_state.drop_columns_last = cols_to_drop
                st.session_state.drop_before_preview_df = before_df.head(20).copy()
                st.session_state.drop_after_preview_df = st.session_state.df.head(20).copy()
                st.session_state.drop_before_shape = before_df.shape
                st.session_state.drop_after_shape = st.session_state.df.shape
                st.session_state.df.to_csv("Data_Dropped_Columns.csv", index=False, encoding="utf-8-sig")
                st.success("Columns dropped successfully.")
                st.rerun()

        if "drop_columns_last" in st.session_state:
            st.markdown("**Last Applied Drop Result**")
            st.caption(f"Dropped columns: {st.session_state.get('drop_columns_last', [])}")
            b_shape = st.session_state.get("drop_before_shape", ("?", "?"))
            a_shape = st.session_state.get("drop_after_shape", ("?", "?"))
            st.caption(f"Shape before: {b_shape[0]} x {b_shape[1]} | after: {a_shape[0]} x {a_shape[1]}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("Before (head)")
                st.dataframe(
                    st.session_state.get("drop_before_preview_df", pd.DataFrame()),
                    use_container_width=True,
                    height=260,
                )
            with c2:
                st.write("After (head)")
                st.dataframe(
                    st.session_state.get("drop_after_preview_df", pd.DataFrame()),
                    use_container_width=True,
                    height=260,
                )

    # --- 2. Encoding ---
    with st.expander("2) Encode Text Columns (One-Hot / Label / Target / Binary)"):
        st.caption("Convert text categories to numeric form before training ML models.")
        obj_cols = st.session_state.df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            st.write(f"Available object columns: `{obj_cols}`")
            st.metric("Object Columns Count", len(obj_cols))

            object_summary = pd.DataFrame(
                {
                    "column": obj_cols,
                    "non_null": [int(st.session_state.df[col].notna().sum()) for col in obj_cols],
                    "nulls": [int(st.session_state.df[col].isna().sum()) for col in obj_cols],
                    "unique_values": [int(st.session_state.df[col].nunique(dropna=True)) for col in obj_cols],
                }
            )
            st.dataframe(object_summary, use_container_width=True, height=min(320, 38 * (len(obj_cols) + 1)))

            view_col = st.selectbox("Inspect object column values:", options=obj_cols, key="object_view_col")
            if view_col:
                st.write(f"First values in `{view_col}`:")
                st.dataframe(st.session_state.df[[view_col]].head(20), use_container_width=True, height=260)

                st.write("Value frequencies:")
                value_counts_df = (
                    st.session_state.df[view_col].astype(str).value_counts(dropna=False).reset_index()
                )
                value_counts_df.columns = [view_col, "count"]
                st.dataframe(value_counts_df.head(20), use_container_width=True, height=260)

            method = st.radio(
                "Encoding method:",
                [
                    "One-Hot Encoding",
                    "Label Encoding",
                    "Target Encoding (Mean)",
                    "Binary Encoding",
                ],
            )
            st.markdown("**Encoding Methods Discussion**")
            enc_discussion_df = pd.DataFrame(
                [
                    {
                        "method": "One-Hot Encoding",
                        "why_use_it": "Safe default for nominal categories.",
                        "pro": "No fake ranking between categories.",
                        "con": "Creates many columns for high-cardinality features.",
                    },
                    {
                        "method": "Label Encoding",
                        "why_use_it": "Useful for ordinal categories or tree-based quick baselines.",
                        "pro": "Simple and compact.",
                        "con": "Can imply false numeric order for nominal categories.",
                    },
                    {
                        "method": "Target Encoding (Mean)",
                        "why_use_it": "High-cardinality features (e.g., car model) with numeric target like price.",
                        "pro": "Keeps dataset compact (single encoded column).",
                        "con": "Overfitting risk; use smoothing and optional Leave-One-Out.",
                    },
                    {
                        "method": "Binary Encoding",
                        "why_use_it": "Balanced option for high-cardinality categories.",
                        "pro": "Far fewer columns than One-Hot.",
                        "con": "Less directly interpretable than One-Hot.",
                    },
                ]
            )
            st.dataframe(enc_discussion_df, use_container_width=True, height=230)
            st.caption(
                "Binary Encoding note: for ~1,167 categories, about 11 binary columns can represent all values."
            )
            selected_enc = st.multiselect("Select columns to encode:", options=obj_cols)

            target_enc_target_col = None
            target_enc_smoothing = 20.0
            target_enc_use_loo = False
            if method == "Target Encoding (Mean)":
                all_cols_for_target = st.session_state.df.columns.tolist()
                default_target_idx = len(all_cols_for_target) - 1 if all_cols_for_target else 0
                preferred_target = st.session_state.get("model_output_col")
                if preferred_target in all_cols_for_target:
                    default_target_idx = all_cols_for_target.index(preferred_target)
                target_enc_target_col = st.selectbox(
                    "Target column for mean encoding (numeric target recommended):",
                    options=all_cols_for_target,
                    index=default_target_idx,
                    key="target_enc_target_col",
                )
                target_enc_smoothing = st.slider(
                    "Smoothing strength (higher = less overfitting):",
                    min_value=1.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    key="target_enc_smoothing",
                )
                target_enc_use_loo = st.checkbox(
                    "Use Leave-One-Out style target encoding",
                    value=True,
                    key="target_enc_use_loo",
                )

            output_file_name = st.text_input(
                "Final processed file name (CSV):",
                value=st.session_state.get("encoded_output_file", "final_process_encoded_data.csv"),
            )
            if not output_file_name.strip():
                output_file_name = "final_process_encoded_data.csv"
            if not output_file_name.lower().endswith(".csv"):
                output_file_name = f"{output_file_name}.csv"

            if selected_enc:
                st.write("Sample before encoding:")
                st.dataframe(st.session_state.df[selected_enc].head(10), use_container_width=True, height=230)

                if method == "One-Hot Encoding":
                    onehot_preview = pd.get_dummies(
                        st.session_state.df[selected_enc].head(10),
                        columns=selected_enc,
                        drop_first=True,
                        dtype=int,
                    )
                    st.write("Sample after One-Hot Encoding:")
                    st.dataframe(onehot_preview, use_container_width=True, height=260)
                elif method == "Label Encoding":
                    st.write("Label mapping per selected column:")
                    for col in selected_enc:
                        series_as_str = st.session_state.df[col].astype(str).str.strip().str.lower()
                        if col.lower() == "salary":
                            salary_map = {"low": 0, "medium": 1, "high": 2}
                            mapping_df = pd.DataFrame(
                                {"value": ["low", "medium", "high"], "encoded": [0, 1, 2]}
                            )
                        else:
                            le_preview = LabelEncoder()
                            le_preview.fit(series_as_str)
                            mapping_df = pd.DataFrame(
                                {
                                    "value": le_preview.classes_,
                                    "encoded": range(len(le_preview.classes_)),
                                }
                            )
                        st.write(f"`{col}` mapping")
                        st.dataframe(mapping_df, use_container_width=True, height=220)

                    label_preview = st.session_state.df[selected_enc].head(10).copy()
                    for col in selected_enc:
                        if col.lower() == "salary":
                            salary_map = {"low": 0, "medium": 1, "high": 2}
                            label_preview[col] = (
                                label_preview[col]
                                .astype(str)
                                .str.strip()
                                .str.lower()
                                .map(salary_map)
                            )
                        else:
                            le_preview = LabelEncoder()
                            label_preview[col] = le_preview.fit_transform(
                                label_preview[col].astype(str).str.strip().str.lower()
                            )
                    st.write("Sample after Label Encoding:")
                    st.dataframe(label_preview, use_container_width=True, height=230)
                elif method == "Target Encoding (Mean)":
                    if target_enc_target_col in selected_enc:
                        st.warning("Do not include target column itself in selected encoding columns.")
                    else:
                        te_preview_df = st.session_state.df.copy()
                        y_num = pd.to_numeric(te_preview_df[target_enc_target_col], errors="coerce")
                        if y_num.notna().sum() == 0:
                            st.warning("Target Encoding needs a numeric target (or convertible to numeric).")
                        elif not HAS_CATEGORY_ENCODERS:
                            st.warning("`category_encoders` is not installed. Using internal fallback encoding.")
                            global_mean = float(y_num.mean())
                            sample_te = te_preview_df[selected_enc].head(10).copy()
                            for col in selected_enc:
                                cat = te_preview_df[col].where(te_preview_df[col].notna(), "__missing__").astype(str).str.strip().str.lower()
                                grp = pd.DataFrame({"cat": cat, "y": y_num})
                                agg = grp.groupby("cat")["y"].agg(["mean", "count", "sum"])
                                smooth_map = (
                                    (agg["count"] * agg["mean"] + target_enc_smoothing * global_mean)
                                    / (agg["count"] + target_enc_smoothing)
                                )
                                cat_head = (
                                    te_preview_df[col]
                                    .head(10)
                                    .where(te_preview_df[col].head(10).notna(), "__missing__")
                                    .astype(str)
                                    .str.strip()
                                    .str.lower()
                                )
                                if target_enc_use_loo:
                                    y_head = pd.to_numeric(te_preview_df[target_enc_target_col].head(10), errors="coerce")
                                    loo_vals = []
                                    for idx_row, cat_val in cat_head.items():
                                        if cat_val not in agg.index or pd.isna(y_head.loc[idx_row]):
                                            loo_vals.append(global_mean)
                                            continue
                                        c = float(agg.loc[cat_val, "count"])
                                        s = float(agg.loc[cat_val, "sum"])
                                        yv = float(y_head.loc[idx_row])
                                        denom = (c - 1.0) + target_enc_smoothing
                                        if denom <= 0:
                                            loo_vals.append(global_mean)
                                        else:
                                            loo_vals.append(((s - yv) + target_enc_smoothing * global_mean) / denom)
                                    sample_te[col] = loo_vals
                                else:
                                    sample_te[col] = cat_head.map(smooth_map).fillna(global_mean)
                            st.write("Sample after Target Encoding (Mean):")
                            st.dataframe(sample_te, use_container_width=True, height=230)
                        else:
                            x_te = (
                                te_preview_df[selected_enc]
                                .where(te_preview_df[selected_enc].notna(), "__missing__")
                                .astype(str)
                                .apply(lambda s: s.str.strip().str.lower())
                            )
                            valid_mask = y_num.notna()
                            if valid_mask.sum() == 0:
                                st.warning("Target Encoding needs non-null numeric target values.")
                            else:
                                if target_enc_use_loo:
                                    enc = ce.LeaveOneOutEncoder(cols=selected_enc)
                                else:
                                    enc = ce.TargetEncoder(cols=selected_enc, smoothing=target_enc_smoothing)
                                enc.fit(x_te.loc[valid_mask], y_num.loc[valid_mask])
                                sample_te = enc.transform(x_te).head(10)
                            st.write("Sample after Target Encoding (Mean):")
                            st.dataframe(sample_te, use_container_width=True, height=230)
                else:
                    st.write("Binary encoding summary per selected column:")
                    bin_rows = []
                    sample_binary = pd.DataFrame(index=st.session_state.df.head(10).index)
                    for col in selected_enc:
                        cat = (
                            st.session_state.df[col]
                            .where(st.session_state.df[col].notna(), "__missing__")
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )
                        uniq = sorted(cat.dropna().unique().tolist())
                        code_map = {v: i + 1 for i, v in enumerate(uniq)}
                        bits = max(1, int(np.ceil(np.log2(len(code_map) + 1))))
                        codes_head = cat.head(10).map(code_map).fillna(0).astype(int)
                        codes_head_arr = codes_head.to_numpy(dtype=np.int64)
                        for b in range(bits):
                            sample_binary[f"{col}_bin_{b+1}"] = np.right_shift(codes_head_arr, b) & 1
                        bin_rows.append(
                            {
                                "column": col,
                                "unique_categories": len(code_map),
                                "binary_columns_created": bits,
                            }
                        )
                    st.dataframe(pd.DataFrame(bin_rows), use_container_width=True, height=220)
                    st.write("Sample after Binary Encoding:")
                    st.dataframe(sample_binary, use_container_width=True, height=230)

            if one_shot_checkbox("Apply Encoding", "apply_encoding_chk"):
                if selected_enc:
                    apply_ok = True
                    if method == "One-Hot Encoding":
                        st.session_state.df = pd.get_dummies(
                            st.session_state.df,
                            columns=selected_enc,
                            drop_first=True,
                            dtype=int,
                        )
                    elif method == "Label Encoding":
                        le = LabelEncoder()
                        for col in selected_enc:
                            if col.lower() == "salary":
                                salary_map = {"low": 0, "medium": 1, "high": 2}
                                st.session_state.df[col] = (
                                    st.session_state.df[col]
                                    .astype(str)
                                    .str.strip()
                                    .str.lower()
                                    .map(salary_map)
                                )
                            else:
                                st.session_state.df[col] = le.fit_transform(
                                    st.session_state.df[col].astype(str).str.strip().str.lower()
                                )
                    elif method == "Target Encoding (Mean)":
                        if target_enc_target_col in selected_enc:
                            st.error("Target column cannot be encoded by itself. Remove it from selected columns.")
                            apply_ok = False
                        y_num = pd.to_numeric(st.session_state.df[target_enc_target_col], errors="coerce")
                        if apply_ok and y_num.notna().sum() == 0:
                            st.error("Target Encoding requires numeric target values.")
                            apply_ok = False
                        if apply_ok:
                            x_te = (
                                st.session_state.df[selected_enc]
                                .where(st.session_state.df[selected_enc].notna(), "__missing__")
                                .astype(str)
                                .apply(lambda s: s.str.strip().str.lower())
                            )
                            valid_mask = y_num.notna()
                            if valid_mask.sum() == 0:
                                st.error("Target Encoding needs non-null numeric target values.")
                                apply_ok = False
                            elif HAS_CATEGORY_ENCODERS:
                                if target_enc_use_loo:
                                    enc = ce.LeaveOneOutEncoder(cols=selected_enc)
                                else:
                                    enc = ce.TargetEncoder(cols=selected_enc, smoothing=target_enc_smoothing)
                                enc.fit(x_te.loc[valid_mask], y_num.loc[valid_mask])
                                transformed = enc.transform(x_te)
                                st.session_state.df[selected_enc] = transformed[selected_enc]
                            else:
                                global_mean = float(y_num.mean())
                                for col in selected_enc:
                                    cat = (
                                        st.session_state.df[col]
                                        .where(st.session_state.df[col].notna(), "__missing__")
                                        .astype(str)
                                        .str.strip()
                                        .str.lower()
                                    )
                                    grp = pd.DataFrame({"cat": cat, "y": y_num})
                                    agg = grp.groupby("cat")["y"].agg(["mean", "count", "sum"])
                                    if target_enc_use_loo:
                                        te_vals = []
                                        for idx_row, cat_val in cat.items():
                                            yv = y_num.loc[idx_row]
                                            if cat_val not in agg.index or pd.isna(yv):
                                                te_vals.append(global_mean)
                                                continue
                                            c = float(agg.loc[cat_val, "count"])
                                            s = float(agg.loc[cat_val, "sum"])
                                            denom = (c - 1.0) + target_enc_smoothing
                                            if denom <= 0:
                                                te_vals.append(global_mean)
                                            else:
                                                te_vals.append(((s - float(yv)) + target_enc_smoothing * global_mean) / denom)
                                        st.session_state.df[col] = te_vals
                                    else:
                                        smooth_map = (
                                            (agg["count"] * agg["mean"] + target_enc_smoothing * global_mean)
                                            / (agg["count"] + target_enc_smoothing)
                                        )
                                        st.session_state.df[col] = cat.map(smooth_map).fillna(global_mean)
                    else:
                        for col in selected_enc:
                            cat = (
                                st.session_state.df[col]
                                .where(st.session_state.df[col].notna(), "__missing__")
                                .astype(str)
                                .str.strip()
                                .str.lower()
                            )
                            uniq = sorted(cat.dropna().unique().tolist())
                            code_map = {v: i + 1 for i, v in enumerate(uniq)}
                            bits = max(1, int(np.ceil(np.log2(len(code_map) + 1))))
                            codes = cat.map(code_map).fillna(0).astype(int)
                            codes_arr = codes.to_numpy(dtype=np.int64)
                            for b in range(bits):
                                st.session_state.df[f"{col}_bin_{b+1}"] = np.right_shift(codes_arr, b) & 1
                        st.session_state.df.drop(columns=selected_enc, inplace=True, errors="ignore")

                    if apply_ok:
                        st.session_state.df.to_csv("Data_Encoded.csv", index=False, encoding="utf-8-sig")
                        st.session_state.encoding_applied = True
                        st.session_state.last_encoding_method = method
                        st.session_state.encoded_output_file = output_file_name
                        st.session_state.encoding_result_df = st.session_state.df.copy()
                        st.session_state.show_encoded_preview = False
                        st.success("Encoding applied successfully.")
                        st.rerun()
        else:
            st.write("No object columns found.")

        if st.session_state.get("encoding_applied", False):
            st.markdown("---")
            st.subheader("Encoded Data Result")
            st.caption(f"Method: {st.session_state.get('last_encoding_method', 'Unknown')}")
            if one_shot_checkbox("Load and Preview Encoded Data", "load_encoded_preview_chk"):
                st.session_state.show_encoded_preview = True

            if st.session_state.get("show_encoded_preview", False):
                encoded_preview_df = st.session_state.get("encoding_result_df", st.session_state.df)
                st.dataframe(encoded_preview_df.head(30), use_container_width=True, height=320)
                render_stage_eda(encoded_preview_df, "encoded_preview", "Encoded Preview")
                default_name = st.session_state.get("encoded_output_file", "final_process_encoded_data.csv")
                project_folders = list_project_folders(os.getcwd(), max_depth=4)
                selected_folder = st.selectbox(
                    "Save inside project folder:",
                    options=project_folders,
                    index=0,
                    key="encoded_project_folder",
                )
                custom_name = st.text_input(
                    "File name:",
                    value=default_name,
                    key="encoded_project_file_name",
                ).strip()
                if not custom_name:
                    custom_name = default_name
                if not custom_name.lower().endswith(".csv"):
                    custom_name = f"{custom_name}.csv"

                target_path = os.path.join(os.getcwd(), selected_folder, custom_name)
                st.caption(f"Target path: {target_path}")

                if st.button("Save inside Project", key="save_encoded_to_project_btn"):
                    try:
                        target_dir = os.path.dirname(target_path)
                        if target_dir:
                            os.makedirs(target_dir, exist_ok=True)
                        encoded_save_df = st.session_state.get("encoding_result_df", st.session_state.df)
                        encoded_save_df.to_csv(target_path, index=False, encoding="utf-8-sig")
                        st.success(f"File saved to: {target_path}")
                    except Exception as e:
                        st.error(f"Failed to save file: {e}")

        st.markdown("---")
        with st.expander("Read Before Transformation: Binary vs Continuous Features"):
            st.markdown(
                """
**1) Why not Power-Transform binary features (0/1)?**
- A binary feature has only two values (0 and 1), not a real continuous distribution.
- High skewness in binary flags usually means class imbalance (many 0, few 1), not a shape problem.
- Power/standard/min-max transforms on binary flags often just convert them to decimals and can reduce interpretability.

**2) How to handle skewness in binary features**
- Keep binary flags as `0/1` so the model reads them as Yes/No indicators.
- Handle imbalance at the target/class level with balancing methods such as `SMOTE`, oversampling, or undersampling.

**3) What to transform vs what to keep**
- Transform continuous numeric measurements when needed (scaling or power transform).
- Keep categorical flags in binary form after encoding (for example, one-hot encoded columns and other Yes/No indicator columns).

**Summary**
- Right-skew in binary flags is usually imbalance.
- Use transformation for continuous features.
- Use balancing (including SMOTE) for class imbalance.
"""
            )

        st.markdown("---")
        with st.expander("Optional: Load Dataset Before Transformation / Balance"):
            st.caption(
                "Use this if you want to skip earlier cleaning steps and start directly from "
                "Transformation or Balance."
            )
            upload_col_stage, github_col_stage = st.columns(2)

            with upload_col_stage:
                stage_uploaded_file = st.file_uploader(
                    "Load CSV/Excel from PC",
                    type=["csv", "xlsx", "xls"],
                    key="pre_transform_file_uploader",
                )

            with github_col_stage:
                stage_github_url = st.text_input(
                    "Or paste direct/Raw GitHub file URL",
                    placeholder="https://github.com/user/repo/blob/main/data.csv",
                    key="pre_transform_github_url",
                )

            st.caption("If both are provided, uploaded file is used first.")
            if st.button("Load File for Transformation/Balance", key="pre_transform_load_btn"):
                try:
                    loaded_stage_df = load_dataframe_from_source(stage_uploaded_file, stage_github_url)
                    if loaded_stage_df is None:
                        st.error("Please upload a file or provide a valid GitHub URL.")
                    else:
                        st.session_state.df = loaded_stage_df.copy()
                        st.session_state.df_original = loaded_stage_df.copy()
                        st.session_state.pop("model_output_col", None)
                        st.session_state.pop("model_output_col_prev", None)
                        st.session_state.pop("model_input_cols", None)
                        for k in list(st.session_state.keys()):
                            if str(k).startswith("model_input_chk_"):
                                st.session_state.pop(k, None)
                        for k in [
                            "encoding_applied",
                            "last_encoding_method",
                            "encoding_result_df",
                            "show_encoded_preview",
                            "transformation_applied",
                            "transformation_result_df",
                            "balance_applied",
                            "balance_result_df",
                            "balance_preview_df",
                        ]:
                            st.session_state.pop(k, None)
                        st.success("New dataset loaded. You can now run Transformation and/or Balance directly.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

    # --- 3. Feature Transformation ---
    with st.expander("3) Feature Transformation (Scaling)"):
        transform_num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        if not transform_num_cols:
            st.info("No numeric columns available for transformation.")
        else:
            st.markdown(
                """
**Method Guidance (General):**
- `Power Transform (Yeo-Johnson)`: best when features are strongly non-normal/skewed; reshapes distribution toward normality.
- `Standard Scaling (Z-score)`: best for already fairly symmetric features; useful for linear models and SVM-like models.
- `Min-Max Scaling (0 to 1)`: keeps relative distances in a fixed range; common for neural-network style pipelines.
- `Robust Scaling (Outlier-resistant)`: uses median/IQR; safer when outliers exist and you do not want to drop them.
"""
            )
            st.markdown("**Method Selection Summary (General):**")
            hr_reco_df = pd.DataFrame(
                [
                    {
                        "method": "Power Transform (Yeo-Johnson)",
                        "when_to_use": "Feature is strongly skewed / non-normal",
                        "primary_goal": "Fix skewness (change distribution shape)",
                        "why_not_always": "Unnecessary for already symmetric features.",
                    },
                    {
                        "method": "Standard Scaling (Z-score)",
                        "when_to_use": "Feature is roughly symmetric but on different scale",
                        "primary_goal": "Put features on same scale",
                        "why_not_always": "Sensitive to outliers; does not fix skewness shape.",
                    },
                    {
                        "method": "Min-Max Scaling (0 to 1)",
                        "when_to_use": "Need bounded feature range [0, 1]",
                        "primary_goal": "Boundary control (0 to 1)",
                        "why_not_always": "Strongly affected by extreme outliers.",
                    },
                    {
                        "method": "Robust Scaling (Outlier-resistant)",
                        "when_to_use": "Outliers exist and you want to keep rows",
                        "primary_goal": "Outlier protection",
                        "why_not_always": "Does not normalize shape as strongly as power transform.",
                    },
                ]
            )
            st.dataframe(hr_reco_df, use_container_width=True, height=240)
            st.warning(
                "Important: do not transform binary flag columns (0/1) unless you intentionally want that. "
                "One-hot encoded flag columns are usually already model-ready."
            )
            transform_method = st.radio(
                "Transformation method:",
                [
                    "Min-Max Scaling (0 to 1)",
                    "Standard Scaling (Z-score)",
                    "Robust Scaling (Outlier-resistant)",
                    "Power Transform (Yeo-Johnson)",
                ],
                horizontal=True,
                key="transform_method_radio",
            )
            st.markdown("**Current encoded/working data preview (compact):**")
            st.dataframe(st.session_state.df.head(12), use_container_width=True, height=180)
            skew_threshold = st.slider(
                "Right-skew threshold (skew > threshold):",
                0.3,
                2.5,
                0.5,
                0.1,
                key="transform_skew_threshold",
            )
            skew_values = st.session_state.df[transform_num_cols].skew()
            suggested_cols = [
                c for c in transform_num_cols if float(skew_values.get(c, 0.0)) > float(skew_threshold)
            ]
            binary_cols = []
            for c in transform_num_cols:
                vals = pd.Series(st.session_state.df[c].dropna().unique())
                try:
                    vals = vals.astype(float)
                    if len(vals) > 0 and set(np.round(vals, 10)).issubset({0.0, 1.0}):
                        binary_cols.append(c)
                except Exception:
                    continue

            st.write(f"Suggested right-skewed features: `{suggested_cols}`")
            st.caption("You can select any numeric feature manually, including 0/1 columns if needed.")
            quick_select_mode = st.radio(
                "Select features to transform:",
                ["Suggested", "All", "None", "Manual"],
                horizontal=True,
                key="transform_quick_select_mode",
            )

            prev_mode = st.session_state.get("transform_quick_select_prev_mode")
            if prev_mode != quick_select_mode:
                for c in transform_num_cols:
                    chk_key = f"transform_col_chk_{c}"
                    if quick_select_mode == "Suggested":
                        st.session_state[chk_key] = c in suggested_cols
                    elif quick_select_mode == "All":
                        st.session_state[chk_key] = True
                    elif quick_select_mode == "None":
                        st.session_state[chk_key] = False
                st.session_state["transform_quick_select_prev_mode"] = quick_select_mode

            st.caption("Feature checkboxes:")
            checkbox_cols = st.columns(3)
            for idx, col_name in enumerate(transform_num_cols):
                with checkbox_cols[idx % 3]:
                    st.checkbox(col_name, key=f"transform_col_chk_{col_name}")

            transform_cols = [
                c for c in transform_num_cols if st.session_state.get(f"transform_col_chk_{c}", False)
            ]
            st.caption(f"Detected binary 0/1 features: {binary_cols}")

            st.markdown("**Skewness before transformation (selected features):**")
            if transform_cols:
                skew_selected = pd.DataFrame(
                    {
                        "feature": transform_cols,
                        "skewness": [float(st.session_state.df[c].skew()) for c in transform_cols],
                    }
                )
                skew_selected["abs_skewness"] = skew_selected["skewness"].abs()
                skew_selected["skew_type"] = skew_selected["skewness"].apply(
                    lambda x: "right-skewed" if x > 0.5 else
                    "left-skewed" if x < -0.5 else
                    "approximately symmetric"
                )
                st.dataframe(skew_selected.sort_values("abs_skewness", ascending=False), use_container_width=True, height=240)
            else:
                st.info("Select at least one feature to view skewness values.")

            selected_transform_col = st.selectbox(
                "Preview column:",
                options=transform_cols if transform_cols else transform_num_cols,
                key="transform_preview_col",
            )

            def build_skew_compare_table(df_before: pd.DataFrame, df_after: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                rows = []
                for c in cols:
                    skew_before = float(df_before[c].skew())
                    skew_after = float(df_after[c].skew())
                    abs_before = abs(skew_before)
                    abs_after = abs(skew_after)
                    rows.append(
                        {
                            "feature": c,
                            "skew_before": skew_before,
                            "abs_skew_before": abs_before,
                            "skew_after": skew_after,
                            "abs_skew_after": abs_after,
                            "abs_skew_change": abs_after - abs_before,
                            "improved": abs_after < abs_before,
                        }
                    )
                out = pd.DataFrame(rows)
                return out.sort_values("abs_skew_before", ascending=False).reset_index(drop=True)

            def run_transform(df_in: pd.DataFrame):
                df_out = df_in.copy()
                if not transform_cols:
                    return df_out
                if transform_method == "Min-Max Scaling (0 to 1)":
                    scaler = MinMaxScaler()
                elif transform_method == "Standard Scaling (Z-score)":
                    scaler = StandardScaler()
                elif transform_method == "Robust Scaling (Outlier-resistant)":
                    scaler = RobustScaler()
                else:
                    scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                df_out[transform_cols] = scaler.fit_transform(df_out[transform_cols])
                return df_out

            if one_shot_checkbox("Preview Transformation", "preview_transform_chk"):
                st.session_state.transform_preview_df = run_transform(st.session_state.df)
                st.session_state.transform_preview_method = transform_method
                st.session_state.transform_preview_scope = "user-selected features"
                st.session_state.transform_selected_features = transform_cols

            if "transform_preview_df" in st.session_state:
                st.markdown("---")
                st.write(
                    f"Preview after `{st.session_state.get('transform_preview_method', '')}` "
                    f"on `{st.session_state.get('transform_preview_scope', '')}`"
                )
                tp = st.session_state.transform_preview_df
                tpc1, tpc2 = st.columns(2)
                with tpc1:
                    st.caption("Before")
                    st.dataframe(
                        st.session_state.df[[selected_transform_col]].head(20),
                        use_container_width=True,
                        height=240,
                    )
                with tpc2:
                    st.caption("After (Preview)")
                    st.dataframe(
                        tp[[selected_transform_col]].head(20),
                        use_container_width=True,
                        height=240,
                    )
                if transform_cols:
                    st.markdown("**Skewness check (before vs after preview):**")
                    skew_compare_preview = build_skew_compare_table(st.session_state.df, tp, transform_cols)
                    st.dataframe(skew_compare_preview, use_container_width=True, height=260)

            if one_shot_checkbox("Apply Transformation", "apply_transform_chk"):
                if not transform_cols:
                    st.info("No features selected for transformation.")
                else:
                    before_apply_df = st.session_state.df.copy()
                    transformed_df = run_transform(st.session_state.df)
                    st.session_state.df = transformed_df
                    st.session_state.transformation_applied = True
                    st.session_state.transformation_method = transform_method
                    st.session_state.transformation_scope = "user-selected features"
                    st.session_state.transformation_features = transform_cols
                    st.session_state.transformation_detected_binary = binary_cols
                    st.session_state.transformation_before_df = before_apply_df
                    st.session_state.transformation_result_df = transformed_df.copy()
                    st.session_state.pop("transform_preview_df", None)
                    st.session_state.pop("transform_preview_method", None)
                    st.session_state.pop("transform_preview_scope", None)
                    st.rerun()

            if st.session_state.get("transformation_applied", False):
                st.markdown("---")
                st.subheader("Transformation Result")
                st.caption(
                    f"Applied `{st.session_state.get('transformation_method', '')}` on "
                    f"`{st.session_state.get('transformation_scope', '')}`"
                )
                st.caption(f"Transformed features: {st.session_state.get('transformation_features', [])}")
                st.caption(f"Detected binary 0/1 features: {st.session_state.get('transformation_detected_binary', [])}")
                tr = st.session_state.get("transformation_result_df", st.session_state.df)
                st.write(f"Transformed data rows: {tr.shape[0]} | columns: {tr.shape[1]}")
                st.dataframe(tr.head(30), use_container_width=True, height=320)
                render_stage_eda(tr, "transformed_preview", "Transformation Preview")
                applied_cols = st.session_state.get("transformation_features", [])
                if applied_cols:
                    st.markdown("**Skewness check (before vs after apply):**")
                    before_df = st.session_state.get("transformation_before_df", st.session_state.df_original)
                    skew_compare_applied = build_skew_compare_table(before_df, tr, applied_cols)
                    st.dataframe(skew_compare_applied, use_container_width=True, height=260)
                with st.expander("Show full transformed data table"):
                    st.dataframe(tr, use_container_width=True, height=520)

                default_name = st.session_state.get(
                    "transformation_result_file", "final_process_transformed_StandardScaler_data.csv"
                )
                project_folders = list_project_folders(os.getcwd(), max_depth=4)
                selected_folder = st.selectbox(
                    "Save inside project folder:",
                    options=project_folders,
                    index=0,
                    key="transform_project_folder",
                )
                custom_name = st.text_input(
                    "File name:",
                    value=default_name,
                    key="transform_project_file_name",
                ).strip()
                if not custom_name:
                    custom_name = default_name
                if not custom_name.lower().endswith(".csv"):
                    custom_name = f"{custom_name}.csv"

                target_path = os.path.join(os.getcwd(), selected_folder, custom_name)
                st.caption(f"Target path: {target_path}")
                if st.button("Save inside Project", key="save_transform_to_project_btn"):
                    try:
                        target_dir = os.path.dirname(target_path)
                        if target_dir:
                            os.makedirs(target_dir, exist_ok=True)
                        transform_save_df = st.session_state.get("transformation_result_df", st.session_state.df)
                        transform_save_df.to_csv(target_path, index=False, encoding="utf-8-sig")
                        st.success(f"File saved to: {target_path}")
                    except Exception as e:
                        st.error(f"Failed to save transformed file: {e}")
                        

    # --- 4. Outlier Preview ---
    with st.expander("4) Outlier Preview"):
        st.caption("Preview only: this section helps you inspect extreme values after transformation.")
        num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            outlier_stats = []
            for col in num_cols:
                q1_col, q3_col = st.session_state.df[col].quantile([0.25, 0.75])
                iqr_col = q3_col - q1_col
                low_col, up_col = q1_col - 1.5 * iqr_col, q3_col + 1.5 * iqr_col
                col_mask = (st.session_state.df[col] < low_col) | (st.session_state.df[col] > up_col)
                outlier_stats.append(
                    {
                        "column": col,
                        "outlier_count": int(col_mask.sum()),
                        "marker": "OUTLIERS" if int(col_mask.sum()) > 0 else "",
                    }
                )

            outlier_stats_df = pd.DataFrame(outlier_stats)
            st.write("Outlier summary by numeric column:")
            st.dataframe(outlier_stats_df, use_container_width=True, height=min(320, 38 * (len(outlier_stats_df) + 1)))

            cols_with_outliers = outlier_stats_df.loc[outlier_stats_df["outlier_count"] > 0, "column"].tolist()
            if cols_with_outliers:
                st.caption(f"Columns with detected outliers: {cols_with_outliers}")
            else:
                st.caption("No outliers detected by IQR rule in the current working data.")

            only_outlier_cols = st.checkbox(
                "Show only columns that currently have outliers",
                value=True if cols_with_outliers else False,
                key="only_outlier_cols_chk",
            )
            selectable_cols = cols_with_outliers if (only_outlier_cols and cols_with_outliers) else num_cols
            selected_col = st.selectbox(
                "Select column to view outlier rows:",
                options=selectable_cols,
                key="outlier_preview_col",
            )
            q1, q3 = st.session_state.df[selected_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            st.caption(
                f"IQR limits for `{selected_col}` -> lower: {low:.5f}, upper: {up:.5f}"
            )
            outliers = st.session_state.df[(st.session_state.df[selected_col] < low) | (st.session_state.df[selected_col] > up)]
            st.metric(f"Outliers in {selected_col}", int(outliers.shape[0]))

            with st.expander("Show outlier rows"):
                if outliers.empty:
                    st.info(
                        "No outlier rows for this selected column in the current data "
                        "(based on IQR rule: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR])."
                    )
                else:
                    st.dataframe(outliers, use_container_width=True, height=260)
        else:
            st.info("No numeric columns available for outlier preview.")

    # --- 5. Class Balancing ---
    with st.expander("5) Class Balancing (Complete Balance)"):
        st.caption("Use this step for classification targets to make class counts equal.")
        available_cols = st.session_state.df.columns.tolist()
        target_col = st.session_state.get("model_output_col")
        if target_col not in available_cols and available_cols:
            target_col = available_cols[-1]
        input_cols = [
            c for c in st.session_state.get("model_input_cols", [])
            if c in available_cols and c != target_col
        ]
        if not input_cols:
            input_cols = [c for c in available_cols if c != target_col]

        st.caption(f"Auto-selected output from first selector: `{target_col}`")
        st.caption(f"Auto-selected input features count: `{len(input_cols)}`")

        if not target_col:
            st.warning("Please select an output/target column after the first table preview.")
        else:
            before_counts = st.session_state.df[target_col].value_counts(dropna=False).reset_index()
            before_counts.columns = ["class", "count_before"]
            st.write("Class distribution (before):")
            st.dataframe(before_counts, use_container_width=True, height=240)

            balance_method = st.radio(
                "Balancing method:",
                [
                    "Oversample minority classes to complete balance",
                    "Undersample majority classes to complete balance",
                    "SMOTE (synthetic samples on numeric input features-(but it's a good for Calssification not for Regression))",
                ],
                horizontal=True,
                key="balance_method_radio",
            )

            def build_balanced_df(work_df: pd.DataFrame):
                grouped = [g for _, g in work_df.groupby(target_col, dropna=False)]
                if len(grouped) <= 1:
                    return None, "Need at least 2 classes for balancing.", None, None

                if balance_method.startswith("SMOTE"):
                    if not HAS_SMOTE:
                        return None, "SMOTE requires `imbalanced-learn`. Install it with: pip install imbalanced-learn", None, None
                    smote_input_cols = [c for c in input_cols if c in work_df.columns and c != target_col]
                    if not smote_input_cols:
                        return None, "No valid input features selected for SMOTE.", None, None

                    smote_df = work_df[smote_input_cols + [target_col]].copy()
                    smote_df = smote_df.dropna(subset=[target_col])
                    X_all = smote_df[smote_input_cols]
                    y = smote_df[target_col]
                    numeric_input_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
                    excluded_non_numeric = [c for c in smote_input_cols if c not in numeric_input_cols]

                    if not numeric_input_cols:
                        return None, "SMOTE needs numeric input features. Encode text columns first.", None, None

                    X = X_all[numeric_input_cols].copy()
                    X = X.fillna(X.median(numeric_only=True))
                    y_counts = y.value_counts()
                    min_class_count = int(y_counts.min())
                    if min_class_count < 2:
                        return None, "SMOTE needs at least 2 samples in each class.", None, None

                    k_neighbors = min(5, min_class_count - 1)
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_res, y_res = smote.fit_resample(X, y)
                    balanced_df = pd.DataFrame(X_res, columns=numeric_input_cols)
                    balanced_df[target_col] = y_res.values
                    return balanced_df, None, numeric_input_cols, excluded_non_numeric

                counts = [len(g) for g in grouped]
                if balance_method.startswith("Oversample"):
                    target_n = max(counts)
                    balanced_parts = [
                        resample(g, replace=True, n_samples=target_n, random_state=42)
                        for g in grouped
                    ]
                else:
                    target_n = min(counts)
                    balanced_parts = [
                        resample(g, replace=False, n_samples=target_n, random_state=42)
                        for g in grouped
                    ]

                balanced_df = (
                    pd.concat(balanced_parts, axis=0)
                    .sample(frac=1.0, random_state=42)
                    .reset_index(drop=True)
                )
                return balanced_df, None, [], []

            if one_shot_checkbox("Preview Balance Result", "preview_balance_chk"):
                work_df = st.session_state.df.copy()
                if target_col not in work_df.columns:
                    st.error("Selected output/target column does not exist in current data.")
                else:
                    preview_df, preview_err, smote_used, smote_excluded = build_balanced_df(work_df)
                    if preview_err:
                        st.warning(preview_err)
                    else:
                        st.session_state.balance_preview_df = preview_df
                        st.session_state.balance_preview_target = target_col
                        st.session_state.balance_preview_method = balance_method
                        st.session_state.balance_preview_smote_used = smote_used
                        st.session_state.balance_preview_smote_excluded = smote_excluded

            if "balance_preview_df" in st.session_state:
                st.markdown("**Preview before apply:**")
                bp = st.session_state.balance_preview_df
                bp_target = st.session_state.get("balance_preview_target", target_col)
                bp_counts = bp[bp_target].value_counts(dropna=False).reset_index()
                bp_counts.columns = ["class", "count_preview"]
                st.write("Class distribution (preview):")
                st.dataframe(bp_counts, use_container_width=True, height=220)
                if st.session_state.get("balance_preview_method", "").startswith("SMOTE"):
                    st.caption(f"SMOTE numeric input features used: {st.session_state.get('balance_preview_smote_used', [])}")
                    st.caption(f"SMOTE input features excluded (non-numeric): {st.session_state.get('balance_preview_smote_excluded', [])}")
                st.dataframe(bp.head(30), use_container_width=True, height=260)

            if one_shot_checkbox("Apply Complete Balance", "apply_balance_chk"):
                work_df = st.session_state.df.copy()
                if target_col not in work_df.columns:
                    st.error("Selected output/target column does not exist in current data.")
                else:
                    balanced_df, apply_err, smote_used, smote_excluded = build_balanced_df(work_df)
                    if apply_err:
                        st.warning(apply_err)
                    else:
                        st.session_state.df = balanced_df
                        st.session_state.balance_applied = True
                        st.session_state.balance_target = target_col
                        st.session_state.balance_method = balance_method
                        st.session_state.balance_result_df = balanced_df.copy()
                        st.session_state.balance_smote_used_features = smote_used
                        st.session_state.balance_smote_excluded_features = smote_excluded
                        st.session_state.pop("balance_preview_df", None)
                        st.session_state.pop("balance_preview_target", None)
                        st.session_state.pop("balance_preview_method", None)
                        st.session_state.pop("balance_preview_smote_used", None)
                        st.session_state.pop("balance_preview_smote_excluded", None)
                        st.rerun()

        st.markdown("---")
        with st.expander("Optional: Load Dataset Before Class Balancing Result"):
            st.caption(
                "Use this if you already have a transformed/balanced dataset and want to continue from here."
            )
            bal_up_col, bal_git_col = st.columns(2)
            with bal_up_col:
                balance_uploaded_file = st.file_uploader(
                    "Load CSV/Excel from PC",
                    type=["csv", "xlsx", "xls"],
                    key="pre_balance_file_uploader",
                )
            with bal_git_col:
                balance_github_url = st.text_input(
                    "Or paste direct/Raw GitHub file URL",
                    placeholder="https://github.com/user/repo/blob/main/data.csv",
                    key="pre_balance_github_url",
                )
            if st.button("Load File Before Balance Result", key="pre_balance_load_btn"):
                try:
                    loaded_balance_df = load_dataframe_from_source(balance_uploaded_file, balance_github_url)
                    if loaded_balance_df is None:
                        st.error("Please upload a file or provide a valid GitHub URL.")
                    else:
                        st.session_state.df = loaded_balance_df.copy()
                        st.session_state.df_original = loaded_balance_df.copy()
                        loaded_cols = loaded_balance_df.columns.tolist()
                        loaded_target = None
                        preferred_target = st.session_state.get("model_output_col")
                        if preferred_target in loaded_cols:
                            loaded_target = preferred_target
                        elif loaded_cols:
                            loaded_target = loaded_cols[-1]
                        st.session_state.balance_applied = True
                        st.session_state.balance_result_df = loaded_balance_df.copy()
                        st.session_state.balance_target = loaded_target
                        st.session_state.balance_method = "Loaded from file (skip balance step)"
                        st.session_state.balance_smote_used_features = []
                        st.session_state.balance_smote_excluded_features = []
                        st.success("Dataset loaded. Class Balancing Result is now based on the loaded file.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

        if st.session_state.get("balance_applied", False):
            st.markdown("---")
            st.subheader("Class Balancing Result")
            st.caption(
                f"Applied `{st.session_state.get('balance_method', '')}` on "
                f"`{st.session_state.get('balance_target', '')}`"
            )
            bdf = st.session_state.get("balance_result_df", st.session_state.df)
            if st.session_state.get("balance_method", "").startswith("SMOTE"):
                st.caption(f"SMOTE numeric input features used: {st.session_state.get('balance_smote_used_features', [])}")
                st.caption(f"SMOTE input features excluded (non-numeric): {st.session_state.get('balance_smote_excluded_features', [])}")
            after_counts = bdf[st.session_state.get("balance_target")].value_counts(dropna=False).reset_index()
            after_counts.columns = ["class", "count_after"]
            st.write("Class distribution (after):")
            st.dataframe(after_counts, use_container_width=True, height=240)
            st.dataframe(bdf.head(30), use_container_width=True, height=300)
            render_stage_eda(bdf, "balanced_preview", "Balanced Preview")

            default_name = st.session_state.get("balance_output_file", "final_process_balanced_data.csv")
            project_folders = list_project_folders(os.getcwd(), max_depth=4)
            selected_folder = st.selectbox(
                "Save inside project folder:",
                options=project_folders,
                index=0,
                key="balance_project_folder",
            )
            custom_name = st.text_input(
                "File name:",
                value=default_name,
                key="balance_project_file_name",
            ).strip()
            if not custom_name:
                custom_name = default_name
            if not custom_name.lower().endswith(".csv"):
                custom_name = f"{custom_name}.csv"

            st.session_state.balance_output_file = custom_name
            target_path = os.path.join(os.getcwd(), selected_folder, custom_name)
            st.caption(f"Target path: {target_path}")
            if st.button("Save inside Project", key="save_balance_to_project_btn"):
                try:
                    target_dir = os.path.dirname(target_path)
                    if target_dir:
                        os.makedirs(target_dir, exist_ok=True)
                    balance_save_df = st.session_state.get("balance_result_df", st.session_state.df)
                    balance_save_df.to_csv(target_path, index=False, encoding="utf-8-sig")
                    st.success(f"File saved to: {target_path}")
                except Exception as e:
                    st.error(f"Failed to save balanced file: {e}")

            st.markdown("---")
            st.subheader("Next Step Hint")
            st.caption(
                "Data is now ready for splitting and training. Since balancing is already applied, "
                "you can use a standard Train-Test split."
            )
            hint_df = pd.DataFrame(
                [
                    {
                        "model": "Logistic Regression",
                        "category": "Linear",
                        "when_to_choose": "When interpretability and simplicity are priority.",
                        "why": "Fast, easy to explain, and strong baseline for classification.",
                    },
                    {
                        "model": "Decision Tree",
                        "category": "Tree-based",
                        "when_to_choose": "When you want rule-like decisions and easy visualization.",
                        "why": "Simple to interpret but can overfit without tuning.",
                    },
                    {
                        "model": "Random Forest",
                        "category": "Ensemble Trees",
                        "when_to_choose": "Strong baseline for mixed numeric + binary tabular data.",
                        "why": "Handles non-linear interactions well.",
                    },
                    {
                        "model": "Extra Trees",
                        "category": "Ensemble Trees",
                        "when_to_choose": "When you want a fast robust tree ensemble baseline.",
                        "why": "Often similar to Random Forest with different randomization.",
                    },
                    {
                        "model": "Gradient Boosting",
                        "category": "Boosting",
                        "when_to_choose": "When you want stronger performance than single trees.",
                        "why": "Builds trees sequentially to reduce residual errors.",
                    },
                    {
                        "model": "XGBoost",
                        "category": "Boosting",
                        "when_to_choose": "When you want top predictive accuracy on tabular data.",
                        "why": "Often achieves the best performance after tuning.",
                    },
                    {
                        "model": "LightGBM / CatBoost",
                        "category": "Boosting",
                        "when_to_choose": "When handling large tabular datasets or many categorical patterns.",
                        "why": "Efficient gradient-boosting alternatives with strong performance.",
                    },
                    {
                        "model": "SVM (Linear/RBF)",
                        "category": "Kernel-based",
                        "when_to_choose": "When feature space boundaries are complex and data size is moderate.",
                        "why": "Can model non-linear separation with kernels.",
                    },
                    {
                        "model": "KNN",
                        "category": "Distance-based",
                        "when_to_choose": "When local neighborhood structure is important.",
                        "why": "Simple non-parametric baseline; sensitive to scaling/noise.",
                    },
                    {
                        "model": "Naive Bayes",
                        "category": "Probabilistic",
                        "when_to_choose": "When you need a very fast baseline on high-dimensional data.",
                        "why": "Simple assumptions, fast training and inference.",
                    },
                    {
                        "model": "MLP (Neural Network)",
                        "category": "Neural",
                        "when_to_choose": "When you want to test deep non-linear patterns.",
                        "why": "Can capture complex structure but needs careful tuning.",
                    },
                ]
            )
            st.dataframe(hint_df, use_container_width=True, height=320)
            st.info("Go to the Model Training page to split the data and train models.")

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def find_default_sheet(sheet_names, keywords):
    lower_map = {name.lower(): name for name in sheet_names}
    for key in keywords:
        for low, original in lower_map.items():
            if key in low:
                return original
    return sheet_names[0] if sheet_names else None


def load_excel_three_tables(uploaded_excel):
    xls = pd.ExcelFile(uploaded_excel)
    sheets = xls.sheet_names

    st.sidebar.write("Map workbook sheets to tables:")
    main_default = find_default_sheet(sheets, ["main", "original", "raw"])
    enc_default = find_default_sheet(sheets, ["encod", "encoded"])
    out_default = find_default_sheet(sheets, ["outlier", "clean", "treated"])

    main_sheet = st.sidebar.selectbox(
        "Main table sheet",
        options=sheets,
        index=sheets.index(main_default) if main_default in sheets else 0,
        key="main_sheet_pick",
    )
    encoded_sheet = st.sidebar.selectbox(
        "Encoded table sheet",
        options=sheets,
        index=sheets.index(enc_default) if enc_default in sheets else 0,
        key="encoded_sheet_pick",
    )
    outliers_sheet = st.sidebar.selectbox(
        "Outliers table sheet",
        options=sheets,
        index=sheets.index(out_default) if out_default in sheets else 0,
        key="outliers_sheet_pick",
    )

    return {
        "Main Table": pd.read_excel(uploaded_excel, sheet_name=main_sheet),
        "Encoded Table": pd.read_excel(uploaded_excel, sheet_name=encoded_sheet),
        "Outliers Table": pd.read_excel(uploaded_excel, sheet_name=outliers_sheet),
    }


def render_eda_for_table(df: pd.DataFrame, table_name: str):
    st.subheader(f"{table_name} - EDA")
    st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    with st.expander("1) Full Data Preview"):
        st.dataframe(df, use_container_width=True, height=320)

    st.markdown("---")
    numeric_df = df.select_dtypes(include=["number"])

    st.subheader("2) Correlation Heatmap")
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(max(10, numeric_df.shape[1] * 0.8), max(6, numeric_df.shape[1] * 0.6)))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.4)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        st.info("Need at least 2 numeric columns for heatmap.")

    st.markdown("---")
    st.subheader("3) Scatter Plot Matrix")
    if numeric_df.shape[1] < 2:
        st.info("Need at least 2 numeric columns for a scatter plot matrix.")
    else:
        scatter_mode = st.radio(
            "Scatter matrix mode:",
            ["All numeric features", "Selected features", "One feature vs others"],
            horizontal=True,
            key=f"scatter_mode_{table_name}",
        )
        numeric_cols = numeric_df.columns.tolist()
        plot_cols = []

        if scatter_mode == "All numeric features":
            max_cols = min(8, len(numeric_cols))
            if max_cols <= 2:
                use_cols_count = 2
                st.caption("Using 2 features (available numeric features).")
            else:
                use_cols_count = st.slider(
                    "Number of numeric features to include",
                    min_value=2,
                    max_value=max_cols,
                    value=min(5, max_cols),
                    key=f"scatter_all_count_{table_name}",
                )
            plot_cols = numeric_cols[:use_cols_count]
        elif scatter_mode == "Selected features":
            selected_cols = st.multiselect(
                "Select numeric features (min 2)",
                options=numeric_cols,
                default=numeric_cols[: min(4, len(numeric_cols))],
                key=f"scatter_selected_cols_{table_name}",
            )
            plot_cols = selected_cols
        else:
            focus_col = st.selectbox(
                "Focus feature",
                options=numeric_cols,
                key=f"scatter_focus_col_{table_name}",
            )
            compare_candidates = [c for c in numeric_cols if c != focus_col]
            compare_cols = st.multiselect(
                "Compare focus feature with",
                options=compare_candidates,
                default=compare_candidates[: min(3, len(compare_candidates))],
                key=f"scatter_compare_cols_{table_name}",
            )
            plot_cols = [focus_col] + compare_cols

        if len(plot_cols) < 2:
            st.info("Please select at least 2 features.")
        else:
            max_rows = min(3000, len(df))
            min_rows = min(200, len(df))
            if max_rows <= min_rows:
                sample_rows = max_rows
                st.caption(f"Using all available rows: {sample_rows}")
            else:
                sample_rows = st.slider(
                    "Rows used for matrix (sampling for speed)",
                    min_value=min_rows,
                    max_value=max_rows,
                    value=min(1000, max_rows),
                    key=f"scatter_rows_{table_name}",
                )
            plot_df = df[plot_cols].dropna()
            if len(plot_df) > sample_rows:
                plot_df = plot_df.sample(sample_rows, random_state=42)
            if plot_df.empty:
                st.info("No rows left after dropping missing values for selected features.")
            else:
                g = sns.pairplot(plot_df, corner=True, diag_kind="hist")
                st.pyplot(g.fig)
                plt.close(g.fig)

    st.markdown("---")
    st.subheader("4) Skewness Plots")
    if numeric_df.shape[1] == 0:
        st.info("No numeric columns available for skew plots.")
    else:
        skew_mode = st.radio(
            "Skew plot mode:",
            ["One feature", "All numeric features"],
            horizontal=True,
            key=f"skew_mode_{table_name}",
        )
        if skew_mode == "One feature":
            skew_col = st.selectbox(
                "Select feature for skew plot",
                options=numeric_df.columns.tolist(),
                key=f"skew_col_{table_name}",
            )
            fig_sk, ax_sk = plt.subplots(figsize=(10, 5))
            sns.histplot(numeric_df[skew_col].dropna(), kde=True, ax=ax_sk, color="#ff7f50")
            ax_sk.set_title(f"Skew Plot - {skew_col} (skew={numeric_df[skew_col].skew():.3f})")
            st.pyplot(fig_sk)
        else:
            for col in numeric_df.columns:
                fig_sk, ax_sk = plt.subplots(figsize=(10, 4))
                sns.histplot(numeric_df[col].dropna(), kde=True, ax=ax_sk, color="#4e79a7")
                ax_sk.set_title(f"Skew Plot - {col} (skew={numeric_df[col].skew():.3f})")
                st.pyplot(fig_sk)

    st.markdown("---")
    st.subheader("5) General Statistics")
    t1, t2, t3 = st.tabs(["Describe", "Unique Values", "Missing Values"])
    with t1:
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns.")
    with t2:
        st.dataframe(df.nunique().to_frame("unique_count"), use_container_width=True)
    with t3:
        st.dataframe(df.isnull().sum().to_frame("missing_count"), use_container_width=True)

    st.markdown("---")
    st.subheader("6) Value Frequency")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        freq_col = st.selectbox(
            "Select column",
            options=df.columns.tolist(),
            key=f"freq_col_{table_name}",
        )
        top_n = st.slider("Top N", 5, 30, 10, key=f"topn_{table_name}")

    vc = df[freq_col].astype(str).value_counts().head(top_n)
    with col_a:
        st.dataframe(vc.to_frame("count"), use_container_width=True)
    with col_b:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=vc.values, y=vc.index, ax=ax2, palette="viridis")
        ax2.set_title(f"Top {top_n} values - {freq_col}")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("7) Distribution + Outlier Check")
    if numeric_df.shape[1] == 0:
        st.info("No numeric columns for distribution/outlier analysis.")
        return

    target_col = st.selectbox(
        "Select numeric column",
        options=numeric_df.columns.tolist(),
        key=f"target_num_{table_name}",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        sns.histplot(df[target_col].dropna(), kde=True, element="step", fill=False, linewidth=2, ax=ax3)
        ax3.set_title(f"Distribution - {target_col}")
        st.pyplot(fig3)

    with c2:
        q1 = df[target_col].quantile(0.25)
        q3 = df[target_col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr
        outliers = df[(df[target_col] < low) | (df[target_col] > up)]

        st.metric(f"Outliers in {target_col}", int(outliers.shape[0]))
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        sns.boxplot(y=df[target_col], ax=ax4, color="#ff6b6b")
        ax4.set_title(f"Box Plot - {target_col}")
        st.pyplot(fig4)

st.sidebar.header("Data Source")
source_mode = st.sidebar.radio("Load mode", ["Excel workbook (3 tables)", "Three CSV files"], index=0)

loaded_tables = {}
if "eda_loaded_tables" not in st.session_state:
    st.session_state.eda_loaded_tables = None
if "eda_loaded_mode" not in st.session_state:
    st.session_state.eda_loaded_mode = None

reuse_loaded = st.sidebar.checkbox("Reuse previously loaded tables", value=True, key="eda_reuse_loaded")
clear_loaded = st.sidebar.checkbox("Clear loaded tables", value=False, key="eda_clear_loaded")
if clear_loaded:
    st.session_state.eda_loaded_tables = None
    st.session_state.eda_loaded_mode = None
    st.session_state.eda_clear_loaded = False

if (
    reuse_loaded
    and st.session_state.eda_loaded_tables is not None
    and st.session_state.eda_loaded_mode == source_mode
):
    loaded_tables = st.session_state.eda_loaded_tables
    st.info("Using previously loaded tables from this session.")

if not loaded_tables and source_mode == "Excel workbook (3 tables)":
    uploaded_excel = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="eda_excel_upload")
    if uploaded_excel is not None:
        try:
            loaded_tables = load_excel_three_tables(uploaded_excel)
            st.session_state.eda_loaded_tables = loaded_tables
            st.session_state.eda_loaded_mode = source_mode
            st.success("Excel loaded successfully. All three tables are ready.")
        except Exception as e:
            st.error(f"Failed to read Excel workbook: {e}")
elif not loaded_tables:
    main_csv = st.sidebar.file_uploader("Upload Main table CSV", type=["csv"], key="eda_main_csv")
    encoded_csv = st.sidebar.file_uploader("Upload Encoded table CSV", type=["csv"], key="eda_encoded_csv")
    outliers_csv = st.sidebar.file_uploader("Upload Outliers table CSV", type=["csv"], key="eda_outliers_csv")
    if main_csv is not None and encoded_csv is not None and outliers_csv is not None:
        try:
            loaded_tables = {
                "Main Table": pd.read_csv(main_csv),
                "Encoded Table": pd.read_csv(encoded_csv),
                "Outliers Table": pd.read_csv(outliers_csv),
            }
            st.session_state.eda_loaded_tables = loaded_tables
            st.session_state.eda_loaded_mode = source_mode
            st.success("All three CSV tables loaded successfully.")
        except Exception as e:
            st.error(f"Failed to read CSV files: {e}")

if loaded_tables:
    st.markdown("---")
    tabs = st.tabs(list(loaded_tables.keys()))

import os
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


st.markdown("---")
st.header("Step 3: Model_Training")
st.caption("Run this after Step 2 (EDA_Data).")


def normalize_github_url(url: str) -> str:
    url = (url or "").strip()
    if "github.com" in url and "/blob/" in url:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return url


def load_dataframe_from_source(uploaded_file, github_url: str):
    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file, encoding="utf-8", encoding_errors="ignore")

    if github_url and github_url.strip():
        raw_url = normalize_github_url(github_url)
        if raw_url.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(raw_url)
        return pd.read_csv(raw_url, encoding="utf-8", encoding_errors="ignore")

    return None


def list_project_folders(base_path: str, max_depth: int = 3):
    folders = ["."]
    base_path = os.path.abspath(base_path)
    base_depth = base_path.rstrip(os.sep).count(os.sep)
    for root, dirnames, _ in os.walk(base_path):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in {"venv", "__pycache__", ".git"}
        ]
        for d in dirnames:
            rel = os.path.relpath(os.path.join(root, d), base_path)
            folders.append(rel)
    return sorted(set(folders))


def model_discussion_table(problem_type: str) -> pd.DataFrame:
    if problem_type == "Classification":
        return pd.DataFrame(
            [
                {"model": "Decision Tree", "best_for": "Simple rules", "notes": "Easy to interpret, can overfit."},
                {"model": "Random Forest", "best_for": "Strong baseline", "notes": "Robust and handles non-linearity."},
                {"model": "Extra Trees", "best_for": "Fast ensemble", "notes": "High variance reduction with random splits."},
                {"model": "Gradient Boosting", "best_for": "Accuracy-focused", "notes": "Strong performance with tuning."},
                {"model": "AdaBoost", "best_for": "Weak learners boost", "notes": "Good on clean tabular datasets."},
                {"model": "Logistic Regression", "best_for": "Interpretability", "notes": "Linear decision boundary baseline."},
                {"model": "SVM (RBF)", "best_for": "Complex boundaries", "notes": "Needs scaling; slower on large data."},
                {"model": "KNN", "best_for": "Local patterns", "notes": "Needs scaling; sensitive to noise."},
                {"model": "Naive Bayes", "best_for": "Very fast baseline", "notes": "Strong assumptions, fast training."},
                {"model": "MLP (Neural Network)", "best_for": "Non-linear structure", "notes": "Needs tuning and scaling."},
                {"model": "XGBoost", "best_for": "Top tabular accuracy", "notes": "Often best after tuning; optional package."},
            ]
        )
    return pd.DataFrame(
        [
            {"model": "Decision Tree", "best_for": "Simple non-linear rules", "notes": "Interpretable, can overfit."},
            {"model": "Random Forest", "best_for": "Robust regression baseline", "notes": "Handles interactions well."},
            {"model": "Extra Trees", "best_for": "Fast robust ensemble", "notes": "Similar to RF with more randomization."},
            {"model": "Gradient Boosting", "best_for": "Lower error targets", "notes": "Strong but can overfit if untuned."},
            {"model": "AdaBoost", "best_for": "Boosted weak regressors", "notes": "Simple boosting baseline."},
            {"model": "Linear Regression", "best_for": "Linear relationships", "notes": "Interpretable and fast."},
            {"model": "SVR (RBF)", "best_for": "Complex smooth functions", "notes": "Needs scaling; can be slow."},
            {"model": "KNN", "best_for": "Local regression trends", "notes": "Needs scaling; weak in high dimensions."},
            {"model": "MLP (Neural Network)", "best_for": "Non-linear function fit", "notes": "Needs tuning and scaling."},
            {"model": "XGBoost", "best_for": "High-performance tabular regression", "notes": "Often strong with tuning."},
        ]
    )


def build_pipeline(problem_type: str, model_name: str):
    needs_scaling = model_name in {
        "Logistic Regression",
        "SVM (RBF)",
        "KNN",
        "MLP (Neural Network)",
        "Linear Regression",
        "SVR (RBF)",
    }
    steps = [("scaler", StandardScaler())] if needs_scaling else []

    if problem_type == "Classification":
        if model_name == "Decision Tree":
            estimator = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            estimator = RandomForestClassifier(random_state=42)
        elif model_name == "Extra Trees":
            estimator = ExtraTreesClassifier(random_state=42)
        elif model_name == "Gradient Boosting":
            estimator = GradientBoostingClassifier(random_state=42)
        elif model_name == "AdaBoost":
            estimator = AdaBoostClassifier(random_state=42)
        elif model_name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=2000)
        elif model_name == "SVM (RBF)":
            estimator = SVC(kernel="rbf", probability=False)
        elif model_name == "KNN":
            estimator = KNeighborsClassifier()
        elif model_name == "Naive Bayes":
            estimator = GaussianNB()
        elif model_name == "MLP (Neural Network)":
            estimator = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        else:
            if not HAS_XGBOOST:
                raise ValueError("XGBoost is not installed. Run: pip install xgboost")
            estimator = XGBClassifier(random_state=42, eval_metric="logloss")
    else:
        if model_name == "Decision Tree":
            estimator = DecisionTreeRegressor(random_state=42)
        elif model_name == "Random Forest":
            estimator = RandomForestRegressor(random_state=42)
        elif model_name == "Extra Trees":
            estimator = ExtraTreesRegressor(random_state=42)
        elif model_name == "Gradient Boosting":
            estimator = GradientBoostingRegressor(random_state=42)
        elif model_name == "AdaBoost":
            estimator = AdaBoostRegressor(random_state=42)
        elif model_name == "Linear Regression":
            estimator = LinearRegression()
        elif model_name == "SVR (RBF)":
            estimator = SVR(kernel="rbf")
        elif model_name == "KNN":
            estimator = KNeighborsRegressor()
        elif model_name == "MLP (Neural Network)":
            estimator = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        else:
            if not HAS_XGBOOST:
                raise ValueError("XGBoost is not installed. Run: pip install xgboost")
            estimator = XGBRegressor(random_state=42)

    steps.append(("model", estimator))
    return Pipeline(steps)


def train_live_model(df: pd.DataFrame, input_cols: list[str], target_col: str, problem_type: str, model_name: str):
    X_raw = df[input_cols].copy()
    y = df[target_col].copy()
    X_encoded = pd.get_dummies(X_raw, drop_first=True, dtype=int)
    pipeline = build_pipeline(problem_type, model_name)

    label_encoder = None
    if problem_type == "Classification" and model_name == "XGBoost":
        label_encoder = LabelEncoder()
        y_fit = label_encoder.fit_transform(y.astype(str))
        pipeline.fit(X_encoded, y_fit)
    else:
        pipeline.fit(X_encoded, y)

    return {
        "pipeline": pipeline,
        "train_columns": X_encoded.columns.tolist(),
        "label_encoder": label_encoder,
    }


def show_swal_popup(title: str, text: str, icon: str = "success"):
    title_js = json.dumps(title)
    text_js = json.dumps(text)
    icon_js = json.dumps(icon)
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
      Swal.fire({{
        title: {title_js},
        text: {text_js},
        icon: {icon_js},
        confirmButtonText: "OK"
      }});
    </script>
    """
    components.html(html, height=0)


def format_salary_label(values) -> pd.Series:
    s = pd.Series(values).astype(str).str.strip().str.lower()
    label_to_code = {"low": 0, "medium": 1, "high": 2}
    code_to_label = {0: "low", 1: "medium", 2: "high"}
    numeric = pd.to_numeric(s, errors="coerce")
    out = []
    for raw, num in zip(s.tolist(), numeric.tolist()):
        if pd.notna(num):
            code = int(round(float(num)))
            label = code_to_label.get(code)
            out.append(label if label is not None else str(raw))
        elif raw in label_to_code:
            out.append(raw)
        else:
            out.append(str(raw))
    return pd.Series(out)


# ------------------------------------------------------------------
# 1) Load data (PC upload or GitHub)
# ------------------------------------------------------------------
st.subheader("1) Load Dataset")
source_mode = st.radio(
    "Data source:",
    ["Use current cleaned dataset", "Upload from PC", "Load from GitHub URL"],
    horizontal=True,
)

if source_mode == "Use current cleaned dataset":
    if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame):
        st.session_state.mt_df = st.session_state.df.copy()
        st.success("Loaded current cleaned dataset from session.")
    else:
        st.warning("No cleaned dataset found in session. Upload from PC or GitHub URL.")

elif source_mode == "Upload from PC":
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="mt_file_upload")
    if uploaded_file is not None and st.button("Load Uploaded File", key="mt_load_upload_btn"):
        try:
            st.session_state.mt_df = load_dataframe_from_source(uploaded_file, "")
            st.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to load uploaded file: {e}")

else:
    github_url = st.text_input("GitHub raw URL (CSV/Excel):", key="mt_github_url")
    if st.button("Load from GitHub", key="mt_load_github_btn"):
        try:
            df_github = load_dataframe_from_source(None, github_url)
            if df_github is None:
                st.warning("Please provide a valid URL.")
            else:
                st.session_state.mt_df = df_github
                st.success("Loaded dataset from GitHub URL.")
        except Exception as e:
            st.error(f"Failed to load from GitHub: {e}")


if "mt_df" in st.session_state and isinstance(st.session_state.mt_df, pd.DataFrame):
    df = st.session_state.mt_df.copy()

    st.markdown("---")
    st.subheader("2) Data Preview and X / y Selection")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head(30), use_container_width=True, height=320)

    all_cols = df.columns.tolist()
    default_target_idx = len(all_cols) - 1 if all_cols else 0
    target_col = st.selectbox(
        "Select output/target column (y):",
        options=all_cols,
        index=default_target_idx,
        key="mt_target_col",
    )

    input_cols = [c for c in all_cols if c != target_col]
    st.session_state.mt_input_cols = input_cols
    st.caption(f"Input features are auto-selected from all columns except `{target_col}`.")
    st.caption(f"Input features count: {len(input_cols)}")

    if not input_cols:
        st.warning("Select at least one input feature column.")
    else:
        X = df[input_cols].copy()
        y = df[target_col].copy()

        st.markdown("---")
        st.subheader("3) Split Data")
        test_size = st.slider("Test size:", 0.05, 0.4, 0.2, 0.05)
        val_size = st.slider("Validation size:", 0.05, 0.3, 0.1, 0.05)
        split_shuffle = st.checkbox("Shuffle before split", value=True, key="mt_split_shuffle")
        split_seed = st.number_input(
            "Random seed:",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            key="mt_split_seed",
        )
        train_size = 1.0 - test_size - val_size

        if train_size <= 0:
            st.error("Train size must be > 0. Reduce test/validation sizes.")
        else:
            st.caption(f"Split plan: Train={train_size:.0%}, Validation={val_size:.0%}, Test={test_size:.0%}")

            if st.button("Run Split", key="mt_run_split_btn"):
                try:
                    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
                    holdout_size = test_size + val_size
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_encoded,
                        y,
                        test_size=holdout_size,
                        random_state=int(split_seed),
                        shuffle=bool(split_shuffle),
                    )
                    test_ratio_in_temp = test_size / holdout_size
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp,
                        y_temp,
                        test_size=test_ratio_in_temp,
                        random_state=int(split_seed),
                        shuffle=bool(split_shuffle),
                    )

                    st.session_state.mt_split = {
                        "X_train": X_train,
                        "X_val": X_val,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_val": y_val,
                        "y_test": y_test,
                        "target_col": target_col,
                    }
                    st.success("Split completed.")
                except Exception as e:
                    st.error(f"Failed to split data: {e}")

        if "mt_split" in st.session_state:
            split = st.session_state.mt_split
            st.markdown("---")
            st.subheader("Split Preview")
            st.write(
                f"X_train: {split['X_train'].shape} | X_val: {split['X_val'].shape} | X_test: {split['X_test'].shape}"
            )
            t1, t2, t3 = st.tabs(["Train", "Validation", "Test"])
            with t1:
                st.dataframe(split["X_train"].head(10), use_container_width=True, height=220)
                st.dataframe(split["y_train"].head(10).to_frame(name=split["target_col"]), use_container_width=True, height=160)
            with t2:
                st.dataframe(split["X_val"].head(10), use_container_width=True, height=220)
                st.dataframe(split["y_val"].head(10).to_frame(name=split["target_col"]), use_container_width=True, height=160)
            with t3:
                st.dataframe(split["X_test"].head(10), use_container_width=True, height=220)
                st.dataframe(split["y_test"].head(10).to_frame(name=split["target_col"]), use_container_width=True, height=160)

            st.markdown("---")
            st.subheader("Save Split Files Inside Project")
            project_folders = list_project_folders(os.getcwd(), max_depth=4)
            selected_folder = st.selectbox(
                "Save inside project folder:",
                options=project_folders,
                index=0,
                key="mt_split_folder",
            )
            file_prefix = st.text_input("File name prefix:", value="final_process_split", key="mt_split_prefix").strip()
            if not file_prefix:
                file_prefix = "final_process_split"

            split_file_map = {
                "X_train": f"{file_prefix}_X_train.csv",
                "X_val": f"{file_prefix}_X_val.csv",
                "X_test": f"{file_prefix}_X_test.csv",
                "y_train": f"{file_prefix}_y_train.csv",
                "y_val": f"{file_prefix}_y_val.csv",
                "y_test": f"{file_prefix}_y_test.csv",
            }
            for key_name, file_name in split_file_map.items():
                st.caption(f"{key_name} -> {os.path.join(os.getcwd(), selected_folder, file_name)}")

            if st.button("Save inside Project", key="mt_save_split_btn"):
                try:
                    target_dir = os.path.join(os.getcwd(), selected_folder)
                    os.makedirs(target_dir, exist_ok=True)
                    for key_name, df_part in split.items():
                        if key_name in split_file_map:
                            out_path = os.path.join(target_dir, split_file_map[key_name])
                            if key_name.startswith("y_"):
                                df_part.to_frame(name=split["target_col"]).to_csv(out_path, index=False, encoding="utf-8-sig")
                            else:
                                df_part.to_csv(out_path, index=False, encoding="utf-8-sig")
                    st.success(f"Split files saved to: {target_dir}")
                except Exception as e:
                    st.error(f"Failed to save split files: {e}")

            st.markdown("---")
            st.subheader("4) Train Model")

            y_train = split["y_train"]
            classification_auto = (y_train.dtype == "object") or (y_train.nunique(dropna=True) <= 20)
            problem_type = st.radio(
                "Problem type:",
                ["Classification", "Regression"],
                index=0 if classification_auto else 1,
                horizontal=True,
            )

            if problem_type == "Classification":
                model_options = [
                    "Decision Tree",
                    "Random Forest",
                    "Extra Trees",
                    "Gradient Boosting",
                    "AdaBoost",
                    "Logistic Regression",
                    "SVM (RBF)",
                    "KNN",
                    "Naive Bayes",
                    "MLP (Neural Network)",
                    "XGBoost",
                ]
            else:
                model_options = [
                    "Decision Tree",
                    "Random Forest",
                    "Extra Trees",
                    "Gradient Boosting",
                    "AdaBoost",
                    "Linear Regression",
                    "SVR (RBF)",
                    "KNN",
                    "MLP (Neural Network)",
                    "XGBoost",
                ]

            st.markdown("**Model Discussion (when to use each):**")
            st.dataframe(model_discussion_table(problem_type), use_container_width=True, height=280)
            st.caption("All models are trained through a scikit-learn Pipeline in this page.")

            st.markdown("**Select models to evaluate:**")
            select_mode = st.radio(
                "Selection mode:",
                ["All", "None", "Manual"],
                horizontal=True,
                key="mt_model_select_mode",
            )
            prev_select_mode = st.session_state.get("mt_model_select_mode_prev")
            if prev_select_mode != select_mode:
                for m in model_options:
                    key = f"mt_model_chk_{problem_type}_{m}"
                    if select_mode == "All":
                        st.session_state[key] = True
                    elif select_mode == "None":
                        st.session_state[key] = False
                st.session_state["mt_model_select_mode_prev"] = select_mode

            model_cols = st.columns(3)
            for idx, m in enumerate(model_options):
                with model_cols[idx % 3]:
                    st.checkbox(m, key=f"mt_model_chk_{problem_type}_{m}")

            selected_models = [
                m for m in model_options
                if st.session_state.get(f"mt_model_chk_{problem_type}_{m}", False)
            ]
            st.caption(f"Selected models: {len(selected_models)}")

            if st.button("Evaluate Selected Models", key="mt_train_btn"):
                try:
                    if not selected_models:
                        st.warning("Select at least one model.")
                        st.stop()

                    X_train = split["X_train"]
                    X_test = split["X_test"]
                    y_train = split["y_train"]
                    y_test = split["y_test"]
                    target_name = str(split.get("target_col", "")).strip().lower()
                    results = []
                    predictions_frames = []

                    for model_name in selected_models:
                        try:
                            pipeline = build_pipeline(problem_type, model_name)
                            if problem_type == "Classification":
                                if model_name == "XGBoost":
                                    le = LabelEncoder()
                                    y_train_fit = le.fit_transform(y_train.astype(str))
                                    y_test_eval = le.transform(y_test.astype(str))
                                    pipeline.fit(X_train, y_train_fit)
                                    train_preds_enc = pipeline.predict(X_train)
                                    test_preds_enc = pipeline.predict(X_test)
                                    train_preds = le.inverse_transform(train_preds_enc.astype(int))
                                    test_preds = le.inverse_transform(test_preds_enc.astype(int))
                                    train_acc = accuracy_score(y_train.astype(str), train_preds.astype(str))
                                    test_acc = accuracy_score(y_test.astype(str), test_preds.astype(str))
                                    train_f1 = f1_score(y_train.astype(str), train_preds.astype(str), average="weighted")
                                    test_f1 = f1_score(y_test.astype(str), test_preds.astype(str), average="weighted")
                                else:
                                    pipeline.fit(X_train, y_train)
                                    train_preds = pipeline.predict(X_train)
                                    test_preds = pipeline.predict(X_test)
                                    train_acc = accuracy_score(y_train, train_preds)
                                    test_acc = accuracy_score(y_test, test_preds)
                                    train_f1 = f1_score(y_train, train_preds, average="weighted")
                                    test_f1 = f1_score(y_test, test_preds, average="weighted")

                                gap = float(train_acc - test_acc)
                                if train_acc >= 0.90 and gap >= 0.08:
                                    fit_status = "Likely overfitting"
                                elif train_acc < 0.75 and test_acc < 0.70:
                                    fit_status = "Likely underfitting"
                                else:
                                    fit_status = "Generalization acceptable"

                                results.append(
                                    {
                                        "model": model_name,
                                        "train_accuracy": float(train_acc),
                                        "test_accuracy": float(test_acc),
                                        "acc_gap_train_minus_test": gap,
                                        "train_f1_weighted": float(train_f1),
                                        "test_f1_weighted": float(test_f1),
                                        "fit_status": fit_status,
                                    }
                                )

                                pred_df = pd.DataFrame(
                                    {
                                        "row_index": X_test.index,
                                        "model": model_name,
                                        "y_true": y_test.astype(str).values,
                                        "y_pred": pd.Series(test_preds).astype(str).values,
                                    }
                                )
                                if target_name == "salary":
                                    pred_df["y_true"] = format_salary_label(pred_df["y_true"])
                                    pred_df["y_pred"] = format_salary_label(pred_df["y_pred"])
                                predictions_frames.append(pred_df)
                            else:
                                pipeline.fit(X_train, y_train)
                                train_preds = pipeline.predict(X_train)
                                test_preds = pipeline.predict(X_test)
                                train_mse = mean_squared_error(y_train, train_preds)
                                test_mse = mean_squared_error(y_test, test_preds)
                                train_rmse = np.sqrt(train_mse)
                                test_rmse = np.sqrt(test_mse)
                                train_r2 = r2_score(y_train, train_preds)
                                test_r2 = r2_score(y_test, test_preds)
                                gap = float(train_r2 - test_r2)
                                if train_r2 >= 0.90 and gap >= 0.12:
                                    fit_status = "Likely overfitting"
                                elif train_r2 < 0.45 and test_r2 < 0.35:
                                    fit_status = "Likely underfitting"
                                else:
                                    fit_status = "Generalization acceptable"
                                results.append(
                                    {
                                        "model": model_name,
                                        "train_rmse": float(train_rmse),
                                        "test_rmse": float(test_rmse),
                                        "train_r2": float(train_r2),
                                        "test_r2": float(test_r2),
                                        "r2_gap_train_minus_test": gap,
                                        "fit_status": fit_status,
                                    }
                                )
                                pred_df = pd.DataFrame(
                                    {
                                        "row_index": X_test.index,
                                        "model": model_name,
                                        "y_true": y_test.values,
                                        "y_pred": test_preds,
                                    }
                                )
                                if target_name == "salary":
                                    pred_df["y_true"] = format_salary_label(pred_df["y_true"])
                                    pred_df["y_pred"] = format_salary_label(pred_df["y_pred"])
                                predictions_frames.append(pred_df)
                        except Exception as model_err:
                            results.append({"model": model_name, "error": str(model_err)})

                    results_df = pd.DataFrame(results)
                    preds_all_df = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()
                    if problem_type == "Classification":
                        sort_cols = [c for c in ["test_accuracy", "test_f1_weighted"] if c in results_df.columns]
                        if sort_cols:
                            results_df = results_df.sort_values(sort_cols, ascending=False, na_position="last")
                    else:
                        if "test_r2" in results_df.columns:
                            results_df = results_df.sort_values(["test_r2"], ascending=False, na_position="last")
                    st.session_state.mt_eval_results_df = results_df.copy()
                    st.session_state.mt_eval_problem_type = problem_type
                    st.session_state.mt_predictions_df = preds_all_df.copy()
                    st.success("Evaluation completed.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

            if "mt_eval_results_df" in st.session_state:
                st.markdown("---")
                st.subheader("Save Evaluation Results Inside Project")
                eval_df = st.session_state.mt_eval_results_df.copy()
                eval_clean = eval_df.copy()
                if "error" in eval_clean.columns:
                    eval_clean = eval_clean[eval_clean["error"].isna() | (eval_clean["error"].astype(str).str.strip() == "")]

                if not eval_clean.empty:
                    problem_kind = st.session_state.get("mt_eval_problem_type", "Classification")
                    if "fit_status" in eval_clean.columns:
                        fit_rank_map = {
                            "Generalization acceptable": 0,
                            "Likely underfitting": 1,
                            "Likely overfitting": 2,
                        }
                        eval_clean["fit_rank"] = eval_clean["fit_status"].astype(str).map(fit_rank_map).fillna(99)
                    else:
                        eval_clean["fit_rank"] = 99

                    if problem_kind == "Classification" and "test_accuracy" in eval_clean.columns:
                        best_row = eval_clean.sort_values(
                            ["fit_rank", "test_accuracy", "test_f1_weighted"],
                            ascending=[True, False, False],
                            na_position="last",
                        ).head(1)
                    elif problem_kind == "Regression" and "test_r2" in eval_clean.columns:
                        best_row = eval_clean.sort_values(
                            ["fit_rank", "test_r2", "test_rmse"],
                            ascending=[True, False, True],
                            na_position="last",
                        ).head(1)
                    else:
                        best_row = eval_clean.sort_values(["fit_rank"], ascending=[True], na_position="last").head(1)

                    if not best_row.empty and "model" in best_row.columns:
                        best_model_name = str(best_row.iloc[0]["model"])
                        st.success(f"Best model based on current evaluation: {best_model_name}")
                        st.caption("Reference priority: fit_status first, then test metrics.")
                        st.dataframe(best_row.drop(columns=["fit_rank"], errors="ignore"), use_container_width=True, height=120)

                st.write("Evaluation table preview (mark rows to delete, then apply):")
                eval_edit_df = eval_df.copy()
                eval_edit_df["delete_row"] = False
                edited_eval_df = st.data_editor(
                    eval_edit_df,
                    use_container_width=True,
                    height=320,
                    key="mt_eval_editor",
                    column_config={
                        "delete_row": st.column_config.CheckboxColumn("Delete", help="Check to remove this model row")
                    },
                    disabled=[c for c in eval_edit_df.columns if c != "delete_row"],
                )

                if st.button("Apply Row Deletion", key="mt_apply_eval_row_delete_btn"):
                    rows_to_delete = edited_eval_df[edited_eval_df["delete_row"] == True].copy()
                    if rows_to_delete.empty:
                        st.info("No rows selected for deletion.")
                    else:
                        remaining_df = edited_eval_df[edited_eval_df["delete_row"] != True].drop(columns=["delete_row"])
                        st.session_state.mt_eval_results_df = remaining_df.reset_index(drop=True)
                        removed_models = rows_to_delete["model"].astype(str).tolist() if "model" in rows_to_delete.columns else []
                        if "mt_predictions_df" in st.session_state and not st.session_state.mt_predictions_df.empty and removed_models:
                            pred_df = st.session_state.mt_predictions_df.copy()
                            st.session_state.mt_predictions_df = pred_df[~pred_df["model"].astype(str).isin(removed_models)].reset_index(drop=True)
                        st.success(f"Removed {len(rows_to_delete)} row(s) from evaluation table.")
                        st.rerun()

                eval_df = st.session_state.mt_eval_results_df.copy()
                default_name = st.session_state.get(
                    "mt_eval_results_file",
                    f"model_evaluation_results_{st.session_state.get('mt_eval_problem_type', 'classification').lower()}.csv",
                )
                project_folders = list_project_folders(os.getcwd(), max_depth=4)
                selected_folder = st.selectbox(
                    "Save inside project folder:",
                    options=project_folders,
                    index=0,
                    key="mt_eval_folder",
                )
                custom_name = st.text_input(
                    "File name:",
                    value=default_name,
                    key="mt_eval_file_name",
                ).strip()
                if not custom_name:
                    custom_name = default_name
                if not custom_name.lower().endswith(".csv"):
                    custom_name = f"{custom_name}.csv"
                st.session_state.mt_eval_results_file = custom_name

                target_path = os.path.join(os.getcwd(), selected_folder, custom_name)
                st.caption(f"Target path: {target_path}")
                if st.button("Save inside Project", key="mt_save_eval_results_btn"):
                    try:
                        target_dir = os.path.dirname(target_path)
                        if target_dir:
                            os.makedirs(target_dir, exist_ok=True)
                        eval_df.to_csv(target_path, index=False, encoding="utf-8-sig")
                        st.success(f"Evaluation results saved to: {target_path}")
                    except Exception as e:
                        st.error(f"Failed to save evaluation results: {e}")

            if "mt_predictions_df" in st.session_state and not st.session_state.mt_predictions_df.empty:
                st.markdown("---")
                st.subheader("Predictions Preview")
                pred_all_df = st.session_state.mt_predictions_df
                pred_models = pred_all_df["model"].dropna().unique().tolist()
                pred_model_pick = st.selectbox(
                    "Choose model predictions to preview:",
                    options=pred_models,
                    key="mt_pred_model_pick",
                )
                pred_view = pred_all_df[pred_all_df["model"] == pred_model_pick].copy()
                st.dataframe(pred_view.head(200), use_container_width=True, height=320)

                st.subheader("Save Predictions Inside Project")
                default_pred_name = st.session_state.get("mt_predictions_file", "model_predictions.csv")
                pred_folders = list_project_folders(os.getcwd(), max_depth=4)
                pred_folder = st.selectbox(
                    "Save inside project folder:",
                    options=pred_folders,
                    index=0,
                    key="mt_pred_folder",
                )
                pred_file_name = st.text_input(
                    "File name:",
                    value=default_pred_name,
                    key="mt_pred_file_name",
                ).strip()
                if not pred_file_name:
                    pred_file_name = default_pred_name
                if not pred_file_name.lower().endswith(".csv"):
                    pred_file_name = f"{pred_file_name}.csv"
                st.session_state.mt_predictions_file = pred_file_name
                pred_target_path = os.path.join(os.getcwd(), pred_folder, pred_file_name)
                st.caption(f"Target path: {pred_target_path}")
                if st.button("Save inside Project", key="mt_save_predictions_btn"):
                    try:
                        pred_target_dir = os.path.dirname(pred_target_path)
                        if pred_target_dir:
                            os.makedirs(pred_target_dir, exist_ok=True)
                        pred_all_df.to_csv(pred_target_path, index=False, encoding="utf-8-sig")
                        st.success(f"Predictions saved to: {pred_target_path}")
                    except Exception as e:
                        st.error(f"Failed to save predictions: {e}")

            st.markdown("---")
            st.subheader("Live Prediction")
            if "mt_live_alert" in st.session_state:
                alert_data = st.session_state.pop("mt_live_alert", None)
                if isinstance(alert_data, dict):
                    show_swal_popup(
                        title="Prediction Finished",
                        text=(
                            f"Model: {alert_data.get('model', 'N/A')} | "
                            f"Type: {alert_data.get('problem_type', 'N/A')} | "
                            f"Output: {alert_data.get('target', 'N/A')}"
                        ),
                        icon="success",
                    )
            target_for_live = split.get("target_col", target_col)
            input_cols_live = st.session_state.get("mt_input_cols", [])
            if not input_cols_live:
                input_cols_live = [c for c in df.columns.tolist() if c != target_for_live]

            if not input_cols_live:
                st.warning("No input features available for live prediction.")
            else:
                eval_models = []
                if "mt_eval_results_df" in st.session_state and "model" in st.session_state.mt_eval_results_df.columns:
                    eval_models = (
                        st.session_state.mt_eval_results_df["model"]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )
                eval_models = [m for m in eval_models if m]
                live_model_options = eval_models if eval_models else model_options
                selected_live_model = st.selectbox(
                    "Model for live prediction:",
                    options=live_model_options,
                    key="mt_live_model_pick",
                )

                ordered_cols = df.columns.tolist()
                ordered_input_cols = [c for c in ordered_cols if c != target_for_live]
                template_df = pd.DataFrame(columns=ordered_cols)
                for col in ordered_cols:
                    if col != target_for_live and pd.api.types.is_numeric_dtype(df[col]):
                        template_df[col] = template_df[col].astype(float)
                template_df[target_for_live] = None
                if "mt_live_table_df" not in st.session_state or not isinstance(st.session_state.mt_live_table_df, pd.DataFrame):
                    st.session_state.mt_live_table_df = template_df.copy()
                else:
                    st.session_state.mt_live_table_df = st.session_state.mt_live_table_df.reindex(columns=ordered_cols)
                editor_col_config = {}
                for col in ordered_input_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        editor_col_config[col] = st.column_config.NumberColumn(
                            col,
                            step=0.01,
                            format="%.6f",
                        )
                st.subheader("Live Prediction Output")
                live_full_df = st.data_editor(
                    st.session_state.mt_live_table_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    height=280,
                    key="mt_live_input_editor",
                    disabled=[target_for_live],
                    column_config=editor_col_config,
                )

                if st.button("Predict Live Inputs", key="mt_live_predict_btn"):
                    try:
                        if live_full_df.empty:
                            st.warning("Add at least one row to predict.")
                        else:
                            live_model_pack = train_live_model(
                                df=df,
                                input_cols=input_cols_live,
                                target_col=target_for_live,
                                problem_type=problem_type,
                                model_name=selected_live_model,
                            )
                            pipeline = live_model_pack["pipeline"]
                            train_columns = live_model_pack["train_columns"]
                            label_encoder = live_model_pack["label_encoder"]

                            live_input_only = live_full_df.copy()
                            for col in input_cols_live:
                                if col not in live_input_only.columns:
                                    live_input_only[col] = None

                            live_X_encoded = pd.get_dummies(
                                live_input_only[input_cols_live],
                                drop_first=True,
                                dtype=int,
                            )
                            live_X_encoded = live_X_encoded.reindex(columns=train_columns, fill_value=0)
                            live_preds = pipeline.predict(live_X_encoded)

                            if label_encoder is not None:
                                live_preds = label_encoder.inverse_transform(pd.Series(live_preds).astype(int))
                            elif (
                                problem_type == "Classification"
                                and str(target_for_live).strip().lower() == "salary"
                            ):
                                preds_numeric = pd.to_numeric(pd.Series(live_preds), errors="coerce")
                                if preds_numeric.notna().all():
                                    live_preds = preds_numeric.round().astype(int)

                            if (
                                problem_type == "Classification"
                                and str(target_for_live).strip().lower() == "salary"
                            ):
                                live_preds = format_salary_label(live_preds)
                            else:
                                target_series = df[target_for_live]
                                if problem_type == "Classification":
                                    live_preds = pd.Series(live_preds).astype(str).values
                                else:
                                    target_num = pd.to_numeric(target_series, errors="coerce")
                                    is_int_like_target = (
                                        pd.api.types.is_integer_dtype(target_series)
                                        or (
                                            target_num.notna().any()
                                            and np.allclose(target_num.dropna().values, np.round(target_num.dropna().values))
                                        )
                                    )
                                    # If target was transformed earlier, map predictions back to original target scale.
                                    trans_features = st.session_state.get("transformation_features", [])
                                    before_df = st.session_state.get("transformation_before_df")
                                    after_df = st.session_state.get("transformation_result_df")
                                    if (
                                        str(target_for_live) in trans_features
                                        and isinstance(before_df, pd.DataFrame)
                                        and isinstance(after_df, pd.DataFrame)
                                        and target_for_live in before_df.columns
                                        and target_for_live in after_df.columns
                                    ):
                                        x_after = pd.to_numeric(after_df[target_for_live], errors="coerce")
                                        y_before = pd.to_numeric(before_df[target_for_live], errors="coerce")
                                        valid = x_after.notna() & y_before.notna()
                                        if int(valid.sum()) >= 10:
                                            x_sorted = x_after[valid].to_numpy(dtype=float)
                                            y_sorted = y_before[valid].to_numpy(dtype=float)
                                            order_idx = np.argsort(x_sorted)
                                            x_sorted = x_sorted[order_idx]
                                            y_sorted = y_sorted[order_idx]
                                            pred_num = pd.to_numeric(pd.Series(live_preds), errors="coerce").to_numpy(dtype=float)
                                            mapped_pred = np.interp(pred_num, x_sorted, y_sorted)
                                            live_preds = mapped_pred
                                    if is_int_like_target:
                                        live_preds = np.round(pd.to_numeric(pd.Series(live_preds), errors="coerce")).astype("Int64")

                            live_result_df = live_full_df.copy()
                            live_result_df[target_for_live] = live_preds
                            st.session_state.mt_live_table_df = live_full_df.copy()
                            st.session_state.mt_live_table_df[target_for_live] = None
                            st.session_state.mt_live_result_df = live_result_df.copy()
                            st.success("Live prediction completed.")
                            st.session_state.mt_live_alert = {
                                "model": selected_live_model,
                                "problem_type": problem_type,
                                "target": target_for_live,
                            }
                            st.rerun()
                    except Exception as e:
                        st.error(f"Live prediction failed: {e}")

                if "mt_live_result_df" in st.session_state and isinstance(st.session_state.mt_live_result_df, pd.DataFrame):
                    st.markdown("---")
                    st.subheader("Live Prediction (Original Table Style)")
                    st.dataframe(st.session_state.mt_live_result_df, use_container_width=True, height=280)

                if "mt_live_table_df" in st.session_state and isinstance(st.session_state.mt_live_table_df, pd.DataFrame):
                    st.markdown("---")
                    st.subheader("Save Live Table Inside Project")
                    project_folders = list_project_folders(os.getcwd(), max_depth=4)
                    selected_folder = st.selectbox(
                        "Save inside project folder:",
                        options=project_folders,
                        index=0,
                        key="mt_live_table_folder",
                    )
                    export_name = st.text_input(
                        "File name:",
                        value=st.session_state.get("mt_live_export_file", "live_prediction_table.csv"),
                        key="mt_live_export_file_input",
                    ).strip()
                    if not export_name:
                        export_name = "live_prediction_table.csv"
                    if not export_name.lower().endswith(".csv"):
                        export_name = f"{export_name}.csv"
                    st.session_state.mt_live_export_file = export_name

                    export_path = os.path.join(os.getcwd(), selected_folder, export_name)
                    st.caption(f"Target path: {export_path}")
                    if st.button("Save inside Project", key="mt_live_export_btn"):
                        try:
                            target_dir = os.path.dirname(export_path)
                            if target_dir:
                                os.makedirs(target_dir, exist_ok=True)
                            export_df = st.session_state.get("mt_live_result_df", st.session_state.mt_live_table_df)
                            export_df.to_csv(export_path, index=False, encoding="utf-8-sig")
                            st.success(f"Live table saved to: {export_path}")
                        except Exception as e:
                            st.error(f"Failed to save live table: {e}")
else:
    st.info("Load a dataset first from session, PC upload, or GitHub URL.")
