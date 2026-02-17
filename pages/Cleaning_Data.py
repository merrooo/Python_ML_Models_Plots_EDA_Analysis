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

st.set_page_config(page_title="Data Cleaning - NDEDC", layout="wide")
st.title("Data Cleaning and Final Dataset Preparation")


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


# ------------------------------------------------------------------
# 1. Data Source (Upload or GitHub)
# ------------------------------------------------------------------
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

        if one_shot_checkbox("Confirm Drop", "confirm_drop_chk"):
            if cols_to_drop:
                st.session_state.df.drop(columns=cols_to_drop, inplace=True)
                st.session_state.df.to_csv("Data_Dropped_Columns.csv", index=False, encoding="utf-8-sig")
                st.success("Columns dropped successfully.")
                st.rerun()

    # --- 2. Encoding ---
    with st.expander("2) Encode Text Columns (One-Hot / Label)"):
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

            method = st.radio("Encoding method:", ["One-Hot Encoding", "Label Encoding"])
            st.caption(
                "Use One-Hot for nominal categories (no order). Use Label Encoding for ordinal categories "
                "(example: low=0, medium=1, high=2)."
            )
            selected_enc = st.multiselect("Select columns to encode:", options=obj_cols)
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
                else:
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

            if one_shot_checkbox("Apply Encoding", "apply_encoding_chk"):
                if selected_enc:
                    if method == "One-Hot Encoding":
                        st.session_state.df = pd.get_dummies(
                            st.session_state.df,
                            columns=selected_enc,
                            drop_first=True,
                            dtype=int,
                        )
                    else:
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
        with st.expander("Feature Correlation (All Numeric Features)"):
            st.caption("Choose output/target first, then inspect strongest to weakest correlations.")
            corr_method = st.radio(
                "Correlation method:",
                ["pearson", "spearman", "kendall"],
                horizontal=True,
                key="corr_method_radio",
            )
            corr_df = st.session_state.df.select_dtypes(include=[np.number])
            if corr_df.shape[1] >= 2:
                corr_matrix = corr_df.corr(method=corr_method)
                st.write("Correlation matrix:")
                st.dataframe(corr_matrix, use_container_width=True, height=320)

                target_feature = st.selectbox(
                    "Choose output/feature for correlation ranking:",
                    options=corr_df.columns.tolist(),
                    key="corr_target_feature",
                )
                target_corr = corr_matrix[target_feature].drop(labels=[target_feature]).reset_index()
                target_corr.columns = ["feature", "correlation"]
                target_corr["abs_correlation"] = target_corr["correlation"].abs()
                target_corr = target_corr.sort_values("abs_correlation", ascending=False)

                st.write(f"Correlations vs `{target_feature}` (strongest to weakest):")
                st.dataframe(
                    target_corr[["feature", "correlation", "abs_correlation"]],
                    use_container_width=True,
                    height=320,
                )
            else:
                st.info("Need at least 2 numeric columns to compute correlation.")

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
                    "SMOTE (synthetic samples on numeric input features)",
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
