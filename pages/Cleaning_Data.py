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
    with st.expander("Inspect Data Types"):
        st.write(st.session_state.df_original.dtypes)

    st.markdown("---")
    st.subheader("Processing Tools")

    # --- 1. Drop Columns ---
    with st.expander("1) Drop Columns"):
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
        with st.expander("Feature Correlation (All Numeric Features)"):
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

        with st.expander("Skewness Check (After Encoding)"):
            skew_df = st.session_state.df.select_dtypes(include=[np.number])
            if skew_df.shape[1] == 0:
                st.info("No numeric columns available for skewness check.")
            else:
                skew_mode = st.radio(
                    "Skewness mode:",
                    ["One feature", "All numeric features"],
                    horizontal=True,
                    key="skew_mode_after_encoding",
                )
                if skew_mode == "One feature":
                    skew_col = st.selectbox(
                        "Select feature:",
                        options=skew_df.columns.tolist(),
                        key="skew_col_after_encoding",
                    )
                    skew_value = float(skew_df[skew_col].skew())
                    st.metric("Skewness Value", f"{skew_value:.6f}")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "feature": [skew_col],
                                "skewness": [skew_value],
                                "skew_type": [
                                    "right-skewed" if skew_value > 0.5 else
                                    "left-skewed" if skew_value < -0.5 else
                                    "approximately symmetric"
                                ],
                            }
                        ),
                        use_container_width=True,
                        height=120,
                    )
                else:
                    skew_summary = pd.DataFrame(
                        {
                            "feature": skew_df.columns.tolist(),
                            "skewness": [float(skew_df[c].skew()) for c in skew_df.columns],
                        }
                    )
                    skew_summary["abs_skewness"] = skew_summary["skewness"].abs()
                    skew_summary["skew_type"] = skew_summary["skewness"].apply(
                        lambda x: "right-skewed" if x > 0.5 else
                        "left-skewed" if x < -0.5 else
                        "approximately symmetric"
                    )
                    skew_summary = skew_summary.sort_values("abs_skewness", ascending=False)
                    st.dataframe(skew_summary, use_container_width=True, height=340)

    # --- 3. Date Conversion ---
    with st.expander("3) Convert Date Columns"):
        if one_shot_checkbox("Extract Date Parts", "extract_date_parts_chk"):
            df_temp = st.session_state.df.copy()
            converted_cols = []
            candidate_cols = df_temp.select_dtypes(include=["object", "string", "category"]).columns.tolist()

            for col in candidate_cols:
                series = df_temp[col]
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                parsed_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
                if parsed_ratio >= 0.6 and parsed.notna().sum() > 0:
                    df_temp[f"{col}_year"] = parsed.dt.year
                    df_temp[f"{col}_month"] = parsed.dt.month
                    df_temp[f"{col}_day"] = parsed.dt.day
                    df_temp.drop(columns=[col], inplace=True)
                    converted_cols.append(col)

            if converted_cols:
                st.session_state.df = df_temp
                st.session_state.date_applied = True
                st.session_state.date_result_df = df_temp.copy()
                st.session_state.date_converted_cols = converted_cols
                st.success(f"Date conversion completed for: {', '.join(converted_cols)}")
                st.rerun()
            else:
                st.warning("No date-like columns were detected in the current working data.")

        if st.session_state.get("date_applied", False):
            st.markdown("---")
            st.subheader("Date Conversion Result")
            st.caption(
                "Converted columns: "
                + ", ".join(st.session_state.get("date_converted_cols", []))
            )
            date_preview_df = st.session_state.get("date_result_df", st.session_state.df)
            st.dataframe(date_preview_df.head(30), use_container_width=True, height=320)

     # --- 5. Feature Transformation ---
    with st.expander("5) Feature Transformation (Scaling)"):
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
            transform_cols = st.multiselect(
                "Select features to transform:",
                options=transform_num_cols,
                default=suggested_cols,
                key="transform_feature_selection",
            )
            st.caption(f"Detected binary 0/1 features: {binary_cols}")
            selected_transform_col = st.selectbox(
                "Preview column:",
                options=transform_cols if transform_cols else transform_num_cols,
                key="transform_preview_col",
            )

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

            if one_shot_checkbox("Apply Transformation", "apply_transform_chk"):
                if not transform_cols:
                    st.info("No features selected for transformation.")
                else:
                    transformed_df = run_transform(st.session_state.df)
                    st.session_state.df = transformed_df
                    st.session_state.transformation_applied = True
                    st.session_state.transformation_method = transform_method
                    st.session_state.transformation_scope = "user-selected features"
                    st.session_state.transformation_features = transform_cols
                    st.session_state.transformation_skipped_binary = binary_cols
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
                st.caption(f"Skipped binary 0/1 features: {st.session_state.get('transformation_skipped_binary', [])}")
                tr = st.session_state.get("transformation_result_df", st.session_state.df)
                st.write(f"Transformed data rows: {tr.shape[0]} | columns: {tr.shape[1]}")
                st.dataframe(tr.head(30), use_container_width=True, height=320)
                with st.expander("Show full transformed data table"):
                    st.dataframe(tr, use_container_width=True, height=520)

                default_name = st.session_state.get(
                    "transformation_result_file", "final_process_transformed_data.csv"
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
                        

    # --- 4. Outliers ---
    with st.expander("4) Outlier Handling"):
        num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            outlier_stats = []
            for col in num_cols:
                q1_col, q3_col = st.session_state.df[col].quantile([0.25, 0.75])
                iqr_col = q3_col - q1_col
                low_col, up_col = q1_col - 1.5 * iqr_col, q3_col + 1.5 * iqr_col
                col_mask = (st.session_state.df[col] < low_col) | (st.session_state.df[col] > up_col)
                col_outlier_count = int(col_mask.sum())
                outlier_stats.append(
                    {
                        "column": col,
                        "outlier_count": col_outlier_count,
                        "has_outliers": col_outlier_count > 0,
                    }
                )

            outlier_stats_df = pd.DataFrame(outlier_stats)
            outlier_stats_df["marker"] = outlier_stats_df["has_outliers"].map(
                lambda x: "OUTLIERS" if x else ""
            )
            st.write("Outlier summary by numeric column:")
            st.dataframe(
                outlier_stats_df[["column", "outlier_count", "marker"]],
                use_container_width=True,
                height=min(320, 38 * (len(outlier_stats_df) + 1)),
            )

            col_option_map = {}
            for _, row in outlier_stats_df.iterrows():
                label = (
                    f"{row['column']} [OUTLIERS: {int(row['outlier_count'])}]"
                    if row["has_outliers"]
                    else f"{row['column']}"
                )
                col_option_map[label] = row["column"]
            outlier_cols_only = outlier_stats_df.loc[
                outlier_stats_df["has_outliers"], "column"
            ].tolist()

            selected_label = st.selectbox("Select numeric column:", options=list(col_option_map.keys()))
            target_col = col_option_map[selected_label]

            q1, q3 = st.session_state.df[target_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (st.session_state.df[target_col] < low) | (st.session_state.df[target_col] > up)
            outliers = st.session_state.df[mask]

            st.metric(f"Outliers in {target_col}", outliers.shape[0])
            if not outliers.empty:
                with st.expander("Show outlier rows"):
                    st.dataframe(outliers, use_container_width=True, height=260)

                st.write("Critical outlier samples (most extreme values):")
                q1_center, q3_center = st.session_state.df[target_col].quantile([0.25, 0.75])
                iqr_center = q3_center - q1_center
                median_center = st.session_state.df[target_col].median()
                scale = iqr_center if iqr_center and not np.isnan(iqr_center) else 1.0

                critical_samples = outliers.copy()
                critical_samples["_outlier_score"] = (
                    (critical_samples[target_col] - median_center).abs() / scale
                )
                critical_samples = critical_samples.sort_values(
                    by="_outlier_score", ascending=False
                )

                st.dataframe(
                    critical_samples[[target_col, "_outlier_score"]].head(10),
                    use_container_width=True,
                    height=260,
                )

                low_outliers = outliers.sort_values(by=target_col, ascending=True).head(5)
                high_outliers = outliers.sort_values(by=target_col, ascending=False).head(5)
                c_low, c_high = st.columns(2)
                with c_low:
                    st.write("Lowest outlier samples:")
                    st.dataframe(low_outliers[[target_col]], use_container_width=True, height=180)
                with c_high:
                    st.write("Highest outlier samples:")
                    st.dataframe(high_outliers[[target_col]], use_container_width=True, height=180)

            st.divider()
            strat = st.radio(
                "Choose strategy:",
                [
                    "Mean Replace",
                    "Drop Rows",
                    "Quantile Transform",
                    "IQR Capping (Winsorize)",
                    "Percentile Clipping (1%-99%)",
                    "MAD Clipping (Robust)",
                    "Log Transform + IQR Capping",
                    "Isolation Forest (Drop Anomalies)",
                ],
                horizontal=True,
                key="outlier_strategy_selected",
            )
            st.caption(f"Selected strategy: {strat}")

            apply_scope = st.radio(
                "Apply scope:",
                ["Selected column only", "All columns with outliers"],
                horizontal=True,
            )

            if one_shot_checkbox("Apply Selected Strategy", "apply_outlier_strategy_chk"):
                if apply_scope == "Selected column only":
                    updated_df, used_iters = apply_outlier_strategy_until_stable(
                        st.session_state.df, target_col, strat
                    )
                    st.session_state.df = updated_df
                    st.session_state.outlier_result_df = updated_df.copy()
                    st.session_state.outlier_last_scope = target_col
                else:
                    df_work = st.session_state.df.copy()
                    total_iters = 0
                    rounds = 0
                    max_rounds = 10
                    while rounds < max_rounds:
                        rounds += 1
                        current_num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
                        current_outlier_cols = []
                        for c in current_num_cols:
                            cmask, _, _ = get_iqr_mask(df_work[c])
                            if int(cmask.sum()) > 0:
                                current_outlier_cols.append(c)

                        if not current_outlier_cols:
                            break

                        for c in current_outlier_cols:
                            df_work, used_iters_col = apply_outlier_strategy_until_stable(
                                df_work, c, strat
                            )
                            total_iters += used_iters_col

                    if not outlier_cols_only:
                        st.info("No columns with outliers were found to apply this strategy.")
                    else:
                        st.session_state.df = df_work
                        st.session_state.outlier_result_df = df_work.copy()
                        st.session_state.outlier_last_scope = "all columns with outliers"
                        used_iters = total_iters

                if apply_scope == "Selected column only" or outlier_cols_only:
                    st.session_state.outlier_applied = True
                    st.session_state.outlier_last_strategy = strat
                    st.session_state.outlier_last_iterations = used_iters
                    st.session_state.outlier_result_file = "final_process_outlier_handled_data.csv"
                    st.session_state.pop("outlier_preview_df", None)
                    st.session_state.pop("outlier_preview_col", None)
                    st.session_state.pop("outlier_preview_strategy", None)
                    st.session_state.pop("outlier_preview_iters", None)
                    st.rerun()

            if one_shot_checkbox("Preview Strategy Result", "preview_outlier_strategy_chk"):
                preview_df, preview_iters = apply_outlier_strategy_until_stable(
                    st.session_state.df, target_col, strat
                )
                st.session_state.outlier_preview_df = preview_df
                st.session_state.outlier_preview_col = target_col
                st.session_state.outlier_preview_strategy = strat
                st.session_state.outlier_preview_iters = preview_iters

            if "outlier_preview_df" in st.session_state:
                prev_col = st.session_state.get("outlier_preview_col", target_col)
                prev_strat = st.session_state.get("outlier_preview_strategy", strat)
                st.markdown("---")
                st.write(f"Preview after `{prev_strat}` on `{prev_col}`:")
                st.caption(f"Iterations used: {st.session_state.get('outlier_preview_iters', 0)}")
                c_prev1, c_prev2 = st.columns(2)
                with c_prev1:
                    st.caption("Before")
                    st.dataframe(
                        st.session_state.df[[prev_col]].head(20),
                        use_container_width=True,
                        height=240,
                    )
                with c_prev2:
                    st.caption("After (Preview)")
                    st.dataframe(
                        st.session_state.outlier_preview_df[[prev_col]].head(20),
                        use_container_width=True,
                        height=240,
                    )

            if st.session_state.get("outlier_applied", False):
                st.markdown("---")
                st.subheader("Outlier Handling Result")
                st.caption(
                    f"Applied `{st.session_state.get('outlier_last_strategy', '')}` on "
                    f"`{st.session_state.get('outlier_last_scope', '')}`"
                )
                st.caption(f"Iterations used: {st.session_state.get('outlier_last_iterations', 0)}")
                outlier_result_df = st.session_state.get("outlier_result_df", st.session_state.df)
                st.dataframe(outlier_result_df.head(30), use_container_width=True, height=320)

                result_num_cols = outlier_result_df.select_dtypes(include=[np.number]).columns.tolist()
                if result_num_cols:
                    result_stats = []
                    for c in result_num_cols:
                        rq1, rq3 = outlier_result_df[c].quantile([0.25, 0.75])
                        riqr = rq3 - rq1
                        rlow, rup = rq1 - 1.5 * riqr, rq3 + 1.5 * riqr
                        rmask = (outlier_result_df[c] < rlow) | (outlier_result_df[c] > rup)
                        rc = int(rmask.sum())
                        result_stats.append(
                            {"column": c, "outlier_count": rc, "marker": "OUTLIERS" if rc > 0 else ""}
                        )
                    st.write("Outlier summary by numeric column (after apply):")
                    st.caption("This table is recalculated from the current working data after each apply.")
                    st.dataframe(
                        pd.DataFrame(result_stats),
                        use_container_width=True,
                        height=min(320, 38 * (len(result_stats) + 1)),
                    )

                default_name = st.session_state.get(
                    "outlier_result_file", "final_process_outlier_handled_data.csv"
                )
                project_folders = list_project_folders(os.getcwd(), max_depth=4)
                selected_folder = st.selectbox(
                    "Save inside project folder:",
                    options=project_folders,
                    index=0,
                    key="outlier_project_folder",
                )
                custom_name = st.text_input(
                    "File name:",
                    value=default_name,
                    key="outlier_project_file_name",
                ).strip()
                if not custom_name:
                    custom_name = default_name
                if not custom_name.lower().endswith(".csv"):
                    custom_name = f"{custom_name}.csv"
                target_path = os.path.join(os.getcwd(), selected_folder, custom_name)
                st.caption(f"Target path: {target_path}")
                if st.button("Save inside Project", key="save_outlier_to_project_btn"):
                    try:
                        target_dir = os.path.dirname(target_path)
                        if target_dir:
                            os.makedirs(target_dir, exist_ok=True)
                        outlier_save_df = st.session_state.get("outlier_result_df", st.session_state.df)
                        outlier_save_df.to_csv(target_path, index=False, encoding="utf-8-sig")
                        st.success(f"File saved to: {target_path}")
                    except Exception as e:
                        st.error(f"Failed to save file: {e}")

   

   
