import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

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
        load_from_github = st.button("Load from GitHub")

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
    if st.sidebar.button("Upload Different File"):
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

        if st.button("Confirm Drop"):
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
                        le_preview = LabelEncoder()
                        series_as_str = st.session_state.df[col].astype(str)
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
                        le_preview = LabelEncoder()
                        label_preview[col] = le_preview.fit_transform(label_preview[col].astype(str))
                    st.write("Sample after Label Encoding:")
                    st.dataframe(label_preview, use_container_width=True, height=230)

            if st.button("Apply Encoding"):
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
                            st.session_state.df[col] = le.fit_transform(st.session_state.df[col].astype(str))

                    st.session_state.df.to_csv("Data_Encoded.csv", index=False, encoding="utf-8-sig")
                    st.session_state.encoding_applied = True
                    st.session_state.last_encoding_method = method
                    st.session_state.encoded_output_file = output_file_name
                    st.session_state.show_encoded_preview = False
                    st.success("Encoding applied successfully.")
                    st.rerun()
        else:
            st.write("No object columns found.")

        if st.session_state.get("encoding_applied", False):
            st.markdown("---")
            st.subheader("Encoded Data Result")
            st.caption(f"Method: {st.session_state.get('last_encoding_method', 'Unknown')}")
            if st.button("Load and Preview Encoded Data", key="load_encoded_preview_btn"):
                st.session_state.show_encoded_preview = True

            if st.session_state.get("show_encoded_preview", False):
                st.dataframe(st.session_state.df.head(30), use_container_width=True, height=320)
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
                        st.session_state.df.to_csv(target_path, index=False, encoding="utf-8-sig")
                        st.success(f"File saved to: {target_path}")
                    except Exception as e:
                        st.error(f"Failed to save file: {e}")

        # --- 3. Date Conversion ---
        with st.expander("3) Convert Date Columns"):
            if st.button("Extract Date Parts"):
                df_temp = st.session_state.df.copy()
                for col in df_temp.columns:
                    if df_temp[col].dtype == "object":
                        try:
                            df_temp[col] = pd.to_datetime(df_temp[col])
                            df_temp[f"{col}_year"] = df_temp[col].dt.year
                            df_temp[f"{col}_month"] = df_temp[col].dt.month
                            df_temp[f"{col}_day"] = df_temp[col].dt.day
                            df_temp.drop(columns=[col], inplace=True)
                        except Exception:
                            continue
                st.session_state.df = df_temp
                st.success("Date conversion completed.")
                st.rerun()

        # --- 4. Outliers ---
        with st.expander("4) Outlier Handling"):
            num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                target_col = st.selectbox("Select numeric column:", options=num_cols)

                q1, q3 = st.session_state.df[target_col].quantile([0.25, 0.75])
                iqr = q3 - q1
                low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (st.session_state.df[target_col] < low) | (st.session_state.df[target_col] > up)
                outliers = st.session_state.df[mask]

                st.metric(f"Outliers in {target_col}", outliers.shape[0])
                if not outliers.empty:
                    st.dataframe(outliers, height=200)

                st.divider()
                strat = st.radio(
                    "Strategy:",
                    ["Mean Replace", "Drop Rows", "Quantile Transform"],
                    horizontal=True,
                )

                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"Apply on {target_col}"):
                        if strat == "Mean Replace":
                            st.session_state.df.loc[mask, target_col] = st.session_state.df[target_col].mean()
                        elif strat == "Drop Rows":
                            st.session_state.df = st.session_state.df[~mask]
                        else:
                            qt = QuantileTransformer(
                                output_distribution="normal",
                                n_quantiles=min(len(st.session_state.df), 100),
                            )
                            st.session_state.df[target_col] = qt.fit_transform(
                                st.session_state.df[[target_col]].values
                            ).flatten()
                        st.rerun()

                with c2:
                    if st.button("Apply to All Numeric Columns"):
                        df_work = st.session_state.df.copy()
                        for c in num_cols:
                            cq1, cq3 = df_work[c].quantile([0.25, 0.75])
                            ciqr = cq3 - cq1
                            cl, cu = cq1 - 1.5 * ciqr, cq3 + 1.5 * ciqr
                            cm = (df_work[c] < cl) | (df_work[c] > cu)
                            if strat == "Mean Replace":
                                df_work.loc[cm, c] = df_work[c].mean()
                            elif strat == "Drop Rows":
                                df_work = df_work[~cm]
                            else:
                                qt = QuantileTransformer(
                                    output_distribution="normal",
                                    n_quantiles=min(len(df_work), 100),
                                )
                                df_work[c] = qt.fit_transform(df_work[[c]].values).flatten()
                        st.session_state.df = df_work
                        st.rerun()

        # --- 5. Train/Test Split ---
        with st.expander("5) Split Data"):
            target_var = st.selectbox("Target column (y):", options=st.session_state.df.columns.tolist())
            size = st.slider("Test size:", 0.1, 0.5, 0.2)
            if st.button("Run Final Split"):
                X = st.session_state.df.drop(columns=[target_var])
                y = st.session_state.df[target_var]
                X = pd.get_dummies(X, drop_first=True, dtype=int)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=size, random_state=42
                )

                X_train.to_csv("X_train.csv", index=False)
                X_test.to_csv("X_test.csv", index=False)
                y_train.to_csv("y_train.csv", index=False)
                y_test.to_csv("y_test.csv", index=False)

                st.session_state.split_done = True
                st.success("Train/test files saved successfully.")
