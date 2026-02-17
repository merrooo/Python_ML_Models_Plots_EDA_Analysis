import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="EDA - NDEDC Dashboard", layout="wide")
st.title("EDA: Multi-Table Visual Analysis")


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
    st.subheader("3) Skewness Plots")
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
    st.subheader("4) General Statistics")
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
    st.subheader("5) Value Frequency")
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
    st.subheader("6) Distribution + Outlier Check")
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
    for tab, (name, table_df) in zip(tabs, loaded_tables.items()):
        with tab:
            render_eda_for_table(table_df, name)
else:
    st.info("Upload data to start EDA for Main, Encoded, and Outliers tables.")
