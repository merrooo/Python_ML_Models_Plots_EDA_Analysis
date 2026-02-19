import os
import pandas as pd
import streamlit as st
from utils.helpers import one_shot_checkbox


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

    if github_url and str(github_url).strip():
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
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {"venv", "__pycache__", ".git"}]
        for d in dirnames:
            rel = os.path.relpath(os.path.join(root, d), base_path)
            folders.append(rel)
    return sorted(set(folders))


def save_dataframe(df: pd.DataFrame, folder: str, file_name: str):
    try:
        if not str(file_name).lower().endswith(".csv"):
            file_name = f"{file_name}.csv"
        target_path = os.path.join(os.getcwd(), folder, file_name)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
        return True, target_path
    except Exception as e:
        return False, str(e)


def render_step_table_loader(step_key: str, label: str | None = None):
    label = label or "Step"
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader(
            "Load CSV/Excel from PC",
            type=["csv", "xlsx", "xls"],
            key=f"{step_key}_file_uploader",
        )
    with c2:
        url = st.text_input(
            "Or paste direct/Raw GitHub URL",
            placeholder="https://github.com/user/repo/blob/main/data.csv",
            key=f"{step_key}_github_url",
        )
    if one_shot_checkbox("Load table for this step", key=f"{step_key}_load_chk"):
        df = load_dataframe_from_source(up, url)
        if df is None:
            st.error("Please upload a file or provide a valid GitHub URL.")
        else:
            st.session_state.df = df.copy()
            st.session_state.df_original = df.copy()
            st.success("Step table loaded.")
            st.rerun()

