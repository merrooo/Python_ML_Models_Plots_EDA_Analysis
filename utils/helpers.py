import numpy as np
import pandas as pd
import streamlit as st


def one_shot_checkbox(label: str, key: str) -> bool:
    fired = bool(st.session_state.get(key, False))
    if fired:
        st.session_state[key] = False
    st.checkbox(label, key=key)
    return fired


def checkbox_select_columns(options: list[str], key_prefix: str, columns_per_row: int = 3):
    if not options:
        return []
    mode = st.radio(
        "Select columns mode:",
        ["All", "None", "Manual"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )
    prev_mode = st.session_state.get(f"{key_prefix}_mode_prev")
    if prev_mode != mode:
        for col_name in options:
            st.session_state[f"{key_prefix}_chk_{col_name}"] = (mode == "All")
        st.session_state[f"{key_prefix}_mode_prev"] = mode
    grid = st.columns(columns_per_row)
    for idx, col_name in enumerate(options):
        chk_key = f"{key_prefix}_chk_{col_name}"
        if chk_key not in st.session_state:
            st.session_state[chk_key] = (mode == "All")
        with grid[idx % columns_per_row]:
            st.checkbox(col_name, key=chk_key)
    return [c for c in options if st.session_state.get(f"{key_prefix}_chk_{c}", False)]


def detect_binary_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        vals = pd.Series(df[c].dropna().unique())
        try:
            vals = vals.astype(float)
            if len(vals) > 0 and set(np.round(vals, 10)).issubset({0.0, 1.0}):
                cols.append(c)
        except Exception:
            continue
    return cols


def format_salary_labels(values) -> pd.Series:
    s = pd.Series(values).astype(str).str.strip().str.lower()
    code_to_label = {0: "low", 1: "medium", 2: "high"}
    numeric = pd.to_numeric(s, errors="coerce")
    out = []
    for raw, num in zip(s.tolist(), numeric.tolist()):
        if pd.notna(num):
            out.append(code_to_label.get(int(round(float(num))), str(raw)))
        else:
            out.append(raw)
    return pd.Series(out)

