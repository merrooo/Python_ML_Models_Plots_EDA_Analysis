import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import (
    list_project_folders,
    load_dataframe_from_source,
    render_step_table_loader,
    save_dataframe,
)
from utils.helpers import (
    checkbox_select_columns,
    detect_binary_columns,
    format_salary_labels,
    one_shot_checkbox,
)

from config.settings import (
    BALANCING_DISCUSSION,
    CLASSIFICATION_MODELS,
    ENCODING_DISCUSSION,
    REGRESSION_MODELS,
    TRANSFORMATION_DISCUSSION,
)

def render_date_features():
    """Render date features (split) interface"""
    st.markdown("Date Features (Split)")
    st.caption("Split date columns into day, month, year and one time column (hour+minute as total minutes).")
    
    # Optional: Load new table
    render_step_table_loader("date_split", "Date Split")
    
    if st.session_state.df is None:
        st.warning("No data loaded")
        return
    
    df = st.session_state.df
    all_cols = df.columns.tolist()
    
    date_cols = checkbox_select_columns(all_cols, "date_split", columns_per_row=3)
    
    if not date_cols:
        st.info("Select at least one date column")
        return
    
    drop_original = st.checkbox("Drop original date columns after split", value=True)
    
    # Preview date parsing
    with st.expander("Date Parsing Preview"):
        parse_summary = []
        for col in date_cols:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                valid = parsed.notna().sum()
                invalid = len(df) - valid
                parse_summary.append({
                    'Column': col,
                    'Valid Dates': valid,
                    'Invalid/Missing': invalid,
                    'Valid %': f"{(valid/len(df)*100):.1f}%"
                })
            except:
                parse_summary.append({
                    'Column': col,
                    'Valid Dates': 0,
                    'Invalid/Missing': len(df),
                    'Valid %': "0%"
                })
        
        st.dataframe(pd.DataFrame(parse_summary), use_container_width=True)
    
    # Preview after split
    preview_df = df.head(10).copy()
    for col in date_cols:
        try:
            parsed = pd.to_datetime(preview_df[col], errors='coerce')
            preview_df[f"{col}_day"] = parsed.dt.day
            preview_df[f"{col}_month"] = parsed.dt.month
            preview_df[f"{col}_year"] = parsed.dt.year
            preview_df[f"{col}_hour_minute"] = parsed.dt.hour * 60 + parsed.dt.minute
        except:
            preview_df[f"{col}_day"] = np.nan
            preview_df[f"{col}_month"] = np.nan
            preview_df[f"{col}_year"] = np.nan
            preview_df[f"{col}_hour_minute"] = np.nan
    
    if drop_original:
        preview_df = preview_df.drop(columns=date_cols, errors='ignore')
    
    st.markdown("**Preview after split:**")
    st.dataframe(preview_df, use_container_width=True)
    
    # Save options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        folders = list_project_folders(os.getcwd(), max_depth=4)
        save_folder = st.selectbox("Save folder:", folders, key="date_split_folder")
    
    with col2:
        default_name = st.session_state.get("date_split_output_file", "date_split_data.csv")
        file_name = st.text_input("File name:", value=default_name, key="date_split_filename")
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        st.session_state.date_split_output_file = file_name
    
    # Apply button
    if st.button("Apply Date Split", key="apply_date_split", type="primary"):
        try:
            work_df = df.copy()
            
            for col in date_cols:
                try:
                    parsed = pd.to_datetime(work_df[col], errors='coerce')
                    work_df[f"{col}_day"] = parsed.dt.day
                    work_df[f"{col}_month"] = parsed.dt.month
                    work_df[f"{col}_year"] = parsed.dt.year
                    work_df[f"{col}_hour_minute"] = parsed.dt.hour * 60 + parsed.dt.minute
                except:
                    st.warning(f"Could not parse column: {col}")
            
            if drop_original:
                work_df = work_df.drop(columns=date_cols, errors='ignore')
            
            # Update session state
            st.session_state.df = work_df
            st.session_state.date_split_applied = True
            st.session_state.date_split_columns = date_cols
            st.session_state.date_split_result_df = work_df.copy()
            
            # Save file
            success, path = save_dataframe(work_df, save_folder, file_name)
            if success:
                st.success(f"Date split applied and saved to {path}")
            else:
                st.warning(f"Date split applied but failed to save: {path}")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Show previous result
    if st.session_state.get("date_split_applied", False):
        st.markdown("---")
        st.subheader("ًPrevious Date Split Result")
        st.caption(f"Processed columns: {st.session_state.date_split_columns}")
        
        result_df = st.session_state.date_split_result_df
        st.dataframe(result_df.head(20), use_container_width=True)


def render_date_recency():
    """Render date recency/freshness interface"""
    st.markdown("Date Recency/Freshness")
    st.caption("Calculate years since last update (recency feature)")
    
    # Optional: Load new table
    render_step_table_loader("recency", "Date Recency")
    
    if st.session_state.df is None:
        st.warning("No data loaded")
        return
    
    df = st.session_state.df
    all_cols = df.columns.tolist()
    
    if not all_cols:
        st.info("No columns available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_year = st.selectbox(
            "Base year column (e.g., built/start year):",
            all_cols,
            key="recency_base"
        )
    
    with col2:
        update_year = st.selectbox(
            "Update year column (0 or null means no update):",
            all_cols,
            key="recency_update"
        )
    
    # Output feature name
    output_name = st.text_input(
        "Output feature name:",
        value="years_since_last_update",
        key="recency_output"
    ).strip()
    
    if not output_name:
        output_name = "years_since_last_update"
    
    # Reference year mode
    ref_mode = st.radio(
        "Reference year:",
        ["Fixed year", "Use reference column"],
        horizontal=True,
        key="recency_ref_mode"
    )
    
    if ref_mode == "Fixed year":
        ref_year = st.number_input(
            "Reference year:",
            min_value=1900,
            max_value=2100,
            value=2026,
            step=1,
            key="recency_fixed"
        )
        ref_values = None
    else:
        ref_values = st.selectbox(
            "Reference year column:",
            all_cols,
            key="recency_ref_col"
        )
        ref_year = None
    
    remove_old_dates = st.checkbox(
        "Remove old date columns after creating recency",
        value=False,
        key="recency_remove_old",
    )
    drop_candidates = [base_year, update_year]
    if ref_mode == "Use reference column" and ref_values is not None:
        drop_candidates.append(ref_values)
    drop_candidates = list(dict.fromkeys(drop_candidates))
    cols_to_remove = []
    if remove_old_dates:
        cols_to_remove = st.multiselect(
            "Select old date columns to remove:",
            options=drop_candidates,
            default=drop_candidates,
            key="recency_drop_cols",
        )
    
    # Preview
    preview_df = df.head(20).copy()
    
    base_s = pd.to_numeric(preview_df[base_year], errors='coerce')
    upd_s = pd.to_numeric(preview_df[update_year], errors='coerce')
    
    last_update = np.where((upd_s > 0) & upd_s.notna(), upd_s, base_s)
    
    if ref_mode == "Fixed year":
        ref_s = pd.Series(ref_year, index=preview_df.index)
    else:
        ref_s = pd.to_numeric(preview_df[ref_values], errors='coerce')
    
    recency = pd.Series(ref_s - last_update, index=preview_df.index).clip(lower=0)
    preview_df[output_name] = recency
    
    if remove_old_dates and cols_to_remove:
        preview_df = preview_df.drop(columns=cols_to_remove, errors='ignore')
    
    st.dataframe(preview_df, use_container_width=True)
    
    # Save options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        folders = list_project_folders(os.getcwd(), max_depth=4)
        save_folder = st.selectbox("Save folder:", folders, key="recency_folder")
    
    with col2:
        default_name = st.session_state.get("recency_output_file", "recency_data.csv")
        file_name = st.text_input("File name:", value=default_name, key="recency_filename")
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        st.session_state.recency_output_file = file_name
    
    # Apply button
    if st.button(" Apply Recency Transform", key="apply_recency", type="primary"):
        try:
            work_df = df.copy()
            
            base_s = pd.to_numeric(work_df[base_year], errors='coerce')
            upd_s = pd.to_numeric(work_df[update_year], errors='coerce')
            
            last_update = np.where((upd_s > 0) & upd_s.notna(), upd_s, base_s)
            
            if ref_mode == "Fixed year":
                ref_s = pd.Series(ref_year, index=work_df.index)
            else:
                ref_s = pd.to_numeric(work_df[ref_values], errors='coerce')
            
            work_df[output_name] = pd.Series(ref_s - last_update, index=work_df.index).clip(lower=0)
            
            if remove_old_dates and cols_to_remove:
                work_df = work_df.drop(columns=cols_to_remove, errors='ignore')
            
            # Update session state
            st.session_state.df = work_df
            st.session_state.recency_applied = True
            st.session_state.recency_result_df = work_df.copy()
            
            # Save file
            success, path = save_dataframe(work_df, save_folder, file_name)
            if success:
                st.success(f"Recency transform applied and saved to {path}")
            else:
                st.warning(f"Recency applied but failed to save: {path}")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Show previous result
    if st.session_state.get("recency_applied", False):
        st.markdown("---")
        st.subheader("ًPrevious Recency Result")
        
        result_df = st.session_state.recency_result_df
        st.dataframe(result_df.head(20), use_container_width=True)


def render_encoding():
    """Render encoding interface"""
    st.markdown("Encode Text Columns")
    st.caption("Convert categorical text to numeric values for ML models")
    
    # Optional: Load new table
    render_step_table_loader("encoding", "Encoding")
    
    if st.session_state.df is None:
        st.warning("No data loaded")
        return
    
    df = st.session_state.df
    
    # Get object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    if not obj_cols:
        st.success("No text columns found - all columns are already numeric")
        return
    
    # Display object columns summary
    st.info(f"Found {len(obj_cols)} text columns")
    
    summary_df = pd.DataFrame({
        'Column': obj_cols,
        'Non-Null': [df[col].notna().sum() for col in obj_cols],
        'Nulls': [df[col].isna().sum() for col in obj_cols],
        'Unique Values': [df[col].nunique() for col in obj_cols],
        'Sample': [str(df[col].dropna().iloc[0])[:50] if len(df[col].dropna()) > 0 else "NA" for col in obj_cols]
    })
    st.dataframe(summary_df, use_container_width=True)
    
    # Inspect column values
    with st.expander("Inspect Column Values"):
        view_col = st.selectbox("Select column to inspect:", obj_cols, key="enc_view_col")
        if view_col:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 20 values:**")
                st.dataframe(df[[view_col]].head(20), use_container_width=True)
            with col2:
                st.write("**Value frequencies (top 20):**")
                value_counts = df[view_col].astype(str).value_counts().head(20)
                st.dataframe(value_counts.to_frame("count"), use_container_width=True)
    
    # Encoding method selection
    method = st.radio(
        "Encoding method:",
        ["One-Hot Encoding", "Label Encoding", "Binary Encoding"],
        horizontal=True,
        key="enc_method"
    )
    
    # Method discussion
    with st.expander("Encoding Methods Explained"):
        st.dataframe(ENCODING_DISCUSSION, use_container_width=True, hide_index=True)
    
    # Column selection
    st.markdown("**Select columns to encode:**")
    selected_cols = checkbox_select_columns(obj_cols, "encoding", columns_per_row=3)
    
    if not selected_cols:
        st.info("Select at least one column to encode")
        return
    
    # Preview based on selected method
    preview_df = df.head(10).copy()
    
    if method == "One-Hot Encoding":
        preview_encoded = pd.get_dummies(preview_df[selected_cols], columns=selected_cols, drop_first=True, dtype=int)
        st.markdown("**Preview after One-Hot Encoding:**")
        st.dataframe(preview_encoded, use_container_width=True)
        st.caption(f"Columns will expand from {len(selected_cols)} to {preview_encoded.shape[1]} columns")
    
    elif method == "Label Encoding":
        st.markdown("**Label mappings (sample):**")
        for col in selected_cols[:3]:  # Show first 3 columns
            le = LabelEncoder()
            le.fit(df[col].astype(str).str.strip().str.lower())
            mapping_df = pd.DataFrame({
                'Original': le.classes_[:10],
                'Encoded': list(range(min(10, len(le.classes_))))
            })
            st.write(f"**{col}** - {len(le.classes_)} unique values")
            st.dataframe(mapping_df, use_container_width=True)
    
    elif method == "Binary Encoding":
        binary_preview = pd.DataFrame()
        bin_summary = []
        
        for col in selected_cols:
            col_data = df[col].astype(str).str.strip().str.lower()
            uniq = sorted(col_data.unique())
            code_map = {v: i+1 for i, v in enumerate(uniq)}
            bits = max(1, int(np.ceil(np.log2(len(code_map) + 1))))
            
            # Preview first 10 rows
            codes = col_data.head(10).map(code_map).fillna(0).astype(int)
            codes_arr = codes.to_numpy(dtype=np.int64)
            for b in range(bits):
                binary_preview[f"{col}_bin_{b+1}"] = np.right_shift(codes_arr, b) & 1
            
            bin_summary.append({
                'Column': col,
                'Categories': len(code_map),
                'Binary Columns': bits
            })
        
        st.markdown("**Preview after Binary Encoding:**")
        st.dataframe(binary_preview, use_container_width=True)
        st.dataframe(pd.DataFrame(bin_summary), use_container_width=True)
    
    # Save options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        folders = list_project_folders(os.getcwd(), max_depth=4)
        save_folder = st.selectbox("Save folder:", folders, key="encoding_folder")
    
    with col2:
        default_name = st.session_state.get("encoding_output_file", "encoded_data.csv")
        file_name = st.text_input("File name:", value=default_name, key="encoding_filename")
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        st.session_state.encoding_output_file = file_name
    
    # Apply button
    if st.button(" Apply Encoding", key="apply_encoding", type="primary"):
        try:
            work_df = df.copy()
            
            if method == "One-Hot Encoding":
                work_df = pd.get_dummies(work_df, columns=selected_cols, drop_first=True, dtype=int)
            
            elif method == "Label Encoding":
                le = LabelEncoder()
                for col in selected_cols:
                    if col.lower() == "salary":
                        salary_map = {"low": 0, "medium": 1, "high": 2}
                        work_df[col] = work_df[col].astype(str).str.strip().str.lower().map(salary_map)
                    else:
                        work_df[col] = le.fit_transform(
                            work_df[col].astype(str).str.strip().str.lower()
                        )
            
            elif method == "Binary Encoding":
                for col in selected_cols:
                    col_data = work_df[col].astype(str).str.strip().str.lower()
                    uniq = sorted(col_data.unique())
                    code_map = {v: i+1 for i, v in enumerate(uniq)}
                    bits = max(1, int(np.ceil(np.log2(len(code_map) + 1))))
                    codes = col_data.map(code_map).fillna(0).astype(int)
                    codes_arr = codes.to_numpy(dtype=np.int64)
                    
                    for b in range(bits):
                        work_df[f"{col}_bin_{b+1}"] = np.right_shift(codes_arr, b) & 1
                    
                    work_df = work_df.drop(columns=[col])
            
            # Update session state
            st.session_state.df = work_df
            st.session_state.encoding_applied = True
            st.session_state.encoding_method = method
            st.session_state.encoding_result_df = work_df.copy()
            
            # Save file
            success, path = save_dataframe(work_df, save_folder, file_name)
            if success:
                st.success(f"Encoding applied and saved to {path}")
            else:
                st.warning(f"Encoding applied but failed to save: {path}")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error applying encoding: {e}")
    
    # Show previous result
    if st.session_state.get("encoding_applied", False):
        st.markdown("---")
        st.subheader("ًPrevious Encoding Result")
        st.caption(f"Method: {st.session_state.encoding_method}")
        
        result_df = st.session_state.encoding_result_df
        st.dataframe(result_df.head(20), use_container_width=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer


def build_skew_comparison(df_before, df_after, cols):
    """Build skewness comparison table"""
    rows = []
    for c in cols:
        if c in df_before.columns and c in df_after.columns:
            skew_before = float(df_before[c].skew())
            skew_after = float(df_after[c].skew())
            rows.append({
                'Feature': c,
                'Skew Before': round(skew_before, 3),
                'Skew After': round(skew_after, 3),
                'Change': round(abs(skew_after) - abs(skew_before), 3),
                'Improved': 'Yes' if abs(skew_after) < abs(skew_before) else 'NO'
            })
    return pd.DataFrame(rows).sort_values('Skew Before', ascending=False)

def apply_transformation(df, cols, method):
    """Apply selected transformation to specified columns"""
    df_out = df.copy()
    
    if not cols:
        return df_out
    
    if method == "Min-Max Scaling (0 to 1)":
        scaler = MinMaxScaler()
        df_out[cols] = scaler.fit_transform(df_out[cols])
    
    elif method == "Standard Scaling (Z-score)":
        scaler = StandardScaler()
        df_out[cols] = scaler.fit_transform(df_out[cols])
    
    elif method == "Robust Scaling (Outlier-resistant)":
        scaler = RobustScaler()
        df_out[cols] = scaler.fit_transform(df_out[cols])
    
    elif method == "Log Transformation":
        for c in cols:
            s = pd.to_numeric(df_out[c], errors='coerce')
            if s.notna().any():
                min_val = float(s.min())
                shift = 0.0 if min_val > -1.0 else abs(min_val) + 1.001
                df_out[c] = np.log1p(s + shift)
    
    elif method == "Power Transform (Yeo-Johnson)":
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        df_out[cols] = scaler.fit_transform(df_out[cols])
    
    return df_out

def render_transformation():
    """Render transformation interface"""
    st.markdown("### ًں“گ 1.1.5 Feature Transformation")
    st.caption("Scale and transform numerical features to improve model performance")
    
    # Optional: Load new table
    render_step_table_loader("transform", "Transformation")
    
    if st.session_state.df is None:
        st.warning("No data loaded")
        return
    
    df = st.session_state.df
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns found for transformation")
        return
    
    st.info(f"Found {len(numeric_cols)} numeric columns")
    
    # Detect binary columns (should not be transformed)
    binary_cols = detect_binary_columns(df)
    if binary_cols:
        st.warning(f"Detected {len(binary_cols)} binary columns (0/1). These should NOT be transformed.")
        with st.expander("View binary columns"):
            st.write(binary_cols)
    
    # Method selection
    method = st.radio(
        "Transformation method:",
        [
            "Min-Max Scaling (0 to 1)",
            "Standard Scaling (Z-score)",
            "Robust Scaling (Outlier-resistant)",
            "Power Transform (Yeo-Johnson)",
            "Log Transformation"
        ],
        horizontal=True,
        key="transform_method"
    )
    
    # Method guidance
    with st.expander("When to use each method"):
        st.dataframe(TRANSFORMATION_DISCUSSION, use_container_width=True, hide_index=True)
    
    # Skewness analysis
    skew_threshold = st.slider("Skewness threshold:", 0.3, 2.0, 0.5, 0.1, key="skew_threshold")
    
    skew_values = df[numeric_cols].skew()
    right_skewed = [c for c in numeric_cols if skew_values[c] > skew_threshold]
    left_skewed = [c for c in numeric_cols if skew_values[c] < -skew_threshold]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Right-skewed (>{skew_threshold}):** {len(right_skewed)} features")
        if right_skewed:
            st.write(", ".join(right_skewed[:5]) + ("..." if len(right_skewed) > 5 else ""))
    with col2:
        st.markdown(f"**Left-skewed (<-{skew_threshold}):** {len(left_skewed)} features")
        if left_skewed:
            st.write(", ".join(left_skewed[:5]) + ("..." if len(left_skewed) > 5 else ""))
    
    # Column selection
    st.markdown("**Select features to transform:**")
    st.dataframe(df[numeric_cols].head(15), use_container_width=True, height=260)
    
    quick_select = st.radio(
        "Quick select:",
        ["All", "Right-skewed only", "None", "Manual"],
        horizontal=True,
        key="transform_quick_select"
    )
    
    # Sync checkbox state when quick-select mode changes
    prev_quick_select = st.session_state.get("transform_quick_select_prev")
    if prev_quick_select != quick_select:
        for col in numeric_cols:
            chk_key = f"transform_chk_{col}"
            if quick_select == "All":
                st.session_state[chk_key] = True
            elif quick_select == "Right-skewed only":
                st.session_state[chk_key] = col in right_skewed
            elif quick_select == "None":
                st.session_state[chk_key] = False
            elif chk_key not in st.session_state:
                st.session_state[chk_key] = False
        st.session_state.transform_quick_select_prev = quick_select
    
    # Create checkboxes
    cols_per_row = 4
    
    for i, col in enumerate(numeric_cols):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        chk_key = f"transform_chk_{col}"
        if chk_key not in st.session_state:
            if quick_select == "All":
                st.session_state[chk_key] = True
            elif quick_select == "Right-skewed only":
                st.session_state[chk_key] = col in right_skewed
            else:
                st.session_state[chk_key] = False
        
        with cols[i % cols_per_row]:
            st.checkbox(col, key=chk_key)
    
    selected_cols = [c for c in numeric_cols if st.session_state.get(f"transform_chk_{c}", False)]
    
    st.caption(f"Selected {len(selected_cols)} features for transformation")
    
    if not selected_cols:
        st.info("Select at least one feature to transform")
        return
    
    # Show skewness of selected features
    skew_selected = pd.DataFrame({
        'Feature': selected_cols,
        'Skewness': [df[c].skew() for c in selected_cols],
        '|Skewness|': [abs(df[c].skew()) for c in selected_cols]
    }).sort_values('|Skewness|', ascending=False)
    
    st.markdown("**Skewness of selected features:**")
    st.dataframe(skew_selected[['Feature', 'Skewness']], use_container_width=True)
    
    # Preview transformation
    preview_col = st.selectbox("Preview column:", selected_cols, key="transform_preview_col")
    
    if st.button("Preview Transformation", key="preview_transform"):
        preview_df = apply_transformation(df.head(100), selected_cols, method)
        st.session_state.transform_preview_df = preview_df
        st.session_state.transform_preview_method = method
    
    if "transform_preview_df" in st.session_state:
        st.markdown("**Transformation Preview:**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Before")
            st.dataframe(df[[preview_col]].head(10), use_container_width=True)
        with col2:
            st.caption("After")
            preview_result = st.session_state.transform_preview_df
            st.dataframe(preview_result[[preview_col]].head(10), use_container_width=True)
        
        # Skewness comparison
        if st.checkbox("Show skewness comparison", key="show_skew_compare"):
            compare_df = build_skew_comparison(
                df.head(100), 
                st.session_state.transform_preview_df, 
                selected_cols
            )
            st.dataframe(compare_df, use_container_width=True)
    
    # Save options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        folders = list_project_folders(os.getcwd(), max_depth=4)
        save_folder = st.selectbox("Save folder:", folders, key="transform_folder")
    
    with col2:
        default_name = st.session_state.get("transform_output_file", "transformed_data.csv")
        file_name = st.text_input("File name:", value=default_name, key="transform_filename")
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        st.session_state.transform_output_file = file_name
    
    # Apply button
    if st.button("Apply Transformation", key="apply_transform", type="primary"):
        try:
            # Store before state
            before_df = df.copy()
            
            # Apply transformation
            transformed_df = apply_transformation(df, selected_cols, method)
            
            # Update session state
            st.session_state.df = transformed_df
            st.session_state.transformation_applied = True
            st.session_state.transformation_method = method
            st.session_state.transformation_features = selected_cols
            st.session_state.transformation_before_df = before_df
            st.session_state.transformation_result_df = transformed_df.copy()
            
            # Save file
            success, path = save_dataframe(transformed_df, save_folder, file_name)
            if success:
                st.success(f"Transformation applied and saved to {path}")
            else:
                st.warning(f"Transformation applied but failed to save: {path}")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error applying transformation: {e}")
    
    # Show previous result
    if st.session_state.get("transformation_applied", False):
        st.markdown("---")
        st.subheader("Previous Transformation Result")
        st.caption(f"Method: {st.session_state.transformation_method}")
        
        result_df = st.session_state.transformation_result_df
        st.dataframe(result_df.head(20), use_container_width=True)
        
        # Skewness improvement
        if st.checkbox("Show skewness improvement", key="show_skew_improve"):
            before_df = st.session_state.transformation_before_df
            compare_df = build_skew_comparison(
                before_df, 
                result_df, 
                st.session_state.transformation_features
            )
            st.dataframe(compare_df, use_container_width=True)

def render_zero_handling():
    """Render zero handling interface"""
    st.markdown("Zero Handling (Replace 0 with Mean)")
    st.caption("Replace zero values with column mean (computed from non-zero values)")
    
    # Optional: Load new table
    render_step_table_loader("zero", "Zero Handling")
    
    if st.session_state.df is None:
        st.warning("No data loaded")
        return
    
    df = st.session_state.df
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns found")
        return
    
    # Find columns with zeros
    cols_with_zero = []
    zero_counts = {}
    zero_percentages = {}
    
    for col in numeric_cols:
        zero_count = (pd.to_numeric(df[col], errors='coerce') == 0).sum()
        if zero_count > 0:
            cols_with_zero.append(col)
            zero_counts[col] = zero_count
            zero_percentages[col] = (zero_count / len(df) * 100)
    
    if not cols_with_zero:
        st.success("No zero values found in numeric columns")
        return
    
    st.warning(f"Found {len(cols_with_zero)} columns with zero values")
    
    # Display zero statistics
    zero_stats = pd.DataFrame({
        'Column': cols_with_zero,
        'Zero Count': [zero_counts[col] for col in cols_with_zero],
        'Zero %': [f"{zero_percentages[col]:.2f}%" for col in cols_with_zero]
    }).sort_values('Zero Count', ascending=False)
    
    st.dataframe(zero_stats, use_container_width=True)
    
    # Column selection
    st.markdown("**Select columns to process:**")
    selected_cols = checkbox_select_columns(cols_with_zero, "zero", columns_per_row=3)
    
    if not selected_cols:
        st.info("Select at least one column to process")
        return
    
    # Calculate means (excluding zeros)
    means = {}
    for col in selected_cols:
        non_zero = pd.to_numeric(df[col], errors='coerce')
        non_zero = non_zero[non_zero != 0].dropna()
        means[col] = non_zero.mean() if len(non_zero) > 0 else np.nan
    
    # Preview effect
    preview_col = st.selectbox("Preview column:", selected_cols, key="zero_preview_col")
    
    preview_df = df.head(20).copy()
    for col in selected_cols:
        mask = (pd.to_numeric(preview_df[col], errors='coerce') == 0)
        preview_df.loc[mask, col] = means.get(col, np.nan)
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Before")
        st.dataframe(df[[preview_col]].head(20), use_container_width=True)
    with col2:
        st.caption("After Preview")
        st.dataframe(preview_df[[preview_col]].head(20), use_container_width=True)
    
    # Save options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        folders = list_project_folders(os.getcwd(), max_depth=4)
        save_folder = st.selectbox("Save folder:", folders, key="zero_folder")
    
    with col2:
        default_name = st.session_state.get("zero_output_file", "zero_handled_data.csv")
        file_name = st.text_input("File name:", value=default_name, key="zero_filename")
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        st.session_state.zero_output_file = file_name
    
    # Apply button
    if st.button("Apply Zero Handling", key="apply_zero", type="primary"):
        try:
            work_df = df.copy()
            applied_cols = []
            
            for col in selected_cols:
                s = pd.to_numeric(work_df[col], errors='coerce')
                zero_mask = (s == 0)
                
                if zero_mask.any():
                    non_zero_mean = s[~zero_mask & s.notna()].mean()
                    if pd.notna(non_zero_mean):
                        work_df.loc[zero_mask, col] = non_zero_mean
                        applied_cols.append(col)
            
            if applied_cols:
                # Update session state
                st.session_state.df = work_df
                st.session_state.zero_applied = True
                st.session_state.zero_columns = applied_cols
                st.session_state.zero_result_df = work_df.copy()
                
                # Save file
                success, path = save_dataframe(work_df, save_folder, file_name)
                if success:
                    st.success(f"Zero handling applied to {len(applied_cols)} columns and saved to {path}")
                else:
                    st.warning(f"Zero handling applied but failed to save: {path}")
                
                st.rerun()
            else:
                st.warning("No columns were modified")
                
        except Exception as e:
            st.error(f"Error applying zero handling: {e}")
    
    # Show previous result
    if st.session_state.get("zero_applied", False):
        st.markdown("---")
        st.subheader("ًPrevious Zero Handling Result")
        st.caption(f"Processed columns: {st.session_state.zero_columns}")
        
        result_df = st.session_state.zero_result_df
        st.dataframe(result_df.head(20), use_container_width=True)

from sklearn.utils import resample


# Check for SMOTE availability
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

def apply_smote_balancing(df, target_col, input_cols):
    """Apply SMOTE balancing"""
    if not HAS_SMOTE:
        return None, "SMOTE requires imbalanced-learn. Install with: pip install imbalanced-learn"
    
    # Get numeric input columns
    numeric_inputs = [c for c in input_cols if pd.api.types.is_numeric_dtype(df[c])]
    excluded = [c for c in input_cols if c not in numeric_inputs]
    
    if not numeric_inputs:
        return None, "SMOTE needs numeric input features. Encode text columns first."
    
    # Prepare data
    X = df[numeric_inputs].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Check class sizes
    class_counts = y.value_counts()
    min_class_size = class_counts.min()
    
    if min_class_size < 2:
        return None, f"SMOTE needs at least 2 samples in each class. Smallest class has {min_class_size} samples."
    
    # Apply SMOTE
    k_neighbors = min(5, min_class_size - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Reconstruct dataframe
        balanced_df = pd.DataFrame(X_resampled, columns=numeric_inputs)
        balanced_df[target_col] = y_resampled
        
        return balanced_df, None, numeric_inputs, excluded
    except Exception as e:
        return None, f"SMOTE failed: {str(e)}", None, None

def apply_resampling(df, target_col, method):
    """Apply oversampling or undersampling"""
    # Group by target class
    groups = [group for _, group in df.groupby(target_col)]
    
    if method == "Oversample":
        target_size = max(len(g) for g in groups)
        balanced_groups = [
            resample(g, replace=True, n_samples=target_size, random_state=42)
            for g in groups
        ]
    else:  # Undersample
        target_size = min(len(g) for g in groups)
        balanced_groups = [
            resample(g, replace=False, n_samples=target_size, random_state=42)
            for g in groups
        ]
    
    # Combine and shuffle
    balanced_df = pd.concat(balanced_groups, axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return balanced_df, None, [], []

def render_balancing():
    """Render class balancing interface"""
    st.markdown("Class Balancing")
    st.caption("Balance target classes for classification problems")
    
    # Optional: Load new table
    render_step_table_loader("balance", "Class Balancing")
    
    if st.session_state.df is None:
        st.warning("No data loaded")
        return
    
    df = st.session_state.df
    
    # Get target column
    target_col = st.session_state.get("model_output_col")
    if not target_col or target_col not in df.columns:
        available_cols = df.columns.tolist()
        target_col = st.selectbox(
            "Select target column for balancing:",
            available_cols,
            index=len(available_cols)-1 if available_cols else 0,
            key="balance_target_select"
        )
    else:
        st.info(f"Using target column: **{target_col}**")
    
    # Show class distribution before balancing
    st.markdown("**Class Distribution (Before):**")
    before_counts = df[target_col].value_counts().reset_index()
    before_counts.columns = ['Class', 'Count']
    before_counts['Percentage'] = (before_counts['Count'] / len(df) * 100).round(2)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(before_counts, use_container_width=True)
    with col2:
        # Simple bar chart
        st.bar_chart(before_counts.set_index('Class')['Count'])
    
    # Check if balancing is needed
    unique_classes = before_counts['Class'].nunique()
    if unique_classes <= 1:
        st.warning("Only one class found. Balancing requires at least 2 classes.")
        return
    
    # Input features selection
    available_inputs = [c for c in df.columns if c != target_col]
    st.markdown("**Select input features for balancing:**")
    input_cols = checkbox_select_columns(available_inputs, "balance_inputs", columns_per_row=3)
    
    if not input_cols:
        st.warning("Select at least one input feature")
        return
    
    # Balancing method selection
    method = st.radio(
        "Balancing method:",
        ["Oversample", "Undersample", "SMOTE"],
        horizontal=True,
        key="balance_method"
    )
    
    # Method explanation
    with st.expander("When to use each method"):
        st.dataframe(BALANCING_DISCUSSION, use_container_width=True, hide_index=True)
    
    # Preview button
    if st.button("Preview Balance", key="preview_balance"):
        with st.spinner("Generating preview..."):
            if method == "SMOTE":
                preview_df, error, used_features, excluded = apply_smote_balancing(
                    df, target_col, input_cols
                )
            else:
                preview_df, error, used_features, excluded = apply_resampling(
                    df, target_col, method
                )
            
            if error:
                st.error(error)
            else:
                st.session_state.balance_preview_df = preview_df
                st.session_state.balance_preview_method = method
                st.session_state.balance_preview_used = used_features
                st.session_state.balance_preview_excluded = excluded
    
    # Show preview if available
    if "balance_preview_df" in st.session_state:
        st.markdown("---")
        st.subheader("Preview Result")
        
        preview_df = st.session_state.balance_preview_df
        preview_method = st.session_state.balance_preview_method
        
        # Show class distribution after
        after_counts = preview_df[target_col].value_counts().reset_index()
        after_counts.columns = ['Class', 'Count']
        after_counts['Percentage'] = (after_counts['Count'] / len(preview_df) * 100).round(2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Class Distribution (After):**")
            st.dataframe(after_counts, use_container_width=True)
        with col2:
            st.markdown("**Preview Stats:**")
            st.metric("Total Rows", preview_df.shape[0])
            st.metric("Change", f"{preview_df.shape[0] - df.shape[0]:+d} rows")
        
        if preview_method == "SMOTE":
            if st.session_state.balance_preview_used:
                st.caption(f"SMOTE used numeric features: {st.session_state.balance_preview_used[:5]}")
            if st.session_state.balance_preview_excluded:
                st.caption(f"Excluded non-numeric features: {st.session_state.balance_preview_excluded[:5]}")
        
        st.dataframe(preview_df.head(20), use_container_width=True)
    
    # Save options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        folders = list_project_folders(os.getcwd(), max_depth=4)
        save_folder = st.selectbox("Save folder:", folders, key="balance_folder")
    
    with col2:
        default_name = st.session_state.get("balance_output_file", "balanced_data.csv")
        file_name = st.text_input("File name:", value=default_name, key="balance_filename")
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        st.session_state.balance_output_file = file_name
    
    # Apply button
    if st.button("Apply Balancing", key="apply_balance", type="primary"):
        with st.spinner("Applying balancing..."):
            if method == "SMOTE":
                balanced_df, error, used_features, excluded = apply_smote_balancing(
                    df, target_col, input_cols
                )
            else:
                balanced_df, error, used_features, excluded = apply_resampling(
                    df, target_col, method
                )
            
            if error:
                st.error(error)
            else:
                # Update session state
                st.session_state.df = balanced_df
                st.session_state.balance_applied = True
                st.session_state.balance_method = method
                st.session_state.balance_target = target_col
                st.session_state.balance_result_df = balanced_df.copy()
                st.session_state.balance_used_features = used_features
                st.session_state.balance_excluded_features = excluded
                
                # Save file
                success, path = save_dataframe(balanced_df, save_folder, file_name)
                if success:
                    st.success(f"Balancing applied and saved to {path}")
                else:
                    st.warning(f"Balancing applied but failed to save: {path}")
                
                # Clear preview
                if "balance_preview_df" in st.session_state:
                    del st.session_state.balance_preview_df
                
                st.rerun()
    
    # Show previous balancing result
    if st.session_state.get("balance_applied", False):
        st.markdown("---")
        st.subheader("ًPrevious Balancing Result")
        st.caption(f"Method: {st.session_state.balance_method}")
        
        result_df = st.session_state.balance_result_df
        
        # Show class distribution
        after_counts = result_df[st.session_state.balance_target].value_counts().reset_index()
        after_counts.columns = ['Class', 'Count']
        st.dataframe(after_counts, use_container_width=True)
        
        st.dataframe(result_df.head(20), use_container_width=True)

import matplotlib.pyplot as plt
import seaborn as sns


def render_eda_step():
    """Main EDA step renderer"""
    st.header("Step 2: EDA & Visualization")
    
    # =====================================================================
    # Data Loading for EDA
    # =====================================================================
    with st.expander("Load Data for EDA", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Upload from PC**")
            eda_file = st.file_uploader(
                "Choose CSV or Excel",
                type=["csv", "xlsx", "xls"],
                key="eda_upload"
            )
        
        with col2:
            st.markdown("**Load from GitHub**")
            eda_url = st.text_input(
                "GitHub URL:",
                placeholder="https://github.com/.../data.csv",
                key="eda_github"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Load for EDA", key="load_eda"):
                try:
                    df = load_dataframe_from_source(eda_file, eda_url)
                    if df is not None:
                        st.session_state.eda_df = df.copy()
                        st.success("Data loaded for EDA")
                        st.rerun()
                    else:
                        st.error("Please provide a file or valid URL")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
        
        with col2:
            if st.button("Use Current Dataset", key="use_current"):
                if st.session_state.df is not None:
                    st.session_state.eda_df = st.session_state.df.copy()
                    st.success("Using current dataset")
                    st.rerun()
                else:
                    st.warning("No current dataset found")
        
        with col3:
            if st.button("Clear EDA Data", key="clear_eda"):
                if "eda_df" in st.session_state:
                    del st.session_state.eda_df
                st.rerun()
    
    # =====================================================================
    # EDA Visualizations
    # =====================================================================
    if "eda_df" in st.session_state and st.session_state.eda_df is not None:
        df = st.session_state.eda_df
        
        # Dataset overview
        st.markdown("---")
        st.subheader(" Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data info table
        with st.expander("Data Info", expanded=True):
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Null %': (df.isnull().sum() / len(df) * 100).round(2).values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(info_df, use_container_width=True)
        
        # Full data preview
        with st.expander("Full Data Preview"):
            st.dataframe(df, use_container_width=True, height=400)
        
        # Get numeric columns for visualizations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for visualizations")
            return
        
        # =====================================================================
        # 1.3.1 Correlation Heatmap
        # =====================================================================
        st.markdown("---")
        st.subheader("1.3.1 Correlation Heatmap")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                corr_method = st.radio(
                    "Correlation method:",
                    ["pearson", "spearman", "kendall"],
                    horizontal=False,
                    key="corr_method"
                )
                
                show_values = st.checkbox("Show correlation values", value=True)
                
                # Feature selection for correlation
                selected_corr_cols = st.multiselect(
                    "Select features (optional):",
                    numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))]
                )
            
            with col2:
                plot_cols = selected_corr_cols if selected_corr_cols else numeric_cols
                
                if len(plot_cols) >= 2:
                    corr_matrix = df[plot_cols].corr(method=corr_method)
                    
                    fig, ax = plt.subplots(figsize=(max(8, len(plot_cols)), max(6, len(plot_cols)*0.8)))
                    sns.heatmap(
                        corr_matrix,
                        annot=show_values,
                        cmap="coolwarm",
                        center=0,
                        fmt='.2f' if show_values else '',
                        linewidths=0.5,
                        ax=ax
                    )
                    plt.title(f'{corr_method.capitalize()} Correlation Matrix')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show correlation values in table
                    if st.checkbox("Show correlation table", key="show_corr_table"):
                        st.dataframe(corr_matrix, use_container_width=True)
                else:
                    st.info("Select at least 2 features for correlation heatmap")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
        
        # =====================================================================
        # 1.3.2 Outlier Analysis
        # =====================================================================
        st.markdown("---")
        st.subheader("1.3.2 Outlier Analysis")
        
        # Outlier summary table
        outlier_stats = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
            
            outlier_stats.append({
                'Column': col,
                'Outliers': outliers,
                'Outlier %': f"{(outliers/len(df)*100):.2f}%",
                'Lower Bound': f"{lower:.4f}",
                'Upper Bound': f"{upper:.4f}"
            })
        
        outlier_df = pd.DataFrame(outlier_stats).sort_values('Outliers', ascending=False)
        st.dataframe(outlier_df, use_container_width=True)
        
        # Visualize outliers for selected column
        col1, col2 = st.columns(2)
        
        with col1:
            out_col = st.selectbox("Select column for box plot:", numeric_cols, key="outlier_col")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(x=df[out_col], ax=ax, color='#ff6b6b')
            ax.set_title(f'Box Plot - {out_col}')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Show outlier rows
            Q1 = df[out_col].quantile(0.25)
            Q3 = df[out_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[out_col] < lower) | (df[out_col] > upper)]
            
            st.metric(f"Outliers in {out_col}", len(outliers))
            if len(outliers) > 0:
                with st.expander(f"View {len(outliers)} outlier rows"):
                    st.dataframe(outliers, use_container_width=True)
        
        # =====================================================================
        # 1.3.3 Dispersion Analysis
        # =====================================================================
        st.markdown("---")
        st.subheader("1.3.3 Dispersion Analysis")
        
        dispersion_stats = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) > 0:
                mean_val = s.mean()
                std_val = s.std()
                var_val = s.var()
                iqr_val = s.quantile(0.75) - s.quantile(0.25)
                cv_val = std_val / mean_val if mean_val != 0 else np.nan
                range_val = s.max() - s.min()
                
                dispersion_stats.append({
                    'Column': col,
                    'Mean': f"{mean_val:.4f}",
                    'Std': f"{std_val:.4f}",
                    'Variance': f"{var_val:.4f}",
                    'IQR': f"{iqr_val:.4f}",
                    'Range': f"{range_val:.4f}",
                    'CV': f"{cv_val:.4f}" if not np.isnan(cv_val) else "N/A"
                })
        
        st.dataframe(pd.DataFrame(dispersion_stats), use_container_width=True)
        
        # =====================================================================
        # 1.3.4 Skewness Analysis
        # =====================================================================
        st.markdown("---")
        st.subheader("1.3.4 Skewness Analysis")
        
        skew_data = []
        for col in numeric_cols:
            skew = df[col].skew()
            skew_data.append({
                'Column': col,
                'Skewness': round(skew, 4),
                '|Skewness|': round(abs(skew), 4),
                'Direction': 'Right' if skew > 0.5 else 'Left' if skew < -0.5 else 'Symmetric',
                'Severity': 'High' if abs(skew) > 1 else 'Moderate' if abs(skew) > 0.5 else 'Low'
            })
        
        skew_df = pd.DataFrame(skew_data).sort_values('|Skewness|', ascending=False)
        st.dataframe(skew_df, use_container_width=True)
        
        # Visualize skewness
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar plot of skewness
            fig, ax = plt.subplots(figsize=(10, max(4, len(numeric_cols)*0.3)))
            colors = ['red' if x > 0.5 else 'blue' if x < -0.5 else 'gray' 
                     for x in skew_df['Skewness']]
            ax.barh(skew_df['Column'], skew_df['Skewness'], color=colors)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Skewness')
            ax.set_title('Feature Skewness')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Select column for distribution plot
            skew_col = st.selectbox("Select column for distribution plot:", numeric_cols, key="skew_col")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[skew_col].dropna(), kde=True, ax=ax, color='#4e79a7')
            ax.set_title(f'Distribution of {skew_col} (Skewness: {df[skew_col].skew():.3f})')
            st.pyplot(fig)
            plt.close()
        
        # =====================================================================
        # Additional: Distribution plots for all features
        # =====================================================================
        st.markdown("---")
        st.subheader("ًں“ˆ Distribution Plots for All Features")
        
        plot_all = st.checkbox("Show distribution plots for all numeric features", value=False)
        
        if plot_all:
            for col in numeric_cols[:20]:  # Limit to 20 features
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Histogram with KDE
                sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='#4e79a7')
                axes[0].set_title(f'Distribution of {col}')
                
                # Box plot
                sns.boxplot(y=df[col].dropna(), ax=axes[1], color='#ff6b6b')
                axes[1].set_title(f'Box Plot of {col}')
                
                st.pyplot(fig)
                plt.close()
        
        # =====================================================================
        # Additional: Value Frequency Analysis
        # =====================================================================
        st.markdown("---")
        st.subheader(" Value Frequency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            freq_col = st.selectbox("Select column for frequency analysis:", df.columns.tolist())
        
        with col2:
            top_n = st.slider("Number of top values:", 5, 50, 10)
        
        # Get value counts
        value_counts = df[freq_col].astype(str).value_counts().head(top_n)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(value_counts.to_frame('Count'), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, max(4, top_n*0.3)))
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')
            ax.set_title(f'Top {top_n} values in {freq_col}')
            st.pyplot(fig)
            plt.close()
    
    else:
        st.info(" Please load data for EDA analysis")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Import model classes
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def build_pipeline(problem_type, model_name):
    """Build sklearn pipeline with appropriate model"""
    model_name = str(model_name).strip()
    model_key = model_name.lower()
    canonical_names = {
        "decision tree": "Decision Tree",
        "random forest": "Random Forest",
        "gradient boosting": "Gradient Boosting",
        "adaboost": "AdaBoost",
        "extra trees": "Extra Trees",
        "logistic regression": "Logistic Regression",
        "svm": "SVM",
        "knn": "KNN",
        "naive bayes": "Naive Bayes",
        "mlp": "MLP",
        "xgboost": "XGBoost",
        "linear regression": "Linear Regression",
        "svr": "SVR",
    }
    model_name = canonical_names.get(model_key, model_name)

    needs_scaling = model_name in {
        "Logistic Regression", "SVM", "SVR", "KNN", "MLP", "Linear Regression"
    }
    
    steps = [("scaler", StandardScaler())] if needs_scaling else []
    
    if problem_type == "Classification":
        if model_name == "Decision Tree":
            estimator = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == "Gradient Boosting":
            estimator = GradientBoostingClassifier(random_state=42)
        elif model_name == "AdaBoost":
            estimator = AdaBoostClassifier(random_state=42)
        elif model_name == "Extra Trees":
            estimator = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        elif model_name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
        elif model_name == "SVM":
            estimator = SVC(kernel="rbf", random_state=42)
        elif model_name == "KNN":
            estimator = KNeighborsClassifier(n_jobs=-1)
        elif model_name == "Naive Bayes":
            estimator = GaussianNB()
        elif model_name == "MLP":
            estimator = MLPClassifier(hidden_layer_sizes=(100, 50), 
                                      max_iter=1000, random_state=42)
        elif model_name == "XGBoost" and HAS_XGBOOST:
            estimator = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
        elif model_name == "XGBoost" and not HAS_XGBOOST:
            raise ValueError("XGBoost is not installed. Install with: pip install xgboost")
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:  # Regression
        if model_name == "Decision Tree":
            estimator = DecisionTreeRegressor(random_state=42)
        elif model_name == "Random Forest":
            estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif model_name == "Gradient Boosting":
            estimator = GradientBoostingRegressor(random_state=42)
        elif model_name == "AdaBoost":
            estimator = AdaBoostRegressor(random_state=42)
        elif model_name == "Extra Trees":
            estimator = ExtraTreesRegressor(random_state=42, n_jobs=-1)
        elif model_name == "Linear Regression":
            estimator = LinearRegression(n_jobs=-1)
        elif model_name == "SVR":
            estimator = SVR(kernel="rbf")
        elif model_name == "KNN":
            estimator = KNeighborsRegressor(n_jobs=-1)
        elif model_name == "MLP":
            estimator = MLPRegressor(hidden_layer_sizes=(100, 50), 
                                    max_iter=1000, random_state=42)
        elif model_name == "XGBoost" and HAS_XGBOOST:
            estimator = XGBRegressor(random_state=42, n_jobs=-1)
        elif model_name == "XGBoost" and not HAS_XGBOOST:
            raise ValueError("XGBoost is not installed. Install with: pip install xgboost")
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    steps.append(("model", estimator))
    return Pipeline(steps)

def prepare_features(df, input_cols, fit_encoder=None):
    """Prepare features by encoding categorical variables"""
    X = df[input_cols].copy()
    
    if fit_encoder is None:
        # First time: create encoder
        encoder_info = {"maps": {}}
        for col in X.columns:
            if X[col].dtype == 'object':
                # Encode categorical
                unique_vals = sorted(X[col].astype(str).str.lower().unique())
                code_map = {val: i+1 for i, val in enumerate(unique_vals)}
                encoder_info["maps"][col] = code_map
                X[col] = X[col].astype(str).str.lower().map(code_map).fillna(0).astype(int)
        return X, encoder_info
    else:
        # Use existing encoder
        for col in X.columns:
            if col in fit_encoder.get("maps", {}):
                code_map = fit_encoder["maps"][col]
                X[col] = X[col].astype(str).str.lower().map(code_map).fillna(0).astype(int)
        return X, fit_encoder

def render_training_step():
    """Main model training step renderer"""
    st.header(" Step 3: Model Training")
    
    # Initialize session state for this step
    if 'mt_state' not in st.session_state:
        st.session_state.mt_state = {
            'df': None,
            'split': None,
            'results': None,
            'predictions': None
        }
    
    # =====================================================================
    # 1.3.1 Load Dataset
    # =====================================================================
    st.subheader("1.3.1 Load Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Use Current Dataset", key="use_current_mt"):
            if st.session_state.df is not None:
                st.session_state.mt_state['df'] = st.session_state.df.copy()
                st.success("Loaded current dataset")
                st.rerun()
            else:
                st.warning("No current dataset found")
    
    with col2:
        uploaded_file = st.file_uploader(
            "Upload from PC",
            type=["csv", "xlsx", "xls"],
            key="mt_upload",
            label_visibility="collapsed"
        )
        if uploaded_file and st.button("Load", key="load_mt_upload"):
            try:
                df = load_dataframe_from_source(uploaded_file, "")
                if df is not None:
                    st.session_state.mt_state['df'] = df
                    st.success(f"Loaded {uploaded_file.name}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col3:
        github_url = st.text_input(
            "GitHub URL:",
            placeholder="https://...",
            key="mt_github",
            label_visibility="collapsed"
        )
        if github_url and st.button("Load", key="load_mt_github"):
            try:
                df = load_dataframe_from_source(None, github_url)
                if df is not None:
                    st.session_state.mt_state['df'] = df
                    st.success("Loaded from GitHub")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # =====================================================================
    # 1.3.2 Data Preview and X/y Selection
    # =====================================================================
    if st.session_state.mt_state['df'] is not None:
        df = st.session_state.mt_state['df']
        
        st.markdown("---")
        st.subheader("1.3.2 Data Preview and X/y Selection")
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head(100), use_container_width=True, height=300)
        
        # Target column selection
        all_cols = df.columns.tolist()
        target_col = st.selectbox(
            "Select output/target column (y):",
            all_cols,
            index=len(all_cols)-1 if all_cols else 0,
            key="mt_target"
        )
        
        # Input features selection
        input_cols = [c for c in all_cols if c != target_col]
        
        st.markdown("**Select input features:**")
        
        # Quick selection modes
        select_mode = st.radio(
            "Select features:",
            ["Suggested", "All", "Numeric only", "None", "Manual"],
            horizontal=True,
            key="mt_select_mode"
        )
        
        # Calculate suggested features based on correlation
        if select_mode == "Suggested":
            try:
                y_num = pd.to_numeric(df[target_col], errors='coerce')
                numeric_inputs = [c for c in input_cols if pd.api.types.is_numeric_dtype(df[c])]
                
                if y_num.notna().sum() > 0 and numeric_inputs:
                    corr_df = df[numeric_inputs + [target_col]].copy()
                    corr_df[target_col] = y_num
                    corr = corr_df.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors='ignore').abs()
                    
                    if not corr.empty:
                        threshold = corr.median() if corr.notna().any() else 0.0
                        suggested = corr[corr >= threshold].index.tolist()
                        default_inputs = suggested + [c for c in input_cols if c not in numeric_inputs]
                    else:
                        default_inputs = input_cols
                else:
                    default_inputs = input_cols
            except:
                default_inputs = input_cols
        
        elif select_mode == "All":
            default_inputs = input_cols
        elif select_mode == "Numeric only":
            default_inputs = [c for c in input_cols if pd.api.types.is_numeric_dtype(df[c])]
        else:
            default_inputs = []
        
        # Sync checkbox state when selection mode changes
        prev_select_mode = st.session_state.get("mt_select_mode_prev")
        if prev_select_mode != select_mode:
            for col in input_cols:
                chk_key = f"mt_input_{col}"
                if select_mode in {"Suggested", "All", "Numeric only", "None"}:
                    st.session_state[chk_key] = col in default_inputs
                elif chk_key not in st.session_state:
                    st.session_state[chk_key] = False
            st.session_state.mt_select_mode_prev = select_mode

        # Create checkboxes
        cols_per_row = 4
        
        for i, col in enumerate(input_cols):
            if i % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            
            chk_key = f"mt_input_{col}"
            if chk_key not in st.session_state:
                st.session_state[chk_key] = col in default_inputs
            
            with cols[i % cols_per_row]:
                st.checkbox(col, key=chk_key)
        
        selected_inputs = [c for c in input_cols if st.session_state.get(f"mt_input_{c}", False)]
        
        st.caption(f"Selected {len(selected_inputs)} input features")
        st.session_state.mt_input_cols = selected_inputs
        
        if not selected_inputs:
            st.warning("Select at least one input feature")
            return
        
        # =====================================================================
        # 1.3.3 Split Data
        # =====================================================================
        st.markdown("---")
        st.subheader("1.3.3 Split Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test size:", 0.1, 0.4, 0.2, 0.05)
        with col2:
            val_size = st.slider("Validation size:", 0.0, 0.3, 0.1, 0.05)
        with col3:
            random_seed = st.number_input("Random seed:", 1, 9999, 42)
        
        shuffle = st.checkbox("Shuffle data", value=True)
        
        train_size = 1.0 - test_size - val_size
        if train_size <= 0:
            st.error("Train size must be positive. Reduce test/validation sizes.")
            return
        
        st.info(f"Split: Train {train_size:.0%} | Validation {val_size:.0%} | Test {test_size:.0%}")
        
        if st.button("Run Split", key="mt_run_split", type="primary"):
            with st.spinner("Splitting data..."):
                try:
                    # Prepare features
                    X, encoder = prepare_features(df, selected_inputs)
                    y = df[target_col]
                    
                    # Split data
                    if val_size > 0:
                        # Train + (Validation + Test)
                        X_train, X_temp, y_train, y_temp = train_test_split(
                            X, y, test_size=test_size + val_size, 
                            random_state=random_seed, shuffle=shuffle
                        )
                        # Split temp into validation and test
                        val_ratio = val_size / (test_size + val_size)
                        X_val, X_test, y_val, y_test = train_test_split(
                            X_temp, y_temp, test_size=1-val_ratio,
                            random_state=random_seed, shuffle=shuffle
                        )
                    else:
                        # No validation set
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size,
                            random_state=random_seed, shuffle=shuffle
                        )
                        X_val, y_val = None, None
                    
                    # Store split
                    st.session_state.mt_state['split'] = {
                        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                        'target_col': target_col,
                        'feature_encoder': encoder,
                        'input_cols': selected_inputs
                    }
                    
                    st.success("Data split completed")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Split failed: {e}")
        
        # =====================================================================
        # 1.3.4 Split Preview
        # =====================================================================
        if st.session_state.mt_state.get('split') is not None:
            split = st.session_state.mt_state['split']
            
            st.markdown("---")
            st.subheader("1.3.4 Split Preview")
            
            # Show split shapes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train", f"{split['X_train'].shape[0]} rows x {split['X_train'].shape[1]} cols")
            with col2:
                if split['X_val'] is not None:
                    st.metric("Validation", f"{split['X_val'].shape[0]} rows x {split['X_val'].shape[1]} cols")
                else:
                    st.metric("Validation", "None")
            with col3:
                st.metric("Test", f"{split['X_test'].shape[0]} rows x {split['X_test'].shape[1]} cols")
            
            # Preview tabs
            tabs = st.tabs(["Train", "Validation", "Test"])
            
            with tabs[0]:
                st.dataframe(split['X_train'].head(10), use_container_width=True)
                if split['y_train'] is not None:
                    st.dataframe(split['y_train'].head(10).to_frame(name=split['target_col']), 
                               use_container_width=True)
            
            with tabs[1]:
                if split['X_val'] is not None:
                    st.dataframe(split['X_val'].head(10), use_container_width=True)
                    if split['y_val'] is not None:
                        st.dataframe(split['y_val'].head(10).to_frame(name=split['target_col']), 
                                   use_container_width=True)
                else:
                    st.info("No validation set created")
            
            with tabs[2]:
                st.dataframe(split['X_test'].head(10), use_container_width=True)
                if split['y_test'] is not None:
                    st.dataframe(split['y_test'].head(10).to_frame(name=split['target_col']), 
                               use_container_width=True)
            
            # Save split files
            st.markdown("---")
            st.subheader("Save Split Files")
            
            col1, col2 = st.columns(2)
            with col1:
                save_folders = list_project_folders(os.getcwd(), max_depth=4)
                save_folder = st.selectbox("Save folder:", save_folders, key="mt_split_folder")
            with col2:
                file_prefix = st.text_input("File prefix:", value="data_split", key="mt_split_prefix")
            
            if st.button("Save Split Files", key="mt_save_split"):
                try:
                    target_dir = os.path.join(os.getcwd(), save_folder)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Save train
                    split['X_train'].to_csv(
                        os.path.join(target_dir, f"{file_prefix}_X_train.csv"), 
                        index=False, encoding='utf-8-sig'
                    )
                    split['y_train'].to_frame(name=split['target_col']).to_csv(
                        os.path.join(target_dir, f"{file_prefix}_y_train.csv"), 
                        index=False, encoding='utf-8-sig'
                    )
                    
                    # Save validation if exists
                    if split['X_val'] is not None:
                        split['X_val'].to_csv(
                            os.path.join(target_dir, f"{file_prefix}_X_val.csv"), 
                            index=False, encoding='utf-8-sig'
                        )
                        split['y_val'].to_frame(name=split['target_col']).to_csv(
                            os.path.join(target_dir, f"{file_prefix}_y_val.csv"), 
                            index=False, encoding='utf-8-sig'
                        )
                    
                    # Save test
                    split['X_test'].to_csv(
                        os.path.join(target_dir, f"{file_prefix}_X_test.csv"), 
                        index=False, encoding='utf-8-sig'
                    )
                    split['y_test'].to_frame(name=split['target_col']).to_csv(
                        os.path.join(target_dir, f"{file_prefix}_y_test.csv"), 
                        index=False, encoding='utf-8-sig'
                    )
                    
                    st.success(f"Split files saved to {target_dir}")
                    
                except Exception as e:
                    st.error(f"Error saving files: {e}")
            
            # =====================================================================
            # 1.4 Train Model
            # =====================================================================
            st.markdown("---")
            st.subheader("1.4 Train Model")
            
            # Determine problem type
            y_train = split['y_train']
            if y_train.dtype == 'object' or y_train.nunique() <= 20:
                default_problem = "Classification"
            else:
                default_problem = "Regression"
            
            problem_type = st.radio(
                "Problem type:",
                ["Classification", "Regression"],
                index=0 if default_problem == "Classification" else 1,
                horizontal=True,
                key="mt_problem_type"
            )
            
            # Model selection
            if problem_type == "Classification":
                model_options = [
                    "Logistic Regression", "Decision Tree", "Random Forest",
                    "Gradient Boosting", "AdaBoost", "Extra Trees", "SVM",
                    "KNN", "Naive Bayes", "MLP", "XGBoost"
                ]
                model_discussion = CLASSIFICATION_MODELS
            else:
                model_options = [
                    "Linear Regression", "Decision Tree", "Random Forest",
                    "Gradient Boosting", "AdaBoost", "Extra Trees", "SVR",
                    "KNN", "MLP", "XGBoost"
                ]
                model_discussion = REGRESSION_MODELS
            
            # Model discussion
            with st.expander("Model Selection Guide"):
                st.dataframe(model_discussion, use_container_width=True, hide_index=True)
            
            # Model checkboxes
            st.markdown("**Select models to evaluate:**")
            
            select_all = st.checkbox("Select all models", key="mt_select_all")
            prev_select_all = st.session_state.get("mt_select_all_prev")
            if prev_select_all is None or prev_select_all != select_all:
                for model in model_options:
                    st.session_state[f"mt_model_{model}"] = select_all
                st.session_state.mt_select_all_prev = select_all
            
            selected_models = []
            cols_per_row = 3
            
            for i, model in enumerate(model_options):
                if i % cols_per_row == 0:
                    cols = st.columns(cols_per_row)
                
                chk_key = f"mt_model_{model}"
                if chk_key not in st.session_state:
                    st.session_state[chk_key] = select_all
                
                with cols[i % cols_per_row]:
                    st.checkbox(model, key=chk_key)
            
            selected_models = [m for m in model_options if st.session_state.get(f"mt_model_{m}", False)]
            
            if not selected_models:
                st.warning("Select at least one model")
                return
            
            st.caption(f"Selected {len(selected_models)} models")
            
            # Training button
            if st.button("Evaluate Selected Models", key="mt_train", type="primary"):
                with st.spinner("Training models... This may take a moment."):
                    results = []
                    predictions = []
                    
                    X_train = split['X_train']
                    X_test = split['X_test']
                    y_train = split['y_train']
                    y_test = split['y_test']
                    target_name = split['target_col'].lower()
                    
                    for model_name in selected_models:
                        try:
                            # Build and train pipeline
                            pipeline = build_pipeline(problem_type, model_name)
                            
                            # Handle label encoding for classification
                            if problem_type == "Classification" and str(model_name).strip().lower() == "xgboost":
                                le = LabelEncoder()
                                y_train_encoded = le.fit_transform(y_train.astype(str))
                                pipeline.fit(X_train, y_train_encoded)
                                
                                # Predict
                                train_pred = pipeline.predict(X_train)
                                test_pred = pipeline.predict(X_test)
                                
                                # Decode
                                train_pred = le.inverse_transform(train_pred.astype(int))
                                test_pred = le.inverse_transform(test_pred.astype(int))
                                
                                # Calculate metrics
                                train_acc = accuracy_score(y_train.astype(str), train_pred.astype(str))
                                test_acc = accuracy_score(y_test.astype(str), test_pred.astype(str))
                                train_f1 = f1_score(y_train.astype(str), train_pred.astype(str), average='weighted')
                                test_f1 = f1_score(y_test.astype(str), test_pred.astype(str), average='weighted')
                                
                                # Determine fit status
                                gap = train_acc - test_acc
                                if train_acc >= 0.90 and gap >= 0.08:
                                    fit_status = "Likely overfitting"
                                elif train_acc < 0.75 and test_acc < 0.70:
                                    fit_status = "Likely underfitting"
                                else:
                                    fit_status = "Generalization acceptable"
                                
                                results.append({
                                    'Model': model_name,
                                    'Train Accuracy': round(train_acc, 4),
                                    'Test Accuracy': round(test_acc, 4),
                                    'Train F1': round(train_f1, 4),
                                    'Test F1': round(test_f1, 4),
                                    'Gap': round(gap, 4),
                                    'Fit Status': fit_status
                                })
                                
                            elif problem_type == "Classification":
                                pipeline.fit(X_train, y_train)
                                train_pred = pipeline.predict(X_train)
                                test_pred = pipeline.predict(X_test)
                                
                                train_acc = accuracy_score(y_train, train_pred)
                                test_acc = accuracy_score(y_test, test_pred)
                                train_f1 = f1_score(y_train, train_pred, average='weighted')
                                test_f1 = f1_score(y_test, test_pred, average='weighted')
                                
                                gap = train_acc - test_acc
                                if train_acc >= 0.90 and gap >= 0.08:
                                    fit_status = "Likely overfitting"
                                elif train_acc < 0.75 and test_acc < 0.70:
                                    fit_status = "Likely underfitting"
                                else:
                                    fit_status = "Generalization acceptable"
                                
                                results.append({
                                    'Model': model_name,
                                    'Train Accuracy': round(train_acc, 4),
                                    'Test Accuracy': round(test_acc, 4),
                                    'Train F1': round(train_f1, 4),
                                    'Test F1': round(test_f1, 4),
                                    'Gap': round(gap, 4),
                                    'Fit Status': fit_status
                                })
                                
                            else:  # Regression
                                pipeline.fit(X_train, y_train)
                                train_pred = pipeline.predict(X_train)
                                test_pred = pipeline.predict(X_test)
                                
                                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                                train_r2 = r2_score(y_train, train_pred)
                                test_r2 = r2_score(y_test, test_pred)
                                
                                gap = train_r2 - test_r2
                                if train_r2 >= 0.90 and gap >= 0.12:
                                    fit_status = "Likely overfitting"
                                elif train_r2 < 0.45 and test_r2 < 0.35:
                                    fit_status = "Likely underfitting"
                                else:
                                    fit_status = "Generalization acceptable"
                                
                                results.append({
                                    'Model': model_name,
                                    'Train RMSE': round(train_rmse, 4),
                                    'Test RMSE': round(test_rmse, 4),
                                    'Train R2': round(train_r2, 4),
                                    'Test R2': round(test_r2, 4),
                                    'Gap': round(gap, 4),
                                    'Fit Status': fit_status
                                })
                            
                            # Store predictions
                            pred_df = pd.DataFrame({
                                'Model': model_name,
                                'True Value': y_test.values,
                                'Predicted': test_pred
                            })
                            
                            if target_name == 'salary':
                                pred_df['True Value'] = format_salary_labels(pred_df['True Value'])
                                pred_df['Predicted'] = format_salary_labels(pred_df['Predicted'])
                            
                            predictions.append(pred_df)
                            
                        except Exception as e:
                            st.error(f"Error training {model_name}: {e}")
                            results.append({
                                'Model': model_name,
                                'Error': str(e)
                            })
                    
                    # Store results
                    results_df = pd.DataFrame(results)
                    
                    # Sort results
                    if problem_type == "Classification" and 'Test Accuracy' in results_df.columns:
                        results_df = results_df.sort_values('Test Accuracy', ascending=False)
                    elif 'Test Rآ²' in results_df.columns:
                        results_df = results_df.sort_values('Test Rآ²', ascending=False)
                    
                    st.session_state.mt_state['results'] = results_df
                    st.session_state.mt_state['predictions'] = pd.concat(predictions, ignore_index=True) if predictions else None
                    
                    st.success("Model evaluation completed")
                    st.rerun()
            
            # =====================================================================
            # Results Display
            # =====================================================================
            if st.session_state.mt_state.get('results') is not None:
                st.markdown("---")
                st.subheader("Evaluation Results")
                
                results_df = st.session_state.mt_state['results']
                
                # Highlight best model
                if 'Test Accuracy' in results_df.columns:
                    best_idx = results_df['Test Accuracy'].idxmax()
                    st.success(f"Best Model: **{results_df.loc[best_idx, 'Model']}** "
                              f"(Test Accuracy: {results_df.loc[best_idx, 'Test Accuracy']:.2%})")
                elif 'Test Rآ²' in results_df.columns:
                    best_idx = results_df['Test R2'].idxmax()
                    st.success(f"Best Model: **{results_df.loc[best_idx, 'Model']}** "
                              f"(Test R2:{results_df.loc[best_idx, 'Test R2']:.4f})")
                
                # Display results table
                st.dataframe(results_df, use_container_width=True)
                
                # Save results
                st.markdown("---")
                st.subheader("Save Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    save_folders = list_project_folders(os.getcwd(), max_depth=4)
                    save_folder = st.selectbox("Save folder:", save_folders, key="mt_results_folder")
                with col2:
                    file_name = st.text_input(
                        "File name:", 
                        value="model_results.csv",
                        key="mt_results_file"
                    )
                    if not file_name.endswith('.csv'):
                        file_name += '.csv'
                
                if st.button("Save Results", key="mt_save_results"):
                    success, path = save_dataframe(results_df, save_folder, file_name)
                    if success:
                        st.success(f"Results saved to {path}")
                    else:
                        st.error(f"Error saving: {path}")
                
                # =====================================================================
                # 1.4.1 Live Prediction
                # =====================================================================
                st.markdown("---")
                st.subheader("1.4.1 Live Prediction")
                
                # Get best model for live prediction
                if 'Test Accuracy' in results_df.columns:
                    best_model = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
                elif 'Test Rآ²' in results_df.columns:
                    best_model = results_df.loc[results_df['Test Rآ²'].idxmax(), 'Model']
                else:
                    best_model = selected_models[0]
                
                live_model = st.selectbox(
                    "Select model for live prediction:",
                    options=selected_models,
                    index=list(selected_models).index(best_model) if best_model in selected_models else 0,
                    key="mt_live_model"
                )
                
                # Create input template / upload mode
                st.markdown("**Enter values for prediction:**")
                
                # Get original columns
                original_cols = split['input_cols']
                original_df = st.session_state.mt_state['df']
                st.caption("Required input columns: " + ", ".join(original_cols))
                
                input_mode = st.radio(
                    "Prediction input source:",
                    ["Manual entry", "Upload test table"],
                    horizontal=True,
                    key="mt_live_input_source"
                )
                
                # Create input dataframe
                if 'mt_live_input' not in st.session_state:
                    # Create template with correct data types
                    template = pd.DataFrame(columns=original_cols)
                    for col in original_cols:
                        if col in original_df.columns and pd.api.types.is_numeric_dtype(original_df[col]):
                            template[col] = template[col].astype(float)
                    st.session_state.mt_live_input = template
                
                # Configure column editors
                column_config = {}
                for col in original_cols:
                    if col in original_df.columns and pd.api.types.is_numeric_dtype(original_df[col]):
                        column_config[col] = st.column_config.NumberColumn(
                            col,
                            step=0.0001,
                            format="%.6f"
                        )
                
                if input_mode == "Manual entry":
                    # Data editor for manual input
                    live_input_df = st.data_editor(
                        st.session_state.mt_live_input,
                        num_rows="dynamic",
                        use_container_width=True,
                        height=200,
                        key="mt_live_editor",
                        column_config=column_config
                    )
                else:
                    uploaded_live = st.file_uploader(
                        "Upload test table (CSV/Excel)",
                        type=["csv", "xlsx", "xls"],
                        key="mt_live_upload_file",
                    )
                    live_url = st.text_input(
                        "Or paste direct/Raw GitHub URL",
                        placeholder="https://github.com/user/repo/blob/main/test.csv",
                        key="mt_live_upload_url",
                    )
                    loaded_live_df = load_dataframe_from_source(uploaded_live, live_url)
                    if loaded_live_df is None:
                        st.info("Upload a test table or provide a valid GitHub URL.")
                        live_input_df = pd.DataFrame(columns=original_cols)
                    else:
                        st.caption(f"Loaded test table: {loaded_live_df.shape[0]} rows | {loaded_live_df.shape[1]} columns")
                        live_input_df = st.data_editor(
                            loaded_live_df,
                            num_rows="dynamic",
                            use_container_width=True,
                            height=260,
                            key="mt_live_editor_upload",
                            column_config=column_config
                        )
                
                if st.button(" Predict", key="mt_live_predict", type="primary"):
                    if live_input_df.empty:
                        st.warning("Please enter at least one row for prediction")
                    else:
                        try:
                            missing_cols = [c for c in original_cols if c not in live_input_df.columns]
                            if missing_cols:
                                st.error("Uploaded table is missing required columns: " + ", ".join(missing_cols))
                                return
                            
                            pred_input_df = live_input_df[original_cols].copy()
                            
                            # Prepare features
                            X_live, _ = prepare_features(pred_input_df, original_cols, split['feature_encoder'])
                            
                            # Train the selected model
                            pipeline = build_pipeline(problem_type, live_model)
                            
                            if problem_type == "Classification" and str(live_model).strip().lower() == "xgboost":
                                le = LabelEncoder()
                                y_train_encoded = le.fit_transform(split['y_train'].astype(str))
                                pipeline.fit(split['X_train'], y_train_encoded)
                                predictions = pipeline.predict(X_live)
                                predictions = le.inverse_transform(predictions.astype(int))
                            else:
                                pipeline.fit(split['X_train'], split['y_train'])
                                predictions = pipeline.predict(X_live)
                            
                            # Format results
                            result_df = live_input_df.copy()
                            
                            if split['target_col'].lower() == 'salary':
                                result_df['predicted_' + split['target_col']] = format_salary_labels(predictions)
                            elif problem_type == "Classification":
                                result_df['predicted_' + split['target_col']] = predictions.astype(str)
                            else:
                                # Check if target should be integer
                                if pd.api.types.is_integer_dtype(original_df[split['target_col']]):
                                    result_df['predicted_' + split['target_col']] = np.round(predictions).astype(int)
                                else:
                                    result_df['predicted_' + split['target_col']] = predictions
                            
                            st.success("Prediction completed!")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Save predictions
                            if st.button("Save Live Predictions", key="mt_save_live"):
                                success, path = save_dataframe(result_df, save_folder, "live_predictions.csv")
                                if success:
                                    st.success(f"Live predictions saved to {path}")
                                else:
                                    st.error(f"Error saving: {path}")
                            
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
    
    else:
        st.info("Please load a dataset to begin training")



def _ensure_base_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_original" not in st.session_state:
        st.session_state.df_original = None


def _render_base_loader():
    st.subheader("1.1 Data Loading")
    render_step_table_loader("base_cleaning", "Cleaning")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.dataframe(df.head(20), use_container_width=True, height=260)

        with st.expander("Inspect data types", expanded=False):
            info_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Null Count": df.isna().sum().values,
                    "Unique": [df[c].nunique(dropna=True) for c in df.columns],
                }
            )
            st.dataframe(info_df, use_container_width=True)


def render_drop_columns():
    st.markdown("### 1.1.1 Drop Columns")
    st.caption("Remove selected columns from the current table.")

    render_step_table_loader("drop_cols", "Drop Columns")

    if st.session_state.df is None:
        st.warning("No data loaded")
        return

    df = st.session_state.df
    all_cols = df.columns.tolist()

    if not all_cols:
        st.warning("Dataset has no columns")
        return

    cols_to_drop = checkbox_select_columns(all_cols, "drop_cols", columns_per_row=3)
    preview_df = df.drop(columns=cols_to_drop, errors="ignore") if cols_to_drop else df.copy()

    left, right = st.columns(2)
    with left:
        st.markdown("**Table Before**")
        st.dataframe(df.head(20), use_container_width=True, height=260)
    with right:
        st.markdown("**Table After**")
        st.dataframe(preview_df.head(20), use_container_width=True, height=260)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        folders = list_project_folders(os.getcwd(), max_depth=4)
        save_folder = st.selectbox("Save folder:", folders, key="drop_cols_folder")
    with c2:
        default_name = st.session_state.get("drop_cols_output_file", "dropped_columns_data.csv")
        file_name = st.text_input("File name:", value=default_name, key="drop_cols_filename")
        if not file_name.endswith(".csv"):
            file_name += ".csv"
        st.session_state.drop_cols_output_file = file_name

    if st.button("Apply Drop Columns", key="apply_drop_cols", type="primary"):
        if not cols_to_drop:
            st.warning("Select at least one column to drop.")
            return

        try:
            work_df = df.drop(columns=cols_to_drop, errors="ignore")
            st.session_state.df = work_df
            st.session_state.drop_cols_result_df = work_df.copy()
            st.session_state.drop_cols_applied = True

            success, path = save_dataframe(work_df, save_folder, file_name)
            if success:
                st.success(f"Drop columns applied and saved to {path}")
            else:
                st.warning(f"Drop columns applied but failed to save: {path}")
            st.rerun()
        except Exception as e:
            st.error(f"Error applying drop columns: {e}")


def _render_cleaning_step():
    st.header("Step 1: Cleaning Data")
    _render_base_loader()

    st.markdown("---")
    st.subheader("1.1 Processing Tools")
    tabs = st.tabs(
        [
            "Drop Columns",
            "Date Features",
            "Date Recency/Freshness",
            "Encoding",
            "Transformation",
            "Zero Handling",
            "Class Balancing",
        ]
    )

    with tabs[0]:
        render_drop_columns()
    with tabs[1]:
        render_date_features()
    with tabs[2]:
        render_date_recency()
    with tabs[3]:
        render_encoding()
    with tabs[4]:
        render_transformation()
    with tabs[5]:
        render_zero_handling()
    with tabs[6]:
        render_balancing()


_ensure_base_session_state()
st.set_page_config(
    page_title="Cleaning, EDA, and Model Training",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        max-width: 1280px;
    }
    .app-hero {
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 10px;
        background: linear-gradient(120deg, #f4f9ff 0%, #f8fcff 50%, #eef5ff 100%);
        border: 1px solid #dce8fb;
    }
    .app-hero h1 {
        margin: 0;
        font-size: 1.8rem;
        letter-spacing: 0.2px;
        color: #123a67;
    }
    .app-hero p {
        margin: 6px 0 0 0;
        color: #36587c;
        font-size: 0.95rem;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    div[data-baseweb="tab-list"] button {
        border-radius: 10px;
        padding: 0.3rem 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-hero">
        <h1>Cleaning, EDA, and Model Training</h1>
        <p>Prepare data, explore patterns, train models, and run live predictions in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Workflow")
section = st.sidebar.radio(
    "Go to section",
    ["Cleaning Data", "EDA", "Model Training"],
    key="main_section_nav",
)

if section == "Cleaning Data":
    _render_cleaning_step()
elif section == "EDA":
    render_eda_step()
else:
    render_training_step()
