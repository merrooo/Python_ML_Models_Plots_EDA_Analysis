import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

st.set_page_config(page_title="Model Training - NDEDC", layout="wide")
st.title("🤖 تدريب نماذج الذكاء الاصطناعي")

  # --- 6. Train/Test/Validation Split ---
    with st.expander("6) Split Data"):
        target_var = st.selectbox(
            "Choose output/target column (y):",
            options=st.session_state.df.columns.tolist(),
        )
        test_size = st.slider("Test size (final):", 0.05, 0.4, 0.2, 0.05)
        max_val_size = max(0.05, round(0.8 - test_size, 2))
        default_val_size = min(0.1, max_val_size)
        val_size = st.slider(
            "Validation size (final):",
            0.05,
            max_val_size,
            default_val_size,
            0.05,
        )
        train_size = 1.0 - test_size - val_size
        st.caption(
            f"Split plan: Train={train_size:.0%}, Validation={val_size:.0%}, Test={test_size:.0%}"
        )

        if one_shot_checkbox("Run Final Split", "run_final_split_chk"):
            holdout_size = test_size + val_size
            if holdout_size >= 1.0:
                st.error("Invalid split sizes. Train size must be greater than 0.")
            else:
                X = st.session_state.df.drop(columns=[target_var])
                y = st.session_state.df[target_var]
                X = pd.get_dummies(X, drop_first=True, dtype=int)
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=holdout_size, random_state=42
                )
                test_ratio_in_temp = test_size / holdout_size
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=test_ratio_in_temp, random_state=42
                )

                st.session_state.split_done = True
                st.session_state.split_summary = {
                    "rows_total": int(len(X)),
                    "rows_train": int(len(X_train)),
                    "rows_val": int(len(X_val)),
                    "rows_test": int(len(X_test)),
                    "target": target_var,
                    "train_size": train_size,
                    "val_size": val_size,
                    "test_size": test_size,
                }
                st.session_state.split_preview = {
                    "X_train": X_train.head(10),
                    "X_val": X_val.head(10),
                    "X_test": X_test.head(10),
                    "y_train": y_train.head(10).to_frame(name=target_var),
                    "y_val": y_val.head(10).to_frame(name=target_var),
                    "y_test": y_test.head(10).to_frame(name=target_var),
                }
                st.session_state.split_data = {
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train.to_frame(name=target_var),
                    "y_val": y_val.to_frame(name=target_var),
                    "y_test": y_test.to_frame(name=target_var),
                }
                st.success(
                    "Split completed in memory. Review samples below, then choose folder/path and click "
                    "'Save inside Project' to write files."
                )

        if st.session_state.get("split_done", False):
            summary = st.session_state.get("split_summary", {})
            st.markdown("---")
            st.subheader("Split Result Summary")
            st.write(
                f"Target: `{summary.get('target', 'N/A')}` | "
                f"Train={summary.get('train_size', 0):.0%}, "
                f"Validation={summary.get('val_size', 0):.0%}, "
                f"Test={summary.get('test_size', 0):.0%}"
            )
            st.write(
                f"Rows -> Total={summary.get('rows_total', 0)}, "
                f"Train={summary.get('rows_train', 0)}, "
                f"Validation={summary.get('rows_val', 0)}, "
                f"Test={summary.get('rows_test', 0)}"
            )

            preview = st.session_state.get("split_preview", {})
            t1, t2, t3 = st.tabs(["Train Sample", "Validation Sample", "Test Sample"])
            with t1:
                st.write("X_train (sample)")
                st.dataframe(preview.get("X_train", pd.DataFrame()), use_container_width=True, height=240)
                st.write("y_train (sample)")
                st.dataframe(preview.get("y_train", pd.DataFrame()), use_container_width=True, height=180)
            with t2:
                st.write("X_val (sample)")
                st.dataframe(preview.get("X_val", pd.DataFrame()), use_container_width=True, height=240)
                st.write("y_val (sample)")
                st.dataframe(preview.get("y_val", pd.DataFrame()), use_container_width=True, height=180)
            with t3:
                st.write("X_test (sample)")
                st.dataframe(preview.get("X_test", pd.DataFrame()), use_container_width=True, height=240)
                st.write("y_test (sample)")
                st.dataframe(preview.get("y_test", pd.DataFrame()), use_container_width=True, height=180)

            st.markdown("---")
            st.subheader("Save Split Files Inside Project")
            project_folders = list_project_folders(os.getcwd(), max_depth=4)
            selected_folder = st.selectbox(
                "Save inside project folder:",
                options=project_folders,
                index=0,
                key="split_project_folder",
            )
            file_prefix = st.text_input(
                "File name prefix:",
                value="final_process_split",
                key="split_file_prefix",
            ).strip()
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

            if st.button("Save inside Project", key="save_split_to_project_btn"):
                try:
                    target_dir = os.path.join(os.getcwd(), selected_folder)
                    os.makedirs(target_dir, exist_ok=True)
                    split_data = st.session_state.get("split_data", {})
                    for key_name, df_part in split_data.items():
                        out_path = os.path.join(target_dir, split_file_map[key_name])
                        df_part.to_csv(out_path, index=False, encoding="utf-8-sig")
                    st.success(f"Split files saved to: {target_dir}")
                except Exception as e:
                    st.error(f"Failed to save split files: {e}")



# التأكد من وجود ملفات التقسيم
if os.path.exists("X_train.csv"):
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")

    st.success("✅ تم العثور على بيانات التدريب والاختبار بنجاح!")

    col1, col2 = st.columns(2)
    with col1:
        problem_type = st.selectbox("حدد نوع المشكلة:", ["تصنيف (Classification)", "توقع رقمي (Regression)"])
    
    with col2:
        model_name = st.selectbox("اختر النموذج:", ["Random Forest", "Linear/Logistic Regression"])

    start_training_trigger = bool(st.session_state.get("start_training_chk", False))
    if start_training_trigger:
        st.session_state.start_training_chk = False
    st.checkbox("🚀 بدء تدريب النموذج", key="start_training_chk")
    if start_training_trigger:
        with st.spinner("جاري التدريب..."):
            if problem_type == "تصنيف (Classification)":
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train.values.ravel())
                predictions = model.predict(X_test)
                acc = accuracy_score(y_test, predictions)
                st.metric("دقة النموذج (Accuracy)", f"{acc:.2%}")
            else:
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train.values.ravel())
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                st.metric("جودة التوقع (R2 Score)", f"{r2:.2f}")
                
            st.balloons()
            st.success("تم الانتهاء من التدريب والتقييم!")
else:
    st.warning("⚠️ يرجى إجراء عملية Train-Test Split من صفحة التنظيف أولاً.")
