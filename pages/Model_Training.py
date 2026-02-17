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


st.set_page_config(page_title="Model Training - NDEDC", layout="wide")
st.title("Model Training")


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
                                train_rmse = mean_squared_error(y_train, train_preds, squared=False)
                                test_rmse = mean_squared_error(y_test, test_preds, squared=False)
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
            st.subheader("Live Prediction (Original Table Style)")
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
                template_df = pd.DataFrame(columns=ordered_cols)
                for col in ordered_cols:
                    if col != target_for_live and pd.api.types.is_numeric_dtype(df[col]):
                        template_df[col] = template_df[col].astype(float)
                if "mt_live_table_df" not in st.session_state or not isinstance(st.session_state.mt_live_table_df, pd.DataFrame):
                    st.session_state.mt_live_table_df = template_df.copy()
                editor_col_config = {}
                for col in ordered_cols:
                    if col != target_for_live and pd.api.types.is_numeric_dtype(df[col]):
                        editor_col_config[col] = st.column_config.NumberColumn(
                            col,
                            step=0.01,
                            format="%.6f",
                        )
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

                            live_result_df = live_full_df.copy()
                            live_result_df[target_for_live] = live_preds
                            st.session_state.mt_live_table_df = live_result_df.copy()
                            st.success("Live prediction completed.")
                            st.session_state.mt_live_alert = {
                                "model": selected_live_model,
                                "problem_type": problem_type,
                                "target": target_for_live,
                            }
                            st.rerun()
                    except Exception as e:
                        st.error(f"Live prediction failed: {e}")

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
                            st.session_state.mt_live_table_df.to_csv(export_path, index=False, encoding="utf-8-sig")
                            st.success(f"Live table saved to: {export_path}")
                        except Exception as e:
                            st.error(f"Failed to save live table: {e}")
else:
    st.info("Load a dataset first from session, PC upload, or GitHub URL.")
