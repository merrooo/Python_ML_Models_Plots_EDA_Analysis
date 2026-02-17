import os

import pandas as pd
import streamlit as st
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

    input_options = [c for c in all_cols if c != target_col]
    default_inputs = st.session_state.get("mt_input_cols", input_options)
    default_inputs = [c for c in default_inputs if c in input_options]
    input_cols = st.multiselect(
        "Select input/feature columns (X):",
        options=input_options,
        default=default_inputs if default_inputs else input_options,
        key="mt_input_cols",
    )

    if not input_cols:
        st.warning("Select at least one input feature column.")
    else:
        X = df[input_cols].copy()
        y = df[target_col].copy()

        st.markdown("---")
        st.subheader("3) Split Data")
        test_size = st.slider("Test size:", 0.05, 0.4, 0.2, 0.05)
        val_size = st.slider("Validation size:", 0.05, 0.3, 0.1, 0.05)
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
                        random_state=42,
                    )
                    test_ratio_in_temp = test_size / holdout_size
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp,
                        y_temp,
                        test_size=test_ratio_in_temp,
                        random_state=42,
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
                model_name = st.selectbox("Model:", model_options, key="mt_model_cls")
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
                model_name = st.selectbox("Model:", model_options, key="mt_model_reg")

            st.markdown("**Model Discussion (when to use each):**")
            st.dataframe(model_discussion_table(problem_type), use_container_width=True, height=280)
            st.caption("All models are trained through a scikit-learn Pipeline in this page.")

            if st.button("Train and Evaluate", key="mt_train_btn"):
                try:
                    X_train = split["X_train"]
                    X_test = split["X_test"]
                    y_train = split["y_train"]
                    y_test = split["y_test"]
                    pipeline = build_pipeline(problem_type, model_name)

                    if problem_type == "Classification":
                        if model_name == "XGBoost":
                            le = LabelEncoder()
                            y_train_fit = le.fit_transform(y_train.astype(str))
                            y_test_eval = le.transform(y_test.astype(str))
                            pipeline.fit(X_train, y_train_fit)
                            preds = pipeline.predict(X_test)
                            acc = accuracy_score(y_test_eval, preds)
                            f1 = f1_score(y_test_eval, preds, average="weighted")
                        else:
                            pipeline.fit(X_train, y_train)
                            preds = pipeline.predict(X_test)
                            acc = accuracy_score(y_test, preds)
                            f1 = f1_score(y_test, preds, average="weighted")
                        st.metric("Accuracy", f"{acc:.4f}")
                        st.metric("F1 (weighted)", f"{f1:.4f}")
                    else:
                        pipeline.fit(X_train, y_train)
                        preds = pipeline.predict(X_test)
                        rmse = mean_squared_error(y_test, preds, squared=False)
                        r2 = r2_score(y_test, preds)
                        st.metric("RMSE", f"{rmse:.4f}")
                        st.metric("R2", f"{r2:.4f}")

                    st.success("Training completed.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
else:
    st.info("Load a dataset first from session, PC upload, or GitHub URL.")
