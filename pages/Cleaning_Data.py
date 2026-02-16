import pandas as pd
import streamlit as st
import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. إعداد الصفحة
st.set_page_config(page_title="Data Cleaning - NDEDC", layout="wide")

st.title("🛠️ مرحلة التنظيف النهائي وتحميل الملفات")

# دالة مساعدة لتحويل DataFrame إلى CSV جاهز للتحميل
def convert_df(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8-sig')


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
        return pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='ignore')

    if github_url and github_url.strip():
        normalized_url = normalize_github_url(github_url)
        lower_url = normalized_url.lower()
        if lower_url.endswith((".xlsx", ".xls")):
            response = pd.read_excel(normalized_url)
            return response
        return pd.read_csv(normalized_url, encoding='utf-8', encoding_errors='ignore')

    return None

# ------------------------------------------------------------------
# 2. نظام إدارة رفع الملفات (يختفي بعد الرفع)
# ------------------------------------------------------------------
if 'df' not in st.session_state:
    st.info("👋 يرجى رفع ملف بيانات للبدء.")
    upload_col, github_col = st.columns(2)

    with upload_col:
        uploaded_file = st.file_uploader("اختر ملف CSV أو Excel", type=['csv', 'xlsx', 'xls'])

    with github_col:
        github_url = st.text_input(
            "أو ضع رابط GitHub مباشر/Raw للملف",
            placeholder="https://github.com/user/repo/blob/main/data.csv",
        )
        load_from_github = st.button("تحميل من GitHub")

    if uploaded_file is not None:
        st.session_state.df = load_dataframe_from_source(uploaded_file, "")
        st.rerun()

    if load_from_github:
        try:
            df = load_dataframe_from_source(None, github_url)
            if df is None:
                st.error("يرجى إدخال رابط GitHub صحيح.")
            else:
                st.session_state.df = df
                st.rerun()
        except Exception as e:
            st.error(f"فشل التحميل من GitHub: {e}")
else:
    if st.sidebar.button("🔄 رفع ملف مختلف"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ------------------------------------------------------------------
# 3. أدوات المعالجة والمعاينة
# ------------------------------------------------------------------
if 'df' in st.session_state:
    st.markdown("---")
    
    # تقسيم الصفحة
    col_tools, col_preview = st.columns([1, 2])

    with col_tools:
        st.subheader("⚙️ أدوات المعالجة")

        # --- 1. حذف الأعمدة ---
        with st.expander("🗑️ 1- حذف أعمدة"):
            all_columns = st.session_state.df.columns.tolist()
            cols_to_drop = st.multiselect("اختر الأعمدة لحذفها:", options=all_columns)
            
            if st.button("❌ تأكيد الحذف"):
                if cols_to_drop:
                    st.session_state.df.drop(columns=cols_to_drop, inplace=True)
                    st.session_state.df.to_csv("Data_Dropped_Columns.csv", index=False, encoding='utf-8-sig')
                    st.success("تم الحذف!")
                    st.rerun()

        # --- 2. الترميز (One-Hot & Label Encoding) ---
        with st.expander("🔢 2- تحويل النصوص (Encoding)"):
            obj_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
            if obj_cols:
                st.write(f"الأعمدة المتاحة: `{obj_cols}`")
                st.metric("عدد الأعمدة النصية (object)", len(obj_cols))
                object_summary = pd.DataFrame({
                    "column": obj_cols,
                    "non_null": [int(st.session_state.df[col].notna().sum()) for col in obj_cols],
                    "nulls": [int(st.session_state.df[col].isna().sum()) for col in obj_cols],
                    "unique_values": [int(st.session_state.df[col].nunique(dropna=True)) for col in obj_cols],
                })
                st.dataframe(object_summary, use_container_width=True, height=min(320, 38 * (len(obj_cols) + 1)))

                view_col = st.selectbox("عرض قيم عمود نصي:", options=obj_cols, key="object_view_col")
                if view_col:
                    st.write(f"أول قيم من العمود `{view_col}`:")
                    st.dataframe(
                        st.session_state.df[[view_col]].head(20),
                        use_container_width=True,
                        height=260,
                    )
                    st.write("تكرار القيم:")
                    value_counts_df = (
                        st.session_state.df[view_col]
                        .astype(str)
                        .value_counts(dropna=False)
                        .reset_index()
                    )
                    value_counts_df.columns = [view_col, "count"]
                    st.dataframe(value_counts_df.head(20), use_container_width=True, height=260)

                method = st.radio("نوع الترميز:", ["One-Hot Encoding", "Label Encoding"])
                selected_enc = st.multiselect("اختر الأعمدة:", options=obj_cols)
                
                if st.button("⚙️ تنفيذ الترميز"):
                    if selected_enc:
                        if method == "One-Hot Encoding":
                            st.session_state.df = pd.get_dummies(st.session_state.df, columns=selected_enc, drop_first=True, dtype=int)
                        else:
                            le = LabelEncoder()
                            for col in selected_enc:
                                st.session_state.df[col] = le.fit_transform(st.session_state.df[col].astype(str))
                        
                        st.session_state.df.to_csv("Data_Encoded.csv", index=False, encoding='utf-8-sig')
                        st.success("تم التحويل!")
                        st.rerun()
            else:
                st.write("✅ لا توجد أعمدة نصية.")

        # --- 3. تحويل التواريخ ---
        with st.expander("📅 3- تحويل التواريخ"):
            if st.button("🔄 استخراج بيانات الوقت"):
                df_temp = st.session_state.df.copy()
                for col in df_temp.columns:
                    if df_temp[col].dtype == 'object':
                        try:
                            df_temp[col] = pd.to_datetime(df_temp[col])
                            df_temp[f'{col}_year'] = df_temp[col].dt.year
                            df_temp[f'{col}_month'] = df_temp[col].dt.month
                            df_temp[f'{col}_day'] = df_temp[col].dt.day
                            df_temp.drop(columns=[col], inplace=True)
                        except: continue 
                st.session_state.df = df_temp
                st.success("تم تحويل التواريخ!")
                st.rerun()

        # --- 4. التحكم في القيم الشاذة (المحسن) ---
        with st.expander("🚀 4- القيم الشاذة (Outliers)"):
            num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                target_col = st.selectbox("فحص عمود محدد:", options=num_cols)
                
                # حساب الـ IQR
                q1, q3 = st.session_state.df[target_col].quantile([0.25, 0.75])
                iqr = q3 - q1
                low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (st.session_state.df[target_col] < low) | (st.session_state.df[target_col] > up)
                outliers = st.session_state.df[mask]

                st.metric(f"عدد الشواذ في {target_col}", outliers.shape[0])
                if not outliers.empty:
                    st.dataframe(outliers, height=200)

                st.divider()
                strat = st.radio("الاستراتيجية:", ["المتوسط", "الحذف", "Quantile Transform"], horizontal=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"🪄 معالجة {target_col}"):
                        if strat == "المتوسط":
                            st.session_state.df.loc[mask, target_col] = st.session_state.df[target_col].mean()
                        elif strat == "الحذف":
                            st.session_state.df = st.session_state.df[~mask]
                        else:
                            qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(st.session_state.df), 100))
                            st.session_state.df[target_col] = qt.fit_transform(st.session_state.df[[target_col]].values).flatten()
                        st.rerun()

                with c2:
                    if st.button("🔥 معالجة الكل"):
                        df_work = st.session_state.df.copy()
                        for c in num_cols:
                            cq1, cq3 = df_work[c].quantile([0.25, 0.75])
                            ciqr = cq3 - cq1
                            cl, cu = cq1 - 1.5 * ciqr, cq3 + 1.5 * ciqr
                            cm = (df_work[c] < cl) | (df_work[c] > cu)
                            if strat == "المتوسط":
                                df_work.loc[cm, c] = df_work[c].mean()
                            elif strat == "الحذف":
                                df_work = df_work[~cm]
                            else:
                                qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(df_work), 100))
                                df_work[c] = qt.fit_transform(df_work[[c]].values).flatten()
                        st.session_state.df = df_work
                        st.rerun()

        # --- 5. التقسيم (Split) ---
        with st.expander("🤖 5- تقسيم البيانات (Split)"):
            target_var = st.selectbox("الهدف (y):", options=st.session_state.df.columns.tolist())
            size = st.slider("نسبة الاختبار:", 0.1, 0.5, 0.2)
            if st.button("📊 تنفيذ التقسيم النهائي"):
                X = st.session_state.df.drop(columns=[target_var])
                y = st.session_state.df[target_var]
                X = pd.get_dummies(X, drop_first=True, dtype=int)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
                
                X_train.to_csv("X_train.csv", index=False)
                X_test.to_csv("X_test.csv", index=False)
                y_train.to_csv("y_train.csv", index=False)
                y_test.to_csv("y_test.csv", index=False)
                
                st.session_state.split_done = True
                st.success("✅ تم حفظ ملفات التدريب!")

    # // --- عمود معاينة البيانات ---
    with col_preview:
        st.subheader("📋 معاينة البيانات الحالية")
        st.dataframe(st.session_state.df, height=600, use_container_width=True)
        
        with st.expander("🔍 فحص أنواع البيانات"):
            st.write(st.session_state.df.dtypes)
