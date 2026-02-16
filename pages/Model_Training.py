import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

st.set_page_config(page_title="Model Training - NDEDC", layout="wide")
st.title("🤖 تدريب نماذج الذكاء الاصطناعي")

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

    if st.button("🚀 بدء تدريب النموذج"):
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