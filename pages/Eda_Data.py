import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. إعداد الصفحة
st.set_page_config(page_title="EDA - NDEDC Dashboard", layout="wide")

st.title("2️⃣ مرحلة استكشاف البيانات وتحليلها (EDA)")

# --- التعديل الجديد: نظام اختيار الملف يدوياً ---
st.sidebar.header("📁 إعدادات البيانات")

# جلب قائمة بكل ملفات CSV المتاحة في المجلد
all_csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if not all_csv_files:
    st.error("⚠️ لا توجد ملفات CSV في المجلد الحالي.")
    st.info("الرجاء الذهاب إلى صفحة 'Cleaning Data' أولاً وحفظ البيانات.")
    st.stop()

# اختيار الملف من القائمة المنسدلة
selected_file = st.sidebar.selectbox(
    "اختر ملف البيانات للتحليل:", 
    options=all_csv_files,
    index=all_csv_files.index("Data_Dropped_Columns.csv") if "Data_Dropped_Columns.csv" in all_csv_files else 0
)

# تحميل البيانات بناءً على الملف المختار
@st.cache_data 
def load_data(path):
    return pd.read_csv(path)

df = load_data(selected_file)

st.success(f"✅ تم تحميل ملف: `{selected_file}` بنجاح! (صفوف: {df.shape[0]} | أعمدة: {df.shape[1]})")
# --------------------------------------------------



# باقي الكود الخاص بالرسوم البيانية والإحصائيات يكمل هنا كما هو...
# 4. عرض البيانات الكاملة
with st.expander("1-📋 عرض جدول البيانات الكامل"):
    st.dataframe(df, use_container_width=True)
st.markdown("---")

# ------------------------------------------------------------------------------------------------------

# 2. خريطة الارتباط (Heatmap) العامة
st.subheader("2-🗺️ خريطة الارتباط (Correlation Heatmap)")
numeric_df = df.select_dtypes(include=["number"])

if numeric_df.shape[1] >= 2:
    num_cols = numeric_df.shape[1]
    chart_width = max(12, num_cols * 0.9)
    chart_height = max(8, num_cols * 0.7)
    fig, ax = plt.subplots(figsize=(chart_width, chart_height))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                annot_kws={"size": max(6, 12 - num_cols//5)}, linewidths=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
else:
    st.warning("الـ Heatmap يحتاج إلى عمودين رقميين على الأقل.")
st.markdown("---")

# ----------------------------------------------------------------------------------------------

# 3. الإحصائيات العامة
st.subheader("3-📊 إحصائيات عامة")
tab1, tab2, tab3 = st.tabs(["📉 الوصف الإحصائي", "🔢 القيم الفريدة", "❓ القيم المفقودة"])
with tab1:
    st.dataframe(numeric_df.describe(), use_container_width=True)
with tab2:
    st.dataframe(df.nunique(), use_container_width=True)
with tab3:
    st.dataframe(df.isnull().sum(), use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------------------------

# 4. تحليل أكثر القيم تكراراً
st.subheader("4-🔝 تحليل تكرار القيم")
col_select, col_chart = st.columns([1, 2])
with col_select:
    selected_col = st.selectbox("اختر عموداً لتحليله:", options=df.columns.tolist())
    top_n = st.slider("عدد القيم المعروضة", 5, 20, 10)
if selected_col:
    top_values = df[selected_col].value_counts().head(top_n)
    with col_select:
        st.table(top_values)
    with col_chart:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_values.values, y=top_values.index.astype(str), palette="viridis", ax=ax2)
        st.pyplot(fig2)

# ----------------------------------------------------------------------------------------------

# 5. عرض توزيع جميع الأعمدة
st.markdown("---")
with st.expander("5-🔍 عرض تفاصيل التوزيع لجميع الأعمدة"):
    for col in df.columns:
        st.write(f"**العمود: {col}**")
        st.write(df[col].value_counts().head(5))
        st.write("---")

# ----------------------------------------------------------------------------------------------


# 6. اختيار أعمدة محددة للعرض والتحليل (Feature Selection & Analytics)
st.markdown("---")
st.subheader("6-🎯 تحليل أعمدة مختارة (Feature Selection & Analytics)")

# تأمين قائمة الأعمدة ومنع التكرار
all_cols = list(dict.fromkeys(df.columns.tolist())) 
default_selection = [c for c in ["date_time", "spain_market", "output"] if c in all_cols]

# 1. قائمة الاختيار المتعدد (Multiselect)
selected_features = st.multiselect(
    "اختر الأعمدة التي تود عرضها وتحليلها:",
    options=all_cols,
    default=default_selection if default_selection else [all_cols[0]]
)

if selected_features:
    # عرض الجدول المفلتر
    st.write(f"📋 عرض `{len(selected_features)}` أعمدة مختارة:")
    st.dataframe(df[selected_features], use_container_width=True)
    
    # تحديد الأعمدة الرقمية فقط للتحليل البياني
    numeric_features = df[selected_features].select_dtypes(include=["number"]).columns.tolist()
    
    if numeric_features:
        # --- (أ) الرسم البياني للمقارنة الخطية ---
        st.write("📈 **أولاً: الرسم البياني للمقارنة (Time-Series / Line Plot):**")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        for feat in numeric_features:
            ax3.plot(df.index, df[feat], label=feat, linewidth=1.5, alpha=0.8)
        ax3.set_title("مقارنة البيانات المختارة عبر الزمن")
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig3)

        st.markdown("---")

        # --- (ب) رسم التوزيع (Seaborn Distribution Plot) ---
        st.write("📊 **ثانياً: تحليل توزيع عمود مختار (Seaborn Distplot):**")
        col_dist_select, col_dist_plot = st.columns([1, 2])
        
        with col_dist_select:
            dist_target = st.selectbox("اختر عموداً لرسم توزيعه الإحصائي:", options=numeric_features, key="dist_sb")
            st.info(f"عرض التوزيع لـ `{dist_target}` باستخدام Step Histogram و Rug Plot.")

        with col_dist_plot:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=df[dist_target],
                kde=True,
                element="step",
                fill=False,
                color="red",
                linewidth=3,
                ax=ax4
            )
            sns.rugplot(data=df[dist_target], color="g", height=0.05, ax=ax4)
            ax4.set_title(f"Distribution Plot: {dist_target}")
            st.pyplot(fig4)

        st.markdown("---")

        # --- (ج) تحليل القيم الشاذة المتقدم (Outliers Detection) ---
        st.write("🕵️ **ثالثاً: تحليل القيم الشاذة (Outliers Detection):**")
        
        # اختيار العمود للفحص
        outlier_target = st.selectbox("اختر عموداً لفحص القيم الشاذة فيه:", options=numeric_features, key="out_sb")
        
        # حساب حدود القيم الشاذة (IQR Method)
        Q1 = df[outlier_target].quantile(0.25)
        Q3 = df[outlier_target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # تحديد الصفوف الشاذة
        outliers_df = df[(df[outlier_target] < lower_bound) | (df[outlier_target] > upper_bound)]
        outliers_count = outliers_df.shape[0]

        # عرض عداد القيم الشاذة
        st.metric(label=f"إجمالي القيم الشاذة في {outlier_target}", value=outliers_count)

        # إنشاء رسمين بيانيين متجاورين (Box Plot & Scatter Plot)
        col_box, col_scatter = st.columns(2)

        with col_box:
            st.write("📦 **Box Plot:**")
            fig_box, ax_box = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=df[outlier_target], color="#FF4B4B", fliersize=7, ax=ax_box, 
                        flierprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"})
            ax_box.set_title(f"Box Plot: {outlier_target}")
            st.pyplot(fig_box)

        with col_scatter:
            st.write("🌌 **Scatter Plot (Outlier Distribution):**")
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            
            # رسم جميع النقاط باللون الرمادي
            ax_scatter.scatter(df.index, df[outlier_target], color='lightgrey', alpha=0.5, label='Normal')
            
            # رسم القيم الشاذة فقط باللون الأحمر لتبرز
            if not outliers_df.empty:
                ax_scatter.scatter(outliers_df.index, outliers_df[outlier_target], color='red', label='Outlier', s=20)
            
            ax_scatter.set_title(f"Scatter Plot: {outlier_target}")
            ax_scatter.set_xlabel("Index")
            ax_scatter.set_ylabel(outlier_target)
            ax_scatter.legend()
            st.pyplot(fig_scatter)
            
        if outliers_count > 0:
            with st.expander("🔍 عرض بيانات القيم الشاذة فقط"):
                st.dataframe(outliers_df, use_container_width=True)
                st.write("📝 **ملاحظة:** تم تحديد القيم الشاذة بناءً على نطاق الـ IQR (أكبر من Q3+1.5IQR أو أصغر من Q1-1.5IQR).")
                
    else:
        st.info("💡 الأعمدة المختارة لا تحتوي على بيانات رقمية.")
else:
    st.warning("⚠️ الرجاء اختيار عمود واحد على الأقل للبدء.")
    
            
# ----------------------------------------------------------------------------------------------
    
st.markdown("---")
st.subheader("7-🎯 ما هي المتغيرات الأكثر تأثيراً على المخرجات (OUTPUT)؟")

# نستخدم جميع البيانات الرقمية لضمان عدم تفويت أي علاقة مهمة
numeric_only_all = df.select_dtypes(include=["number"])

if 'output' in [c.lower() for c in numeric_only_all.columns]:
    # العثور على الاسم الصحيح لعمود المخرجات (سواء كان Output أو output)
    target_out = next(c for c in numeric_only_all.columns if c.lower() == 'output')
    
    # حساب الارتباط وترتيب القيم تنازلياً كما طلبت
    corr_series = numeric_only_all.corr()[target_out].sort_values(ascending=False)
    
    # استبعاد المخرجات من القائمة لكي لا تظهر علاقتها بنفسها (وهي دائماً 1)
    corr_series = corr_series.drop(labels=[target_out])
    
    col_corr1, col_corr2 = st.columns([1, 1.5])
    
    with col_corr1:
        st.write("📊 **جدول معامل الارتباط:**")
        # عرض الجدول بتنسيق ملون
        st.dataframe(corr_series.to_frame().style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    with col_corr2:
        st.write("📈 **تمثيل بصري لقوة التأثير:**")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        # رسم بياني يوضح التأثير الطردي والعكسي
        sns.barplot(x=corr_series.values, y=corr_series.index, palette="coolwarm", ax=ax_corr)
        ax_corr.set_title("Correlation with OUTPUT")
        ax_corr.set_xlabel("Correlation Coefficient")
        st.pyplot(fig_corr)
else:
    st.info("💡 لم يتم العثور على عمود باسم 'OUTPUT' لحساب الارتباط معه.")

# ----------------------------------------------------------------------------------------------

# 8. خريطة الارتباط للأعمدة المختارة فقط
st.markdown("---")
if selected_features:
    selected_numeric = df[selected_features].select_dtypes(include=["number"])
    
    if selected_numeric.shape[1] >= 2:
        st.subheader(f"8-🗺️ خريطة الارتباط للأعمدة المختارة")
        h_size = max(8, len(selected_numeric.columns) * 1.5)
        fig4, ax4 = plt.subplots(figsize=(h_size, h_size * 0.6))
        sns.heatmap(selected_numeric.corr(), annot=True, cmap="RdYlGn", fmt=".2f", ax=ax4, linewidths=1, square=True)
        st.pyplot(fig4)
    else:
        st.info("💡 اختر عمودين رقميين على الأقل لرؤية الارتباط.")

# ----------------------------------------------------------------------------------------------

# 9. عرض القيم القصوى والصغرى في جداول مخصصة
st.markdown("---")
if selected_features:
    st.subheader("9-🔝 تحليل الصفوف القصوى (Max & Min Details)")
    cols_lower = {c.lower(): c for c in df.columns}
    numeric_only = df[selected_features].select_dtypes(include=["number"]).columns.tolist()
    
    if numeric_only:
        target_col = st.selectbox("اختر العمود لعرض تفاصيل القيم القصوى له:", options=numeric_only)
        
        extra_cols = []
        # بحث مرن عن عمود التاريخ وعمود المخرجات
        for k in ['date_time', 'datetime', 'date', 'time']:
            if k in cols_lower:
                extra_cols.append(cols_lower[k])
                break
        if 'output' in cols_lower: extra_cols.append(cols_lower['output'])
        
        display_list = [target_col] + [c for c in extra_cols if c != target_col]

        st.markdown(f"#### 🚀 أقصى قيمة لـ `{target_col}`")
        max_idx = df[target_col].idxmax()
        max_table = df.loc[[max_idx], display_list].copy()
        max_table.insert(0, 'row_index', max_idx)
        st.table(max_table)

        st.markdown(f"#### 📉 أقل قيمة لـ `{target_col}`")
        min_idx = df[target_col].idxmin()
        min_table = df.loc[[min_idx], display_list].copy()
        min_table.insert(0, 'row_index', min_idx)
        st.table(min_table)
    else:
        st.info("💡 يرجى اختيار أعمدة رقمية من القائمة الجانبية.")

# ----------------------------------------------------------------------------------------------

# 10. الجداول الشاملة للقيم القصوى والدنيا (Global Max/Min Tables)
st.markdown("---")
st.subheader("10-🌐 الجداول الملخصه لجميع المتغيرات الرقمية")

cols_map = {c.lower(): c for c in df.columns}
# بحث مرن عن التاريخ والمخرجات لضمان عدم ظهور N/A
actual_dt_col = next((cols_map[k] for k in ['date_time', 'datetime', 'date', 'time'] if k in cols_map), None)
actual_out_col = cols_map.get('output')

all_numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if actual_out_col in all_numeric_cols:
    all_numeric_cols.remove(actual_out_col)

if all_numeric_cols:
    max_summary_data, min_summary_data = [], []

    for col in all_numeric_cols:
        max_idx, min_idx = df[col].idxmax(), df[col].idxmin()
        
        max_summary_data.append({
            "row_index": max_idx,
            "datetime": df.loc[max_idx, actual_dt_col] if actual_dt_col else "N/A",
            "feature": col,
            "max_value_feature": df.loc[max_idx, col],
            "OUTPUT": df.loc[max_idx, actual_out_col] if actual_out_col else "N/A"
        })
        min_summary_data.append({
            "row_index": min_idx,
            "datetime": df.loc[min_idx, actual_dt_col] if actual_dt_col else "N/A",
            "feature": col,
            "min_value_feature": df.loc[min_idx, col],
            "OUTPUT": df.loc[min_idx, actual_out_col] if actual_out_col else "N/A"
        })

    st.markdown("### 🚀 الجدول الأول: القيم العظمى (Global Max Table)")
    st.dataframe(pd.DataFrame(max_summary_data), use_container_width=True)
    st.markdown("### 📉 الجدول الثاني: القيم الصغرى (Global Min Table)")
    st.dataframe(pd.DataFrame(min_summary_data), use_container_width=True)

    # 10. تصدير الجداول (داخل شرط وجود البيانات)
    st.markdown("---")
    st.subheader("10-📥 تحميل تقارير الملخص الشامل")
    csv_max = pd.DataFrame(max_summary_data).to_csv(index=False).encode('utf-8-sig')
    csv_min = pd.DataFrame(min_summary_data).to_csv(index=False).encode('utf-8-sig')
    c1, c2 = st.columns(2)
    c1.download_button("📥 تحميل جدول Max", data=csv_max, file_name='max_summary.csv', use_container_width=True)
    c2.download_button("📥 تحميل جدول Min", data=csv_min, file_name='min_summary.csv', use_container_width=True)
else:
    st.warning("لا توجد أعمدة رقمية كافية.")

# ----------------------------------------------------------------------------------------------

# 11. نظام البحث المتقدم وإعادة عرض الجدول
st.markdown("---")
st.subheader("11-🔍 نظام البحث المتقدم (Advanced Search)")
st.write("### 📋 الجدول الرئيسي للبيانات")

col_search1, col_search2 = st.columns([1, 2])
with col_search1:
    search_type = st.radio("البحث بواسطة:", ["التاريخ (DateTime)", "قيمة في عمود محدد"])

with col_search2:
    if search_type == "التاريخ (DateTime)":
        actual_dt_col = next((cols_map[k] for k in ['date_time', 'datetime', 'date', 'time'] if k in cols_map), None)
        if actual_dt_col:
            query = st.text_input(f"أدخل التاريخ المراد البحث عنه:")
            filtered_df = df[df[actual_dt_col].astype(str).str.contains(query)] if query else df
        else:
            st.warning("لم يتم العثور على عمود تاريخ.")
            filtered_df = df
    else:
        target_s = st.selectbox("اختر العمود:", options=df.columns.tolist())
        query = st.text_input(f"أدخل القيمة للبحث في `{target_s}`:")
        filtered_df = df[df[target_s].astype(str).str.contains(query)] if query else df

st.write(f"✅ النتائج: {len(filtered_df)} صفوف.")
st.dataframe(filtered_df, use_container_width=True)

if 0 < len(filtered_df) < len(df):
    st.write("📊 إحصائيات النتائج المفلترة:")
    st.dataframe(filtered_df.describe().loc[['mean', 'max', 'min']])