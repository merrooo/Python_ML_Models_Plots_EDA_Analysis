ENCODING_DISCUSSION = [
    {
        "Method": "One-Hot Encoding",
        "When to use": "Nominal categories with low/medium cardinality",
        "Benefits": "Clear interpretation, no ordinal bias",
        "Risks": "Can create many columns",
    },
    {
        "Method": "Label Encoding",
        "When to use": "Ordinal categories or quick baseline for tree models",
        "Benefits": "Simple, compact",
        "Risks": "May inject false order for nominal values",
    },
    {
        "Method": "Binary Encoding",
        "When to use": "High-cardinality categories with multiple bit columns",
        "Benefits": "Lower dimensionality than one-hot",
        "Risks": "Less interpretable than one-hot",
    },
]


TRANSFORMATION_DISCUSSION = [
    {
        "Method": "Min-Max Scaling (0 to 1)",
        "When to use": "Need bounded feature range",
        "Benefits": "Stable range, useful for neural-style models",
        "Risks": "Sensitive to outliers",
    },
    {
        "Method": "Standard Scaling (Z-score)",
        "When to use": "Symmetric-ish distributions with scale differences",
        "Benefits": "Common default for linear/SVM-like models",
        "Risks": "Does not fix heavy skew",
    },
    {
        "Method": "Robust Scaling (Outlier-resistant)",
        "When to use": "Outliers present and rows should be retained",
        "Benefits": "Less sensitive to extreme values",
        "Risks": "May not normalize shape enough",
    },
    {
        "Method": "Power Transform (Yeo-Johnson)",
        "When to use": "Strong skew/non-normal numeric features",
        "Benefits": "Can reduce skew substantially",
        "Risks": "Less intuitive transformed scale",
    },
    {
        "Method": "Log Transformation",
        "When to use": "Right-skewed positive-tail distributions",
        "Benefits": "Compresses large values",
        "Risks": "Needs shift handling for non-positive values",
    },
]


BALANCING_DISCUSSION = [
    {
        "Method": "Oversample Minority",
        "When to use": "Small minority class and enough memory budget",
        "Benefits": "Keeps all majority samples",
        "Risks": "Potential overfitting on duplicated minority rows",
    },
    {
        "Method": "Undersample Majority",
        "When to use": "Majority class very large",
        "Benefits": "Fast and simple",
        "Risks": "May discard useful information",
    },
    {
        "Method": "SMOTE",
        "When to use": "Numeric features with imbalanced classification target",
        "Benefits": "Synthetic minority examples improve balance",
        "Risks": "Requires numeric inputs and enough minority samples",
    },
]


CLASSIFICATION_MODELS = [
    {"Model": "Decision Tree", "Why": "Interpretable, rule-based baseline"},
    {"Model": "Random Forest", "Why": "Strong general tabular baseline"},
    {"Model": "Gradient Boosting", "Why": "Often better accuracy with tuning"},
    {"Model": "AdaBoost", "Why": "Simple boosting baseline"},
    {"Model": "Extra Trees", "Why": "Fast ensemble with random splits"},
    {"Model": "Logistic Regression", "Why": "Linear and interpretable baseline"},
    {"Model": "SVM", "Why": "Good on non-linear boundaries (scaled inputs)"},
    {"Model": "KNN", "Why": "Local neighborhood baseline"},
    {"Model": "Naive Bayes", "Why": "Very fast probabilistic baseline"},
    {"Model": "MLP", "Why": "Captures non-linear relations"},
    {"Model": "XGBoost", "Why": "Strong tabular performance when available"},
]


REGRESSION_MODELS = [
    {"Model": "Decision Tree", "Why": "Simple non-linear baseline"},
    {"Model": "Random Forest", "Why": "Robust regression baseline"},
    {"Model": "Gradient Boosting", "Why": "Often strong low-error model"},
    {"Model": "AdaBoost", "Why": "Simple boosted regressor baseline"},
    {"Model": "Extra Trees", "Why": "Fast variance-reducing ensemble"},
    {"Model": "Linear Regression", "Why": "Interpretable linear baseline"},
    {"Model": "SVR", "Why": "Kernel regression for complex patterns"},
    {"Model": "KNN", "Why": "Local trend baseline"},
    {"Model": "MLP", "Why": "Flexible non-linear function fit"},
    {"Model": "XGBoost", "Why": "Strong tabular regression performance"},
]

