"""
This script provides interpretability analysis for the tuned Logistic Regression model
used for customer churn prediction. It focuses on understanding which features drive
predictions, both globally and for individual customers.

Functionalities included:

1. SHAP Global Explainability
   - Computes feature importance and effect direction for all test samples.
   - Generates a summary bar plot showing the top features contributing to churn predictions.

2. SHAP Local Explainability
   - Computes SHAP values for a single customer to understand why the model predicted
     churn or non-churn for that instance.
   - Produces a force plot for visualization.

3. Logistic Regression Coefficient Analysis
   - Extracts and ranks feature coefficients from the tuned LR model as a simple global
     measure of feature importance.
   - Produces a bar chart of the top 10 features.
"""





import os
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
from preprocessing import load_data, clean_data, preprocess_features, split_data

os.makedirs("reports", exist_ok=True)


# ________________________________________________________________________________________
# Load and preprocess
df = load_data()
df = clean_data(df)
X, y, preprocessor = preprocess_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Load tuned Logistic Regression
model = joblib.load("models/logistic_regression_tuned.pkl")

# Transform test features using the pipeline
X_test_transformed = model.named_steps['preprocessing'].transform(X_test)

# ________________________________________________________________________________________
# SHAP Explainablilty (Global; basically which features influence the prediction the most. In my case I found it to be tenure)

# Create SHAP explainer for Logistic Regression
explainer = shap.LinearExplainer(model.named_steps['classifier'], X_test_transformed, feature_perturbation="correlation_dependent")
shap_values = explainer.shap_values(X_test_transformed) #how much each feature contributes to the prediction of each sample


# Global feature importance (summary plot)
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=model.named_steps['preprocessing'].get_feature_names_out(),
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("reports/shap_summary_plot.png")
plt.close()

# ________________________________________________________________________________________
# SHAP Local Explainability (Single Customer)

"""
Insight for customer 0:
# The model's baseline churn score is around -0.65, meaning customers \
# generally tend to stay. For customer 0, the final prediction drops to \
# -2.93, far below the baseline, indicating a very low churn risk driven \
# mainly by strong retention factors such as longer tenure and a long-term \
# contract.”
"""

# Pick one customer from the test set
sample_index = 0                                    # CHANGE THIS AS DESIRED
sample = X_test.iloc[[sample_index]]    # get the sample

# Transform sample using preprocessing
sample_transformed = model.named_steps['preprocessing'].transform(sample)

# Compute SHAP values for this instance
shap_values_single = explainer.shap_values(sample_transformed)

# Force plot (local explanation)
shap.force_plot(
    explainer.expected_value,
    shap_values_single,
    sample_transformed,
    feature_names=model.named_steps['preprocessing'].get_feature_names_out(),
    matplotlib=True,
    show=False
)

plt.tight_layout()
plt.savefig("reports/shap_local_force_plot.png")
plt.close()





# ________________________________________________________________________________________
# Feature importance for Logistic Regression

model = joblib.load("models/logistic_regression_tuned.pkl")

# Get feature names after preprocessing
feature_names = model.named_steps['preprocessing'].get_feature_names_out()

# Get coefficients
coef = model.named_steps['classifier'].coef_[0]

# Sort features by absolute coefficient
sorted_idx = np.argsort(np.abs(coef))[::-1]

# Plot top 10 features
plt.figure(figsize=(10,6))
bars = plt.barh([feature_names[i] for i in sorted_idx[:10]], coef[sorted_idx[:10]], color='skyblue')
plt.xlabel("Coefficient (Feature Importance)")
plt.title("Top 10 Logistic Regression Feature Importances")
plt.gca().invert_yaxis()


# # Add explanatory text on the side
# explanation = (
#     "Longer bars = stronger impact on churn prediction\n"
#     "Right (positive) = feature increases probability of churn\n"
#     "Left (negative) = feature decreases probability of churn (more likely to stay)"
# )
# plt.text(x=min(coef[sorted_idx[:10]])*1.1, y=9.5, s=explanation, fontsize=9, va='top', color='gray')

plt.tight_layout()
plt.savefig("reports/logreg_feature_importance.png")
plt.show()