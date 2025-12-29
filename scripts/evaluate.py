# Here I will deeply analyze, then SHAP explainability which is a way to explain the output of any machine learning model
# We evaluate the saved models, and then generate visuals, reports, insight, etc..

"""
This script evaluates multiple trained models on the churn dataset, providing:
1. Classification metrics (precision, recall, F1-score) for each model.
2. Confusion matrices for visual comparison of predictions.
3. ROC curves + AUC to compare how well models distinguish churners from non-churners.
4. SHAP explainability for the best model (Logistic Regression) to identify which features most influence churn predictions.


Key Insight from SHAP

Top predictors of churn:
1. tenure → short-tenure customers are more likely to churn.
2. Contract_Month-to-Month → month-to-month contracts increase churn risk.
3. Contract_Two year → long-term contracts reduce churn risk.

- Other features have smaller but non-negligible contributions.
- Business takeaway: Focus retention efforts on new, month-to-month customers to reduce churn.

"""



# ________________________________________________________________________________________
# Imports
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, classification_report
from preprocessing import load_data, clean_data, preprocess_features, split_data

import shap


os.makedirs("reports", exist_ok=True) # report folder to save visuals


# ________________________________________________________________________________________
# Load and preprocess
df = load_data()
df = clean_data(df) 
X, y, preprocessor = preprocess_features(df) # preprocess
X_train, X_test, y_train, y_test = split_data(X, y) # split

# List of models to evaluate
model_files = [
    "models/logistic_regression.pkl",
    "models/random_forest.pkl",
    "models/svm.pkl",
    "models/knn.pkl",
    "models/decision_tree.pkl"
]



# ________________________________________________________________________________________
# Evaluation loop
# Collect info for plots
roc_info = {}

plt.figure(figsize=(15, 10))  # For confusion matrices
for i, model_file in enumerate(model_files, 1):
    model = joblib.load(model_file) # load model
    y_pred = model.predict(X_test)  # generated prediction labels, for each customer either 0 or 1
    y_proba = model.predict_proba(X_test)[:, 1] # this give probability predictions instead of either 0 or 1 (we only get for the churn)

    # Print classification report
    print(f"\n===== {model_file.split('/')[-1]} =====")
    print(classification_report(y_test, y_pred)) # the metrics: precision, recall, ...

    # Confusion matrix subplot
    plt.subplot(3, 2, i)  # 3 rows, 2 cols
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=plt.gca(), colorbar=False) # Confusion matrix for the model
    plt.title(model_file.split('/')[-1])


    # Save ROC info for later
    fpr, tpr, _ = roc_curve(y_test, y_proba) # false positive, true positive
    roc_auc = auc(fpr, tpr)                 # area under curve
    roc_info[model_file.split('/')[-1]] = (fpr, tpr, roc_auc)   # saved this data in the dictionary to be created later

plt.tight_layout()
plt.savefig("reports/confusion_matrices.png")
plt.show()


# Receiver Operating Characteristic (aka ROC) explains how well a model separates two classes
# Area under Curve (aka AUC) single number summarizing the ROC curve. 0.5 is random guessing, 1.0 is perfect separation
# Plot all ROC curves together
plt.figure(figsize=(8,6))
for name, (fpr, tpr, roc_auc) in roc_info.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()

plt.savefig("reports/roc_curves.png")

plt.show()


# ________________________________________________________________________________________
# SHAP Explainablilty (Global; basically which features influence the prediction the most. In my case I found it to be tenure)
# for: Importance + effect on predictions (positive/negative)

# Load the best model
model = joblib.load("models/logistic_regression.pkl")

# Transform test features using the pipeline preprocessing
X_test_transformed = model.named_steps['preprocessing'].transform(X_test)

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
# for: Only importance ranking


import numpy as np
model = joblib.load("models/logistic_regression_tuned.pkl")

# Get feature names after preprocessing
feature_names = model.named_steps['preprocessing'].get_feature_names_out()

# Get coefficients
coef = model.named_steps['classifier'].coef_[0]

# Sort features by absolute coefficient
sorted_idx = np.argsort(np.abs(coef))[::-1]

# Plot top 10 features
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.barh([feature_names[i] for i in sorted_idx[:10]], coef[sorted_idx[:10]])
plt.xlabel("Coefficient (Feature Importance)")
plt.title("Top 10 Logistic Regression Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("reports/logreg_feature_importance.png")
plt.show()



"""
For Logistic Regression, coefficients are a simple global feature importance, 
while SHAP gives you the same idea but richer, including effect direction and 
per-sample explanation.
"""