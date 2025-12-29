"""
This script evaluates multiple trained ML models on the churn dataset, providing:

1. Classification metrics (precision, recall, F1-score) for each model.
2. Confusion matrices for visual comparison of predictions.
3. ROC curves + AUC to compare how well models distinguish churners from non-churners.

Key Notes:
- All models evaluated here are the untuned baseline models.
- SHAP explainability and feature importance are handled separately in 'explainability.py'.
- Focus is on understanding model performance and comparing metrics across models.
"""



# ________________________________________________________________________________________
# Imports
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, classification_report
from preprocessing import load_data, clean_data, preprocess_features, split_data



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
    print(classification_report(y_test, y_pred, digits=4)) # the metrics: precision, recall, ...

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


