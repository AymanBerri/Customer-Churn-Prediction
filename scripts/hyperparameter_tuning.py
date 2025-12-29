"""
This script performs simple hyperparameter tuning for the Logistic Regression model
using GridSearchCV. The tuned model is saved for later evaluation and explainability.
"""

import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import load_data, clean_data, preprocess_features, split_data

os.makedirs("models", exist_ok=True)

#_____________________________
# Load and preprocess
df = load_data()
df = clean_data(df)
X, y, preprocessor = preprocess_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

#_____________________________
# Define pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

#_____________________________
# Define hyperparameter grid
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}

#_____________________________
# GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='recall',  # we focus most on recall for churners
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best recall (CV):", grid_search.best_score_)

#_____________________________
# Save tuned model
joblib.dump(grid_search.best_estimator_, "models/logistic_regression_tuned.pkl")
print("Saved tuned Logistic Regression model to models/logistic_regression_tuned.pkl")
