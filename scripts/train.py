# train.py fits multiple models using the preprocessing pipeline, evaluates them, and saves them




# ___________________________________________________________________________________________
# Imports
from preprocessing import load_data, clean_data, preprocess_features, split_data
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


# ___________________________________________________________________________________________
os.makedirs("models", exist_ok=True) #making sure the models folder exists




# ___________________________________________________________________________________________
# Load and preprocess (this is the preprocessing step `preprocessing.py`)
df = load_data()
df = clean_data(df)
X, y, preprocessor = preprocess_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)



# ___________________________________________________________________________________________
# Models

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight='balanced'), # Input is num and output is Binary, S-shape curve used to calc a probability
    "random_forest": RandomForestClassifier(random_state=42, class_weight='balanced'), # Train many DTs, then majority vote wins
    "svm": SVC(probability=True, random_state=42), # Support Vector Machines, use hyperplanes 
    "knn": KNeighborsClassifier(), # Having a new point on the graph, we decide by looking at its neighbors, majority label decides the label of new point
    "decision_tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'), # simple tree
}

# ___________________________________________________________________________________________
# Training loop, this is where we loop through each model defined earlier and train them on the data and then save them.

for name, model in models.items():
    print(f"\nTraining {name}...")

    # A pipeline guarantees that preprocessing steps are fitted only on the training data \
    # and then applied consistently to both training and test data. This is very important \ 
    # bec one of the greatest advantages here is preventing Data Leakage (fitting the test data...)
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"Classification report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Save trained model
    joblib.dump(pipeline, f"models/{name}.pkl")
    print(f"Saved {name} model to models/{name}.pkl")




# ___________________________________________________________________________________________
# From the results:

# In this project, recall is the most critical metric, especially for class 1 (customers who churn).
# False negatives (customers predicted to stay but who actually leave) are costly

# Among all evaluated models, Logistic Regression achieved the highest recall for the churn class.

"""
Logistic Regression Performance (Class 1 - Churn):

Precision: 0.49
Recall:    0.80 <--
F1-score:  0.61
Accuracy:  0.73
"""

# Interpretation:
# The model successfully identifies 80% of customers who actually churn.
# Although this comes at the cost of lower precision, this trade-off is acceptable
# in a churn prediction context where capturing as many churners as possible is the priority.
