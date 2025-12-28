# Preprocessing step

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#_____________________________________________________________________
# Loading the data (same as the EDA step, from the github raw file)
# 
def load_data():
    """
    Load the churn dataset from the github raw url
    """
    url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df


#_____________________________________________________________________
# Data Cleaning
def clean_data(df):
    """
    We drop customerID (identifier, useless)
    Convert TotalCharges to numeric (having some empty field might change its type to Obj (String), "")
    Handle missing values, There were none, but just for best practice
    """
    df = df.copy()

    # Drop  identifier
    df = df.drop(columns=['customerID'])

    # Convert TotalCharges (currently object) to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') # coerce turn anything that failed to NaN

    # Drop rows where TotalCharges conversion failed
    df = df.dropna(subset=['TotalCharges'])

    return df




#_____________________________________________________________________
# Preprocessing (Speration)
def preprocess_features(df):
    """
    Separate features (X) and target (y)
    Encode target to binary
    Preprocess categorical & numerical features:
        Seperate them
        One-hot encode categorical features
        Scale necessary numeric features
    """
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes':1, 'No':0})  # Binary target, simple mapping to binary

    # Numerical columns: keep SeniorCitizen + tenure + charges
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    # Categorical columns: all object types except numerical ones
    categorical_cols = [col for col in X.columns if col not in numerical_cols]




    # TRANSFORMERS ---

    # One-Hot Encoder for categorical variables
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Standard Scaler for numerical variables
    numeric_transformer = StandardScaler()

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numeric_transformer, numerical_cols)
        ])

    return X, y, preprocessor



#_____________________________________________________________________
# Split data

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Train-test split with stratified sampling (80% train)
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)




#_____________________________________________________________________
#_____________________________________________________________________



# Quick testing
if __name__ == "__main__":
    # url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    print("Loading data...")
    df = load_data()
    print("Original shape:", df.shape)

    print("\nCleaning data...")
    df = clean_data(df)
    print("Cleaned shape:", df.shape)

    print("\nPreprocessing features...")
    X, y, preprocessor = preprocess_features(df)
    print("Feature sample:")
    print(X.head())
    print("Target sample:")
    print(y.head())

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")