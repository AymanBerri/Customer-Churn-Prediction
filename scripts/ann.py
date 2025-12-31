# Here i will create a deep learning model, ANN.


"""

ANN Script Overview

This script builds a minimal Artificial Neural Network (ANN) for churn prediction:
1. Load and preprocess data using the same pipeline as the tuned Logistic Regression (ensures consistent scaling and encoding).
2. Define ANN architecture: two hidden layers (64 → 32 neurons, ReLU) and one output layer (sigmoid) for binary classification.
3. Compile with Adam optimizer, binary cross-entropy loss, and accuracy metric.
4. Train with early stopping to prevent overfitting (20% validation split, batch size 32, max 50 epochs).
5. Evaluate on test set to report accuracy.
6. Plot loss curves to diagnose learning behavior (overfitting, underfitting, or good fit).
"""

# ___________________________________________
#Imports

import numpy as np
import matplotlib.pyplot as plt
import joblib

#Neural network toolkit "Tensorflow"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping # to stop training before the model starts overfitting

# Pipeline
from preprocessing import load_data, clean_data, preprocess_features, split_data


# ___________________________________________
# Load and Preprocess
df = load_data()
df = clean_data(df)

X, y, preprocessor = preprocess_features(df)
# X:            input features
# y:            target
# preprocessor: obj that knows how to encode and scale data

X_train, X_test, y_train, y_test = split_data(X, y) # Splitting data


# ___________________________________________

# Use preprocessing from Logistic Regression pipeline
lr_pipeline = joblib.load("models/logistic_regression_tuned.pkl")
X_train_prep = lr_pipeline.named_steps['preprocessing'].transform(X_train) # transform directly, the train
X_test_prep = lr_pipeline.named_steps['preprocessing'].transform(X_test) #                      and test data



# ___________________________________________
# Building the ANN (Artificial Neural Network)

# using sequential layers, layers are stacked in order [Layer 1 → Layer 2 → Layer 3]
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_prep.shape[1],)),  # FIRST layer, having 64 neurons, Each recieving all the input features
    Dense(32, activation='relu'), # Second layer (Hidden), 32 neurons, forces compression
    Dense(1, activation='sigmoid') # output layer, 1 neuron, sigmoid forces output into [0, 1] (0.85, means customer predicted to churn)
])

model.compile(      # compling the model is telling the model how to learn and how to measure mistakes
    optimizer=Adam(learning_rate=0.001), # adjusts weights 
    loss='binary_crossentropy', # measures how wrong was the probability
    metrics=['accuracy']        # just to monitor
)

# early stopping to prevent overfitting
# this is used to prevent learning the data instead of understanding.
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)



# ___________________________________________
# Train, this is where the learning happens
history = model.fit(
    X_train_prep,               # input
    y_train,                    # input

    validation_split=0.2,       # 20% data to test

    epochs=50,                  # 50 full passes over the training data, unless interrupted by the early stop
    batch_size=32,              # how many samples at once

    callbacks=[early_stop],
    verbose=1
)



# ___________________________________________
# Evaluate
loss, acc = model.evaluate(X_test_prep, y_test, verbose=0)      # evaluate with never seen data
print(f"ANN Test Accuracy: {acc:.3f}")





# ___________________________________________
# Plot loss curves, diagnose training behavior
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("reports/ann_loss_curve.png")
plt.show()

"""
both go down: GOOD
Train \/, Val/\: OVERFITTING
Train /\, Val\/: UNDERFITTING

for overfitting or underfitting the dec or inc must be sharp



Chart Summary `reports\ann_loss_curve.png`:
- Train loss starts high (~0.45) → decreases (~0.38): the model is learning patterns.
- Validation loss fluctuates (~0.41 → 0.43): model’s generalization is limited; minor underfitting or batch noise.

Interpretation:
- The ANN learns something but is not as strong as the logistic regression baseline (main model).
- Minor fluctuations are expected with small datasets (~7000 samples).
- EarlyStopping prevents overfitting by restoring best weights.

"""

