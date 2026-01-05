import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- STEP 1: LOAD AND CLEAN FRAMINGHAM DATA ---
# Replace 'framingham.csv' with your actual file path if it's in a different folder
try:
    df = pd.read_csv('framingham.csv')
except FileNotFoundError:
    print("Error: framingham.csv not found. Please ensure the file is in the same directory.")

# The Framingham dataset has several missing values (NaNs)
# To keep the implementation clean, we will drop rows with missing data
df = df.dropna()

# Features and Target (TenYearCHD is the prediction goal)
X = df.drop("TenYearCHD", axis=1).values
y = df["TenYearCHD"].values

# Split and Scale
# Scale is CRITICAL for the scratch model to prevent "overflow" in the exp() function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- STEP 2: LOGISTIC REGRESSION FROM SCRATCH ---
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            # Model prediction: y = sigmoid(Xw + b)
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Gradient Descent equations
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_probs = self.sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_probs])

# --- STEP 3: EXECUTION ---

# 1. Scratch Implementation
scratch_model = LogisticRegressionScratch(lr=0.1, iterations=3000)
scratch_model.fit(X_train, y_train)
scratch_preds = scratch_model.predict(X_test)

# 2. Scikit-Learn Implementation
sk_model = LogisticRegression(max_iter=3000)
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

# --- STEP 4: COMPARISON ---

print("=== RESULTS: SCRATCH MODEL ===")
print(classification_report(y_test, scratch_preds))

print("=== RESULTS: SCIKIT-LEARN ===")
print(classification_report(y_test, sk_preds))

# Confusion Matrix Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, scratch_preds, ax=ax[0], cmap='Blues')
ax[0].set_title("Scratch Model")
ConfusionMatrixDisplay.from_predictions(y_test, sk_preds, ax=ax[1], cmap='Greens')
ax[1].set_title("Scikit-Learn")
plt.show()