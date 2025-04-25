import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess_data
import os

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data("Bank_Transaction_Fraud_Detection.csv")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/logistic_model.pkl")
print("Model saved to models/logistic_model.pkl")
