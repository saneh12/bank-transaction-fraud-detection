import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)

    # Drop high-cardinality and identifier columns
    data.drop(columns=[
        'Customer_ID', 'Customer_Name', 'Transaction_ID', 'Transaction_Date',
        'Transaction_Time', 'Merchant_ID', 'Customer_Contact', 'Customer_Email'
    ], inplace=True)

    # Encode categorical features
    categorical_cols = [
        'Gender', 'State', 'City', 'Bank_Branch', 'Account_Type',
        'Transaction_Type', 'Merchant_Category', 'Transaction_Device',
        'Transaction_Location', 'Device_Type', 'Transaction_Currency',
        'Transaction_Description'
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Features and target
    X = data.drop('Is_Fraud', axis=1)
    y = data['Is_Fraud']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data("Bank_Transaction_Fraud_Detection.csv")
    print("Data loaded and preprocessed successfully.")
