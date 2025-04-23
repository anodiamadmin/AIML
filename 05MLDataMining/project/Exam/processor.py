# process.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load and clean data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.replace(['?', '$'], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)
    return df

# Normalize features and one-hot encode target
def preprocess_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_encoded

# Main execution
if __name__ == "__main__":
    file_path = "data.txt"
    df = load_and_clean_data(file_path)
    X, y = preprocess_data(df)
    np.savez("viz/processed_data.npz", X=X, y=y)
    print("Data preprocessed and saved to 'processed_data.npz'")
