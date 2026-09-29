import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing():
    print("[INFO] Starting Preprocessing Pipeline for World Happiness...")
    
    # 1. Load the raw data
    data_path = 'data/raw/happiness_2015.csv'
    df = pd.read_csv(data_path)
    
    # 2. Data Cleaning & Feature Engineering
    def happiness_status(score):
        if score < 4.0: return "Sad / Low Happiness"
        elif score < 5.0: return "Less Happy"
        elif score < 6.0: return "Moderately Happy"
        elif score < 7.0: return "Happy"
        else: return "Very Happy"
        
    df["Happiness Status"] = df["Happiness Score"].apply(happiness_status)
    
    # Drop irrelevant and target-derived columns
    drop_cols = ['Country', 'Region', 'Happiness Rank', 'Happiness Score']
    X = df.drop(columns=drop_cols + ['Happiness Status'])
    
    # Encode Target Variable 
    le = LabelEncoder()
    y = le.fit_transform(df['Happiness Status'])
    
    # 3. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    num_cols = X.columns.tolist()
    
    # 4. Scale Numerical Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Save Artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    np.save('data/processed/X_train_final.npy', X_train_scaled)
    np.save('data/processed/X_test_final.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train.astype(np.int64))
    np.save('data/processed/y_test.npy', y_test.astype(np.int64))
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # 6. Save Metadata
    metadata = {
        "dataset_name": "World Happiness 2015",
        "train_shape": list(X_train_scaled.shape),
        "test_shape": list(X_test_scaled.shape),
        "numerical_features": num_cols,
        "categorical_features": [],
        "target_classes": list(le.classes_)
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("[SUCCESS] Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()