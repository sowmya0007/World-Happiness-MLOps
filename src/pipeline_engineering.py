import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def build_ml_pipeline():
    print("[INFO] Starting Lab 7: ML Pipeline Engineering...")
    
    # 1. Load Processed Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # 2. Create a Modular Pipeline (Scaler + Model kalipi okate step ga)
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'))
    ])
    
    # 3. Train the Pipeline
    print("[INFO] Training the Pipeline...")
    pipeline.fit(X_train, y_train)
    
    # 4. Evaluate the Pipeline
    y_pred = pipeline.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"[INFO] Pipeline F1 Score: {score:.4f}")
    
    # 5. Save the Modular Pipeline
    os.makedirs('models', exist_ok=True)
    pipeline_path = 'models/happiness_pipeline.pkl'
    joblib.dump(pipeline, pipeline_path)
    print(f"[SUCCESS] Modular Pipeline saved to {pipeline_path}")

if __name__ == "__main__":
    build_ml_pipeline()