import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_evaluation():
    print("[INFO] Starting Model Evaluation...")
    
    # 1. Load test data and the trained model
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    model = joblib.load('models/random_forest_baseline.pkl')
    
    # 2. Make predictions
    y_pred = model.predict(X_test_final)
    
    # 3. Calculate metrics (Using 'weighted' for multi-class)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n--- World Happiness Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-----------------------------------------------\n")

if __name__ == "__main__":
    run_evaluation()