import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def run_deterministic_test():
    print("[INFO] Running Reproducibility Validation for World Happiness...")
    
    # Load Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Fixed Parameters
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced"}
    
    # Execution 1
    model_1 = RandomForestClassifier(**params)
    model_1.fit(X_train, y_train)
    score_1 = f1_score(y_test, model_1.predict(X_test), average='weighted', zero_division=0)
    
    # Execution 2 
    model_2 = RandomForestClassifier(**params)
    model_2.fit(X_train, y_train)
    score_2 = f1_score(y_test, model_2.predict(X_test), average='weighted', zero_division=0)
    
    is_reproducible = (score_1 == score_2)
    
    print(f"Execution 1 F1: {score_1:.6f}")
    print(f"Execution 2 F1: {score_2:.6f}")
    if is_reproducible:
        print("[SUCCESS] Pipeline is 100% reproducible.")
    else:
        print("[FAILED] Pipeline is non-deterministic.")

if __name__ == "__main__":
    run_deterministic_test()