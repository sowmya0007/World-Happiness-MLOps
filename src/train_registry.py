import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    
    # 1. Set up MLflow
    mlflow.set_experiment("World_Happiness_Prediction")
    
    # 2. Load Processed Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # 3. Define New Hyperparameters
    params = {"n_estimators": 200, "max_depth": 15, "random_state": 42, "class_weight": "balanced"}
    
    with mlflow.start_run(run_name="RandomForest_Registry_Run") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")
        
        # 4. Train Model
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)
        
        # 5. Evaluate
        y_pred = rf_model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        mlflow.log_metrics(metrics)
        
        # 6. LOG AND REGISTER THE MODEL
        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="random_forest_model",
            registered_model_name="World_Happiness_Production_Model" 
        )
        
        print(f"[SUCCESS] Metrics logged: F1 = {metrics['f1_score']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: 'World_Happiness_Production_Model'")

if __name__ == "__main__":
    train_and_register_model()