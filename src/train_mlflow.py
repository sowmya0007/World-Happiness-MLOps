import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

def train_and_track(run_name="RandomForest_Baseline", params=None):
    if params is None:
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced"}

    print(f"\n[INFO] Starting MLflow Run: {run_name} ...")
    
    # 1. Load Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    # Set MLflow Experiment
    mlflow.set_experiment("World_Happiness_Prediction")

    with mlflow.start_run(run_name=run_name):
        # 2. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        # 3. Train Model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Evaluate Predictions
        y_pred = model.predict(X_test)
        
        # 'weighted' is used for multi-class classification metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # 5. Log Metrics to MLflow
        mlflow.log_metrics(metrics)

        # 6. Generate and Log Confusion Matrix Plot
        os.makedirs("artifacts", exist_ok=True)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix - {run_name}")
        cm_path = "artifacts/confusion_matrix.png"
        fig_cm.savefig(cm_path, bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # 7. Log Metadata and Model Artifacts
        if os.path.exists('data/processed/dataset_metadata.json'):
            mlflow.log_artifact('data/processed/dataset_metadata.json', artifact_path="metadata")

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        joblib.dump(model, 'models/random_forest_model.pkl')
        print(f"[SUCCESS] Run '{run_name}' successfully tracked! F1 Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    train_and_track(run_name="RandomForest_V1")