# World Happiness Prediction - MLOps Pipeline

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline for predicting **Happiness Status** based on the 2015 World Happiness Report dataset. The project demonstrates a structured transition from raw data to a fully registered and production-ready machine learning model across 6 structured experiments.

## 🚀 Project Overview

The goal of this project is to classify a country's happiness status (Sad to Very Happy) using socio-economic factors such as GDP, Health (Life Expectancy), Freedom, Trust, and Generosity. The project is built following strict MLOps principles, ensuring reproducibility, validation, and automated lifecycle management.

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.12
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **MLOps & Tracking:** MLflow
* **Data Validation:** Pandera
* **Version Control:** Git & GitHub

## 📂 Project Structure
```text
happiness-prediction/
│
├── data/
│   ├── raw/                 # Contains happiness_2015.csv
│   └── processed/           # Scaled features, encoded targets, and metadata JSON
│
├── src/
│   ├── preprocess.py                 # Exp 1: Cleans data, creates target, scales/splits
│   ├── train.py                      # Exp 2: Baseline RandomForest training
│   ├── evaluate.py                   # Exp 2: Evaluates weighted accuracy, precision, recall, F1
│   ├── train_mlflow.py               # Exp 4: MLflow tracking (parameters, metrics, plots)
│   ├── validate_data.py              # Exp 5: Pandera schema validation
│   ├── validate_reproducibility.py   # Exp 5: Deterministic execution testing
│   ├── train_registry.py             # Exp 6: Trains and registers a new model version
│   └── automate_lifecycle.py         # Exp 6: Automated staging-to-production promotion
│
├── models/                  # Serialized models (.pkl)
├── artifacts/               # Generated reports, confusion matrix plots
├── logs/                    # Execution logs
├── mlruns/                  # MLflow tracking database (Ignored in Git)
├── requirements.txt         # Project dependencies
└── .gitignore               # Ignored files/environments
🧪 The 6 MLOps Experiments
Experiment 1: ML Project Initialization & Data Prep (src/preprocess.py)
Engineered a multi-class target variable (Happiness Status) from the continuous Happiness Score feature.

Dropped target-leaking columns (Country, Region, Happiness Rank, Happiness Score).

Performed stratified train-test splits and scaled numerical features using StandardScaler.

Saved processed arrays and feature metadata (dataset_metadata.json).

Experiment 2: Baseline Model Development (src/train.py, src/evaluate.py)
Trained a baseline RandomForestClassifier with balanced class weights.

Evaluated multi-class predictions using average='weighted' for Precision, Recall, and F1-score.

Saved the baseline model artifact using joblib.

Experiment 3: Git-Based Version Control
Established a clean version control system using .gitignore to prevent tracking of large datasets, environments (.venv), and local model binaries.

Maintained a clean, traceable commit history for collaborative workflows.

Experiment 4: Experiment Tracking & Reproducibility (src/train_mlflow.py)
Integrated MLflow to automatically track hyperparameters (max_depth, n_estimators).

Logged scalar metrics and generated visual artifacts (ConfusionMatrixDisplay).

Tied dataset metadata and model objects to specific MLflow run IDs.

Experiment 5: Data Validation & Pipeline Quality (src/validate_data.py, src/validate_reproducibility.py)
Implemented strict schema validation using Pandera to ensure incoming data matches expected types and ranges (e.g., GDP >= 0.0, Happiness Rank >= 1).

Wrote a deterministic validation script to prove that fixed random seeds generate 100% identical F1-scores across repeated executions.

Experiment 6: Model Registry & Automated Lifecycle (src/train_registry.py, src/automate_lifecycle.py)
Trained a highly tuned Random Forest model and pushed it directly to the MLflow Model Registry.

Created an automated lifecycle manager that detects new models, pushes them to the Staging environment, and automatically promotes the best version to Production.

💻 How to Run Locally
1. Clone the repository & create a virtual environment:

Bash
git clone [https://github.com/sowmya0007/World-Happiness-MLOps.git](https://github.com/sowmya0007/World-Happiness-MLOps.git)
cd World-Happiness-MLOps
python -m venv .venv
2. Activate the environment:

Windows PowerShell: .venv\Scripts\Activate.ps1

Windows CMD: .venv\Scripts\activate.bat

Linux/Mac: source .venv/bin/activate

3. Install dependencies:

Bash
pip install -r requirements.txt
4. Execute the Pipeline:
Run the scripts in sequence from the root directory to reproduce the entire MLOps workflow:

Bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/train_mlflow.py
python src/validate_data.py
python src/validate_reproducibility.py
python src/train_registry.py
python src/automate_lifecycle.py
5. View MLflow Dashboard:

Bash
mlflow ui
Open http://127.0.0.1:5000 in your web browser to view the tracked experiments, parameters, metrics, and registered models.