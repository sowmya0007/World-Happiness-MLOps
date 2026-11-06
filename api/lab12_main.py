from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Lab 12: API Monitoring & Data Drift UI")

class HappinessFeatures(BaseModel):
    Standard_Error: float = 0.03
    Economy_GDP_per_Capita: float = 1.39
    Family: float = 1.34
    Health_Life_Expectancy: float = 0.94
    Freedom: float = 0.66
    Trust_Government_Corruption: float = 0.41
    Generosity: float = 0.29
    Dystopia_Residual: float = 2.51

@app.post("/predict_with_drift_monitoring")
def predict_and_monitor(features: HappinessFeatures):
    drift_detected = False
    alerts = []
    
    if features.Economy_GDP_per_Capita > 10 or features.Economy_GDP_per_Capita < 0:
        drift_detected = True
        alerts.append("GDP value is out of expected bounds (0-10).")
        
    if features.Family < 0:
        drift_detected = True
        alerts.append("Family value cannot be negative.")
        
    return {
        "status": "Success",
        "data_drift_detected": drift_detected,
        "alerts": alerts,
        "action_required": "Trigger Lab 13 Retraining Pipeline" if drift_detected else "None"
    }