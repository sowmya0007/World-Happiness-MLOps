from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize FastAPI App
app = FastAPI(title="World Happiness Prediction API")

# 2. Load the trained pipeline and label encoder
try:
    pipeline = joblib.load('models/happiness_pipeline.pkl')
    le = joblib.load('models/label_encoder.pkl')
except Exception as e:
    pipeline = None
    le = None
    print(f"Error loading models: {e}")

# 3. Define the Input Data Schema using Pydantic
class HappinessFeatures(BaseModel):
    Standard_Error: float = 0.03
    Economy_GDP_per_Capita: float = 1.39
    Family: float = 1.34
    Health_Life_Expectancy: float = 0.94
    Freedom: float = 0.66
    Trust_Government_Corruption: float = 0.41
    Generosity: float = 0.29
    Dystopia_Residual: float = 2.51

# 4. Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "active", "message": "World Happiness ML API is up and running!"}

# 5. Prediction Endpoint
@app.post("/predict")
def predict_happiness(features: HappinessFeatures):
    if not pipeline or not le:
        raise HTTPException(status_code=500, detail="ML Models not loaded properly.")
    
    # Format the input data for the model
    input_data = pd.DataFrame([{
        'Standard Error': features.Standard_Error,
        'Economy (GDP per Capita)': features.Economy_GDP_per_Capita,
        'Family': features.Family,
        'Health (Life Expectancy)': features.Health_Life_Expectancy,
        'Freedom': features.Freedom,
        'Trust (Government Corruption)': features.Trust_Government_Corruption,
        'Generosity': features.Generosity,
        'Dystopia Residual': features.Dystopia_Residual
    }])
    
    try:
        # Get prediction from pipeline and decode the text label
        pred_encoded = pipeline.predict(input_data)
        pred_status = le.inverse_transform(pred_encoded)
        
        return {
            "prediction": pred_status[0],
            "message": "Successfully predicted happiness status!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))