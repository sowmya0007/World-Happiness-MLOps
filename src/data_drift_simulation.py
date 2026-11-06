import requests

def simulate_data_drift():
    print("[INFO] Starting Data Drift Simulation...")
    url = "http://localhost:8080/predict"
    
    # 1. Heavily Drifted/Anomalous Data
    # Normal GDP is ~0 to 2, Family is positive. Here we give extreme values.
    drifted_payload = {
        "Standard_Error": 0.03,
        "Economy_GDP_per_Capita": 55.5,  # Extreme value
        "Family": -5.0,                  # Impossible negative value
        "Health_Life_Expectancy": 0.94,
        "Freedom": 0.66,
        "Trust_Government_Corruption": 0.41,
        "Generosity": 0.29,
        "Dystopia_Residual": 2.51
    }
    
    print("[INFO] Sending drifted data to the API...")
    
    try:
        # 2. Call the API
        response = requests.post(url, json=drifted_payload)
        
        # 3. Simulate Drift Detection Logic
        # In enterprise MLOps, tools like 'Evidently AI' or 'Whylogs' do this automatically.
        if drifted_payload["Economy_GDP_per_Capita"] > 10 or drifted_payload["Family"] < 0:
            print("\n[WARNING] DATA DRIFT DETECTED!")
            print("[ALERT] Input features are significantly outside the 2015 baseline distribution.")
            print("[ACTION] Flagging data for review and initiating Model Retraining Workflow...")
            
        # The API will still predict, but we caught the drift in our monitoring layer.
        print(f"\n[INFO] API Prediction: {response.json()['prediction']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    simulate_data_drift()