import pandas as pd
import numpy as np

def simulate_failure():
    print("[INFO] Starting Lab 9: Pipeline Failure Handling and Recovery...")
    
    try:
        # 1. Load Data
        df = pd.read_csv('data/raw/happiness_2015.csv')
        
        # 2. Inject Failures (Corrupt the data purposely)
        print("[INFO] Injecting data corruption (Negative GDP & Missing Values)...")
        df_corrupted = df.copy()
        
        # Injecting a negative GDP which is impossible
        df_corrupted.loc[0, 'Economy (GDP per Capita)'] = -5.0 
        # Injecting a missing value
        df_corrupted.loc[1, 'Happiness Score'] = np.nan 
        
        # 3. Simulate Pipeline Validation Check
        print("[INFO] Running pre-flight data validation checks...")
        
        if df_corrupted['Economy (GDP per Capita)'].min() < 0:
            raise ValueError("Schema Error: Negative GDP per Capita is not allowed!")
            
        if df_corrupted['Happiness Score'].isnull().any():
            raise ValueError("Data Error: Missing values detected in Happiness Score!")
            
        print("[SUCCESS] Data is clean. Proceeding to training...")
        
    except Exception as e:
        print(f"[ERROR] Pipeline execution halted: {e}")
        print("[INFO] Initiating Recovery Workflow...")
        print("[INFO] Recovery: Alerting Data Engineering team and rolling back to last stable model.")
        
    print("[INFO] Lab 9 Failure simulation safely completed.")

if __name__ == "__main__":
    simulate_failure()