import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

def validate_schema(df):
    print("[INFO] Validating World Happiness dataset schema...")
    
    # Define exact expected columns and rules
    schema = DataFrameSchema({
        "Country": Column(str),
        "Region": Column(str),
        "Happiness Rank": Column(int, Check.ge(1)),
        "Happiness Score": Column(float, Check.in_range(0.0, 10.0)),
        "Economy (GDP per Capita)": Column(float, Check.ge(0.0)),
        "Family": Column(float, Check.ge(0.0)),
        "Health (Life Expectancy)": Column(float, Check.ge(0.0)),
        "Freedom": Column(float, Check.ge(0.0)),
        "Trust (Government Corruption)": Column(float, Check.ge(0.0)),
        "Generosity": Column(float, Check.ge(0.0)),
        "Dystopia Residual": Column(float, Check.ge(0.0))
    })
    
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED.")
        print(err.failure_cases)

if __name__ == "__main__":
    data_path = 'data/raw/happiness_2015.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        validate_schema(df)
    else:
        print(f"[ERROR] File not found: {data_path}")