import mlflow
from mlflow.tracking import MlflowClient

def automate_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")
    
    client = MlflowClient()
    model_name = "World_Happiness_Production_Model"
    
    try:
        # Get all versions of our registered model
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        # 1. Push newly tracked models to 'Staging'
        for mv in all_versions:
            if mv.current_stage == "None":
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                print(f"[INFO] Version {mv.version} moved to Staging.")
                
        # 2. Promote the latest Staging model to 'Production'
        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]
        
        if staging_models:
            best_version = staging_models[0].version
            print(f"[INFO] Promoting Version {best_version} to Production...")
            client.transition_model_version_stage(
                name=model_name, version=best_version, stage="Production", archive_existing_versions=True
            )
            print("[SUCCESS] Model successfully deployed to Production!")
            
    except Exception as e:
        print(f"[ERROR] Failed to automate lifecycle: {e}")

if __name__ == "__main__":
    automate_lifecycle()