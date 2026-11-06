import requests
import time

def monitor_api_performance():
    print("[INFO] Starting Lab 12: API Performance Monitoring...")
    # Using port 8080 as discussed
    url = "http://localhost:8080/predict" 
    
    payload = {
        "Standard_Error": 0.03,
        "Economy_GDP_per_Capita": 1.39,
        "Family": 1.34,
        "Health_Life_Expectancy": 0.94,
        "Freedom": 0.66,
        "Trust_Government_Corruption": 0.41,
        "Generosity": 0.29,
        "Dystopia_Residual": 2.51
    }
    
    success_count = 0
    total_requests = 10
    total_time = 0
    
    print("[INFO] Sending 10 requests to Dockerized API on port 8080...")
    for i in range(total_requests):
        try:
            start_time = time.time()
            response = requests.post(url, json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                success_count += 1
                total_time += (end_time - start_time)
                print(f"Request {i+1}: Success, Latency: {(end_time - start_time):.4f}s")
            else:
                print(f"Request {i+1}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"Request {i+1}: Error - {e}")
            
        time.sleep(0.5) 
        
    avg_latency = total_time / success_count if success_count > 0 else 0
    print(f"\n[SUMMARY] Total Requests: {total_requests}")
    print(f"[SUMMARY] Successful: {success_count}")
    print(f"[SUMMARY] Average Latency: {avg_latency:.4f}s")

if __name__ == "__main__":
    monitor_api_performance()