import requests
import numpy as np
import time
import json
import base64

API_URL = "http://localhost:3000"

class SimulatedWorker:
    def __init__(self, worker_id, data_size=100):
        self.worker_id = worker_id
        self.data_size = data_size
        self.current_loss = 10.0  # Initial loss
        
    def register(self):
        response = requests.post(
            f"{API_URL}/register_worker",
            json={"worker_id": self.worker_id, "data_size": self.data_size}
        )
        return response.json()
        
    def train_step(self):
        # Simulate training by decreasing loss
        self.current_loss *= 0.8
        
        # Create dummy parameters (2x1 linear model)
        weights = np.random.randn(2, 1).astype(np.float32)
        bias = np.random.randn(1).astype(np.float32)
        
        # Create a simple model structure
        model_params = {
            "weights": weights.tolist(),
            "bias": bias.tolist()
        }
        
        try:
            # Flatten the parameters into a single array
            flattened_params = np.concatenate([
                weights.reshape(-1),  # 2x1 -> 2
                bias.reshape(-1)      # 1x1 -> 1
            ]).tolist()
            
            response = requests.post(
                f"{API_URL}/submit_update",
                json={
                    "worker_id": self.worker_id,
                    "loss": float(self.current_loss),
                    "parameters": flattened_params
                }
            )
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Response content: {response.content}")
            return {"status": "error", "message": str(e)}

def main():
    # Create workers
    workers = [
        SimulatedWorker("worker1", 100),
        SimulatedWorker("worker2", 150),
        SimulatedWorker("worker3", 120)
    ]
    
    # Register workers
    for worker in workers:
        print(f"Registering {worker.worker_id}...")
        print(worker.register())
    
    # Training loop
    for epoch in range(10):
        print(f"\nEpoch {epoch}")
        for worker in workers:
            print(f"{worker.worker_id} training...")
            result = worker.train_step()
            print(result)
            time.sleep(1)  # Add delay between worker updates

if __name__ == "__main__":
    main()
