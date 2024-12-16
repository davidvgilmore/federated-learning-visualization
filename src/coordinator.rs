use crate::model::{LinearModel, federated_averaging};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use ndarray::{Array1, Array2, ArrayBase, OwnedRepr, Dim};

#[derive(Clone)]
pub struct Coordinator {
    pub model: Arc<RwLock<LinearModel>>,
    pub workers: Arc<RwLock<HashMap<String, usize>>>,
    pub current_epoch: Arc<RwLock<i32>>,
    pub updates_this_epoch: Arc<RwLock<HashMap<String, (LinearModel, usize)>>>,
}

impl Coordinator {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            model: Arc::new(RwLock::new(LinearModel::new(input_dim, output_dim))),
            workers: Arc::new(RwLock::new(HashMap::new())),
            current_epoch: Arc::new(RwLock::new(0)),
            updates_this_epoch: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_worker(&self, worker_id: String, data_size: usize) {
        let mut workers = self.workers.write().await;
        workers.insert(worker_id, data_size);
    }

    pub async fn get_current_model(&self) -> Vec<u8> {
        let model = self.model.read().await;
        model.serialize().unwrap()
    }

    pub async fn submit_update(
        &self,
        worker_id: String,
        parameters: serde_json::Value,
        epoch: i32,
    ) -> Result<(), String> {
        let current_epoch = *self.current_epoch.read().await;
        if epoch != current_epoch {
            return Err("Epoch mismatch".to_string());
        }

        // Convert JSON parameters to Vec<f64>
        let params = parameters.as_array()
            .ok_or("Parameters must be an array")?
            .iter()
            .map(|v| v.as_f64().ok_or("Invalid parameter value"))
            .collect::<Result<Vec<f64>, &str>>()?;

        if params.len() != 3 {  // 2 weights + 1 bias
            return Err("Parameters must have length 3 (2 weights + 1 bias)".to_string());
        }

        // Convert f64 to f32 and split into weights and bias
        let params_f32: Vec<f32> = params.iter().map(|&x| x as f32).collect();
        
        // Split parameters into weights and bias
        let weights_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = 
            Array2::from_shape_vec((2, 1), params_f32[..2].to_vec())
            .map_err(|e| e.to_string())?;
        
        let bias_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> = 
            Array1::from_vec(params_f32[2..].to_vec());

        let model = LinearModel {
            weights: weights_array,
            bias: bias_array,
        };

        let worker_data_size = {
            let workers = self.workers.read().await;
            *workers.get(&worker_id).ok_or("Worker not registered")?
        };

        let mut updates = self.updates_this_epoch.write().await;
        updates.insert(worker_id, (model, worker_data_size));

        // Check if we have all updates
        if updates.len() == self.workers.read().await.len() {
            // Perform federated averaging
            let updates_vec: Vec<_> = updates.drain().map(|(_, v)| v).collect();
            let averaged_model = federated_averaging(updates_vec);
            
            // Update global model
            *self.model.write().await = averaged_model;
            
            // Increment epoch
            let mut current_epoch = self.current_epoch.write().await;
            *current_epoch += 1;
        }

        Ok(())
    }
}
