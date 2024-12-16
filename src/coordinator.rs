use crate::model::{LinearModel, federated_averaging};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct Coordinator {
    model: Arc<RwLock<LinearModel>>,
    workers: Arc<RwLock<HashMap<String, usize>>>,
    current_epoch: Arc<RwLock<i32>>,
    updates_this_epoch: Arc<RwLock<HashMap<String, (LinearModel, usize)>>>,
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
        parameters: Vec<u8>,
        epoch: i32,
    ) -> Result<(), String> {
        let current_epoch = *self.current_epoch.read().await;
        if epoch != current_epoch {
            return Err("Epoch mismatch".to_string());
        }

        let model = LinearModel::deserialize(&parameters)
            .map_err(|e| e.to_string())?;

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