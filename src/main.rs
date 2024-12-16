mod coordinator;
mod model;
mod worker;

use axum::{
    routing::{get, post},
    Router, Json, extract::State,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use tracing_subscriber;

// API Types
#[derive(Serialize)]
struct TrainingStatus {
    current_epoch: i32,
    worker_losses: HashMap<String, f32>,
    global_loss: Option<f32>,
    active_workers: Vec<String>,
}

#[derive(Deserialize)]
struct WorkerUpdate {
    worker_id: String,
    loss: f32,
    parameters: serde_json::Value,  // Accept any JSON value and validate in coordinator
}

#[derive(Clone)]
struct AppState {
    coordinator: Arc<coordinator::Coordinator>,
    worker_losses: Arc<RwLock<HashMap<String, f32>>>,
    global_loss: Arc<RwLock<Option<f32>>>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create coordinator with 2D input and 1D output
    let coordinator = Arc::new(coordinator::Coordinator::new(2, 1));
    
    let app_state = AppState {
        coordinator,
        worker_losses: Arc::new(RwLock::new(HashMap::new())),
        global_loss: Arc::new(RwLock::new(None)),
    };

    // Build our application with routes
    let app = Router::new()
        .route("/status", get(get_status))
        .route("/register_worker", post(register_worker))
        .route("/submit_update", post(submit_update))
        .with_state(app_state);

    // Run it
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    println!("Listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn get_status(
    State(state): State<AppState>,
) -> Json<TrainingStatus> {
    let current_epoch = *state.coordinator.current_epoch.read().await;
    let worker_losses = state.worker_losses.read().await.clone();
    let global_loss = *state.global_loss.read().await;
    let workers = state.coordinator.workers.read().await;
    let active_workers = workers.keys().cloned().collect();

    Json(TrainingStatus {
        current_epoch,
        worker_losses,
        global_loss,
        active_workers,
    })
}

async fn register_worker(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let worker_id = payload["worker_id"].as_str().unwrap().to_string();
    let data_size = payload["data_size"].as_u64().unwrap() as usize;

    state.coordinator.register_worker(worker_id.clone(), data_size).await;

    Json(serde_json::json!({
        "status": "success",
        "message": format!("Worker {} registered", worker_id)
    }))
}

async fn submit_update(
    State(state): State<AppState>,
    Json(update): Json<WorkerUpdate>,
) -> Json<serde_json::Value> {
    let current_epoch = *state.coordinator.current_epoch.read().await;
    
    match state.coordinator.submit_update(
        update.worker_id.clone(),
        update.parameters,
        current_epoch,
    ).await {
        Ok(()) => {
            // Update worker loss
            state.worker_losses.write().await.insert(update.worker_id, update.loss);
            
            // Check if we completed an epoch
            let updates = state.coordinator.updates_this_epoch.read().await;
            if updates.len() == state.coordinator.workers.read().await.len() {
                // Calculate and store global loss
                let total_loss: f32 = state.worker_losses.read().await.values().sum();
                let avg_loss = total_loss / state.worker_losses.read().await.len() as f32;
                *state.global_loss.write().await = Some(avg_loss);
            }

            Json(serde_json::json!({
                "status": "success",
                "message": "Update accepted"
            }))
        },
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": e
        }))
    }
}
