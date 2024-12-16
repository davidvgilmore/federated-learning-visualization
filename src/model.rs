use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Clone, Serialize, Deserialize)]
pub struct LinearModel {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

impl LinearModel {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
            rng.gen::<f32>() * 0.1
        });
        
        let bias = Array1::from_shape_fn(output_dim, |_| {
            rng.gen::<f32>() * 0.1
        });

        Self { weights, bias }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.weights.t()) + &self.bias
    }

    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(serde_json::to_vec(&self)?)
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

pub fn federated_averaging(models: Vec<(LinearModel, usize)>) -> LinearModel {
    let total_samples: usize = models.iter().map(|(_, size)| size).sum();
    
    let (first_model, _) = &models[0];
    let mut avg_weights = Array2::zeros(first_model.weights.raw_dim());
    let mut avg_bias = Array1::zeros(first_model.bias.raw_dim());

    for (model, size) in models {
        let weight = size as f32 / total_samples as f32;
        avg_weights += &(model.weights * weight);
        avg_bias += &(model.bias * weight);
    }

    LinearModel {
        weights: avg_weights,
        bias: avg_bias,
    }
}