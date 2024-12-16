use crate::model::LinearModel;
use ndarray::Array2;
use std::error::Error;

pub struct Worker {
    id: String,
    model: LinearModel,
    data: Array2<f32>,
    labels: Array2<f32>,
}

impl Worker {
    pub fn new(
        id: String,
        input_dim: usize,
        output_dim: usize,
        data: Array2<f32>,
        labels: Array2<f32>,
    ) -> Self {
        Self {
            id,
            model: LinearModel::new(input_dim, output_dim),
            data,
            labels,
        }
    }

    pub fn train_epoch(&mut self, learning_rate: f32) -> f32 {
        let predictions = self.model.forward(&self.data);
        let error = &predictions - &self.labels;
        
        // Compute gradients
        let grad_weights = self.data.t().dot(&error);
        let grad_bias = error.sum_axis(ndarray::Axis(0));
        
        // Update parameters
        self.model.weights -= &(&grad_weights.t() * learning_rate);
        self.model.bias -= &(&grad_bias * learning_rate);
        
        // Return loss
        error.map(|x| x * x).sum() / (self.data.nrows() as f32)
    }

    pub fn get_model_parameters(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        self.model.serialize()
    }

    pub fn update_model(&mut self, parameters: &[u8]) -> Result<(), Box<dyn Error>> {
        self.model = LinearModel::deserialize(parameters)?;
        Ok(())
    }
}
