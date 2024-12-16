mod coordinator;
mod model;
mod worker;

use ndarray::{Array2, arr2};
use crate::worker::Worker;

fn main() {
    // Test data for worker 1
    let data1 = arr2(&[
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ]);
    let labels1 = arr2(&[
        [3.0],
        [6.0],
        [9.0],
    ]);

    // Test data for worker 2
    let data2 = arr2(&[
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ]);
    let labels2 = arr2(&[
        [6.0],
        [9.0],
        [12.0],
    ]);

    // Create workers
    let mut worker1 = Worker::new(
        "worker1".to_string(),
        2,  // input dimension
        1,  // output dimension
        data1,
        labels1,
    );

    let mut worker2 = Worker::new(
        "worker2".to_string(),
        2,
        1,
        data2,
        labels2,
    );

    // Train for a few epochs
    println!("Training worker 1:");
    for epoch in 0..5 {
        let loss = worker1.train_epoch(0.01);
        println!("Epoch {}: Loss = {}", epoch, loss);
    }

    println!("\nTraining worker 2:");
    for epoch in 0..5 {
        let loss = worker2.train_epoch(0.01);
        println!("Epoch {}: Loss = {}", epoch, loss);
    }

    // Serialize and deserialize test
    match worker1.get_model_parameters() {
        Ok(params) => {
            println!("\nModel parameters serialization successful");
            match worker2.update_model(&params) {
                Ok(_) => println!("Model parameters update successful"),
                Err(e) => println!("Error updating model parameters: {}", e),
            }
        },
        Err(e) => println!("Error serializing model parameters: {}", e),
    }
}
