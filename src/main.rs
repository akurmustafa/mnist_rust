use std::time::Instant;

mod utils;
use utils::dataloader::DataLoader;
use utils::losses::{cross_entropy_grad, cross_entropy_loss};
use utils::model::get_linear_model;

fn main() {
    let start = Instant::now();
    ////// ------Parameters -------- //////
    let training_csv_path = "/Users/akur/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/akurmustafa/docs/courses/2025_2-spring/BMI643/project/mlp/data/mnist/mnist_train.csv";
    let test_csv_path = "/Users/akur/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/akurmustafa/docs/courses/2025_2-spring/BMI643/project/mlp/data/mnist/mnist_test.csv";
    let batch_size = 32;
    let same_batch = false;
    let learning_rate = 0.01;
    let print_every = 20; // Print loss every 20 batches
    let n_iter = 2000; // Number of iteration for training

    ////// ------Parameters -------- //////

    let mut train_dataloader = DataLoader::new(training_csv_path, batch_size, same_batch);
    let layers = vec![
        (784, "identity"), // Input layer
        (128, "relu"),     // Hidden layer 1
        (64, "relu"),      // Hidden layer 2
        (10, "identity"),  // Output layer
    ];
    let mut model = get_linear_model(layers);
    let mut count = 0;
    // Training loop
    while let Some((x_batch, y_batch)) = train_dataloader.next_batch() {
        // println!("Batch X shape: {:?}", x_batch.shape());
        // println!("Batch Y shape: {:?}", y_batch.shape());
        let output = model.forward(&x_batch);
        // println!("Output shape: {:?}", output.shape());
        let loss = cross_entropy_loss(&output, &y_batch, true);
        let d_loss = cross_entropy_grad(&output, &y_batch);
        model.backward(&d_loss, learning_rate);
        if count % print_every == 0 {
            println!("Loss: {:?}", loss.to_vec());
            println!("Progress: {}/{}", count, train_dataloader.len());
        }
        count += 1;
        if count >= n_iter {
            break; // Limit to 100 batches for testing
        }
    }

    // Evaluate the model on test data
    // TODO: Add evaluation logic
    let mut test_dataloader = DataLoader::new(test_csv_path, batch_size, same_batch);
    let mut y_preds = vec![];
    let mut y_targets = vec![];
    let mut count = 0;
    while let Some((x_batch, y_batch)) = test_dataloader.next_batch() {
        let output = model.forward(&x_batch);
        let y_pred_batch = output.argmax(1);
        println!("output shape: {:?}", output.shape());
        println!("y_batch shape: {:?}", y_batch.shape());
        println!("y_batch shape: {:?}", y_pred_batch.shape());
        y_preds.extend(y_pred_batch.to_vec());
        y_targets.extend(y_batch.to_vec());
        count += 1;
        println!("Test batch progress: {}/{}", count, test_dataloader.len());
    }
    // Find the accuracy
    let correct_predictions: usize = y_preds
        .iter()
        .zip(y_targets.iter())
        .filter(|(pred, target)| pred == target)
        .count();
    let accuracy = correct_predictions as f64 / y_preds.len() as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    println!("Completed!");
    let duration = start.elapsed();
    println!("Elapsed time: {:?}", duration);
}
