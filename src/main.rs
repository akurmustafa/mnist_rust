use clap::Parser;
use std::time::Instant;

mod utils;
use utils::dataloader::DataLoader;
use utils::losses::{cross_entropy_grad, cross_entropy_loss};
use utils::model::get_linear_model;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to MNIST folder
    #[arg(short, long)]
    mnist: String,

    /// Batch size
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Use same batch for every iteration
    #[arg(short = 's', long, default_value_t = false)]
    same_batch: bool,

    /// Learning rate
    #[arg(short = 'r', long, default_value_t = 0.01)]
    learning_rate: f64,

    /// Print loss every N batches
    #[arg(short = 'p', long, default_value_t = 20)]
    print_every: usize,

    /// Number of training iterations
    #[arg(short = 'i', long, default_value_t = -1)]
    n_iter: i64,
}

fn main() {
    let start = Instant::now();
    let args = Args::parse();

    ////// ------ Parameters from command line -------- //////
    let mnist_folder = args.mnist.as_str();
    let batch_size = args.batch_size;
    let same_batch = args.same_batch;
    let learning_rate = args.learning_rate;
    let print_every = args.print_every;
    let n_iter = args.n_iter;
    ////// ------ End of parameters -------- //////

    let training_csv_path = format!("{}/mnist_train.csv", mnist_folder);
    let test_csv_path = format!("{}/mnist_test.csv", mnist_folder);
    let mut train_dataloader = DataLoader::new(training_csv_path.as_str(), batch_size, same_batch);
    let layers = vec![
        (784, "identity"), // Input layer
        (128, "relu"),     // Hidden layer 1
        (64, "relu"),      // Hidden layer 2
        (10, "identity"),  // Output layer
    ];
    let mut model = get_linear_model(layers);
    let mut count = 0;
    let mut losses = vec![];
    // Training loop
    while let Some((x_batch, y_batch)) = train_dataloader.next_batch() {
        // println!("Batch X shape: {:?}", x_batch.shape());
        // println!("Batch Y shape: {:?}", y_batch.shape());
        let output = model.forward(&x_batch);
        // println!("Output shape: {:?}", output.shape());
        let loss = cross_entropy_loss(&output, &y_batch, true);
        losses.push(loss.to_vec()[0].clone());
        let d_loss = cross_entropy_grad(&output, &y_batch);
        model.backward(&d_loss, learning_rate);
        if count % print_every == 0 {
            println!("Loss: {:?}", loss.to_vec());
            println!("Progress: {}/{}", count, train_dataloader.len());
        }
        count += 1;
        if n_iter != -1 && count as i64 >= n_iter {
            break; // Limit to 100 batches for testing
        }
    }
    // Save losses to a file
    let loss_txt = format!("./losses.txt");
    std::fs::write(
        loss_txt,
        losses
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<String>>()
            .join("\n"),
    )
    .expect("Unable to write losses to file");
    println!("Training completed. Losses saved to losses.txt");

    // Evaluate the model on test data
    // TODO: Add evaluation logic
    let mut test_dataloader = DataLoader::new(test_csv_path.as_str(), batch_size, same_batch);
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
