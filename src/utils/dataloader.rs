use std::fs::File;

use super::tensor::Tensor2D;

use rand::prelude::SliceRandom;
use rand::thread_rng;

pub struct DataLoader {
    data: Tensor2D,
    labels: Tensor2D,
    batch_size: usize,
    current_index: usize,
    n_samples: usize,
    shuffled_indices: Vec<usize>,
    same_batch: bool,
}

impl DataLoader {
    pub fn new(csv_path: &str, batch_size: usize, same_batch: bool) -> Self {
        let file = File::open(csv_path).expect("Failed to open CSV file");
        let mut rdr = csv::Reader::from_reader(file);

        let mut input_data = Vec::new();
        let mut label_data = Vec::new();
        for result in rdr.records() {
            let record = result.expect("Failed to read record");
            let values: Vec<f64> = record
                .iter()
                .map(|s| s.parse().expect("Failed to parse value"))
                .collect();
            label_data.push(values[0]); // the first value is the label
            input_data.extend(values[1..].to_vec());
            // println!("{:?}", values.len());
            // println!("{:?}", record);
        }
        let n_rows = label_data.len();
        let n_cols = input_data.len() / n_rows;
        println!("n_rows: {}, n_cols: {}", n_rows, n_cols);
        let data = Tensor2D::new(n_rows, n_cols, input_data);
        let labels = Tensor2D::new(n_rows, 1, label_data);
        // create a shuffled indices vector
        let mut indices: Vec<usize> = (0..n_rows).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        DataLoader {
            data,
            labels,
            batch_size,
            current_index: 0,
            n_samples: n_rows,
            shuffled_indices: indices,
            same_batch,
        }
    }

    pub fn next_batch(&mut self) -> Option<(Tensor2D, Tensor2D)> {
        if self.current_index >= self.n_samples {
            return None; // No more data
        }

        let end_index = (self.current_index + self.batch_size).min(self.n_samples);
        let batch_indices = &self.shuffled_indices[self.current_index..end_index];
        let batch_data = self.data.get_rows(batch_indices);
        let batch_labels = self.labels.get_rows(batch_indices);
        if !self.same_batch {
            self.current_index = end_index;
        }

        Some((batch_data, batch_labels))
    }

    pub fn len(&self) -> usize {
        self.n_samples / self.batch_size + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader() {
        let mut dataloader = DataLoader::new("./data/mnist/mnist_test.csv", 32, false);
        assert_eq!(dataloader.batch_size, 32);

        for (x_batch, y_batch) in dataloader.next_batch() {
            assert_eq!(x_batch.shape(), (32, 784));
            assert_eq!(y_batch.shape(), (32, 1));
            break; // Just test the first batch
        }

        let mut dataloader = DataLoader::new("./data/mnist/mnist_test.csv", 32, false);
        let mut count = 0;
        while let Some((_x_batch, _y_batch)) = dataloader.next_batch() {
            count += 1;
            println!("Batch count: {}", count);
        }
        assert_eq!(count, 10000 / 32 + 1); // 10000 samples in MNIST test set, last batch may be smaller
        assert_eq!(dataloader.len(), 10000 / 32 + 1); // 10000 samples in MNIST test set, last batch may be smaller
    }
}
