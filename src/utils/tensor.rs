use std::fmt::Display;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor2D {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Tensor2D {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length does not match matrix dimensions"
        );
        Tensor2D { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor2D {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let seed: u64 = 23;
        let mut rng = StdRng::seed_from_u64(seed);
        let range = -0.01..0.01;
        let data = (0..rows * cols)
            .map(|_| rng.gen_range(range.clone()))
            .collect();
        Tensor2D { rows, cols, data }
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    pub fn get_rows(&self, row_indices: &[usize]) -> Self {
        let mut row_data = Vec::new();
        for &row in row_indices {
            assert!(row < self.rows, "Row index out of bounds");
            row_data.extend(self.get_row_helper(row));
        }
        Self::new(row_indices.len(), self.cols, row_data)
    }

    pub fn get_row(&self, row: usize) -> Self {
        Self::new(1, self.cols, self.get_row_helper(row))
    }

    fn get_row_helper(&self, row: usize) -> Vec<f64> {
        let start = row * self.cols;
        let end = start + self.cols;
        self.data[start..end].to_vec()
    }

    pub fn get_col(&self, col: usize) -> Self {
        let col_data = (0..self.rows).map(|row| self.get(row, col)).collect();
        Self::new(self.rows, 1, col_data)
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }

    pub fn matmul(&self, other: &Tensor2D) -> Tensor2D {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions do not match for multiplication"
        );
        let mut result = Tensor2D::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn argmax(&self, axis: usize) -> Tensor2D {
        if axis == 0 {
            let mut result = Tensor2D::zeros(1, self.cols);
            for j in 0..self.cols {
                let col_vec = self.get_col(j).to_vec();
                let max_idx = max_indices_helper(&col_vec);
                result.set(0, j, max_idx as f64);
            }
            result
        } else if axis == 1 {
            let mut result = Tensor2D::zeros(self.rows, 1);
            for i in 0..self.rows {
                let max_idx = max_indices_helper(&self.data[i * self.cols..(i + 1) * self.cols]);
                result.set(i, 0, max_idx as f64);
            }
            result
        } else {
            panic!("Invalid axis: {}", axis);
        }
    }

    pub fn sum(&self, axis: usize) -> Tensor2D {
        if axis == 0 {
            let mut result = Tensor2D::zeros(1, self.cols);
            for j in 0..self.cols {
                let sum: f64 = (0..self.rows).map(|i| self.get(i, j)).sum();
                result.set(0, j, sum);
            }
            result
        } else if axis == 1 {
            let mut result = Tensor2D::zeros(self.rows, 1);
            for i in 0..self.rows {
                let sum: f64 = (0..self.cols).map(|j| self.get(i, j)).sum();
                result.set(i, 0, sum);
            }
            result
        } else {
            panic!("Invalid axis: {}", axis);
        }
    }

    pub fn mean(&self, axis: usize) -> Tensor2D {
        let sum = self.sum(axis);
        if axis == 0 {
            sum.divide(self.rows as f64)
        } else if axis == 1 {
            sum.divide(self.cols as f64)
        } else {
            panic!("Invalid axis: {}", axis);
        }
    }

    fn divide(&self, denominator: f64) -> Tensor2D {
        assert!(denominator != 0.0, "Cannot divide by zero");
        let mut result = self.clone();
        for value in &mut result.data {
            *value /= denominator;
        }
        result
    }

    pub fn multiply(&self, scalar: f64) -> Tensor2D {
        let mut result = self.clone();
        for value in &mut result.data {
            *value *= scalar;
        }
        result
    }

    pub fn subtract(&self, other: &Tensor2D) -> Tensor2D {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shapes do not match for subtraction"
        );
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] -= other.data[i];
        }
        result
    }

    pub fn transpose(&self) -> Tensor2D {
        let mut transposed = Tensor2D::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.set(j, i, self.get(i, j));
            }
        }
        transposed
    }

    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> Tensor2D {
        assert_eq!(
            self.rows * self.cols,
            new_rows * new_cols,
            "New shape does not match total number of elements"
        );
        Tensor2D::new(new_rows, new_cols, self.data.clone())
    }
}

impl Display for Tensor2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols - 1 {
                write!(f, "{:2.2}, ", self.get(i, j))?;
            }
            write!(f, "{:2.2}", self.get(i, self.cols - 1))?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Eq for Tensor2D {}

fn max_indices_helper(slice: &[f64]) -> usize {
    if slice.is_empty() {
        panic!("Cannot find max index of an empty slice");
    }
    slice.iter().enumerate().fold(0, |max_index, (i, &value)| {
        if value > slice[max_index] {
            i
        } else {
            max_index
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Tensor2D::zeros(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 6);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn test_matrix_set_get() {
        let mut m = Tensor2D::zeros(2, 2);
        m.set(0, 1, 5.0);
        assert_eq!(m.get(0, 1), 5.0);
    }

    #[test]
    fn test_get_row() {
        let mut m = Tensor2D::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let row = m.get_row(0).to_vec();
        assert_eq!(row, vec![1.0, 2.0]);
        let row = m.get_row(1).to_vec();
        assert_eq!(row, vec![3.0, 4.0]);
    }

    #[test]
    fn test_get_col() {
        let mut m = Tensor2D::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let col = m.get_col(0).to_vec();
        assert_eq!(col, vec![1.0, 3.0]);
        let col = m.get_col(1).to_vec();
        assert_eq!(col, vec![2.0, 4.0]);
    }

    #[test]
    fn test_matrix_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m = Tensor2D::new(2, 2, data);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 3.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Tensor2D::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor2D::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = a.matmul(&b);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.get(0, 0), 58.0);
        assert_eq!(result.get(0, 1), 64.0);
        assert_eq!(result.get(1, 0), 139.0);
        assert_eq!(result.get(1, 1), 154.0);
    }

    #[test]
    fn test_matrix_display() {
        let m = Tensor2D::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let display_string = format!("{}", m);
        println!("{}", display_string);
        assert_eq!(display_string, "1.00, 2.00\n3.00, 4.00\n");
    }

    #[test]
    fn test_matrix_shape() {
        let m = Tensor2D::new(3, 4, vec![1.0; 12]);
        assert_eq!(m.shape(), (3, 4));
    }

    #[test]
    fn test_matrix_equality() {
        let m1 = Tensor2D::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Tensor2D::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m3 = Tensor2D::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(m1, m2);
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_matrix_sum() {
        let m = Tensor2D::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let sum_row = m.sum(0);
        assert_eq!(sum_row.get(0, 0), 5.0);
        assert_eq!(sum_row.get(0, 1), 7.0);
        assert_eq!(sum_row.get(0, 2), 9.0);

        let sum_col = m.sum(1);
        assert_eq!(sum_col.get(0, 0), 6.0);
        assert_eq!(sum_col.get(1, 0), 15.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Tensor2D::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = m.transpose();
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed.get(0, 0), 1.0);
        assert_eq!(transposed.get(0, 1), 4.0);
        assert_eq!(transposed.get(1, 0), 2.0);
        assert_eq!(transposed.get(1, 1), 5.0);
        assert_eq!(transposed.get(2, 0), 3.0);
        assert_eq!(transposed.get(2, 1), 6.0);
    }

    #[test]
    fn test_max_indices() {
        let data = vec![1.0, 3.0, 2.0];
        assert_eq!(
            max_indices_helper(&data),
            1,
            "Max index should be 1 for [1.0, 3.0, 2.0]"
        );
        let data = vec![5.0, 5.0, 5.0];
        assert_eq!(
            max_indices_helper(&data),
            0,
            "Max index should be 0 for [5.0, 5.0, 5.0]"
        );
        let data = vec![1.0];
        assert_eq!(
            max_indices_helper(&data),
            0,
            "Max index should be 0 for single element [1.0]"
        );
    }
}
