pub fn relu(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

pub fn relu_derivative(x: f64) -> f64 {
    // println!("relu_derivative called with x: {}", x);
    if x < 0.0 { 0.0 } else { 1.0 }
}

pub fn softmax(input: &[f64]) -> Vec<f64> {
    // Prevent overflow by subtracting max
    let max = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = input.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

pub fn softmax_derivative(softmax_output: &[f64]) -> Vec<Vec<f64>> {
    let n = softmax_output.len();
    let mut jacobian = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                jacobian[i][j] = softmax_output[i] * (1.0 - softmax_output[i]);
            } else {
                jacobian[i][j] = -softmax_output[i] * softmax_output[j];
            }
        }
    }

    jacobian
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
    }

    #[test]
    fn test_relu_derivative() {
        assert_eq!(relu_derivative(-1.0), 0.0);
        assert_eq!(relu_derivative(0.0), 1.0);
        assert_eq!(relu_derivative(1.0), 1.0);
    }

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let output = softmax(&input);
        assert!((output[0] - 0.09003057).abs() < 1e-8);
        assert!((output[1] - 0.24472847).abs() < 1e-8);
        assert!((output[2] - 0.66524096).abs() < 1e-8);
    }

    #[test]
    fn test_softmax_derivative() {
        let softmax_output = vec![0.09003057, 0.24472847, 0.66524096];
        let jacobian = softmax_derivative(&softmax_output);
        assert_eq!(jacobian.len(), 3);
        assert_eq!(jacobian[0].len(), 3);
        assert!((jacobian[0][0] - 0.08192507).abs() < 1e-4);
        assert!((jacobian[0][1] + 0.02203664).abs() < 1e-4);
        assert!((jacobian[0][2] + 0.05988843).abs() < 1e-4);
    }
}
