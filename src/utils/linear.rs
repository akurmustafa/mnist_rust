use super::activations::{relu, relu_derivative};
use super::tensor::Tensor2D;

#[derive(Clone)]
pub struct LinearLayer {
    in_features: usize,
    out_features: usize,
    weights: Tensor2D,
    bias: Tensor2D,
    activation: fn(f64) -> f64,        // Callable activation function
    d_activation: fn(f64) -> f64,      // Derivative of activation function
    cache_before_activation: Tensor2D, // Cache for input before activation
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize, activation: &str) -> Self {
        let weights = Tensor2D::random(out_features, in_features);
        let bias = Tensor2D::random(out_features, 1);
        let activation_fn = match activation {
            "relu" => {
                // Initialize weights and bias for ReLU activation
                // (This is just a placeholder, actual initialization may vary)
                relu
            }
            "identity" => {
                // Identity activation does not change the output
                |x| x
            }
            _ => panic!("Unsupported activation function: {}", activation),
        };
        let d_activation = match activation {
            "relu" => {
                // Initialize weights and bias for ReLU activation
                // (This is just a placeholder, actual initialization may vary)
                relu_derivative
            }
            "identity" => {
                // Identity activation derivative is 1
                |x| 1.0
            }
            _ => panic!("Unsupported activation function: {}", activation),
        };
        let cache_before_activation = Tensor2D::zeros(0, 0); // Placeholder for cache before activation

        LinearLayer {
            in_features,
            out_features,
            weights,
            bias,
            activation: activation_fn,
            d_activation,
            cache_before_activation,
        }
    }

    pub fn set_weights(&mut self, weights: Tensor2D) {
        assert_eq!(
            weights.shape().0,
            self.out_features,
            "Weights rows must match out_features"
        );
        assert_eq!(
            weights.shape().1,
            self.in_features,
            "Weights columns must match in_features"
        );
        self.weights = weights;
    }

    pub fn set_bias(&mut self, bias: Tensor2D) {
        assert_eq!(bias.shape().1, 1, "Bias must have one column");
        assert_eq!(
            bias.shape().0,
            self.out_features,
            "Bias rows must match out_features"
        );
        self.bias = bias;
    }

    pub fn update_weights(&mut self, d_w: &Tensor2D, d_b: &Tensor2D, learning_rate: f64) {
        assert_eq!(
            d_w.shape(),
            (self.out_features, self.in_features),
            "Weight gradient shape mismatch"
        );
        assert_eq!(
            d_b.shape(),
            (self.out_features, 1),
            "Bias gradient shape mismatch"
        );
        self.weights = self.weights.subtract(&d_w.multiply(learning_rate));
        self.bias = self.bias.subtract(&d_b.multiply(learning_rate));
    }

    pub fn forward(&mut self, input: &Tensor2D) -> Tensor2D {
        assert_eq!(
            input.shape().1,
            self.in_features,
            "Input features do not match layer's in_features"
        );
        let (B, _) = input.shape();
        let mut output = Tensor2D::zeros(B, self.out_features);
        let mut cache = Tensor2D::zeros(B, self.out_features);
        for b in 0..B {
            for i in 0..self.out_features {
                let mut sum = 0.0;
                for j in 0..self.in_features {
                    sum += input.get(b, j) * self.weights.get(i, j);
                }
                let value = sum + self.bias.get(i, 0);
                cache.set(b, i, value);
                // Apply activation function
                let act_value = (self.activation)(value);
                output.set(b, i, act_value);
            }
        }
        self.cache_before_activation = cache; // Store cache before activation
        output
    }

    // TODO: Add caching
    pub fn backward(
        &self,
        input: &Tensor2D,
        delta_out: &Tensor2D,
    ) -> (Tensor2D, Tensor2D, Tensor2D) {
        assert_eq!(
            delta_out.shape(),
            self.cache_before_activation.shape(),
            "Delta output shape must match cache before activation shape"
        );
        let cache_data = self.cache_before_activation.to_vec();
        let delta_out_data = delta_out.to_vec();
        // println!("\n\n-------delta out start: {:?}", delta_out_data);
        // println!("-------cache data start: {:?}", cache_data);
        let new_delta_data = cache_data
            .iter()
            .zip(delta_out_data.iter())
            .map(|(c, d)| (self.d_activation)(*c) * d)
            .collect::<Vec<f64>>();
        let (B, N) = delta_out.shape();
        // println!("-------delta out after: {:?}", new_delta_data);
        let delta_out = Tensor2D::new(B, N, new_delta_data);

        // TODO: Add activation gradient also. delta_out should be modified
        // according to the activation result before activation function.
        // This function should compute gradients w.r.t. weights and bias, and input
        // and return them as Tensor2D
        let d_in = delta_out.matmul(&self.weights);
        let d_w = delta_out.transpose().matmul(&input);
        let d_b = delta_out.sum(0).reshape(self.out_features, 1);
        (d_in, d_w, d_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let layer = LinearLayer::new(3, 2, "identity");
        assert_eq!(layer.in_features, 3);
        assert_eq!(layer.out_features, 2);
        assert_eq!(layer.weights.shape(), (2, 3)); // 3 * 2
        assert_eq!(layer.bias.shape(), (2, 1)); // 2
    }

    #[test]
    fn test_forward() {
        let mut layer = LinearLayer::new(3, 2, "identity");
        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor2D::new(1, 3, input_data);
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor2D::new(2, 3, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![7.0, 8.0];
        let bias = Tensor2D::new(2, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let output_expected_data = vec![21.0, 40.0];
        let output_expected = Tensor2D::new(1, 2, output_expected_data);
        assert_eq!(output, output_expected);
    }

    #[test]
    fn test_forward_batch() {
        let mut layer = LinearLayer::new(3, 2, "identity");
        let input_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let input = Tensor2D::new(2, 3, input_data);
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor2D::new(2, 3, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![7.0, 8.0];
        let bias = Tensor2D::new(2, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let output_expected_data = vec![21.0, 40.0, 21.0, 40.0];
        let output_expected = Tensor2D::new(2, 2, output_expected_data);
        assert_eq!(output, output_expected);
    }

    #[test]
    fn test_forward2() {
        let mut layer = LinearLayer::new(2, 1, "identity");
        let input_data = vec![21.0, 40.0];
        let input = Tensor2D::new(1, 2, input_data);
        let weight_data = vec![1.0, 2.0];
        let weights = Tensor2D::new(1, 2, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![3.0];
        let bias = Tensor2D::new(1, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let output_expected_data = vec![104.0];
        let output_expected = Tensor2D::new(1, 1, output_expected_data);
        assert_eq!(output, output_expected);
    }

    #[test]
    fn test_forward2_batch() {
        let mut layer = LinearLayer::new(2, 1, "identity");
        let input_data = vec![21.0, 40.0, 21.0, 40.0];
        let input = Tensor2D::new(2, 2, input_data);
        let weight_data = vec![1.0, 2.0];
        let weights = Tensor2D::new(1, 2, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![3.0];
        let bias = Tensor2D::new(1, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let output_expected_data = vec![104.0, 104.0];
        let output_expected = Tensor2D::new(2, 1, output_expected_data);
        assert_eq!(output, output_expected);
    }

    #[test]
    fn test_forward3() {
        let mut layer = LinearLayer::new(3, 2, "relu");
        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor2D::new(1, 3, input_data);
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor2D::new(2, 3, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![-20.0, 8.0];
        let bias = Tensor2D::new(2, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let output_expected_data = vec![0.0, 40.0];
        let output_expected = Tensor2D::new(1, 2, output_expected_data);
        assert_eq!(output, output_expected);
    }

    #[test]
    fn test_forward3_batch() {
        let mut layer = LinearLayer::new(3, 2, "relu");
        let input_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let input = Tensor2D::new(3, 3, input_data);
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor2D::new(2, 3, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![-20.0, 8.0];
        let bias = Tensor2D::new(2, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let output_expected_data = vec![0.0, 40.0, 0.0, 40.0, 0.0, 40.0];
        let output_expected = Tensor2D::new(3, 2, output_expected_data);
        assert_eq!(output, output_expected);
    }

    #[test]
    fn test_backward() {
        let mut layer = LinearLayer::new(3, 2, "identity");
        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor2D::new(1, 3, input_data);
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor2D::new(2, 3, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![7.0, 8.0];
        let bias = Tensor2D::new(2, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let d_output = Tensor2D::new(1, 2, vec![1.0, 1.0]); // Dummy gradient for output
        assert_eq!(output.shape(), d_output.shape());

        // TODO: Verify values below matches with torch results
        let (d_in, d_w, d_b) = layer.backward(&input, &d_output);
        let d_in_expected = Tensor2D::new(1, 3, vec![5.0, 7.0, 9.0]); // d_in = d_output * weights
        let d_w_expected = Tensor2D::new(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]); // d_w = d_output^T * input
        let d_b_expected = Tensor2D::new(2, 1, vec![1.0, 1.0]); // d_b = sum(d_output, axis=0)
        println!("Input Gradient: {:?}", d_in);
        println!("Weight Gradient: {:?}", d_w);
        println!("Bias Gradient: {:?}", d_b);
        assert_eq!(d_in, d_in_expected);
        assert_eq!(d_w, d_w_expected);
        assert_eq!(d_b, d_b_expected);
        println!("Backward pass test passed!");
    }

    #[test]
    fn test_backward_batch() {
        let mut layer = LinearLayer::new(3, 2, "identity");
        let input_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let input = Tensor2D::new(3, 3, input_data);
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor2D::new(2, 3, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![7.0, 8.0];
        let bias = Tensor2D::new(2, 1, bias_data);
        layer.set_bias(bias);
        let output = layer.forward(&input);
        let d_output = Tensor2D::new(3, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]); // Dummy gradient for output
        assert_eq!(output.shape(), d_output.shape());

        // TODO: Verify values below matches with torch results
        let (d_in, d_w, d_b) = layer.backward(&input, &d_output);
        let d_in_expected = Tensor2D::new(3, 3, vec![5.0, 7.0, 9.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0]); // d_in = d_output * weights
        let d_w_expected = Tensor2D::new(2, 3, vec![3.0, 6.0, 9.0, 3.0, 6.0, 9.0]); // d_w = d_output^T * input
        let d_b_expected = Tensor2D::new(2, 1, vec![3.0, 3.0]); // d_b = sum(d_output, axis=0)
        println!("Input Gradient: {:?}", d_in);
        println!("Weight Gradient: {:?}", d_w);
        println!("Bias Gradient: {:?}", d_b);
        assert_eq!(d_in, d_in_expected);
        assert_eq!(d_w, d_w_expected);
        assert_eq!(d_b, d_b_expected);
        println!("Backward pass test passed!");
    }
}
