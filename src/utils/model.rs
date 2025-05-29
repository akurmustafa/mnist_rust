use super::linear::LinearLayer;
use super::tensor::Tensor2D;

pub struct Sequential {
    layers: Vec<LinearLayer>,
    caches: Vec<Tensor2D>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
            caches: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer: LinearLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &Tensor2D) -> Tensor2D {
        let mut output = input.clone();
        self.caches.clear();
        self.caches.push(output.clone());
        for layer in &mut self.layers {
            output = layer.forward(&output);
            self.caches.push(output.clone());
        }
        output
    }

    pub fn backward(&mut self, out_gradient: &Tensor2D, learning_rate: f64) -> () {
        let mut gradient = out_gradient.clone();
        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];
            let cache = &self.caches[i];
            let (d_in, d_w, d_b) = layer.backward(cache, &gradient);
            gradient = d_in;
            layer.update_weights(&d_w, &d_b, learning_rate);
        }
    }
}

pub fn get_linear_model(layers: Vec<(usize, &str)>) -> Sequential {
    let mut model = Sequential::new();
    for i in 0..layers.len() - 1 {
        let (in_features, _) = layers[i];
        let (out_features, activation) = layers[i + 1];
        // TODO: Add random weight initialization
        let layer = LinearLayer::new(in_features, out_features, activation);
        model.add_layer(layer);
    }
    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tensor::Tensor2D;

    #[test]
    fn test_sequential_forward() {
        // Init sequential model
        let mut model = Sequential::new();
        // Input
        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor2D::new(1, 3, input_data);

        // Layer 1
        let mut layer = LinearLayer::new(3, 2, "identity");
        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor2D::new(2, 3, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![7.0, 8.0];
        let bias = Tensor2D::new(2, 1, bias_data);
        layer.set_bias(bias);
        model.add_layer(layer);

        // Layer 2
        let mut layer = LinearLayer::new(2, 1, "identity");
        let weight_data = vec![1.0, 2.0];
        let weights = Tensor2D::new(1, 2, weight_data);
        layer.set_weights(weights);
        let bias_data = vec![3.0];
        let bias = Tensor2D::new(1, 1, bias_data);
        layer.set_bias(bias);
        model.add_layer(layer);

        // Forward pass
        let output = model.forward(&input);
        let output_expected_data = vec![104.0];
        let output_expected = Tensor2D::new(1, 1, output_expected_data);
        assert_eq!(output, output_expected);
    }
}
