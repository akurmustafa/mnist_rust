use super::tensor::Tensor2D;

pub fn cross_entropy_loss(logits: &Tensor2D, targets: &Tensor2D, reduce: bool) -> Tensor2D {
    let (B, N) = logits.shape();
    assert_eq!(targets.shape(), (B, 1), "Targets must be a column vector");
    let mut loss = Tensor2D::zeros(B, 1);

    for i in 0..B {
        let logit = logits.get_row(i).to_vec();
        let target = targets.get(i, 0) as usize;
        assert!(target < N, "Target index out of bounds");
        // Compute the softmax
        let max_logit = logit.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logit.iter().map(|&x| (x - max_logit).exp()).sum();
        let softmax: Vec<f64> = logit
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect();

        // Compute the loss
        loss.set(i, 0, -softmax[target].ln());
    }
    if reduce { loss.mean(0) } else { loss }
}

pub fn cross_entropy_grad(logits: &Tensor2D, targets: &Tensor2D) -> Tensor2D {
    let (B, N) = logits.shape();
    assert_eq!(targets.shape(), (B, 1), "Targets must be a column vector");

    let mut grad = Tensor2D::zeros(B, N);

    for i in 0..B {
        let logit = logits.get_row(i).to_vec();
        let target = targets.get(i, 0) as usize;
        assert!(target < N, "Target index out of bounds");

        // Compute softmax
        let max_logit = logit.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logit.iter().map(|&x| (x - max_logit).exp()).sum();
        let softmax: Vec<f64> = logit
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect();

        // Gradient: softmax - one_hot(target)
        for j in 0..N {
            let mut grad_val = softmax[j];
            if j == target {
                grad_val -= 1.0;
            }
            grad.set(i, j, grad_val / (B as f64));
        }
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss() {
        let logits = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let logits = Tensor2D::new(2, 3, logits);
        let labels = vec![0.0, 1.0];
        let labels = Tensor2D::new(2, 1, labels);

        let loss = cross_entropy_loss(&logits, &labels, false);
        let loss_data = loss.to_vec();
        println!("Loss: {:?}", loss_data);
        assert_eq!(loss_data.len(), 2);
        assert!((loss_data[0] - 2.4076).abs() < 1e-4);
        assert!((loss_data[1] - 1.4076).abs() < 1e-4);

        let loss = cross_entropy_loss(&logits, &labels, true);
        let loss_data = loss.to_vec();
        assert_eq!(loss_data.len(), 1);
        println!("Loss: {:?}", loss_data);
        assert!((loss_data[0] - 1.9076).abs() < 1e-4);
    }

    #[test]
    fn test_cross_entropy_grad() {
        let logits = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let logits = Tensor2D::new(2, 3, logits);
        let labels = vec![0.0, 1.0];
        let labels = Tensor2D::new(2, 1, labels);

        let grad = cross_entropy_grad(&logits, &labels);
        let grad_data = grad.to_vec();
        println!("Gradient: {:?}", grad_data);
        assert_eq!(grad_data.len(), 6);
        assert!((grad_data[0] + 0.4550).abs() < 1e-4);
        assert!((grad_data[1] - 0.1224).abs() < 1e-4);
        assert!((grad_data[2] - 0.3326).abs() < 1e-4);
        assert!((grad_data[3] - 0.0450).abs() < 1e-4);
        assert!((grad_data[4] + 0.3776).abs() < 1e-4);
        assert!((grad_data[5] - 0.3326).abs() < 1e-4);
    }
}
