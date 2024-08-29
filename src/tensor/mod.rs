use ndarray::{Array, ArrayD, IxDyn};
use num_traits::Float;
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor<T: Float> {
    pub data: Arc<ArrayD<T>>,
    pub requires_grad: bool,
}

impl<T: Float> Tensor<T> {
    pub fn new(data: ArrayD<T>, requires_grad: bool) -> Self {
        Self {
            data: Arc::new(data),
            requires_grad,
        }
    }

    pub fn from_vec(data: Vec<T>, shape: &[usize], requires_grad: bool) -> Self {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Shape mismatch");
        Self::new(array, requires_grad)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }
}