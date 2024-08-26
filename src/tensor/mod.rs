use ndarray::{Array, IxDyn};
use num_traits::Num;

#[derive(Clone, Debug)]
pub struct Tensor<T: Num> {
    pub data: Array<T, IxDyn>,
    pub requires_grad: bool,
    // We'll add more fields later for autograd
}

impl<T: Num> Tensor<T> {
    pub fn new(data: Array<T, IxDyn>, requires_grad: bool) -> Self {
        Self {
            data,
            requires_grad,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    // More methods will be added here
}
