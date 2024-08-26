use crate::gpu;
use crate::tensor::Tensor;

pub fn add_vectors(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    assert_eq!(a.ndim(), 1, "First argument must be a 1D tensor");
    assert_eq!(b.ndim(), 1, "Second argument must be a 1D tensor");
    assert_eq!(a.shape(), b.shape(), "Vectors must have the same shape");

    let result = gpu::add_vectors_gpu(a.data.as_slice().unwrap(), b.data.as_slice().unwrap())
        .unwrap_or_else(|_| {
            // Fallback to CPU implementation if GPU fails
            a.data.as_slice().unwrap().iter()
                .zip(b.data.as_slice().unwrap().iter())
                .map(|(&x, &y)| x + y)
                .collect()
        });

    Tensor::new(ndarray::Array::from(result).into_dyn(), false)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_add_vectors() {
        let a = Tensor::new(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].into_dyn(), false);
        let b = Tensor::new(array![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].into_dyn(), false);
        let result = add_vectors(&a, &b);
        assert_eq!(result.data, array![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_dyn());
    }
}