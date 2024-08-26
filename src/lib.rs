
#[macro_use]
extern crate lazy_static;
pub mod tensor;
pub mod ops;
pub mod autograd;
pub mod optimizer;
pub mod gpu;

pub use tensor::Tensor;
pub use ops::*;
pub use autograd::*;
pub use optimizer::*;
pub use gpu::*;
