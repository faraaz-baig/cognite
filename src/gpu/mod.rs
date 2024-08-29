mod memory_pool;

use ocl::ProQue;
use std::sync::Arc;
pub use self::memory_pool::MemoryPool;
use lazy_static::lazy_static;

lazy_static! {
    static ref PRO_QUE: Arc<ProQue> = Arc::new(ProQue::builder()
        .src(r#"
            __kernel void add(__global const float* a, __global const float* b, __global float* c) {
                int gid = get_global_id(0);
                c[gid] = a[gid] + b[gid];
            }
        "#)
        .build()
        .expect("Failed to build ProQue"));

    pub static ref MEMORY_POOL: MemoryPool = MemoryPool::new(PRO_QUE.clone());
}

pub fn add_vectors_gpu(a: &[f32], b: &[f32]) -> Result<Vec<f32>, ocl::Error> {
    let n = a.len();
    
    let buffer_a = MEMORY_POOL.get_buffer(n)?;
    let buffer_b = MEMORY_POOL.get_buffer(n)?;
    let buffer_c = MEMORY_POOL.get_buffer(n)?;

    buffer_a.write(a).enq()?;
    buffer_b.write(b).enq()?;

    let kernel = PRO_QUE.kernel_builder("add")
        .arg(&*buffer_a)
        .arg(&*buffer_b)
        .arg(&*buffer_c)
        .global_work_size(n)
        .build()?;

    unsafe { kernel.enq()?; }

    let mut result = vec![0.0f32; n];
    buffer_c.read(&mut result).enq()?;

    // Return buffers to the pool
    MEMORY_POOL.return_buffer(buffer_a)?;
    MEMORY_POOL.return_buffer(buffer_b)?;
    MEMORY_POOL.return_buffer(buffer_c)?;

    Ok(result)
}