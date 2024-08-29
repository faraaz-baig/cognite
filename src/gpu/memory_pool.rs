use ocl::{ProQue, Buffer, Result as OclResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct MemoryPool {
    pro_que: Arc<ProQue>,
    pools: Mutex<HashMap<usize, Vec<Arc<Buffer<f32>>>>>,
}

impl MemoryPool {
    pub fn new(pro_que: Arc<ProQue>) -> Self {
        MemoryPool {
            pro_que,
            pools: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_buffer(&self, size: usize) -> OclResult<Arc<Buffer<f32>>> {
        let mut pools = self.pools.lock().map_err(|_| ocl::Error::from("Failed to lock mutex"))?;
        if let Some(pool) = pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                return Ok(buffer);
            }
        }
        let buffer = self.pro_que.create_buffer::<f32>()?;
        Ok(Arc::new(buffer))
    }

    pub fn return_buffer(&self, buffer: Arc<Buffer<f32>>) -> OclResult<()> {
        let size = buffer.len();
        let mut pools = self.pools.lock().map_err(|_| ocl::Error::from("Failed to lock mutex"))?;
        pools.entry(size).or_insert_with(Vec::new).push(buffer);
        Ok(())
    }
}